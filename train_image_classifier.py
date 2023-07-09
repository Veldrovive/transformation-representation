"""
Training the image classifier:
The image classifier takes as input the image and the context vector and outputs a probability distribution over all classes.
In order to generate the context vector, for each batch we:
1. embed all images in the each sample
2. Compute the combined embedding using the gamma model
3. Classify each of the images_per_sample*batch_size images using the same combined embedding for each sample
    1. We may want to consider using a leave n out approach for generating the combined embedding to insert more variation into the training
4. Compute the cross entropy loss for each image and backpropagate
"""

import subprocess
from pathlib import Path
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from models.image_embedder import SimpleImgClassifierLateFusion, ResNetClassifierAttentionFusion, ResNetImgClassifierLateFusion
from train_contrastive_transformation import create_model_and_load_checkpoint 
from transformations import image_transformation
from transformations.image_dataloader import create_image_transformation_dataset

WANDB_PROJECT_NAME = "transformed-image-classification"
LOSS_QUEUE_SIZE = 20
LOSS_LOGGING_WARMUP = 10

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode("utf-8")
    except:
        return "Unknown"

def save_model(args, status, model, optimizer, postfixes=["latest"]):
    """
    Saves the model with the format:
    {
        model: model.state_dict(),
        optimizer: optimizer.state_dict(),
        metadata: {
            status: status,
            args: args
        }
    }
    """
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "status": status,
            "args": args
        }
    }
    for postfix in postfixes:
        torch.save(save_dict, status['checkpoint_dir'] / f"model_{postfix}.pt")

def load_model(args, status, model: nn.Module, optimizer: nn.Module, checkpoint_path: Path):
    """
    Loads the model from the checkpoint path
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint_status = checkpoint["metadata"]["status"]
    checkpoint_args = checkpoint["metadata"]["args"]

    if checkpoint_status["git_hash"] != status["git_hash"]:
        print(f"\n******************\nWarning: git hash of checkpoint ({checkpoint_status['git_hash']}) does not match current git hash ({status['git_hash']})\n******************\n")

    print("Checking training argument consistency...")
    any_different = False
    for key, item in checkpoint_args.items():
        try:
            if args[key] != item:
                print(f"\t- Warning: checkpoint arg {key} ({item}) does not match current arg ({args[key]})")
                any_different = True
        except KeyError:
            print(f"\t- Warning: checkpoint arg {key} ({item}) not found in current args")
            any_different = True
    if not any_different:
        print("\tAll checkpoint args match current args")

    return checkpoint_args, checkpoint_status

def get_model(args, status):
    """
    Creates the correct type of model and moves it to the correct device
    """
    if args["classifier_type"] == "simple":
        return SimpleImgClassifierLateFusion(args["num_classes"], args["transformation_embedding_size"]).to(status["device"])
    elif args["classifier_type"] == "attention_resnet":
        return ResNetClassifierAttentionFusion(args["num_classes"], args["transformation_embedding_size"]).to(args["device"])
    elif args["classifier_type"] == "resnet":
        return ResNetImgClassifierLateFusion(args["num_classes"], args["transformation_embedding_size"]).to(args["device"])
    else:
        raise ValueError(f"Invalid classifier type {args['classifier_type']}")
    

def get_dataloader(args, status, val=False):
    """
    Returns the correct dataloader
    """
    if val:
        d = create_image_transformation_dataset(
            seed=0,
            transformation_types=status["transformation_classes"],
            num_classes_per_transformation=1000,
            anchor_dir=args["val_data_dir"],
            example_dir=None,
            num_positive_input_examples=0,
            num_negative_input_examples=0,
            separate_neg_examples=False,
            anchor_limit=None,
            val=True,
            num_anchors=args["images_per_sample"],
        )
        return DataLoader(
            d,
            batch_size=args["val_batch_size"],
            shuffle=True,  # TODO: Make sure that different classes are being used when this is false
            num_workers=args["num_workers"],
            pin_memory=True
        )
    else:
        d = create_image_transformation_dataset(
            seed=0,
            transformation_types=status["transformation_classes"],
            num_classes_per_transformation=1000,
            anchor_dir=args["train_data_dir"],
            example_dir=None,
            num_positive_input_examples=0,
            num_negative_input_examples=0,
            separate_neg_examples=False,
            anchor_limit=None,
            val=True,
            num_anchors=args["images_per_sample"],
        )
        return DataLoader(
            d,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args["num_workers"],
            pin_memory=True
        )


def evaluate(args, status, model: nn.Module, embedder: nn.Module, gamma: nn.Module, val_dataloader):
    """
    We compute the accuracy and nll loss on the validation set
    """
    with torch.no_grad():
        model.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        val_batch_count = min(args["val_length"], len(val_dataloader))
        progress_bar = tqdm(range(val_batch_count), desc="Evaluating")
        val_dataset_iter = iter(val_dataloader)
        for i in progress_bar:
            batch = next(val_dataset_iter)
            transformed_images, _, _, _, image_classes = batch
            transformed_images = transformed_images.to(status["device"])
            image_classes = image_classes.to(status["device"])
            if args["use_dummy_transformation_representation"]:
                # Then we will fake the combined embeddings and set them all to 0
                combined_embeddings = torch.zeros(args["val_batch_size"], args["transformation_embedding_size"], device=status["device"])
                embedding_size = args["transformation_embedding_size"]
            else:
                transformed_images = transformed_images.view(-1, 3, 224, 224)
                embeddings = embedder(transformed_images)
                embeddings = embeddings.view(args["val_batch_size"], args["images_per_sample"], -1)
                embedding_size = embeddings.shape[-1]
                assert embedding_size == args["transformation_embedding_size"], f"Embedding size {embedding_size} does not match expected size {args['transformation_embedding_size']}"

                # Get the combined embeddings
                combined_embeddings = gamma(embeddings) # [batch_size, embedding_size]
        
            # Now we want to get the logits
            # We are first going to reshape everything into a single batch
            transformed_images = transformed_images.view(-1, 3, 224, 224) # Does nothing if we are using real embeddings since we already did this
            image_classes = image_classes.view(-1)
            combined_embeddings = combined_embeddings.unsqueeze(1).repeat(1, args["images_per_sample"], 1).view(-1, embedding_size)

            # Now we can get the logits
            logits = model(transformed_images, combined_embeddings) # [batch_size * images_per_sample, num_classes]

            # Now we can get the loss
            loss = F.cross_entropy(logits, image_classes)
            total_loss += loss.item()

            # Now we can get the accuracy
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == image_classes).sum().item()
            total += image_classes.size(0)

            # Update the progress bar
            progress_bar.set_postfix({"loss": total_loss / (i + 1), "accuracy": total_correct / total})

        # Now we can get the final loss and accuracy
        final_loss = total_loss / val_batch_count
        final_accuracy = total_correct / total

        return {
            "evaluation_loss": final_loss,
            "evaluation_accuracy": final_accuracy
        }

    
def count_parameters(model: nn.Module):
    """
    Counts the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    status = {}
    status["device"] = torch.device(args["device"]) if "device" in args else ("cuda" if torch.cuda.is_available() else "cpu")
    status["git_hash"] = get_git_hash()
    status["epoch"] = None
    status["step"] = None
    status["run_id"] = None
    status["artifact_dir"] = None
    status["checkpoint_dir"] = None
    status["transformation_classes"] = [image_transformation.transformation_name_map[trans_name] for trans_name in args["transformation_types"]]

    model = get_model(args, status)
    print(f"Model is using {count_parameters(model)} trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    wandb_initialized = False
    if args["resume_checkpoint"]:
        print("Resuming checkpoint...")
        checkpoint_args, checkpoint_status = load_model(args, status, model, optimizer, args["checkpoint_path"])
        # We take the epoch and step from the checkpoint, but the rest of the status is from the current run
        status["epoch"] = checkpoint_status["epoch"]
        status["step"] = checkpoint_status["step"]
        if args["resume_wandb_run"]:
            # Then we initialize the wandb run 
            wandb_initialized = True
            checkpoint_run_id = checkpoint_status["run_id"]
            assert checkpoint_run_id is not None, "Run cannot be resume. No run id was saved in the checkpoint"
            wandb.init(project=WANDB_PROJECT_NAME, id=checkpoint_run_id, resume="must")
    else:
        status["epoch"] = 0
        status["step"] = 0
    if not wandb_initialized:
        # Then we just initialize with default settings
        wandb.init(project=WANDB_PROJECT_NAME)

    wandb.config.update(args, allow_val_change=True)
    status["run_id"] = wandb.run.id
    status["artifact_dir"] = Path(args["save_dir"]) / status["run_id"]
    status["checkpoint_dir"] = status["artifact_dir"] / "checkpoints"
    status["artifact_dir"].mkdir(parents=True, exist_ok=True)
    status["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(args, status)
    val_dataloader = get_dataloader(args, status, val=True)

    loss_queue = []
    def loss_enqueue(loss):
        loss_queue.append(loss)
        if len(loss_queue) > LOSS_QUEUE_SIZE:
            loss_queue.pop(0)
    epoch_loss = 0

    embedder, gamma, _, embedder_metadata = create_model_and_load_checkpoint(args["representation_model_checkpoint"], device=status["device"], load_optimizer=False)
    print(f"Using embedder from epoch {embedder_metadata['epoch']}")
    embedder.eval()
    gamma.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    for param in gamma.parameters():
        param.requires_grad = False

    epoch_len = min(len(dataloader), args["epoch_len"])

    evaluation_obj = evaluate(args, status, model, embedder, gamma, val_dataloader)
    wandb.log(evaluation_obj, step=status["step"])
    for epoch in range(status["epoch"]+1, args["epochs"]+1):
        status["epoch"] = epoch
        model.train()
        progress_bar = tqdm(range(epoch_len))
        dataset_iter = iter(dataloader)
        epoch_loss = 0
        for i in progress_bar:
            batch = next(dataset_iter)
            transformed_images, _, _, _, image_classes = batch
            # Transformed images: [batch_size, images_per_sample, 3, 224, 224]
            # Image classes: [batch_size, images_per_sample]
            transformed_images: torch.Tensor = transformed_images.to(status["device"])
            image_classes: torch.Tensor = image_classes.to(status["device"])

            # Get the embeddings
            if args["use_dummy_transformation_representation"]:
                # Then we will fake the combined embeddings and set them all to 0
                combined_embeddings = torch.zeros(args["batch_size"], args["transformation_embedding_size"], device=status["device"])
                embedding_size = args["transformation_embedding_size"]
            else:
                with torch.no_grad():
                    transformed_images = transformed_images.view(-1, 3, 224, 224)
                    embeddings = embedder(transformed_images)
                    embeddings = embeddings.view(args["batch_size"], args["images_per_sample"], -1)
                    embedding_size = embeddings.shape[-1]
                    assert embedding_size == args["transformation_embedding_size"], f"Embedding size {embedding_size} does not match expected size {args['transformation_embedding_size']}"

                    # Get the combined embeddings
                    combined_embeddings = gamma(embeddings) # [batch_size, embedding_size]

            # Now we want to get the logits
            # We are first going to reshape everything into a single batch
            transformed_images = transformed_images.view(-1, 3, 224, 224) # Does nothing if we are using real embeddings since we already did this
            image_classes = image_classes.view(-1)
            combined_embeddings = combined_embeddings.unsqueeze(1).repeat(1, args["images_per_sample"], 1).view(-1, embedding_size)

            # Now we can get the logits
            logits = model(transformed_images, combined_embeddings) # [batch_size * images_per_sample, num_classes]

            # Now we can get the loss
            loss = F.cross_entropy(logits, image_classes)

            # Now we can do backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar
            if status["step"] > LOSS_LOGGING_WARMUP:
                loss_enqueue(loss.item())
                queue_avg_loss = np.mean(loss_queue)
                epoch_loss += loss.item()
                epoch_avg_loss = epoch_loss / (i+1)
                progress_bar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f} | Loss Queue: {queue_avg_loss:.4f} | Epoch Avg Loss: {epoch_avg_loss:.4f}")

                log_obj = {
                    "loss": loss.item(),
                    "queue_avg_loss": queue_avg_loss,
                }
                wandb.log(log_obj, step=status["step"])
            else:
                progress_bar.set_description(f"Epoch {epoch} | Warmup Loss: {loss.item():.4f}")

            status["step"] += 1
        wandb.log({"epoch_loss": epoch_loss}, step=status["step"])
        save_model(args, status, model, optimizer, postfixes=["latest"])

        # Now we can do validation
        evaluation_obj = evaluate(args, status, model, embedder, gamma, val_dataloader)
        wandb.log(evaluation_obj, step=status["step"])


if __name__ == "__main__":
    args = {
        "transformation_types": ["gaussian", "median", "noise", "erosion", "dilation", "perspective"],
        "device": "mps",
        "classifier_type": "simple", # "simple", "attention_resnet", "resnet"
        "transformation_embedding_size": 128,  # int
        "images_per_sample": 10,  # int
        "batch_size": 8,  # int
        "val_batch_size": 4,  # int
        "val_length": 500,  # int
        "epochs": 5,  # int
        "epoch_len": 5000,  # int
        "representation_model_checkpoint": Path("/Users/aidan/projects/2023/summer/trans-rep/artifacts/complete_no_fisheye.pt"),
        "use_dummy_transformation_representation": True, # Bool
        "save_dir": Path("./image_classification_artifacts"),
        "resume_checkpoint": False,  # Boolean
        "checkpoint_path": Path("/Users/aidan/projects/2023/summer/trans-rep/image_classification_artifacts/fmzk02dw/checkpoints/model_latest.pt"),  # Path object
        "resume_wandb_run": False,  # Bool (if true, resume the wandb run from the checkpoint)
        "learning_rate": 1e-4,  # float
        "num_classes": 10,  # int
        "num_workers": 4,
        "train_data_dir": Path("/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320/train"),
        "val_data_dir": Path("/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320/val"),
    }

    train(args)
