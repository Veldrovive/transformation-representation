import wandb
from transformations import image_transformation
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from models.image_embedder import ConvTransEmbedder, Gamma, AttentionGamma, AvgGamma
from typing import Dict, List, Tuple, Optional
from PIL import Image

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import random
import subprocess

import matplotlib.pyplot as plt

# Prevent a bunch of deprecation warnings from umap
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import umap
import umap.plot

from sklearn.neighbors import KNeighborsClassifier

"""
TODO:
1. Check whether the memory leak happens if I just run validation
    1. Just generally improve validation
2. Allow for choosing which transformations should be used
2. Implement a downstream task like regressing the transformation parameters
3. Change the validation to use different transformations and images than training
4. Change the best model to be the one that performs best and clustering the classes
5. Implement a more effective loss function that is less memory intensive and requires fewer forward passes
6. Change validation to also into account gamma combination
"""

UPLOAD_CHECKPOINTS = False

def get_dataloader(args, transformation_classes):
    d = create_image_transformation_dataset(
        seed=0,
        transformation_types=transformation_classes,
        num_classes_per_transformation=args["num_classes_per_transformation"],
        anchor_dir=Path(args["anchor_dir"]),
        example_dir=Path(args["example_dir"]) if args["example_dir"] is not None else None,
        num_positive_input_examples=args["max_num_positive_input_examples"],
        num_negative_input_examples=args["max_num_negative_input_examples"],
        separate_neg_examples=args["sep_neg_examples"],
        num_anchors=args["max_num_anchors"],
    )
    return DataLoader(
        d,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=args["device"] != "cpu"
    ), d

def get_val_dataloader(args, transformation_classes):
    d = create_image_transformation_dataset(
        seed=1,
        transformation_types=transformation_classes,
        num_classes_per_transformation=args["num_validation_classes"],
        anchor_dir=Path(args["validation_dir"]),
        example_dir=None,
        num_positive_input_examples=0,
        num_negative_input_examples=0,
        separate_neg_examples=False,
        anchor_limit=args["num_validation_images"],
        val=True,
        num_anchors=args["val_num_anchors"],
    )
    return DataLoader(
        d,
        batch_size=args["val_batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=args["device"] != "cpu"
    ), d

def get_models(args):
    gamma = None
    if args["gamma"] == "none":
        # Then gamma just takes the first embedding as there should only be one
        # Use a pytorch module so that it can be moved to the GPU
        print("Using no gamma")
        gamma = nn.Identity()
    elif args["gamma"] == "mlp":
        print("Using MLP gamma")
        gamma = Gamma()
    elif args["gamma"] == "attention":
        print("Using attention gamma")
        gamma = AttentionGamma()
    elif args["gamma"] == "avg":
        print("Using average gamma")
        gamma = AvgGamma()
    else:
        raise Exception(f"Unknown gamma model {args['gamma']}")
    gamma.to(args["device"])

    embedder = None
    if args["embedder"] == "conv":
        print("Using conv embedder")
        embedder = ConvTransEmbedder()
    else:
        raise Exception(f"Unknown embedder model {args['embedder']}")
    embedder.to(args["device"])

    return embedder, gamma

def triplet_loss(anchor, positive, negative, margin=0.2):
    positive_dist = (anchor - positive).pow(2).sum(1)
    negative_dist = (anchor - negative).pow(2).sum(1)
    losses = F.relu(positive_dist - negative_dist + margin)
    return losses.mean()

def pairwise_distance(anchor, positive, negative, metric='euclidean'):
    assert metric in ['euclidean', 'cosine'], 'Unsupported metric'

    if metric == 'euclidean':
        positive_dist = (anchor - positive).pow(2).sum(1)
        negative_dist = (anchor.unsqueeze(1) - negative).pow(2).sum(-1)
    else:
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive.squeeze(1), p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=-1)
        
        positive_dist = 1 - torch.sum(anchor * positive, dim=1)
        negative_dist = 1 - torch.sum(anchor.unsqueeze(1) * negative, dim=-1)

    return positive_dist, negative_dist

def tuplet_loss(anchor, positive, negative, margin=0.2, metric='euclidean'):
    # print(f"Anchor shape: {anchor.shape}, positive shape: {positive.shape}, negative shape: {negative.shape}")
    assert positive.shape[1] == 1, "Positive should only have one example"
    positive = positive[:, 0, :]
    positive_dist, negative_dist = pairwise_distance(anchor, positive, negative, metric)

    losses = torch.stack([F.relu(positive_dist[i] - negative_dist[i] + margin) for i in range(anchor.shape[0])])

    return losses.mean()

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode("utf-8")
    except:
        return "Unknown"

def save_checkpoint(save_path, embedder, gamma, optimizer, epoch: int, step: int, run_id: str, args: Dict):
    git_hash = get_git_hash()
    embedder_dict = embedder.state_dict()
    gamma_dict = gamma.state_dict()
    optimizer_dict = optimizer.state_dict()

    run_metadata = {
        "epoch": epoch,
        "run_id": run_id,
        "step": step,
        "args": args,
        "git_hash": git_hash,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}")
    torch.save({
        "embedder": embedder_dict,
        "gamma": gamma_dict,
        "optimizer": optimizer_dict,
        "metadata": run_metadata,
    }, save_path)

def load_checkpoint(args: Dict, checkpoint_path, embedder, gamma, optimizer: Optional[nn.Module] = None, device="cpu", load_optimizer=True):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Check the git hash
    git_hash = get_git_hash()
    if checkpoint["metadata"]["git_hash"] != git_hash:
        print(f"\n******************\nWarning: git hash of checkpoint ({checkpoint['metadata']['git_hash']}) does not match current git hash ({git_hash})\n******************\n")

    embedder.load_state_dict(checkpoint["embedder"])
    gamma.load_state_dict(checkpoint["gamma"])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Checking training argument consistency...")
    any_different = False
    for key, item in checkpoint["metadata"]["args"].items():
        try:
            if args[key] != item:
                print(f"\t- Warning: checkpoint arg {key} ({item}) does not match current arg ({args[key]})")
                any_different = True
        except KeyError:
            print(f"\t- Warning: checkpoint arg {key} ({item}) not found in current args")
            any_different = True
    if not any_different:
        print("\tAll checkpoint args match current args")
            
    return checkpoint["metadata"]

def create_model_and_load_checkpoint(checkpoint_path, device="cpu", load_optimizer=True):
    """
    Gets the args from the checkpoint and then creates the models
    """
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    t = torch.load(checkpoint_path, map_location="cpu")
    args = t["metadata"]["args"]
    del t

    embedder, gamma = get_models(args)
    optimizer = None
    metadata = load_checkpoint(args, checkpoint_path, embedder, gamma, optimizer, device, load_optimizer)
    return embedder, gamma, optimizer, metadata


def evaluate_model(args, epoch, artifact_path, embedder: ConvTransEmbedder, gamma: Gamma, val_dataloader: DataLoader):
    with torch.no_grad():
        device = args["device"]
        embedder.eval()
        gamma.eval()

        # Each sample from val dataset
        # has the format (anchor: np.ndarray[C, H, W], class_idx: int, transformation_id: str, anchor_index: int)
        # We can also recover the transformation from class_idx with val_dataset.trans_classes[class_idx]

        num_classes = len(val_dataloader.dataset.trans_classes)
        print(f"Getting {len(val_dataloader.dataset)} evaluation embeddings from {num_classes} classes...")
        all_embeddings, all_classes, all_transformations, all_anchor_idxs = [], [], [], []
        # for anchors, classes, transformations in iterate_val_dataset(val_dataset, batch_size=batch_size):
        for anchors, classes, transformations, anchor_idxs, image_classes in tqdm(val_dataloader):
            num_anchors = anchors.shape[1]
            anchors = anchors.reshape(-1, * anchors.shape[2:])
            embeddings = embedder(anchors.to(device))
            embeddings = embeddings.reshape(-1, num_anchors, embeddings.shape[1])
            if args["apply_gamma_anchor"]:
                embeddings = gamma(embeddings)
            # embeddings = torch.from_numpy(np.random.random(embeddings.shape).astype(np.float32)).to(device)
            all_embeddings.extend(embeddings.detach().cpu().numpy())
            all_classes.extend(classes)
            all_transformations.extend(transformations)
            all_anchor_idxs.extend([str(idxs) for idxs in anchor_idxs])
        np_all_embeddings = np.stack(all_embeddings)
        np_all_classes = np.array(all_classes)
        np_all_transformations = np.array(all_transformations)
        np_all_anchor_idxs = np.array(all_anchor_idxs)

        # Fit the two knn classifiers for class, transformation, and anchor index
        # We expect both class and transformation to grow in accuracy as the model learns to differentiate transformations,
        # but the anchor index should decrease since we have to pressure the model to learn to differentiate between images
        print("Fitting knn classifiers...")
        class_knn = KNeighborsClassifier(n_neighbors=5)
        class_knn.fit(np_all_embeddings, np_all_classes)

        transformation_knn = KNeighborsClassifier(n_neighbors=5)
        transformation_knn.fit(np_all_embeddings, np_all_transformations)

        anchor_knn = KNeighborsClassifier(n_neighbors=5)
        anchor_knn.fit(np_all_embeddings, np_all_anchor_idxs)

        print("Predicting classes and transformations...")
        all_class_predictions = class_knn.predict(np_all_embeddings)
        all_transformation_predictions = transformation_knn.predict(np_all_embeddings)
        all_anchor_idx_predictions = anchor_knn.predict(np_all_embeddings)

        num_correct_class = (np_all_classes == all_class_predictions).sum()
        num_correct_transformation = (np_all_transformations == all_transformation_predictions).sum()
        num_correct_anchor_idx = (np_all_anchor_idxs == all_anchor_idx_predictions).sum()

        class_accuracy = num_correct_class / len(np_all_classes)
        transformation_accuracy = num_correct_transformation / len(np_all_transformations)
        anchor_idx_accuracy = num_correct_anchor_idx / len(np_all_anchor_idxs)

        class_visualization_path = artifact_path / f"reduced_dim_classes_{epoch}.png"
        transformation_visualization_path = artifact_path / f"reduced_dim_transformations_{epoch}.png"
        anchor_visualization_path = artifact_path / f"reduced_dim_anchor_idxs_{epoch}.png"

        print("Graphing reduced dimension representations...")
        mapper = umap.UMAP().fit(np_all_embeddings)
        graph_reduced_dimensions(mapper, labels=np_all_classes, path=class_visualization_path, show_legend=False)
        graph_reduced_dimensions(mapper, labels=np_all_transformations, path=transformation_visualization_path, show_legend=True)
        graph_reduced_dimensions(mapper, labels=np_all_anchor_idxs, path=anchor_visualization_path, show_legend=False)

        return {
            "class_accuracy": class_accuracy,
            "transformation_accuracy": transformation_accuracy,
            "anchor_idx_accuracy": anchor_idx_accuracy,
        }, transformation_visualization_path, class_visualization_path, anchor_visualization_path

def graph_reduced_dimensions(mapper, labels, path, show_legend=True):
    plt.clf()
    assert type(labels) == np.ndarray, f"Labels must be a numpy array, but got {type(labels)}"
    umap.plot.points(mapper, labels=labels)
    
    if not show_legend:
        plt.gca().get_legend().remove()

    plt.savefig(path.absolute().as_posix())
    plt.clf()
    plt.close()

def main(args):
    device = args["device"]
    embedder, gamma = get_models(args)
    optimizer = torch.optim.Adam(list(embedder.parameters()) + list(gamma.parameters()), lr=args["lr"])

    start_epoch = 0
    step = 0
    metadata = None
    if args["load_checkpoint"]:
        metadata = load_checkpoint(args, Path(args["checkpoint"]), embedder, gamma, optimizer, device=device)
        start_epoch = metadata["epoch"]
        if "step" in metadata:
            # Update the current step to be the step from the checkpoint
            print(f"Setting wandb step to {metadata['step']}")
            step = metadata["step"]
        else:
            print("No wandb step found in checkpoint, setting to 0")
            step = 0
    
    if args["resume_wandb_from_checkpoint"]:
        assert metadata is not None, "Must load checkpoint to resume wandb"
        assert "run_id" in metadata and metadata["run_id"] is not None, "Must have run id in checkpoint metadata to resume wandb"
        # Resume wandb run
        print(f"Resuming wandb run {metadata['run_id']}")
        run = wandb.init(project='transformation-representation', id=metadata["run_id"], resume="must")
    else:
        # Start a new run
        run = wandb.init(project='transformation-representation')
        print(f"Starting new wandb run {run.id}")
    # Add the args to the run
    wandb.config.update(args, allow_val_change=True)
    # Get run id
    run_id = wandb.run.id
    artifact_path = Path(args["artifacts_dir"]) / run_id
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    transformation_classes = [image_transformation.transformation_name_map[trans_name] for trans_name in args["transformation_types"]]
    print("Using transformations: ")
    for trans_class in transformation_classes:
        print(f"\t {trans_class.__name__}")
    dataloader, dataset = get_dataloader(args, transformation_classes)
    val_dataloader, val_dataset = get_val_dataloader(args, transformation_classes)


    loss_queue = []
    loss_queue_max_size = 20

    total_epoch_loss = 0
    best_epoch_loss = float("inf")

    max_epoch_len = len(dataloader)
    epoch_len = min(max_epoch_len, args["epoch_len"])
    logging_warmup = 10

    initial_eval, _, _, _ = evaluate_model(args, start_epoch, artifact_path, embedder, gamma, val_dataloader)
    print(f"Initial class accuracy: {initial_eval['class_accuracy']}, transformation accuracy: {initial_eval['transformation_accuracy']}, anchor accuracy: {initial_eval['anchor_idx_accuracy']}")
    wandb.log(initial_eval, step=step)

    get_num_positive_examples = lambda: args["max_num_positive_input_examples"] if len(args["num_positive_input_examples"]) == 1 else np.random.choice(args["num_positive_input_examples"])
    get_num_negative_examples = lambda: args["max_num_negative_input_examples"] if len(args["num_negative_input_examples"]) == 1 else np.random.choice(args["num_negative_input_examples"])
    get_num_anchors = lambda: args["max_num_anchors"] if len(args["num_anchors"]) == 1 else np.random.choice(args["num_anchors"])

    for i in range(start_epoch+1, args["num_epochs"]+1):
        print("Epoch", i)
        wandb.log({"epoch": i}, step=step)
        epoch_dataloader_iter = iter(dataloader)
        total_epoch_loss = 0
        progress_bar = tqdm(range(epoch_len))

        embedder.train()
        gamma.train()
        for batch_idx in progress_bar:
            batch = next(epoch_dataloader_iter)
            anchor = batch[0].to(device)
            pos = batch[1].to(device)
            neg = batch[2].to(device)

            # # We want to randomly sample the number of positive and negative examples
            anchor = anchor[:, :get_num_anchors()]
            pos = pos[:, :get_num_positive_examples()]
            neg = neg[:, :get_num_negative_examples()]

            # original_pos_shape = pos.shape
            # original_neg_shape = neg.shape
            original_num_pos = pos.shape[1]
            original_num_neg = neg.shape[1]
            original_num_anchors = anchor.shape[1]

            pos = pos.reshape(-1, * pos.shape[2:])
            neg = neg.reshape(-1, * neg.shape[2:])
            anchor = anchor.reshape(-1, * anchor.shape[2:])

            pos_embeddings = embedder(pos)
            neg_embeddings = embedder(neg)
            anchor_embedding = embedder(anchor)

            pos_embeddings = pos_embeddings.reshape(-1, original_num_pos, 128)
            neg_embeddings = neg_embeddings.reshape(-1, original_num_neg, 128)
            anchor_embedding = anchor_embedding.reshape(-1, original_num_anchors, 128)

            # As a speed test, we randomly sample the number of positive and negative examples here instead
            # anchor_embedding = anchor_embedding[:, :get_num_anchors()]
            # pos_embeddings = pos_embeddings[:, :get_num_positive_examples()]
            # neg_embeddings = neg_embeddings[:, :get_num_negative_examples()]

            if args["apply_gamma_anchor"]:
                anchor_embedding = gamma(anchor_embedding)
            pos_embedding = gamma(pos_embeddings)
            neg_embedding = gamma(neg_embeddings)

            if args["loss"] == "triplet":
                loss = triplet_loss(anchor_embedding, pos_embedding, neg_embedding)
            elif args["loss"] == "tuplet":
                loss = tuplet_loss(anchor_embedding, pos_embedding, neg_embedding)

            # Log the loss to wandb
            if logging_warmup == 0:
                loss_queue.append(loss.item())
                if len(loss_queue) > loss_queue_max_size:
                    loss_queue.pop(0)
                avg_loss = sum(loss_queue) / len(loss_queue)
                total_epoch_loss += loss.item()
                current_epoch_loss = total_epoch_loss / (batch_idx + 1)
                # print(loss)
                progress_bar.set_description(f"Loss: {loss.item():.4f} Avg Loss: {avg_loss:.4f} Epoch Loss: {current_epoch_loss:.4f}")
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss, "epoch": i}, step=step)
            else:
                logging_warmup -= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        
        # Save the model checkpoint
        save_checkpoint(
            checkpoint_path / "latest.pt",
            embedder,
            gamma,
            optimizer,
            i,
            step,
            run_id,
            args,
        )
        if UPLOAD_CHECKPOINTS:
            wandb.save((checkpoint_path / "latest.pt").absolute().as_posix())

        # Get the average training loss for the epoch
        avg_epoch_loss = total_epoch_loss / epoch_len

        evaluation, trans_cluster_image_path, class_cluster_image_path, anchor_cluster_image_path = evaluate_model(args, epoch=i, artifact_path=artifact_path, embedder=embedder, gamma=gamma, val_dataloader=val_dataloader)
        evaluation["avg_epoch_loss"] = avg_epoch_loss
        print(evaluation)
        wandb.log(evaluation, step=step)
        # Log the cluster images
        try:
            trans_cluster_image = Image.open(trans_cluster_image_path)
            trains_cluster_image = np.array(trans_cluster_image)

            class_cluster_image = Image.open(class_cluster_image_path)
            class_cluster_image = np.array(class_cluster_image)

            anchor_cluster_image = Image.open(anchor_cluster_image_path)
            anchor_cluster_image = np.array(anchor_cluster_image)

            wandb.log({"trans_cluster_image": wandb.Image(trains_cluster_image), "class_cluster_image": wandb.Image(class_cluster_image), "anchor_cluster_image": wandb.Image(anchor_cluster_image)}, step=step)
        except:
            print("Could not load cluster images")
        

        # TODO: Use one of the evaluation metrics to determine the best model
        # We are not doing this currently as the metrics are unreliable
        if avg_epoch_loss < best_epoch_loss:
            save_checkpoint(
                checkpoint_path / "best.pt",
                embedder,
                gamma,
                optimizer,
                i,
                step,
                run_id,
                args,
            )
            if UPLOAD_CHECKPOINTS:
                wandb.save((checkpoint_path / "best.pt").absolute().as_posix())

    run.finish()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--transformation_types", type=str, nargs="+", default=["gaussian", "median", "noise", "erosion", "dilation", "perspective"])

    args.add_argument("--anchor_dir", type=str, required=True)
    args.add_argument("--example_dir", type=str, default=None)
    args.add_argument("--artifacts_dir", type=str, default="artifacts")

    args.add_argument("--validation_dir", type=str, required=True)
    args.add_argument("--num_validation_images", type=int, default=100)
    args.add_argument("--num_validation_classes", type=int, default=10)
    args.add_argument("--val_batch_size", type=int, default=32)
    args.add_argument("--val_num_anchors", type=int, default=5)

    args.add_argument("--num_positive_input_examples", nargs="+", type=int, default=[1], action="store")
    args.add_argument("--num_negative_input_examples", nargs="+", type=int, default=[3], action="store")
    args.add_argument("--num_anchors", nargs="+", type=int, default=[1], action="store")
    args.add_argument("--num_classes_per_transformation", type=int, default=100)
    args.add_argument("--sep_neg_examples", action="store_true")

    args.add_argument("--load_checkpoint", action="store_true")
    args.add_argument("--checkpoint", type=str, default=None)
    args.add_argument("--resume_wandb_from_checkpoint", action="store_true")

    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--epoch_len", type=int, default=1000)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args.add_argument("--seed", type=int, default=None)
    args.add_argument("--gamma", type=str, default="mlp")
    args.add_argument("--embedder", type=str, default="conv")
    args.add_argument("--loss", type=str, default="triplet")
    args.add_argument("--apply_gamma_anchor", action="store_true")

    # Convert to dict
    args = vars(args.parse_args())
    if args["seed"] is not None:
        set_seed(args["seed"])
    
    args["max_num_positive_input_examples"] = max(args["num_positive_input_examples"])
    args["max_num_negative_input_examples"] = max(args["num_negative_input_examples"])
    args["max_num_anchors"] = max(args["num_anchors"])

    print(args)
    main(args)
