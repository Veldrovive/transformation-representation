from models.image_embedder import ConvTransEmbedder, AttentionGamma
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from train_contrastive_transformation import load_checkpoint, get_models
from transformations import image_transformation
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import wandb

possible_kernel_sizes = [1, 3, 5, 7, 9, 11, 13]
def to_kernel_one_hot(kernel_size):
    """
    Takes an array (n, 1) of kernel sizes and returns a one-hot representation of the kernel size
    """
    one_hot = np.zeros((kernel_size.shape[0], len(possible_kernel_sizes)))
    kernel_size = kernel_size.reshape(-1)
    for i, size in enumerate(possible_kernel_sizes):
        one_hot[kernel_size == size, i] = 1
    return one_hot

def recover_kernel_size(one_hot):
    """
    Takes an array of (n, len(possible_kernel_sizes)) and returns the kernel size
    """
    kernel_indices = np.argmax(one_hot, axis=1)
    return np.array([possible_kernel_sizes[i] for i in kernel_indices])

def normalize_sd(sd, min=2, max=6):
    """
    Takes an array (n, 1) of standard deviations and returns a normalized representation of the standard deviation
    """
    return (sd - min) / (max - min)

def recover_sd(normalized_sd, min=2, max=6):
    """
    Takes an array (n, 1) of normalized standard deviations and returns the standard deviation
    """
    return normalized_sd * (max - min) + min

def get_dataloader():
    d = create_image_transformation_dataset(
        seed=1,
        transformation_types=[image_transformation.transformation_name_map["gaussian"]],
        num_classes_per_transformation=10000,
        anchor_dir=Path("/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320/train"),
        example_dir=None,
        num_positive_input_examples=0,
        num_negative_input_examples=0,
        separate_neg_examples=False,
        val=True,
        num_anchors=10,
    )
    return DataLoader(
        d,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    ), d

def get_val_dataloader():
    d = create_image_transformation_dataset(
        seed=1,
        transformation_types=[image_transformation.transformation_name_map["gaussian"]],
        num_classes_per_transformation=20,
        anchor_dir=Path("/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320/train"),
        example_dir=None,
        num_positive_input_examples=0,
        num_negative_input_examples=0,
        separate_neg_examples=False,
        anchor_limit=32,
        val=True,
        num_anchors=32,
    )
    return DataLoader(
        d,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ), d

def evaluate_model(embedder, gamma, model, val_dataloader):
    """
    We are simultaneously evaluating the model and embedding system.
    Gamma can combine n embeddings into 1 so we want to evaluate the model on different numbers of combined embeddings
    to see if more embeddings implies better performance.
    """
    num_embeddings = [1, 2, 8, 16, 32]
    embedder.eval()
    gamma.eval()
    model.eval()
    evaluation_data = {} # { num_embeddings: [(true_kernel_size, true_sd, predicted_kernel_size, predicted_sd),...]}
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, total=len(val_dataloader))
        for anchor, anchor_classes, transformation, anchor_idx, img_classes in progress_bar:
            # First, we embed the anchors (we assert batch size is 1)
            anchor = anchor[0]
            anchor_class = anchor_classes[0]
            anchor = anchor.to("mps")
            embeddings = embedder(anchor)
            true_parameters = np.array(list(dataset.trans_classes[anchor_class].param.get_parameterization().values()))
            true_sd = true_parameters[0]
            true_kernel_size = true_parameters[1]
            for num_embedding in num_embeddings:
                if num_embedding not in evaluation_data:
                    evaluation_data[num_embedding] = []
                for batch_index in range(32 // num_embedding):
                    batch_embeddings = embeddings[batch_index * num_embedding: (batch_index + 1) * num_embedding]
                    batch_embeddings = batch_embeddings.unsqueeze(0)
                    combined_embedding = gamma(batch_embeddings)
                    predicted_parameters = model(combined_embedding)
                    sd_prediction = predicted_parameters[:, 0]
                    kernel_prediction = predicted_parameters[:, 1:]
                    sd_prediction = recover_sd(sd_prediction.cpu().numpy())[0]
                    kernel_prediction = recover_kernel_size(kernel_prediction.cpu().numpy())[0]
                    evaluation_data[num_embedding].append((true_kernel_size, true_sd, kernel_prediction, sd_prediction))

    # Now we want to produce summary statistics for each num_embedding
    # We will compute the MSE for the standard deviation and the accuracy for the kernel size
    summary_statistics = {}
    for num_embedding in num_embeddings:
        kernel_size_accuracy = 0
        sd_mse = 0
        for true_kernel_size, true_sd, predicted_kernel_size, predicted_sd in evaluation_data[num_embedding]:
            if true_kernel_size == predicted_kernel_size:
                kernel_size_accuracy += 1
            sd_mse += (true_sd - predicted_sd) ** 2
        kernel_size_accuracy /= len(evaluation_data[num_embedding])
        sd_mse /= len(evaluation_data[num_embedding])
        summary_statistics[num_embedding] = (kernel_size_accuracy, sd_mse)
    
    print("Summary Statistics")
    log_obj = {}
    for num_embedding in num_embeddings:
        print(f"\tNum Embeddings: {num_embedding}")
        print(f"\t\tKernel Size Accuracy: {summary_statistics[num_embedding][0]}")
        print(f"\t\tSD MSE: {summary_statistics[num_embedding][1]}")
        log_obj[f"num_embedding_{num_embedding}_kernel_size_accuracy"] = summary_statistics[num_embedding][0]
        log_obj[f"num_embedding_{num_embedding}_sd_mse"] = summary_statistics[num_embedding][1]
    wandb.log(log_obj)

    return summary_statistics

if __name__ == "__main__":
    checkpoint_path = "/Users/aidan/projects/2023/summer/trans-rep/artifacts/complete_no_fisheye.pt"
    t = torch.load(checkpoint_path, map_location="cpu")
    metadata = t["metadata"]
    args = metadata["args"]
    args["device"] = "mps"
    del t

    run = wandb.init(project='gaussian-parameter-regression')

    embedder, gamma = get_models(args)
    embedder.to("mps")
    gamma.to("mps")
    metadata = load_checkpoint(args, checkpoint_path, embedder, gamma, device="mps", load_optimizer=False)

    embedder.requires_grad_(False)
    for p in gamma.parameters():
        p.requires_grad_(False)
    gamma.requires_grad_(False)
    for p in gamma.parameters():
        p.requires_grad_(False)

    dataloader, dataset = get_dataloader()
    val_data, val_dataset = get_val_dataloader()

    max_epoch_len = 1000
    epoch_len = min(len(dataloader), max_epoch_len)

    # model = nn.Sequential(
    #     nn.Linear(128, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 1+len(possible_kernel_sizes)),
    #     nn.Sigmoid()
    # )
    model = nn.Sequential(
        nn.Linear(128, 1+len(possible_kernel_sizes)),
        nn.Sigmoid()
    )
    model.to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # evaluate_model(embedder, gamma, model, val_data)

    for epoch in range(5):
        print(f"Epoch {epoch}")
        epoch_total_loss = 0
        iterator = iter(dataloader)
        progress_bar = tqdm(range(epoch_len), total=epoch_len)
        model.train()
        for i in progress_bar:
            anchor, anchor_classes, transformation, anchor_idx, image_classes = next(iterator)
            anchor = anchor.to("mps")
            original_anchor_shape = anchor.shape
            anchor = anchor.reshape(-1, *anchor.shape[2:])
            embeddings = embedder(anchor)
            embeddings = embeddings.reshape(*original_anchor_shape[:2], -1)
            embeddings = gamma(embeddings)

            parameters = np.array([np.array(list(dataset.trans_classes[trans_class].param.get_parameterization().values())) for trans_class in anchor_classes])
            # parameters = torch.from_numpy(parameters.astype(np.float32))
            one_hot_kernel_size = to_kernel_one_hot(parameters[:, 1].reshape(-1, 1))
            one_hot_kernel_size = torch.from_numpy(one_hot_kernel_size.astype(np.float32))
            normalized_sd = normalize_sd(parameters[:, 0].reshape(-1, 1))
            normalized_sd = torch.from_numpy(normalized_sd.astype(np.float32))
            parameters = torch.cat([normalized_sd, one_hot_kernel_size], dim=1)
            parameters = parameters.to("mps")

            predicted_parameters = model(embeddings)
            loss = torch.mean((parameters - predicted_parameters)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item(), "epoch": epoch})
        avg_loss = epoch_total_loss / epoch_len
        print(f"Average loss: {avg_loss:.4f}")
        wandb.log({"avg_loss": avg_loss, "epoch": epoch})
        evaluate_model(embedder, gamma, model, val_data)


