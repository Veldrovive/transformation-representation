import wandb
from transformations.image_transformation import transformation_classes, Transformation
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from models.image_embedder import ConvTransEmbedder, Gamma
from typing import Dict, List, Tuple
from PIL import Image

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

import matplotlib.pyplot as plt
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
"""

def get_dataloader(args):
    d = create_image_transformation_dataset(
        seed=0,
        transformation_types=transformation_classes,
        num_classes_per_transformation=args["num_classes_per_transformation"],
        anchor_dir=Path(args["anchor_dir"]),
        example_dir=Path(args["example_dir"]) if args["example_dir"] is not None else None,
        num_input_examples=args["num_input_examples"],
        separate_neg_examples=args["sep_neg_examples"]
    )
    return DataLoader(
        d,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=args["device"] != "cpu"
    ), d

def get_val_dataset(args):
    d = create_image_transformation_dataset(
        seed=1,
        transformation_types=transformation_classes,
        num_classes_per_transformation=args["num_validation_classes"],
        anchor_dir=Path(args["validation_dir"]),
        example_dir=None,
        num_input_examples=0,
        separate_neg_examples=False
    )
    return d

def get_models(args):
    return ConvTransEmbedder(), Gamma()

def triplet_loss(anchor, positive, negative, margin=0.2):
    positive_dist = (anchor - positive).pow(2).sum(1)
    negative_dist = (anchor - negative).pow(2).sum(1)
    losses = F.relu(positive_dist - negative_dist + margin)
    return losses.mean()

def load_checkpoint(checkpoint_dir, embedder, gamma, optimizer, post_fix="latest", device="cpu"):
    embedder_file = checkpoint_dir / f"embedder_{post_fix}.pt"
    gamma_file = checkpoint_dir / f"gamma_{post_fix}.pt"
    optimizer_file = checkpoint_dir / f"optimizer_{post_fix}.pt"

    # Ensure all files exist
    assert embedder_file.exists()
    assert gamma_file.exists()
    assert optimizer_file.exists()
    print("Found checkpoint files. Loading...")

    # Load the files
    embedder.load_state_dict(torch.load(embedder_file, map_location=device))
    gamma.load_state_dict(torch.load(gamma_file, map_location=device))
    optimizer.load_state_dict(torch.load(optimizer_file))

def iterate_val_dataset(dataset, batch_size: int = 64):
    end_idx = len(dataset)
    anchors = []
    classes = []
    transformations = []
    for idx in range(end_idx):
        if len(anchors) >= batch_size:
            yield np.stack(anchors), classes, transformations
            anchors = []
            classes = []
            transformations = []
        anchor, class_idx, transformation_id = dataset.__getitem__(idx, val=True)
        anchors.append(anchor)
        classes.append(class_idx)
        transformations.append(transformation_id)
    if len(anchors) > 0:
        yield np.stack(anchors), classes, transformations


def evaluate_model_v2(args, epoch, artifact_path, embedder: ConvTransEmbedder, gamma: Gamma, train_dataset: ImageTransformationContrastiveDataset, batch_size: int = 64):
    device = args["device"]
    
    val_dataset = get_val_dataset(args)
    # Each sample from val dataset obtained with val_dataset.__getitem__(idx, val=True)
    # has the format (anchor: np.ndarray[C, H, W], class_idx: int, transformation_id: str)
    # We can also recover the transformation from class_idx with val_dataset.trans_classes[class_idx]

    all_embeddings, all_classes, all_transformations = [], [], []
    for anchors, classes, transformations in iterate_val_dataset(val_dataset, batch_size=batch_size):
        embeddings = embedder(torch.tensor(anchors, dtype=torch.float32).to(device))
        all_embeddings.extend(embeddings.detach().cpu().numpy())
        all_classes.extend(classes)
        all_transformations.extend(transformations)
    np_all_embeddings = np.stack(all_embeddings)
    np_all_classes = np.array(all_classes)
    np_all_transformations = np.array(all_transformations)

    # Fit the two knn classifiers for class and transformation
    class_knn = KNeighborsClassifier(n_neighbors=5)
    class_knn.fit(np_all_embeddings, np_all_classes)

    transformation_knn = KNeighborsClassifier(n_neighbors=5)
    transformation_knn.fit(np_all_embeddings, np_all_transformations)

    all_class_predictions = class_knn.predict(np_all_embeddings)
    all_transformation_predictions = transformation_knn.predict(np_all_embeddings)

    num_correct_class = (np_all_classes == all_class_predictions).sum()
    num_correct_transformation = (np_all_transformations == all_transformation_predictions).sum()

    class_accuracy = num_correct_class / len(np_all_classes)
    transformation_accuracy = num_correct_transformation / len(np_all_transformations)

    class_visualization_path = artifact_path / f"reduced_dim_classes_{epoch}.png"
    transformation_visualization_path = artifact_path / f"reduced_dim_transformations_{epoch}.png"
    graph_reduced_dimensions(np_all_embeddings, labels=np_all_classes, path=class_visualization_path)
    graph_reduced_dimensions(np_all_embeddings, labels=np_all_transformations, path=transformation_visualization_path)

    return {
        "class_accuracy": class_accuracy,
        "transformation_accuracy": transformation_accuracy,
    }, transformation_visualization_path, class_visualization_path

def graph_reduced_dimensions(embeddings, labels, path):
    plt.clf()
    assert type(labels) == np.ndarray, f"Labels must be a numpy array, but got {type(labels)}"
    mapper = umap.UMAP().fit(embeddings)
    umap.plot.points(mapper, labels=labels)
    plt.savefig(path.absolute().as_posix())
    plt.clf()

def evaluate_model(args, epoch, artifact_path, embedder: ConvTransEmbedder, gamma: Gamma, train_dataset: ImageTransformationContrastiveDataset, batch_size: int = 64):
    """
    
    """
    device = args["device"]

    validation_dir = Path(args["validation_dir"])
    num_validation_images = args["num_validation_images"]  # This means the number of evaluations will be num_validation_images * num_transformation * num_validation_classes
    num_validation_classes = args["num_validation_classes"]

    # Our first step is to load the validation images
    base_validation_files = list(validation_dir.glob("**/*.*"))
    if len(base_validation_files) > num_validation_images:
        base_validation_files = base_validation_files[:num_validation_images]
    print(f"Using {len(base_validation_files)} base validation images")
    base_validation_images: List[np.ndarray] = [train_dataset.load_image(file_path) for file_path in base_validation_files]
    print("Got validation images")

    # Now we want to create a map between the transformation type and the transformation so that we can graph what we think should be clusters
    trans_map: Dict[str, List[Transformation]] = {}
    for transformation in train_dataset.trans_classes:
        id = transformation.id
        if id not in trans_map:
            trans_map[id] = []
        if len(trans_map[id]) >= num_validation_classes:
            continue # We want to limit the number of classes we use during validation because they could be in the thousands
        trans_map[id].append(transformation)

    # Creates a unique map from a transformation class to a transformation object for easy lookup
    class_id_map: Dict[str, Transformation] = {}
    for transformations in trans_map.values():
        for i, transformation in enumerate(transformations):
            class_id_map[f"{transformation.id}-{i}"] = transformation
    inverse_class_id_map: Dict[Transformation, str] = {trans: id for id, trans in class_id_map.items()}


    total_classes = sum(len(classes) for classes in trans_map.values())

    total_validation_inputs = len(base_validation_files) * total_classes
    print(f"Selected transformation classes. {total_classes} total classes. {total_validation_inputs} total validation inputs")

    # Now we need to evaluate the embedder on every image with every class applied.
    # We will also run gamma over each class's embeddings to ensure that they also cluster like we want
    # There is not explicit pressure for those to cluster, but it would make sense for them to be next to each other in the geometry of the latent space
    
    #TODO: Evaluate in batches
    transformed_images: Dict[Transformation, List[np.ndarray]] = {}
    for transformations in trans_map.values():
        for transformation in transformations:
            transformed_images[transformation] = [transformation(base_image) for base_image in base_validation_images]
    print("Transformed images")

    # Now we need to run the embedder on each of the images
    embedded_images: Dict[Transformation, List[np.ndarray]] = {}
    for transformation, images in transformed_images.items():
        embedded_images[transformation] = [embedder(torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy() for image in images]
    print(f"Embedded images. Embedding shape: {embedded_images[transformation][0].shape}")

    # In order to fit our KNN classifier, we will construct a list [embedding, transformation_id, class]
    knn_dataset: List[Tuple[np.ndarray, str, Transformation]] = []
    for transformation, embeddings in embedded_images.items():
        for embedding in embeddings:
            knn_dataset.append((embedding, transformation.id, transformation))
    d_embeddings, d_transformation_ids, d_transformation_classes = zip(*knn_dataset)
    d_transformation_class_ids = [inverse_class_id_map[transformation] for transformation in d_transformation_classes]

    transformation_classifier = KNeighborsClassifier(n_neighbors=5)
    transformation_classifier.fit(d_embeddings, d_transformation_ids)

    class_classifier = KNeighborsClassifier(n_neighbors=5)
    class_classifier.fit(d_embeddings, [inverse_class_id_map[transformation] for transformation in d_transformation_classes])

    # Now we will re-classify all points with their knn prediction
    predictions: List[Tuple[np.ndarray, Tuple[str, str], Tuple[Transformation, Transformation]]] = []
    correct_transformation = 0
    incorrect_transformation = 0
    correct_class = 0
    incorrect_class = 0
    for embedding, true_transformation_id, true_class in knn_dataset:
        transformation_prediction = transformation_classifier.predict(embedding.reshape(1, -1))[0]
        class_prediction = class_classifier.predict(embedding.reshape(1, -1))[0]
        class_prediction = class_id_map[class_prediction]
        if transformation_prediction == true_transformation_id:
            correct_transformation += 1
        else:
            incorrect_transformation += 1
        
        if class_prediction == true_class:
            correct_class += 1
        else:
            incorrect_class += 1

        predictions.append((
            embedding,
            (transformation_prediction, true_transformation_id),
            (class_prediction, true_class)
        ))
    
    transformation_accuracy = correct_transformation / (correct_transformation + incorrect_transformation)
    class_accuracy = correct_class / (correct_class + incorrect_class)

    # We also use umap to plot the points in a reduced dimensional space
    mapper = umap.UMAP().fit(np.array(d_embeddings))
    umap.plot.points(mapper, labels=np.array(d_transformation_class_ids))
    # Save the plot
    trans_save_path = artifact_path / f"umap_{epoch}_trans_class.png"
    # Remove the legend
    plt.gca().get_legend().remove()
    plt.savefig(trans_save_path)
    plt.clf()

    
    umap.plot.points(mapper, labels=np.array(d_transformation_ids))
    # Save the plot
    class_save_path = artifact_path / f"umap_{epoch}_trans.png"
    # Remove the legend
    plt.gca().get_legend().remove()
    plt.savefig(class_save_path)
    plt.clf()


    return {
        "transformation_accuracy": transformation_accuracy,
        "class_accuracy": class_accuracy,
    }, trans_save_path, class_save_path
    
    


def main(args):
    device = args["device"]
    # Start a new run
    run = wandb.init(project='transformation-representation')
    # Add the args to the run
    wandb.config.update(args)
    # Get run id
    run_id = wandb.run.id
    artifact_path = Path("artifacts") / run_id
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    embedder, gamma = get_models(args)
    dataloader, dataset = get_dataloader(args)

    embedder = embedder.to(device)
    gamma = gamma.to(device)

    optimizer = torch.optim.Adam(list(embedder.parameters()) + list(gamma.parameters()), lr=args["lr"])

    if args["load_checkpoint"]:
        load_checkpoint(Path(args["checkpoint_dir"]), embedder, gamma, optimizer, device=device, post_fix=args["checkpoint_postfix"])

    loss_queue = []
    loss_queue_max_size = 20

    total_epoch_loss = 0
    best_epoch_loss = float("inf")

    max_epoch_len = len(dataloader)
    epoch_len = min(max_epoch_len, args["epoch_len"])
    logging_warmup = 10

    for i in range(args["num_epochs"]):
        print("Epoch", i)
        wandb.log({"epoch": i})
        epoch_dataloader_iter = iter(dataloader)
        total_epoch_loss = 0
        progress_bar = tqdm(range(epoch_len))
        for batch_idx in progress_bar:
            batch = next(epoch_dataloader_iter)
            anchor = batch[0].to(device)
            pos = batch[1].to(device)
            neg = batch[2].to(device)

            anchor_embedding = embedder(anchor)

            original_shape = pos.shape
            pos = pos.reshape(-1, * pos.shape[2:])
            neg = neg.reshape(-1, * neg.shape[2:])
            pos_embeddings = embedder(pos)
            neg_embeddings = embedder(neg)
            pos_embeddings = pos_embeddings.reshape(*original_shape[:2], 128)
            neg_embeddings = neg_embeddings.reshape(*original_shape[:2], 128)
            pos_embedding = gamma(pos_embeddings)
            neg_embedding = gamma(neg_embeddings)

            loss = triplet_loss(anchor_embedding, pos_embedding, neg_embedding)

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
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss})
            else:
                logging_warmup -= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save the model checkpoint
        torch.save(embedder.state_dict(), checkpoint_path / "embedder_latest.pt")
        wandb.save((checkpoint_path / "embedder_latest.pt").absolute().as_posix())
        torch.save(gamma.state_dict(), checkpoint_path / "gamma_latest.pt")
        wandb.save((checkpoint_path / "gamma_latest.pt").absolute().as_posix())
        # Save the optimizer checkpoint
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer_latest.pt")
        wandb.save((checkpoint_path / "optimizer_latest.pt").absolute().as_posix())

        # Get the average training loss
        avg_epoch_loss = total_epoch_loss / epoch_len
        wandb.log({"avg_epoch_loss": avg_epoch_loss})
        if avg_epoch_loss < best_epoch_loss:
            best_epoch_loss = avg_epoch_loss
            torch.save(embedder.state_dict(), checkpoint_path / "embedder_best.pt")
            wandb.save((checkpoint_path / "embedder_best.pt").absolute().as_posix())
            torch.save(gamma.state_dict(), checkpoint_path / "gamma_best.pt")
            wandb.save((checkpoint_path / "gamma_best.pt").absolute().as_posix())
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer_best.pt")
            wandb.save((checkpoint_path / "optimizer_best.pt").absolute().as_posix())

        # TODO: Implement validation
        # We will pass n images through each of the m of the classes of transformations and get their embeddings.
        # We will then perform a dimension reduction so that we can graph the embeddings on a 2d surface
        # and compute the accuracy on classification both inside its transformation class and between classes
        # We could also try performing a downstream task such as regressing the transformation parameters
        evaluation, trans_cluster_image_path, class_cluster_image_path = evaluate_model(args, epoch=i, artifact_path=artifact_path, embedder=embedder, gamma=gamma, train_dataset=dataset, batch_size=64)
        print(evaluation)
        wandb.log(evaluation)
        # Log the cluster images
        trans_cluster_image = Image.open(trans_cluster_image_path)
        trains_cluster_image = np.array(trans_cluster_image)

        class_cluster_image = Image.open(class_cluster_image_path)
        class_cluster_image = np.array(class_cluster_image)
        
        wandb.log({"trans_cluster_image": wandb.Image(trains_cluster_image), "class_cluster_image": wandb.Image(class_cluster_image)})

    run.finish()

if __name__ == "__main__":
    args = ArgumentParser()
    # args.add_argument("--transformation_types", type=)

    args.add_argument("--anchor_dir", type=str, required=True)
    args.add_argument("--example_dir", type=str, default=None)

    args.add_argument("--validation_dir", type=str, required=True)
    args.add_argument("--num_validation_images", type=int, default=100)
    args.add_argument("--num_validation_classes", type=int, default=10)

    args.add_argument("--num_input_examples", type=int, default=3)
    args.add_argument("--num_classes_per_transformation", type=int, default=100)
    args.add_argument("--sep_neg_examples", action="store_true")

    args.add_argument("--load_checkpoint", action="store_true")
    args.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args.add_argument("--checkpoint_postfix", type=str, default="latest")

    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--epoch_len", type=int, default=1000)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Convert to dict
    args = vars(args.parse_args())
    print(args)
    main(args)
