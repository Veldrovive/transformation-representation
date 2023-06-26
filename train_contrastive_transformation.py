import wandb
from transformations.image_transformation import transformation_classes
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from models.image_embedder import ConvTransEmbedder, Gamma

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
    return DataLoader(d, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])

def get_models(args):
    return ConvTransEmbedder(), Gamma()

def triplet_loss(anchor, positive, negative, margin=0.2):
    positive_dist = (anchor - positive).pow(2).sum(1)
    negative_dist = (anchor - negative).pow(2).sum(1)
    losses = F.relu(positive_dist - negative_dist + margin)
    return losses.mean()

def main(args):
    # Start a new run
    run = wandb.init(project='transformation-representation')
    # Get run id
    run_id = wandb.run.id
    artifact_path = Path("artifacts") / run_id
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    embedder, gamma = get_models(args)
    dataloader = get_dataloader(args)

    embedder = embedder.to(device)
    gamma = gamma.to(device)

    optimizer = torch.optim.Adam(list(embedder.parameters()) + list(gamma.parameters()), lr=args["lr"])
    loss_queue = []
    loss_queue_max_size = 20

    total_epoch_loss = 0
    best_epoch_loss = float("inf")

    max_epoch_len = len(dataloader)
    epoch_len = min(max_epoch_len, args["epoch_len"])
    logging_warmup = 10

    for i in range(10):
        print("Epoch", i)
        wandb.log({"epoch": i})
        epoch_dataloader_iter = iter(dataloader)
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
                # print(loss)
                progress_bar.set_description(f"Loss: {loss.item():.4f} Avg Loss: {avg_loss:.4f}")
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss})
            else:
                logging_warmup -= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save the model checkpoint
        torch.save(embedder.state_dict(), checkpoint_path / "embedder_latest.pt")
        torch.save(gamma.state_dict(), checkpoint_path / "gamma_latest.pt")
        # Save the optimizer checkpoint
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer_latest.pt")
        # Upload the model checkpoint as an artifact
        run.log_artifact(checkpoint_path / "embedder_latest.pt")
        run.log_artifact(checkpoint_path / "gamma_latest.pt")
        run.log_artifact(checkpoint_path / "optimizer_latest.pt")

        # Get the average training loss
        avg_epoch_loss = total_epoch_loss / epoch_len
        if avg_epoch_loss < best_epoch_loss:
            best_epoch_loss = avg_epoch_loss
            torch.save(embedder.state_dict(), checkpoint_path / "embedder_best.pt")
            torch.save(gamma.state_dict(), checkpoint_path / "gamma_best.pt")
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer_best.pt")
            run.log_artifact(checkpoint_path / "embedder_best.pt")
            run.log_artifact(checkpoint_path / "gamma_best.pt")
            run.log_artifact(checkpoint_path / "optimizer_best.pt")



        # TODO: Implement validation
    # After all the logging is done, don't forget to close your run
    run.finish()

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--anchor_dir", type=str, required=True)
    args.add_argument("--example_dir", type=str, default=None)
    args.add_argument("--num_input_examples", type=int, default=3)
    args.add_argument("--num_classes_per_transformation", type=int, default=20)
    args.add_argument("--sep_neg_examples", action="store_true")

    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--epoch_len", type=int, default=1000)
    args.add_argument("--num_workers", type=int, default=4)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--lr", type=float, default=1e-3)

    # Convert to dict
    args = vars(args.parse_args())
    main(args)
