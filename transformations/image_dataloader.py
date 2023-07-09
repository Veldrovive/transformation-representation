from .image_transformation import *
from typing import List, Dict, Any, Optional, Type, Union
from pathlib import Path
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms
import torch
import random

class ImageTransformationContrastiveDataset(Dataset):
    """
    ImageTransformationContrastiveDataset is a dataset for learning transformation representations using contrastive learning.

    In each item, we supply (anchor, [positive_examples], [negative_examples]) where positive_examples are images 
    that have been transformed by the same transformation as the anchor, and negative_examples are images that have
    been transformed by a different transformation than the anchor.

    The number of examples is controlled by the num_input_examples parameter.

    We hypothesize that the model may find a way to cheat if the example images come from the same set as the anchors so we allow the ability
    to specify a separate directory for the examples.

    We also allow the ability to specify whether the negative examples should be separate from the positive examples. If they are not separate,
    then the base images before transformation for both the positive and negative examples will be the same.
    """
    def __init__(self,
                 trans_classes: List[Transformation],
                 anchor_dir: Path,
                 example_dir: Optional[Path] = None,
                 num_positive_input_examples: int = 1,
                 num_negative_input_examples: int = 3,
                 num_anchors: int = 1,
                 separate_neg_examples: bool = True,
                 img_size: int = 224,
                 anchor_limit: Optional[int] = None,
                 val: bool = False
        ):
        if num_positive_input_examples != num_negative_input_examples:
            assert separate_neg_examples, "If num_positive_input_examples != num_negative_input_examples, then separate_neg_examples must be True."
        self.anchor_dir = Path(anchor_dir)
        self.anchor_files = list(self.anchor_dir.glob("**/*.*"))
        # Shuffle the anchor files
        random.shuffle(self.anchor_files)
        self.img_classes = [entry.name for entry in self.anchor_dir.iterdir() if entry.is_dir()]
        self.img_class_ids = {img_class: i for i, img_class in enumerate(self.img_classes)}
        self.file_idx_to_img_class = {i: self.img_class_ids[f.parent.name] for i, f in enumerate(self.anchor_files)}
        if anchor_limit is not None:
            self.anchor_files = self.anchor_files[:anchor_limit]
        self.num_anchor_examples = len(self.anchor_files)

        if example_dir is None:
            print("No example directory specified. Using anchor directory as example directory.")
        self.example_dir = Path(example_dir) if example_dir is not None else Path(anchor_dir)
        self.example_files = list(self.example_dir.glob("**/*.*"))

        self.num_positive_input_examples = num_positive_input_examples
        self.num_negative_input_examples = num_negative_input_examples
        self.num_anchors = num_anchors
        self.trans_classes = trans_classes
        self.num_trans_classes = len(trans_classes)
        

        if separate_neg_examples:
            print("Separating negative examples. They will not be the same images as positive examples.")
        else:
            print("Not separating negative examples. They will be the same images as positive examples.")
        self.separate_neg_examples = separate_neg_examples

        self.img_size = img_size
        self.val = val

    def __len__(self):
        """
        We arbitrarily decide that the number of batches is equal to num_anchor_examples * num_trans_classes
        as that is the number of unique anchors we can generate.
        """
        return self.num_anchor_examples * self.num_trans_classes
    
    def load_image(self, file_path: Path):
        img = Image.open(file_path)
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # scale the image
            transforms.Pad((0, 0, max(0, self.img_size - img.size[0]), max(0, self.img_size - img.size[1]))),  # pad the image
            transforms.ToTensor()  # convert to tensor
        ])
        img = transform(img).permute(1, 2, 0)
        if img.shape[2] == 1:
            # If the image is grayscale, repeat the channels
            img = img.repeat(1, 1, 3)
        np_img = img.numpy() * 255
        return np_img.astype(np.uint8)
    
    def __getitem__(self, idx, verbose=False):
        """
        To get the item, we first choose the transformation and anchor image. (trans_class=idx%self.num_trans_classes, image=idx//self.num_trans_classes)

        Then we need to choose the positive an negative examples. We select from the example images with a uniform choice of num_input_examples separately for positive and negative examples.
        We already have the positive trans class so we only need to randomly select the negative one. This is easy. We just choose an random int between 0 and num_trans_classes-1
        """
        positive_class = idx % self.num_trans_classes
        base_anchor_index = idx // self.num_trans_classes
        if self.num_anchors > 1:
            # Choose random anchors for the other n-1 anchors
            rng = np.random.default_rng(seed=idx)
            anchor_indices = rng.choice(self.num_anchor_examples, size=self.num_anchors-1, replace=False)
            anchor_indices = np.insert(anchor_indices, 0, base_anchor_index)
        else:
            anchor_indices = np.array([base_anchor_index])

        # Choose the negative class
        rng = np.random.default_rng(seed=idx)
        negative_class = rng.integers(0, self.num_trans_classes-2)
        if negative_class >= positive_class:
            negative_class += 1

        if verbose:
            print("Positive transformation:", self.trans_classes[positive_class])
            print("Negative transformation:", self.trans_classes[negative_class])

        # Choose the positive examples
        positive_examples = rng.choice(self.example_files, size=self.num_positive_input_examples, replace=False)

        # Choose the negative examples
        negative_examples = rng.choice(self.example_files, size=self.num_negative_input_examples, replace=False) if self.separate_neg_examples else positive_examples 

        # Load the anchor image
        anchor_images = [self.load_image(self.anchor_files[i]) for i in anchor_indices]
        anchor_images = np.stack([self.trans_classes[positive_class](i) for i in anchor_images])
        anchor_images = torch.from_numpy(anchor_images.astype(np.float32) / 255).permute(0, 3, 1, 2)

        # Anchor image classes
        anchor_image_classes = [self.file_idx_to_img_class[i] for i in anchor_indices]
        anchor_image_classes = torch.tensor(anchor_image_classes)

        if self.val:
            # Then we don't need any of the contrastive examples. We just want the anchor image
            return anchor_images, positive_class, self.trans_classes[positive_class].id, anchor_indices, anchor_image_classes

        # Load the positive examples
        positive_images = [self.load_image(f) for f in positive_examples]

        # Load the negative examples
        negative_images = [self.load_image(f) for f in negative_examples]

        # Apply the transformations
        positive_images = np.stack([self.trans_classes[positive_class](i) for i in positive_images])
        negative_images = np.stack([self.trans_classes[negative_class](i) for i in negative_images])

        if verbose:
            print("Output shapes:")
            print("\t", anchor_images.shape)
            print("\t", positive_images.shape)
            print("\t", negative_images.shape)

        positive_images = torch.from_numpy(positive_images.astype(np.float32) / 255).permute(0, 3, 1, 2)
        negative_images = torch.from_numpy(negative_images.astype(np.float32) / 255).permute(0, 3, 1, 2)

        return anchor_images, positive_images, negative_images
    

def create_image_transformation_dataset(
        seed: int,
        transformation_types: List[Type[Transformation]],
        num_classes_per_transformation: Union[int, List[int]],
        anchor_dir: Path,
        example_dir: Optional[Path] = None,
        num_positive_input_examples: int = 1,
        num_negative_input_examples: int = 1,
        separate_neg_examples: bool = True,
        anchor_limit: Optional[int] = None,
        val: bool = False,
        num_anchors: int = 1,
    ):
    """
    Creates a dataset for learning transformation representations using contrastive learning.
    """
    rng = np.random.default_rng(seed=seed)
    get_seed = lambda: rng.integers(0, 2**32-1)

    if type(num_classes_per_transformation) != int:
        assert len(num_classes_per_transformation) == len(transformation_types)
    else:
        num_classes_per_transformation = [num_classes_per_transformation] * len(transformation_types)

    transformation_classes = []
    for num_classes, trans_type in zip(num_classes_per_transformation, transformation_types):
        transformation_classes.extend(
            [trans_type(seed=get_seed()) for _ in range(num_classes)]
        )
    
    return ImageTransformationContrastiveDataset(
        trans_classes=transformation_classes,
        anchor_dir=anchor_dir,
        example_dir=example_dir,
        num_positive_input_examples=num_positive_input_examples,
        num_negative_input_examples=num_negative_input_examples,
        num_anchors=num_anchors,
        separate_neg_examples=separate_neg_examples,
        anchor_limit=anchor_limit,
        val=val
    )