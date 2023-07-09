from transformations.image_transformation import *
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from models.image_embedder import ConvTransEmbedder, Gamma
from typing import List, Dict, Any, Optional
from PIL import Image
import os
import torch
from pathlib import Path

def generate_examples(seed=0):
    # Initialize the transformations
    # transformations: List[Transformation] = [x(seed=seed+i) for i, x in enumerate(transformation_classes)]
    transformations: List[Transformation] = [x(seed=seed+i) for i, x in enumerate(imgaug_transformations)]

    # sequential_transformations = []
    # np.random.seed(seed)
    # for i in range(3):
    #     n_transforms = np.random.randint(2, len(imgaug_transformations))
    #     # Now we select n random transformations from the list of transformations without replacement
    #     selected_transformations = np.random.choice(imgaug_transformations, size=n_transforms, replace=False)
    #     # Now we create a sequential transformation from the selected transformations
    #     seq_transformation = SequentialImgAugTransformation(transformation_types=selected_transformations, seed=seed)
    #     sequential_transformations.append(seq_transformation)
    # transformations.extend(sequential_transformations)

    # Base directories
    input_base_dir = "./test_images"
    output_base_dir = "./test_image_output"

    # Add a "parameterization" file to the output directory for each transformation
    for transformation in transformations:
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(output_base_dir, transformation.id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the parameterization to a file
        with open(os.path.join(output_dir, f"{seed}_parameterization.txt"), "w") as f:
            for parameter_id in transformation.param_ids:
                if type(parameter_id) is not str:
                    parameter_id = parameter_id[0]
                f.write(parameter_id + ": " + str(transformation.param[parameter_id]) + "\n")

    # Get all file paths in the input directory
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            new_filename = f"{seed}_{file}"
            print("Processing file: " + file)
            # Skip non-image files
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                continue

            # Load the image
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img = np.array(img)

            for transformation in transformations:
                print("\tApplying transformation: " + transformation.id)
                # Apply the transformation
                output_img = transformation(img)

                # Convert the image back into a format that can be saved by Pillow
                output_img = Image.fromarray(output_img)

                # Create the output directory if it doesn't exist
                output_dir = os.path.join(output_base_dir, transformation.id, os.path.relpath(root, input_base_dir))
                os.makedirs(output_dir, exist_ok=True)

                # Save the image
                output_img.save(os.path.join(output_dir, new_filename))

if __name__ == "__main__":
    for i in range(10):
        generate_examples(seed=i)
    exit()

    d: ImageTransformationContrastiveDataset = create_image_transformation_dataset(
        seed=1,
        transformation_types=transformation_classes,
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
    # Generate a batch and save it to test_image_dataset_output
    output_path = Path("./test_image_dataset_output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Dataset length: " + str(len(d)))
    batch = d.__getitem__(15, verbose=True)
    # imgs = Image.fromarray(batch[0])
    for i, img in enumerate(batch[0]):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)
        image_class_idx = batch[4][i]
        img_class = d.img_classes[image_class_idx]
        save_file = output_path / img_class / f"{i}.png"
        if not os.path.exists(save_file.parent):
            os.makedirs(save_file.parent)
        img.save(save_file)
        # for i, img in enumerate(batch[1]):
        #     img = Image.fromarray(img)
        #     img.save("./test_image_dataset_output/pos_" + str(i) + ".png")
        # for i, img in enumerate(batch[2]):
        #     img = Image.fromarray(img)
        #     img.save("./test_image_dataset_output/neg_" + str(i) + ".png")

    # test_embedder = ConvTransEmbedder()
    # test_embedder.eval()
    # input = torch.from_numpy(batch[0].astype(np.float32) / 255).permute(2, 0, 1)
    # output = test_embedder(input.unsqueeze(0)).squeeze(0).detach().numpy()
    # print("Output shape: " + str(output.shape))

    # g = Gamma()
    # g.eval()
    # input = torch.from_numpy(np.stack(batch[1]).astype(np.float32) / 255).permute(0, 3, 1, 2)
    # c_output = test_embedder(input)
    # g_output = g(c_output)
    # print("Gamma output shape: " + str(g_output.shape))
    

    
