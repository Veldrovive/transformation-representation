from transformations.image_transformation import *
from transformations.image_dataloader import create_image_transformation_dataset, ImageTransformationContrastiveDataset
from models.image_embedder import ConvTransEmbedder, Gamma
from typing import List, Dict, Any, Optional
from PIL import Image
import os
import torch

def generate_examples():
    seed = 1

    # Initialize the transformations
    transformations: List[Transformation] = [x(seed=seed+i) for i, x in enumerate(transformation_classes)]

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
        with open(os.path.join(output_dir, "parameterization.txt"), "w") as f:
            for parameter_id in transformation.param_ids:
                if type(parameter_id) is not str:
                    parameter_id = parameter_id[0]
                f.write(parameter_id + ": " + str(transformation.param[parameter_id]) + "\n")

    # Get all file paths in the input directory
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
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
                output_img.save(os.path.join(output_dir, file))

if __name__ == "__main__":
    generate_examples()

    d = create_image_transformation_dataset(
        seed=0,
        transformation_types=transformation_classes,
        num_classes_per_transformation=1,
        anchor_dir="imagenet/imagenette2-320",
        example_dir=None,
        num_input_examples=3,
        separate_neg_examples=False
    )
    # Generate a batch and save it to test_image_dataset_output
    output_path = "./test_image_dataset_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Dataset length: " + str(len(d)))
    batch = d.__getitem__(15, verbose=True)
    img = Image.fromarray(batch[0])
    img.save("./test_image_dataset_output/anchor.png")
    for i, img in enumerate(batch[1]):
        img = Image.fromarray(img)
        img.save("./test_image_dataset_output/pos_" + str(i) + ".png")
    for i, img in enumerate(batch[2]):
        img = Image.fromarray(img)
        img.save("./test_image_dataset_output/neg_" + str(i) + ".png")

    test_embedder = ConvTransEmbedder()
    test_embedder.eval()
    input = torch.from_numpy(batch[0].astype(np.float32) / 255).permute(2, 0, 1)
    output = test_embedder(input.unsqueeze(0)).squeeze(0).detach().numpy()
    print("Output shape: " + str(output.shape))

    g = Gamma()
    g.eval()
    input = torch.from_numpy(np.stack(batch[1]).astype(np.float32) / 255).permute(0, 3, 1, 2)
    c_output = test_embedder(input)
    g_output = g(c_output)
    print("Gamma output shape: " + str(g_output.shape))
    

    
