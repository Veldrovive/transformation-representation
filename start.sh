# python train_contrastive_transformation.py \
#     --anchor_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
#     --epoch_len 1000 \
#     --num_classes_per_transformation 1000 \
#     --num_epochs 30 \
#     --device mps \
#     --num_workers 4 \
#     --load_checkpoint \
#     --checkpoint_dir "/Users/aidan/projects/2023/summer/trans-rep/artifacts/c7fivxk6/checkpoints"

python train_contrastive_transformation.py \
    --anchor_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
    --epoch_len 1000 \
    --num_classes_per_transformation 1000 \
    --num_epochs 30 \
    --device mps \
    --num_workers 4 \
    --validation_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
    --num_validation_images 100 \
    --num_validation_classes 10 \