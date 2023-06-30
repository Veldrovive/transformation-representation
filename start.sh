# Normal:
# python train_contrastive_transformation.py \
#     --anchor_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
#     --artifacts_dir "artifacts" \
#     --epoch_len 1000 \
#     --num_classes_per_transformation 1000 \
#     --num_epochs 30 \
#     --device mps \
#     --num_workers 4 \
#     --validation_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
#     --num_validation_images 30 \
#     --num_validation_classes 10 \
#     --lr 0.0001 \
#     --seed 0 \
#     --load_checkpoint \
#     --checkpoint_dir "/Users/aidan/projects/2023/summer/trans-rep/artifacts/2pi8yq6e/checkpoints" \

# No Gamma:
python train_contrastive_transformation.py \
    --anchor_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
    --artifacts_dir "artifacts" \
    --epoch_len 1000 \
    --num_positive_input_examples 1 \
    --num_negative_input_examples 5 \
    --num_classes_per_transformation 1000 \
    --num_epochs 30 \
    --device mps \
    --num_workers 4 \
    --validation_dir "/Users/aidan/projects/2023/summer/trans-rep/imagenet/imagenette2-320" \
    --num_validation_images 30 \
    --num_validation_classes 10 \
    --lr 0.00001 \
    --batch_size 32 \
    --seed 0 \
    --sep_neg_examples \
    --gamma "none" \