#!/bin/bash

# List of models
nn_models_list=("alexnet" "resnet50" "resnet152" "unet" "vgg16")

# Loop over each model
for model_name in "${nn_models_list[@]}"; do
    echo "Processing model: $model_name"

    # Source directory (containing 'num_banks = 16')
    src_dir="../../nn_models/${model_name}/single_layers/16banks"

    # Base directory for creating new folders
    base_dir="../../nn_models/${model_name}/single_layers/holistic_exp"

    # For each integer from 1 to 128
    for (( banks=1; banks<=128; banks++ )); do
        dst_dir="${base_dir}/${banks}banks"
        
        mkdir -p "$dst_dir"
        
        for file in "$src_dir"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                cp "$file" "$dst_dir/$filename"
                sed -i "s/num_banks = 16/num_banks = ${banks}/g" "$dst_dir/$filename"
            fi
        done
        
        echo "  Created folder: ${banks}banks"
    done

    echo "Done with model: $model_name"
    echo
done

# We separate the model for bert
remaining_list=("bert")

# Loop over each model
for model_name in "${remaining_list[@]}"; do
    echo "Processing model: $model_name"

    # Source directory (containing 'num_banks = 16')
    src_dir="../../nn_models/${model_name}/single_layers/128banks_full"

    # Base directory for creating new folders
    base_dir="../../nn_models/${model_name}/single_layers/holistic_exp"

    # For each integer from 1 to 128
    for (( banks=1; banks<=128; banks++ )); do
        dst_dir="${base_dir}/${banks}banks"
        
        mkdir -p "$dst_dir"
        
        for file in "$src_dir"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                cp "$file" "$dst_dir/$filename"
                sed -i "s/num_banks = 16/num_banks = ${banks}/g" "$dst_dir/$filename"
            fi
        done
        
        echo "  Created folder: ${banks}banks"
    done

    echo "Done with model: $model_name"
    echo
done