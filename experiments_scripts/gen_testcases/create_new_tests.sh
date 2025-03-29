#!/bin/bash

# Define source and destination directories
src_dir="../../nn_models/alexnet/single_layers/128banks"
dst_dir="../../nn_models/alexnet/single_layers/32banks"

# Create destination directory if it doesn't exist
mkdir -p "$dst_dir"

# Loop over all files in the source directory
for file in "$src_dir"/*; do
    if [ -f "$file" ]; then
        # Get the base filename
        filename=$(basename "$file")
        # Copy the file to the destination directory
        cp "$file" "$dst_dir/$filename"
        # Modify the copied file: change "num_banks = 16" to "num_banks = 32"
        sed -i 's/num_banks = 128/num_banks = 32/g' "$dst_dir/$filename"
    fi
done
