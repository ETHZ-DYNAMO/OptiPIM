#!/bin/bash

# Define the source directory (which already exists and contains num_banks = 128)
src_dir="../../nn_models/unet/single_layers/16banks"

# Define the base directory in which all bank folders will be created
base_dir="../../nn_models/unet/single_layers"

# Loop over all banks from 1 to 128
for (( banks=1; banks<=128; banks++ )); do
    # If you prefer to recreate the 128 directory as well, remove this check
    if [ $banks -eq 128 ]; then
        # Uncomment the 'continue' below if you do NOT want to copy for 128
        # continue
        :
    fi

    # Construct the destination directory name, e.g., '1banks', '2banks', ... '128banks'
    dst_dir="${base_dir}/${banks}banks"
    
    # Create the destination directory if it doesn't exist
    mkdir -p "$dst_dir"

    # Copy each file from the source directory and replace "num_banks = 128" with the new bank number
    for file in "$src_dir"/*; do
        if [ -f "$file" ]; then
            # Get the base filename
            filename=$(basename "$file")
            
            # Copy the file to the new directory
            cp "$file" "$dst_dir/$filename"
            
            # Modify the copied file: change "num_banks = 128" to "num_banks = <banks>"
            sed -i "s/num_banks = 16/num_banks = ${banks}/g" "$dst_dir/$filename"
        fi
    done
done
