import os
import shutil

def copy_and_replace(src_dir, dst_dir, old_str, new_str):
    """
    Copy files from src_dir to dst_dir (creating dst_dir if it doesn't exist),
    and replace old_str with new_str in each file.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over all files in src_dir
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # Only copy if it's a file
        if os.path.isfile(src_path):
            # Copy the file
            shutil.copy(src_path, dst_path)

            # Replace old_str with new_str in the copied file
            with open(dst_path, 'r') as f:
                content = f.read()
            content = content.replace(old_str, new_str)
            with open(dst_path, 'w') as f:
                f.write(content)

def main():
    models = ["alexnet", "bert", "resnet50", "resnet152", "unet", "vgg16"]
    base_dir = "../../nn_models"
    
    # If you want to copy from "128banks" to "32banks"
    src_banks = "16banks"
    dst_banks = "32banks"

    for model in models:
        src_dir = os.path.join(base_dir, model, "single_layers", src_banks)
        dst_dir = os.path.join(base_dir, model, "single_layers", dst_banks)

        # Perform the copy and replace
        copy_and_replace(
            src_dir=src_dir,
            dst_dir=dst_dir,
            old_str="num_banks = 16",
            new_str="num_banks = 32"
        )
        print(f"Processed model {model}: {src_banks} -> {dst_banks}")

if __name__ == "__main__":
    main()
