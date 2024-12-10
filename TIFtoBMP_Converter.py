from PIL import Image
import os

# Define the input directory containing .tif files and output directory
input_dir = "C:\\Users\\nick4\\Documents\\University\\Fall_2024\\Classes\\ENEL_525\\Project\\Code\\Dataset\\UCMerced_LandUse\\Images"
output_dir = "C:\\Users\\nick4\\Documents\\University\\Fall_2024\\Classes\\ENEL_525\\Project\\Code\\BMPDataset"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Traverse through all directories and subdirectories
for root, _, files in os.walk(input_dir):
    # Create a corresponding subdirectory structure in the output directory
    relative_path = os.path.relpath(root, input_dir)
    current_output_dir = os.path.join(output_dir, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    # Loop through all .tif files in the current directory
    for file_name in files:
        if file_name.endswith(".tif"):
            input_path = os.path.join(root, file_name)
            output_path = os.path.join(current_output_dir, file_name.replace(".tif", ".bmp"))  # Change to .bmp for BMP format
            
            # Open the .tif file and save it as .png or .bmp
            with Image.open(input_path) as img:
                img.save(output_path)
                print(f"Converted: {input_path} -> {output_path}")
