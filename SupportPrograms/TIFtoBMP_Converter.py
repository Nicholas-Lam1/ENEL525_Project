from PIL import Image
import os

input_dir = ".\\Dataset\\UCMerced_LandUse\\Images"
output_dir = ".\\BMPDataset"

os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    relative_path = os.path.relpath(root, input_dir)
    current_output_dir = os.path.join(output_dir, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for file_name in files:
        if file_name.endswith(".tif"):
            input_path = os.path.join(root, file_name)
            output_path = os.path.join(current_output_dir, file_name.replace(".tif", ".bmp"))
            
            with Image.open(input_path) as img:
                img.save(output_path)
