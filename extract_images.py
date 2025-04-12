# extract_images.py

import json
import base64
import os
import sys
from PIL import Image
from io import BytesIO

def extract_images_from_notebook(nb_file, output_folder):
    """
    Extracts embedded images from a Jupyter Notebook (.ipynb file)
    and saves them as JPEG files in the specified output folder.
    
    Args:
        nb_file (str): Path to the notebook file.
        output_folder (str): Path to the folder where images will be saved.
        
    Returns:
        int: Number of images extracted.
    """
    
    # Open the notebook file and load it as JSON
    with open(nb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    image_count = 0
    
    # Iterate over each cell in the notebook
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                # Check if the cell output contains display data or execution results
                if output.get('output_type') in ['display_data', 'execute_result']:
                    data = output.get('data', {})
                    # If the output contains an image in PNG format, extract it
                    if 'image/png' in data:
                        image_data = data['image/png']
                        
                        try:
                            # Decode the base64 data
                            img_bytes = base64.b64decode(image_data)
                        except Exception as e:
                            print(f"Error decoding image data: {e}")
                            continue
                        
                        # Convert the bytes to an image using PIL and ensure it's in RGB
                        try:
                            img = Image.open(BytesIO(img_bytes)).convert('RGB')
                        except Exception as e:
                            print(f"Error processing image data: {e}")
                            continue

                        # Save the image as a JPEG file in the output folder
                        output_filename = os.path.join(output_folder, f"extracted_image_{image_count}.jpg")
                        img.save(output_filename, 'JPEG')
                        print(f"Saved image to {output_filename}")
                        image_count += 1
    if image_count == 0:
        print(f"No images found in notebook {nb_file}.")
    return image_count

if __name__ == '__main__':
    # Usage: python extract_images.py <notebook_file> <output_folder>
    if len(sys.argv) < 3:
        print("Usage: python extract_images.py <notebook_file> <output_folder>")
    else:
        nb_file = sys.argv[1]
        output_folder = sys.argv[2]
        count = extract_images_from_notebook(nb_file, output_folder)
        print(f"Extracted {count} images.")
