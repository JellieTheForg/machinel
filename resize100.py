from PIL import Image
import os

def resize_to_32x32(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100))
    return img

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            new_img = resize_to_32x32(input_path)
            new_img.save(output_path)

input_folder = "images_square"
output_folder = "images_100"
process_images(input_folder, output_folder)
