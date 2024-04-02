from PIL import Image
import os

def make_square(image_path):
    img = Image.open(image_path)
    width, height = img.size
    min_dimension = min(width, height)
    left = (width - min_dimension) // 2
    top = (height - min_dimension) // 2
    right = (width + min_dimension) // 2
    bottom = (height + min_dimension) // 2
    new_img = img.crop((left, top, right, bottom))
    return new_img

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            new_img = make_square(input_path)
            new_img.save(output_path)

input_folder = "abstract"
output_folder = "abstract_square"
process_images(input_folder, output_folder)
