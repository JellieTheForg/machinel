from PIL import Image
import os

def crop_center(image, target_size):
    width, height = image.size
    left = (width - target_size) // 2
    top = (height - target_size) // 2
    right = (width + target_size) // 2
    bottom = (height + target_size) // 2
    return image.crop((left, top, right, bottom))

def resize_and_convert_to_grayscale(image_path, target_size=144):
    img = Image.open(image_path)
    img = crop_center(img, min(img.size))
    img = img.resize((target_size, target_size))
    return img

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            new_img = resize_and_convert_to_grayscale(input_path)
            new_img.save(output_path)

input_folder = "art"
output_folder = "art_144"
process_images(input_folder, output_folder)
