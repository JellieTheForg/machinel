import os
import shutil

# Function to move all artwork to source folder and delete subfolders
def move_all_artwork_and_delete_subfolders(source_folder):
    # Iterate through subfolders in source folder
    for subdir, _, files in os.walk(source_folder):
        # Iterate through files in subfolders
        for filename in files:
            # Move file to source folder
            shutil.move(os.path.join(subdir, filename), source_folder)
    
    # Delete subfolders
    for root, dirs, _ in os.walk(source_folder):
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))

source_folder = '/Users/Jellie/Desktop/machinel/images'
move_all_artwork_and_delete_subfolders(source_folder)
