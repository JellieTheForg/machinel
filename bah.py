import os
from collections import defaultdict
import shutil

# Directory containing the abstract art images
directory = "/Users/Jellie/Desktop/machinel/abstract_art_512"

# Dictionary to store image counts per artist
artist_counts = defaultdict(int)

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # Extract artist name from filename
        artist_name = filename.split("_")[1]
        
        # Increment count for this artist
        artist_counts[artist_name] += 1
        
        # If this is the fourth or later image for this artist, delete it
        if artist_counts[artist_name] > 3:
            os.remove(os.path.join(directory, filename))

# Print confirmation message
print("Done! Only three pieces of art per artist are kept.")
