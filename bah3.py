import os
import shutil
from collections import defaultdict

# Function to limit artwork to 5 pieces per artist
def limit_artwork(source_folder):
    # Dictionary to keep track of the number of pieces per artist
    artist_count = defaultdict(int)
    
    # Iterate through files in source folder
    for filename in os.listdir(source_folder):
        # Extract artist name from filename
        parts = filename.split('_')
        artist_name = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        
        # Check if artist count is less than 5
        if artist_count[artist_name] < 5:
            # Increment count for the artist
            artist_count[artist_name] += 1
        else:
            # If artist has more than 5 pieces, delete the extra artwork
            os.remove(os.path.join(source_folder, filename))


source_folder = '/Users/Jellie/Desktop/machinel/images'
limit_artwork(source_folder)
