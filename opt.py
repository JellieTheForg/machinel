import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

y_test = np.load('y_test.npy')
y_pred = np.load('y_pred.npy')

image_indices = np.arange(len(y_test))  # Image indices
actual_labels = y_test  # Actual labels (ground truth)
predicted_labels = y_pred  # Predicted labels by your ML model

# Function to plot images alongside the graph
def plot_images(image_indices, actual_labels, predicted_labels, folder_path, images_per_graph=5):
    num_images = len(image_indices)
    num_graphs = num_images // images_per_graph + (num_images % images_per_graph > 0)  # Calculate number of graphs needed

    for graph_num in range(num_graphs):
        start_index = graph_num * images_per_graph
        end_index = min(start_index + images_per_graph, num_images)
        
        num_rows = (end_index - start_index + 1) // 3 + ((end_index - start_index + 1) % 3 > 0)  # Calculate number of rows needed
        num_cols = min(end_index - start_index + 1, 3)  # Maximum of 3 columns
        
        plt.figure(figsize=(12, 6))
        for i, index in enumerate(range(start_index, end_index)):
            # Choose the nth photo from the folder
            img_filename = os.listdir(folder_path)[index]  # Get the filename corresponding to the index

            # Load image
            img_path = os.path.join(folder_path, img_filename)
            img = Image.open(img_path)

            # Plot image
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Image: {img_filename}\nActual: {actual_labels[index]}, Predicted: {predicted_labels[index]}")
            plt.axis('off')
        
        # Save the graph to a folder
        save_folder = os.path.join(folder_path, "graphs")
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"graph_{graph_num + 1}.png"))
        plt.close()

# Replace 'folder_path' with the path to your images folder
folder_path = "all_grey"

# Plot images and save them to a folder
plot_images(image_indices, actual_labels, predicted_labels, folder_path, images_per_graph=6)