import os
import numpy as np
from keras.models import load_model
from PIL import Image 

model = load_model('best_model.keras')
abstract_folder = 'abstract'
classical_folder = 'classic'
total_images = 0
correct_predictions = 0

#da function
def evaluate_artworks(folder, true_label):
    global correct_predictions
    global total_images
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((144, 144))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict the class of the artwork
        predicted_class_prob = model.predict(image_array)[0][0]
        predicted_class = 1 if predicted_class_prob > 0.5 else 0

        if predicted_class == true_label:
            correct_predictions += 1

        total_images += 1


evaluate_artworks(abstract_folder, 0)
evaluate_artworks(classical_folder, 1)

#total accuracy
accuracy = correct_predictions / total_images

print("Total Accuracy:", accuracy)
