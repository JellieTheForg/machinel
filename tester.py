from keras.models import load_model
from PIL import Image
import numpy as np

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to grayscale
    image = image.resize((144, 144))  # Resize to 32x32 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array.reshape(-1, 144, 144, 3)  # Reshape for model input


model = load_model("best_model.keras")
input_image_path = "classical_test.jpg"

# Preprocess the input image
input_image = preprocess_image(input_image_path)

# Make predictions
prediction = model.predict(input_image)

# Convert prediction to class label
if prediction <= 0.5:
    class_label = "abstract"
    print(prediction)
    pluh = (1-(prediction*2))*100
if prediction >= 0.5:
    class_label = "classical"
    pluh = int((prediction-0.5)*200)
print(f"The AI is {pluh}% sure this is {class_label}")
print("Predicted class:", class_label)
