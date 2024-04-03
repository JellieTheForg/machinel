from keras.models import load_model
from PIL import Image
import numpy as np

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((100, 100))  # Resize to 32x32 pixels
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array.reshape(1, 100, 100, 1)  # Reshape for model input


model = load_model("CNN.keras")
input_image_path = "mona.jpeg"

# Preprocess the input image
input_image = preprocess_image(input_image_path)

# Make predictions
prediction = model.predict(input_image)

# Convert prediction to class label
if prediction <= 0.5:
    class_label = "abstract"
    pluh = prediction
if prediction >= 0.5:
    class_label = "classical"
    pluh = int((prediction-0.5)*200)
print(f"The AI is {pluh}% sure this is {class_label}")
print("Predicted class:", class_label)
