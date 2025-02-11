import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the pre-trained model
model = load_model('model885.keras')

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((144, 144)) 
    image_array = np.array(image) / 255.0  
    return image_array.reshape(-1, 144, 144, 3) 

# Define the function to generate CAM
def generate_cam(model, x):
    x = preprocess_image(x)
    predicted_class_prob = model.predict(x)[0][0]
    print(predicted_class_prob)
    predicted_class = 1 if predicted_class_prob > 0.5 else 0

    # Get the output of the last convolutional layer
    last_conv_layer = model.get_layer('conv2d_3')
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    last_conv_layer_output = last_conv_layer_model(x)

    # Get the weights of the last fully connected layer
    fc_weights = model.get_layer('dense_1').get_weights()[0]

    # Compute the class activation map (CAM)
    cam = np.zeros(last_conv_layer_output.shape[1:3], dtype=np.float32)
    for i, w in enumerate(fc_weights[:, predicted_class]):
        cam += w * last_conv_layer_output[0, :, :, i]

    # Normalize the CAM
    cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))
    print('Predicted class: ' + ('classical' if predicted_class >=0.5 else 'abstract'))
    return cam, predicted_class

# Define the function to plot only the CAM
def plot_cam(cam):
    plt.imshow(cam, cmap='jet')
    plt.axis('off')
    plt.show()

# Define the function to plot CAM overlaid on the original image
def plot_cam_over_image(img_path, cam):
    # Load the input image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((144, 144))  # Resize to match CAM dimensions

    # Rescale the CAM to match the size of the original image
    cam = np.uint8(255 * cam)
    cam = np.expand_dims(cam, axis=-1)  # Add channel dimension
    cam = tf.image.resize(cam, (img.size[1], img.size[0]))  # Resize CAM to match image dimensions

    # Convert PIL image to NumPy array
    img_array = np.array(img)

    # Plot CAM overlaid on the original image
    plt.imshow(img_array)
    plt.imshow(cam[:, :, 0], cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

# Path to the new image
new_img_path = 'photo.jpg'

# Generate CAM for the new image
cam, predicted_class = generate_cam(model, new_img_path)

# Plot only the CAM
plot_cam(cam)

# Plot CAM overlaid on the original image
plot_cam_over_image(new_img_path, cam)
