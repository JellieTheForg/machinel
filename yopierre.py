import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained model
model = load_model('best_model.keras')

# Define the function to generate CAM
def generate_cam(model, img_path, target_size=(144, 144)):
    # Load and preprocess the input image
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)

    # Get the prediction for the input image
    preds = model.predict(x)
    predicted_class_prob = preds[0][0]  # Probability of class 0 (abstract)
    predicted_class = np.argmax(preds[0])

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
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam, predicted_class_prob


# Define the function to plot CAM over the original image
def plot_cam_over_image(img_path, cam):
    # Load the input image
    img = image.load_img(img_path, target_size=(144, 144))
    img = image.img_to_array(img)

    # Rescale the CAM to match the size of the original image
    cam = np.uint8(255 * cam)
    cam = np.expand_dims(cam, axis=-1)  # Add channel dimension
    cam = tf.image.resize(cam, (144, 144))

    plt.imshow(img.astype('uint8'))
    plt.imshow(cam[:,:,0], cmap='jet', alpha=0.5)  # Ensure cam is 2D
    plt.axis('off')
    plt.show()

# Path to the new image
new_img_path = 'test.jpg'

# Generate CAM for the new image
cam, predicted_class = generate_cam(model, new_img_path)

# Print the predicted class
print(predicted_class)
print("Predicted class:", "abstract" if predicted_class <=0.5 else "classical")

# Plot CAM overlaid on the original image
plot_cam_over_image(new_img_path, cam)
