import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('best_model.keras')

# Extract the weights of the first dense layer
weights = model.layers[9].get_weights()[0]

# Transpose the weight matrix
weights = np.transpose(weights)

# Plot the output weights
plt.imshow(weights, cmap='RdYlGn', interpolation='nearest', aspect='auto')

# Resize the plot into 144x144 pixels
plt.gcf().set_size_inches(144 / plt.gcf().dpi, 144 / plt.gcf().dpi)

# Remove axis ticks and labels
plt.axis('off')

# Save the resized plot
plt.savefig('weights_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=144)

# Show the resized plot (optional)
plt.show()
