import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense
import os
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load and preprocess images
image_folder = "all_grey"
images = []
labels = []

for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path)
        images.append(np.array(image))
        if "abstract" in file_name:
            labels.append(0)  # Label as abstract
        else:
            labels.append(1)

X = np.array(images)
y = np.array(labels)
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the images to flatten them
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

encoding_dim = 128
# Define the autoencoder model
input_img = Input(shape=(X_train_scaled.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(X_train_scaled.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=24, batch_size=96, validation_data=(X_test_scaled, X_test_scaled)) # shuffle=True,

# Encode the input data
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Train a classifier using the encoded features
# Train a classifier using the encoded features
classifier = Sequential([
    Dense(128, activation='relu', input_shape=(encoding_dim,)),
    Dense(1, activation='sigmoid')
])


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_encoded, y_train, epochs=56, batch_size=64, validation_split=0.1)

# Evaluate the classifier
test_loss, test_acc = classifier.evaluate(X_test_encoded, y_test)
print(f"Test Accuracy: {test_acc}")

y_pred = classifier.predict(X_test_encoded)
y_pred_binary = (y_pred > 0.5).astype(int)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Labeling the axes
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

# Displaying the values
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='red')

plt.xticks(np.arange(conf_matrix.shape[1]))
plt.yticks(np.arange(conf_matrix.shape[0]))
plt.show()