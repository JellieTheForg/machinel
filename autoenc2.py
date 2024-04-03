import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = image.resize((32, 32))  # Resize to 32x32 pixels
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

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(-1, 32, 32, 1), y_train, epochs=18, batch_size=16, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 32, 32, 1), y_test)
print(f"Test Accuracy: {test_acc}")


plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Generate confusion matrix
y_pred = model.predict(X_test.reshape(-1, 32, 32, 1))
y_pred_binary = (y_pred > 0.5).astype(int)
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
