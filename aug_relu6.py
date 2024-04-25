import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Load and preprocess images
image_folder = "art_144"
images = []
labels = []

fname = "aug_relu6"

for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        images.append(image)
        if "abstract" in file_name:
            labels.append(0)  # 0 is abstract, 1 is classical
        else:
            labels.append(1)

X = np.array(images)
y = np.array(labels)
X = X / 255.0  # Normalizes the arrays

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,  
    zoom_range=0.2, 
    width_shift_range=0.1, 
    height_shift_range=0.1,  
    fill_mode='nearest'  # Fill mode
)

datagen.fit(X_train)

# Neural network architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu6', input_shape=(144, 144, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu6'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (7, 7), activation='relu6'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (9,9), activation='relu6'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the model checkpoint
checkpoint_path = "best_model.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Training the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Load the best model
model = load_model(checkpoint_path)

# Plotting accuracy over epochs
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(fname+"/acc_"+fname)
plt.clf()

# Plotting val_loss over epochs
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(fname+"/valloss_"+fname)

# Confusion matrix
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
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
plt.savefig(fname+"/confmatx_"+fname)
