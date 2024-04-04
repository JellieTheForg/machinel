import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from PIL import Image
from sklearn.metrics import confusion_matrix

# Load and preprocess images
image_folder = "art_144"
images = []
labels = []

for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        images.append(image)
        if "abstract" in file_name:
            labels.append(0)  #0 is abstract, 1 is classical
        else:
            labels.append(1)

X = np.array(images)
y = np.array(labels)
X = X / 255.0 #normalises the arrays

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#this is the actual neural network path
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(144, 144, 3)), #every convd2d thing applies this filter over the previous input
    MaxPooling2D((2, 2)), #every maxpooling reduces the dimensions
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (7, 7), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (9,9), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(), 
    Dense(64, activation='relu'), #these just pass it down to lower and lower dimensions until you have only one(0 or 1) where you use sigmoid
    Dense(32, activation='relu'),  
    Dense(1, activation='sigmoid')
])

#compiling, adam is the most used
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#here is where we can hyperadjust different params
history = model.fit(X_train, y_train, epochs=5, batch_size=80, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

model.save("CNN_RGB.keras")

plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#confusion matrix
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()


plt.xlabel('Predicted labels')
plt.ylabel('True labels')

#displaying the values
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='red')

plt.xticks(np.arange(conf_matrix.shape[1]))
plt.yticks(np.arange(conf_matrix.shape[0]))
plt.show()
