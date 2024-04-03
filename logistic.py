import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load images and labels
def load_images_and_labels(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(folder_path, filename))
            img_array = np.array(img).flatten()  # Flatten the image to a 1D array
            images.append(img_array)
            if "abstract" in filename:
                labels.append(0)  # Assign label 0 for abstract artworks
            else:
                labels.append(1)  # Assign label 1 for classical artworks
    return np.array(images), np.array(labels)

# Load images and labels
folder_path = "all_grey"
images, labels = load_images_and_labels(folder_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

np.save('y_test.npy', y_test)
np.save('y_pred.npy', y_pred)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=["Abstract", "Classical"], yticklabels=["Abstract", "Classical"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
