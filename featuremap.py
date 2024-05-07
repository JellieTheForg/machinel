from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

# reshape array so each channel is the first dimension
def reshapearray(a):
	# create empty array of the desired dimensions
	newarray = np.empty((a.shape[-1],a.shape[-2],a.shape[-3]))
	# get rid of unnecessary first dimension
	a = a[0]
	for i,line in enumerate(a):
		for j,pixel in enumerate(line):
			for k,channel in enumerate(pixel):
				newarray[k][i][j] = channel
	return newarray
				

model = load_model("model885.keras")
image_path = "sean.jpg"

# preprocess
image = Image.open(image_path).convert("RGB") # load in image
image = image.resize((144,144)) # resize to 144 by 144
image = np.array(image) # convert to np array

print(image.shape)

image = image / 255.0 # normalize (max value will now be 1)
image = image.reshape(-1, 144, 144, 3) # Reshape for model input

''' #start
possible layers: 'conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'conv2d_3', 'max_pooling2d_3', 'flatten', 'dense', 'dense_1', 'dense_2'


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(144, 144, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (7, 7), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (9,9), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')

''' # end
# get layer
layer = model.get_layer("conv2d")

# send image through layer and reshape the array that comes out
output = layer(image)
output = np.array(output)
# shape: 1, 142, 142, 32
output2 = reshapearray(output)

''' #start
# graph any layer that isnt just 0



for i,thing in enumerate(output2):
	if len(np.unique(thing)) > 1:
		imgplot = plt.imshow(thing*255)
		plt.title = "channel "+i
		plt.show()

# average all channels together
output = output[0]
averaged = np.empty((output.shape[0],output.shape[1]))
for i,line in enumerate(output):
	for j,pixel in enumerate(line):
		averaged[i][j] = mean(pixel)

# show averaged image
imgplot2 = plt.imshow(averaged*255)
plt.show()

''' #end

output = output[0]

averaged_rgb = np.empty((output.shape[0],output.shape[1],3))


for i,line in enumerate(output):
	for j,pixel in enumerate(line):
		redtemp = []
		greentemp = []
		bluetemp = []
		for k, channel in enumerate(pixel):
			if k%3 == 0:
				redtemp.append(channel*255)
			elif k%3 == 1:
				greentemp.append(channel*255)
			else:
				bluetemp.append(channel*255)
		averaged_rgb[i][j][0] = int(mean(redtemp))
		averaged_rgb[i][j][1] = int(mean(greentemp))
		averaged_rgb[i][j][2] = int(mean(bluetemp))

# show averaged image
imgplot2 = plt.imshow(averaged_rgb)
plt.show()

