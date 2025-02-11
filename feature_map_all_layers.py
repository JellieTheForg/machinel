from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import os

# reshape array so each channel is the first dimension
def reshapearray(a):
	# create empty array of the desired dimensions
	newarray = np.empty((a.shape[-1],a.shape[-2],a.shape[-3]))
	for i,line in enumerate(a):
		for j,pixel in enumerate(line):
			for k,channel in enumerate(pixel):
				newarray[k][i][j] = channel
	return newarray
				

model = load_model("best_model.keras")
image_path = "heart.jpeg"

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
# get layers
layerlist = ['conv2d', 'max_pooling2d', 'conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2', 'conv2d_3', 'max_pooling2d_3', 'flatten', 'dense', 'dense_1', 'dense_2']

layers = []
for lay in layerlist:
	layers.append(model.get_layer(lay))

images = []
rawchannels = []

# send image through layer and reshape the array that comes out. send the output of one layer
# to the next, saving the image at each stage
for i_real,layer in enumerate(layers):
	rawchannels.append([])
	if i_real == 0:
		output = layer(image)
	else:
		output = layer(sendtonext)
		
	output2 = np.array(output)
	output2 = output2[0]
	if i_real < 8:
		output3 = reshapearray(output2)
		averaged = np.empty((output2.shape[0],output2.shape[1]))
		for i,line in enumerate(output2):
			for j,pixel in enumerate(line):
				averaged[i][j] = mean(pixel)
	else:
		output3 = output2
		averaged = mean(output2)
	images.append(averaged)
	sendtonext = output
	rawchannels[i_real] = output3

# display averaged images
for im in images:
	if len(im.shape)>1:
		implot = plt.imshow(im*255,cmap='magma')
		plt.show()
	else:
		print(im)

# make directories for each layer, put each channel in directories

for i,layer in enumerate(layerlist):
	if i < 8:
		try:
			os.mkdir('./featuremap_pics/'+layer)
		except FileExistsError:
			print("folder already exists dw :)")
		for ij,chan in enumerate(rawchannels[i]):
			c = plt.imsave("./featuremap_pics/"+layer+'/'+str(ij)+'.png',chan*255,cmap='magma')

