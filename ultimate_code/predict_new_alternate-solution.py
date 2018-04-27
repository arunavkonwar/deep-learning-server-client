'''http://marubon-ds.blogspot.fr/2017/08/how-to-make-fine-tuning-model.html
'''

def vgg16():
	import keras
	from keras.models import Sequential
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	from keras.layers.normalization import BatchNormalization
	from keras.layers.convolutional import *
	import matplotlib.pyplot as plt
	from keras.utils import plot_model 
	from keras import models
	from keras import layers


	resnet = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), classes=1000)

	model = models.Sequential()
	model.add(resnet)
	model.add(layers.Flatten())
	model.add(Dense(2, activation=None))
	
	
	layer_num = len(resnet.layers)
	
	for layer in resnet.layers[:int(layer_num * 0.9)]:
        	layer.trainable = False
	model_num = len(model.layers)
	'''
	for layer in model.layers[int(model_num * 0.8):]:
        	layer.trainable = True
        '''	
	
	model.summary()
	resnet.summary()
	print len(resnet.layers)
	print(layer_num * 0.8)
	return model
	
	
	
def initializeNetwork():
	from keras.models import load_model
	import h5py

	#initialize_model = load_model('trained_model.h5')
	initialize_model = vgg16()
	initialize_model.load_weights('trained_model_resnet50_90percent_1-50_adam_001.h5')
	return initialize_model

	



if __name__ == '__main__':
	import os
	import numpy as np
	import keras
	from keras import backend as K
	from keras.models import Sequential
	from keras.models import load_model
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	from keras.preprocessing.image import ImageDataGenerator
	from keras.layers.normalization import BatchNormalization
	from keras.layers.convolutional import *
	from sklearn.metrics import confusion_matrix
	import itertools
	import matplotlib.pyplot as plt
	import h5py
	from os import listdir
	from os.path import isfile, join
	import numpy as np
	import cv2
	import time
	import struct
	import codecs
	

	import sys
	import math
	import cPickle
	import signal
	
	import glob
	import skimage.io
	import skimage.exposure

	import matplotlib.pyplot as plt



	model = initializeNetwork()
	print "/nModel initialized..."
	#img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

	file1Path = "/dev/shm/fifo_server"
	file2Path = "/dev/shm/fifo_client"

	print "Server Waiting for Client to connect "
	
	while(1):
		print 'Handshake...'
		f = open(file1Path, 'r')
	
		arraySize = struct.unpack('i', f.read(4))[0] # Reading array size
		print 'Received int = ', arraySize
		strLength = `arraySize`+'i'
		
		array = struct.unpack(strLength, f.read(arraySize*4)) # Reading the actual array
		
		print 'Received array size = ', len(array)
		f.close()
		
		imageSize1 = 224 
    		imageSize2 = 224
    		
		image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
		#print image.shape
		#a=(array[:]).reshape(224,224)

		# Unpacking the array from the vector shape
		imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')/255.0
		#imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')
		image = np.stack((imageTmp,)*3, -1)
		
		'''
		image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
		image[:, :, 0] = imageTmp
		image[:, :, 1] = imageTmp
		image[:, :, 2] = imageTmp 
		'''
		
		'''
		image[:, :, 1] = imageTmp
		image[:, :, 2] = imageTmp 
		'''
		
		start2 = time.clock()
		#desc = getCNNDescriptor(image, 'conv3')
		#desc = getRelativePose(image)
		img_main = image[np.newaxis,...]
		prediction = model.predict(img_main)
		end2 = time.clock() 
		
		print(prediction)
		
		float_formatter = lambda x: "%.6f" % x
		np.set_printoptions(formatter={'float_kind':float_formatter})
		
		yup = prediction[0]
		arraySize = yup.shape[0]
		print "shape:"
		print arraySize
		
		print 'Sending data back ; ArraySize'
		wp = open(file2Path, 'w')
		wp.write(struct.pack('>i',arraySize))
		
		print 'Sending data back ; FLOAT_Array'
		#wp = open(file2Path, 'w')
		packed = struct.pack('<'+`arraySize`+'f', *yup)
		#print 'sending = ' + `packed`
		wp.write(packed)
		wp.close()
		
		print 'Array Sent'

		end1 = time.clock()
		
		#print 'Elapsed server time total = ' + `elapsed1` + 'ms'
		#print 'Elapsed time descriptor computation = ' + `elapsed2` + 'ms\n'
		print 'Ending handshake.' 
		'''
		while(True):
			try:
				mainCommunication(useDualImages)
			except Exception, exc:
				print exc
				
		'''
		
		
