'''http://marubon-ds.blogspot.fr/2017/08/how-to-make-fine-tuning-model.html
'''

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image

WEIGHTS_PATH = 'http://redcatlabs.com/downloads/inception_v1_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'http://redcatlabs.com/downloads/inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5'

# conv2d_bn is similar to (but updated from) inception_v3 version
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer=True,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, (num_row, num_col),
            strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)
    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x
    
# Convenience function for 'standard' Inception concatenated blocks
def concatenated_block(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    
    branch_0 = conv2d_bn(x, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")

    branch_1 = conv2d_bn(x, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")

    branch_2 = conv2d_bn(x, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")

    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3_a_max")(x)  
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x

def InceptionV1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1001):
    """Instantiates the Inception v1 architecture.
    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/abs/1409.4842v1
    
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1001:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1001')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        #default_size=299,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # 'Sequential bit at start'
    x = img_input
    x = conv2d_bn(x,  64, 7, 7, strides=(2, 2), padding='same',  name='Conv2d_1a_7x7')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_2a_3x3')(x)  
    
    x = conv2d_bn(x,  64, 1, 1, strides=(1, 1), padding='same', name='Conv2d_2b_1x1')  
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='same', name='Conv2d_2c_3x3')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_3a_3x3')(x)  
    
    # Now the '3' level inception units
    x = concatenated_block(x, (( 64,), ( 96,128), (16, 32), ( 32,)), channel_axis, 'Mixed_3b')
    x = concatenated_block(x, ((128,), (128,192), (32, 96), ( 64,)), channel_axis, 'Mixed_3c')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_4a_3x3')(x)  

    # Now the '4' level inception units
    x = concatenated_block(x, ((192,), ( 96,208), (16, 48), ( 64,)), channel_axis, 'Mixed_4b')
    x = concatenated_block(x, ((160,), (112,224), (24, 64), ( 64,)), channel_axis, 'Mixed_4c')
    x = concatenated_block(x, ((128,), (128,256), (24, 64), ( 64,)), channel_axis, 'Mixed_4d')
    x = concatenated_block(x, ((112,), (144,288), (32, 64), ( 64,)), channel_axis, 'Mixed_4e')
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_4f')

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='MaxPool_5a_2x2')(x)  

    # Now the '5' level inception units
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_5b')
    x = concatenated_block(x, ((384,), (192,384), (48,128), (128,)), channel_axis, 'Mixed_5c')
    

    if include_top:
        # Classification block
        
        # 'AvgPool_0a_7x7'
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  
        
        # 'Dropout_0b'
        x = Dropout(0.2)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction
        
        #logits = conv2d_bn(x,  classes, 1, 1, strides=(1, 1), padding='valid', name='Logits',
        #                   normalizer=False, activation=None, )  
        
        # Write out the logits explictly, since it is pretty different
        x = Conv2D(classes, (1, 1), strides=(1,1), padding='valid', use_bias=True, name='Logits')(x)
        
        x = Flatten(name='Logits_flat')(x)
        #x = x[:, 1:]  # ??Shift up so that first class ('blank background') vanishes
        # Would be more efficient to strip off position[0] from the weights+bias terms directly in 'Logits'
        
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(    name='global_pooling')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Finally : Create model
    model = Model(inputs, x, name='inception_v1')
    
    # LOAD model weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v1_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='723bf2f662a5c07db50d28c8d35b626d')
        else:
            weights_path = get_file(
                'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='6fa8ecdc5f6c402a59909437f0f5c975')
                
        #load weights file
        model.load_weights(weights_path)
        
        
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
            
    x = Dense(2048, name='last_fc')(x)
            
    trans_fc = Dense(3,name='trans_fc')(x)

    rot_fc = Dense(3,name='rot_fc')(x) 
    
    model1 = Model(inputs=inputs, outputs=[trans_fc,rot_fc])
    
    return model1
	
	
	
def initializeNetwork():
	from keras.models import load_model
	import h5py

	#initialize_model = load_model('trained_model.h5')
	initialize_model = InceptionV1()
	initialize_model.load_weights('googlenet_single-output_imagenet_iter-150.h5')
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

	from PIL import Image
	import csv



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
		#print array
		
		
		print 'Received array size = ', len(array)
		f.close()
		
		imageSize1 = 224 
    		imageSize2 = 224
    		
		#a=(array[:]).reshape(224,224)

		# Unpacking the array from the vector shape
		
		imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')/255.0
		#imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')
		
		cv2.imshow('BOSS image',imageTmp)
		#cv2.waitKey(0)
		#imageTmp = imageTmp.astype(np.uint8)
		
		#imageTmp = skimage.exposure.rescale_intensity(imageTmp * 1.0, out_range=np.float32)
		im = Image.fromarray(imageTmp)
		#im.show()
		print(np.isfortran(imageTmp))
		
		#
		#print(image.shape)
		
		
		image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
		image = np.stack((imageTmp,)*3, -1)
		
		'''
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
		
		
