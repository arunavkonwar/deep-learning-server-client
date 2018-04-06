# From : http://stackoverflow.com/questions/31774953/test-labels-for-regression-caffe-float-not-allowed
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleadi
import h5py, os
import caffe
import numpy as np
from PIL import Image

# For Gerty computer
#rawDataFolderPath = '/media/quentin/OSDisk/trainingSet_10k_txty_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/media/quentin/OSDisk/trainingSet_10k_txty_noZeros/'

# For GANTRY  computer

# For 6DOF
#rawDataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrxryrz_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrxryrz_noZeros/'

# For testSet generation
#rawDataFolderPath = '/local/qbateux/testSet_10k_txtytzrxryrz_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/testSet_10k_txtytzrxryrz_noZeros/'

#rawDataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood/'

#rawDataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_4DOFs/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_4DOFs/'

#rawDataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_2DOFs/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_2DOFs/'

#datasetName = 'trainingSet_DegradedTest_Hollywood_6DOFs'
#datasetName = 'trainingSet_DegradedTest_Hollywood_6DOFs_SinglePos'
#datasetName = 'Afma_Hollywood'
#datasetName = 'Afma_Hollywood_NoDiff' # No difference image, refImage taken from Afma robot directly
#datasetName = 'Afma_Hollywood_NoDiff_degraded' # No difference Image, addition of random localized lights at each images
datasetName = 'Afma_Hollywood_NoDiff_Finer_degraded' # No difference Image, addition of random localized lights at each images, additionnal 1000 samples with much smaller position noise
datasetName = 'Afma_Hollywood_NoDiff_Finer_degraded_occlusions' # No difference Image, addition of random localized lights at each images, additionnal 1000 samples with much smaller position noise
datasetName = 'Afma_Castle_MEDIAN_degraded_occlusions'
datasetName = 'Afma_Castle_MEDIAN2_degraded_occlusions'
#datasetName = 'Afma_Hollywood_NoDiff_Finer_BIG_degraded_occlusions' # 110k, 1occ
#datasetName = 'Afma_Hollywood_NoDiff_Finer_MEDIAN_degraded_occlusions' # 22k, 4occs
#datasetName = 'Afma_Hollywood_NoDiff_Finer_SMALL_degraded_occlusions' # 11k, 4occs

datasetName = 'RealVS_test1' 
datasetName = 'RealVS_test1_degraded_occlusions' 
datasetName = 'RealVS_test1_4DOFs' 
datasetName = 'RealVS_test2'
datasetName = 'RealVS_test2_BIG'
datasetName = 'RealVS_test2_BIGGER'
datasetName = 'RealVS_test3_Coupling'
datasetName = 'RealVS_test3_Coupling4DOFs'
datasetName = 'RealVS_test3_CouplingEasy'
datasetName = 'RealVS_test3_CouplingHard'
datasetName = 'RealVS_test4_CouplingHard_RandZ'
datasetName = 'RealVS_test4_All_RandZ'
datasetName = 'RealVS_test4_All_RandZ2'
datasetName = 'RealVS_test4_All_RandZ2_BIG'

rawDataFolderPath = '/local/qbateux/'+datasetName+'/DatasetIndex'

import random

shuffleSet = True
usePoseNet = False
useDualImages = True #False for PoseNet-style, True for FlowNet-style
Debug = False


lines = open(rawDataFolderPath).readlines()
if shuffleSet:
    random.shuffle(lines)
open(rawDataFolderPath+'_shuffled' , 'w').writelines(lines)

if usePoseNet:
    datasetName = datasetName+'_PoseNet'
    from shutil import copyfile
    copyfile(rawDataFolderPath, '/local/qbateux/'+datasetName+'/DatasetIndex' )
    

hdf5DataFolderPath = '/local/qbateux/'+datasetName+'/'

if not os.path.exists(hdf5DataFolderPath):
    os.makedirs(hdf5DataFolderPath)



# For 4DOF
#rawDataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrz_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrz_noZeros/'


# For 2DOF
#rawDataFolderPath = '/local/qbateux/trainingSet_10k_txty_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txty_noZeros/'

# For 1DOF
#rawDataFolderPath = '/local/qbateux/trainingSet_10k_tx_noZeros/DatasetIndex'
#hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_tx_noZeros/'

SIZE = 227 # fixed size to all images / 227 for AlexNet in caffe model zoo
#if usePoseNet:
#    SIZE = 224 # fixed size to all images / 227 for AlexNet in caffe model zoo
with open( rawDataFolderPath+'_shuffled', 'r' ) as T :
    lines = T.readlines()

#chunkStep = 11000 # For trainingSet
chunkStep = 10000 # For trainingSet
#chunkStep = 100 # For testSet
nbOfChunks = 10
if not Debug:
    if os.path.isfile(hdf5DataFolderPath+'train_h5_list.txt'):
        os.remove(hdf5DataFolderPath+'train_h5_list.txt')
for chunkIndex in range(0, nbOfChunks):
    print 'chunk index = ', chunkIndex
    linesChunks = lines[chunkIndex*chunkStep:(chunkIndex+1)*chunkStep-1]
    #print '1 in chunk = ', linesChunks[0]
    
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    l = linesChunks[0]
    l = l.strip()
    sp = l.split(';') # Take out the extra whitespaces, newlines...etc
        
    X = np.zeros( (len(linesChunks), 3, SIZE, SIZE), dtype='f4' ) 
    y = np.zeros( (len(linesChunks), len(sp)-1), dtype='f4' )
    if usePoseNet:
        y = np.zeros( (len(linesChunks), len(sp)), dtype='f4' ) # Add an extra '0' in the labels (unused, but allow to copy trained-net weights directly 
    for i,l in enumerate(linesChunks):
        if i % 100 == 0:
            print i
        l = l.strip()
        sp = l.split(';') # Take out the extra whitespaces, newlines...etc
        #print 'len(split line) = ', len(sp)
        #print 'type = ', type(len(sp))
        if not useDualImages:
            img = caffe.io.load_image( sp[0])
        else:
            img = caffe.io.load_image( sp[0]+'_1.png'  )
            img2 = caffe.io.load_image( sp[0]+'_2.png'  )
            #print img[0][0][0], ' / ', img[0][0][1], ' / ', img[0][0][2]
            img[:,:,1] = img2[:,:,0]
            img[:,:,2] = 0
            #print 'AFTER = ',img[0][0][0], ' / ', img[0][0][1], ' / ', img[0][0][2]

        
        # first, resize the spatial dimensions, do not touch the channels
        img = caffe.io.resize_image( img, (SIZE,SIZE), interp_order=3 )
        
        if Debug:
            plt.figure()
            plt.imshow(img[:, :, 0])
            plt.show()
            plt.figure()
            plt.imshow(img[:, :, 1])
            plt.show()
        

        # transpose the dimensions from H-W-C to C-H-W
        img = img.transpose( (2,0,1) )
        #img = caffe.io.resize( img, (3, SIZE, SIZE) ) # resize to fixed size # BREAK the channels : BAD !
            
        
       
        
        # you may apply other input transformations here...
        X[i] = img
        for dim  in range(0,len(sp)-1):
            y[i, dim] = float(sp[dim+1])
            #print "y = "    
            #print y[:,i]
    if not Debug:
        with h5py.File(hdf5DataFolderPath+'train'+str(chunkIndex)+'.h5','w') as H:
            H.create_dataset( 'data', data=X ) # note the name X given to the dataset!
            H.create_dataset( 'label', data=y ) # note the name y given to the dataset!
        with open(hdf5DataFolderPath+'train_h5_list.txt','a') as L:
            L.write(hdf5DataFolderPath+'train'+str(chunkIndex)+'.h5\n') # list all h5 files you are going to use


#if not Debug:
#    with h5py.File(hdf5DataFolderPath+'train0.h5','r') as hf:
#        print "List of arrays in this file:"
#        print hf.keys()
#        data = hf.get('data')
#        np_data = np.array(data)
#        print "Shape of the array data: "
#        print np_data.shape
#
#        data = hf.get('label')
#        np_data = np.array(data)
#        print "Shape of the array y: "
#        print np_data.shape
