#!/usr/bin/python


def getCNNDescriptor(img, layerName):
    #mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    #mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    #print 'mean-subtracted values:', zip('BGR', mu)
    # create transformer for the input called 'data'
    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    #transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    #net.blobs['data'].reshape(1,        # batch size
    #                          3,         # 3-channel (BGR) images
    #                          227, 227)  # image size is 227x227
    
    #transformed_image = transformer.preprocess('data', image)
    
    SIZE = 227 # For AlexNet
    #SIZE = 224 # For PoseNet
    # first, resize the spatial dimensions, do not touch the channels
    img = caffe.io.resize_image( img, (SIZE,SIZE), interp_order=3 )
    # transpose the dimensions from H-W-C to C-H-W
    img = img.transpose( (2,0,1) )
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[0, ...] = img
    
    # Passing the image through the CNN
    t1 = time.clock()
    output = net.forward()
    t2 = time.clock()
    #print 'Forward Time = ', (t2-t1)*1000
    # Extracting the filter parameters
    result = net.blobs[layerName].data[0, :] # We get the descriptor associated to the desired image.
    return result.flatten()

def getRelativePose(img, usePoseNet=False):
    SIZE = 227 # For AlexNet
    #SIZE = 224 # For PoseNet
    img = caffe.io.resize_image( img, (SIZE,SIZE), interp_order=3 )
    img = img.transpose( (2,0,1) )
    #net.blobs['data'].data[0, ...] = img
    net.blobs['data'].data[...] = img
    t1 = time.clock()
    net.forward()
    t2 = time.clock()
    print 'Forward Time = ', (t2-t1)*1000

    for layer_name, blob in net.blobs.iteritems():
        name = layer_name
        print layer_name + '\t' + str(blob.data.shape)
    #resultPose = net.blobs['fc8_VS_new'].data[0]
    resultPose = net.blobs[layer_name].data[0]
    print '[Server] Relative Pose =', resultPose
    return resultPose
    
def initializeNetwork(usePoseNet=False):
# The caffe module needs to be on the Python path;
    # we'll add it here explicitly.
    #import sys
    #caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    #sys.path.insert(0, caffe_root + 'python')
    import numpy
    #print numpy.__path__
    import caffe
    # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
    import os
    if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        print 'CaffeNet model not found'
    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet
    
    caffe.set_mode_cpu()
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    
    # Trained with ImageNet->Lena, 10k, 6DOFs
    #model_weights = 'Converged_txtytzrxryrz/weightsVS.pretrained, end-to-end.caffemodel'
    
    # Trained with Lena->Hollywood, 10k, 6DOFs
    #model_weights = 'Hollywood_Converged_6DOFs/weightsVS.pretrained, end-to-end.caffemodel'
    
    # Trained with Lena->Hollywood->HollywoodIllu, 10k, 6DOFs
    #model_weights = 'Hollywood_Converged_Illu_0.3_6DOFs/weightsVS.pretrained, end-to-end.caffemodel'
    # Trained with an Hollywood_Triangle image taken from the Afma Robot
    #model_weights = 'Hollywood_AfmaRef/weightsVS.pretrained, end-to-end.caffemodel'
    
    # Trained with an Hollywood_Triangle image taken from the Afma Robot
    # Trained from 'Hollywood_Converged_6DOFs' net
    #model_weights = 'Hollywood_AfmaRef_LocalLights/weightsVS.pretrained, end-to-end.caffemodel'
    
    # Trained with an Hollywood_Triangle image taken from the Afma Robot, with Locallights added to dataset
    # Trained from 'Hollywood_Converged_6DOFs' net
    #model_weights = '/local/qbateux/Saves_Nets/Hollywood_Finer_LocalLights/weightsVS.pretrained, end-to-end.caffemodel'

    # Trained with an Hollywood_Triangle image taken from the Afma Robot, with Locallights and superpixels occlusions added to dataset
    #model_weights = '/media/quentin/OSDisk/Saves_Nets/Hollywood_Finer_LocalLights_Occlusions/weightsVS.pretrained, end-to-end.caffemodel'
    model_weights = '/media/quentin/ExternalDis/Saves_Nets/Hollywood_Finer_LocalLights_Occlusions/weightsVS.pretrained, end-to-end.caffemodel'
    # Trained from 'Hollywood_Converged_6DOFs' net
    #model_weights = '/local/qbateux/Saves_Nets/Hollywood_Finer_LocalLights_Occlusions/weightsVS.pretrained, end-to-end.caffemodel'

    model_weights = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_BIG_degraded_occlusions/_iter_62000.caffemodel' # TEST

    #model_weights = '/local/qbateux/Afma_Castle_MEDIAN_degraded_occlusions/weightsVS.pretrained, end-to-end.caffemodel'
    
    #model_weights = '/local/qbateux/Afma_Castle_MEDIAN2_degraded_occlusions/weightsVS.pretrained, end-to-end.caffemodel'
    
    #model_weights = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_MEDIAN_degraded_occlusions/weightsVS.pretrained, end-to-end.caffemodel' # TEST
    #model_weights = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_SMALL_degraded_occlusions/weightsVS.pretrained, end-to-end.caffemodel'
    
    #model_weights = '/local/qbateux/RealVS_test1_degraded_occlusions/weightsVS.pretrained, end-to-end.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test1_txty/_iter_42000.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test1_4DOFs/weightsVS.pretrained, end-to-end.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test1/_iter_4000.caffemodel'
    
    #model_weights = '/local/qbateux/RealVS_test2/weightsVS.pretrained, end-to-end.caffemodel'
    
    #model_weights = '/local/qbateux/RealVS_test2_BIG/weightsVS.pretrained, end-to-end.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test3_Coupling/_iter_40000.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test3_Coupling4DOFs/_iter_20000.caffemodel'

    #model_weights = '/local/qbateux/RealVS_test2_BIGGER/weightsVS.pretrained, end-to-end.caffemodel' # 4DOFs coupling training -> 6DOFs

    #model_weights = '/local/qbateux/RealVS_test3_CouplingHard/weightsVS.pretrained, end-to-end.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)

    #model_weights = '/local/qbateux/RealVS_test4_All_RandZ/weightsVS.pretrained, end-to-end.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)

    #model_weights = '/local/qbateux/RealVS_test4_All_RandZ2/weightsVS.pretrained, end-to-end.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)

    #model_weights = '/local/qbateux/RealVS_test4_All_RandZ2/_iter_20000.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)

    #model_weights = '/local/qbateux/RealVS_test4_All_RandZ2_BIG/_iter_100000.caffemodel' # 2DOFs txry -> 6 DOFs txtyrxry (6 outputs), BIG 
    
    #model_weights = '/local/qbateux/RealVS_test4_All_RandZ2_BIG/weightsVS.pretrained, end-to-end.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)

    model_weights = 'weightsVS.pretrained, end-to-end.caffemodel' # 2DOFs txry -> 4 DOFs txtyrxry (6 outputs)
    
    if usePoseNet:
        model_weights = '/local/qbateux/Afma_Hollywood_NoDiff_Finer_degraded_occlusions/weightsVS.pretrained, end-to-end_PoseNet.caffemodel'
        netDef = caffe_root + '/examples/VS/PoseNet/train_chess_Mod2.prototxt'
        netWeights = model_weights
    else:
        useOverwriteLastLayer=True
        #netDef = VS_net(train=False, deploy=True, numClassesVS=6, overwriteLastLayer=useOverwriteLastLayer)
        netDef = 'config.prototxt'
        netWeights = model_weights

    net = caffe.Net(netDef, netWeights, caffe.TEST)
        
    return net


def mainCommunication(useDualImages=False):
    def handler(signum, frame):
        print "Forever is over!"
        raise Exception("Client inactive, resetting server !")
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)
    #time.sleep(1)
    start1 = time.clock()
    #communicate with another process through named pipe
    print "Server Waiting for Client to connect "
    
    ## ~Handshake
    #print 'Sending data back'
    #wp = open(file2Path, 'w')
    #wp.write(struct.pack('>i',123456789))
    #wp.close()
    #time.sleep(100000)
    f = open(file1Path, 'r')
    #rawInput = f.read(3*4)
    #print 'rawInput = '+`rawInput`
    
    arraySize = struct.unpack('i', f.read(4))[0] # Reading array size
    strLength = `arraySize`+'i'
    array = struct.unpack(strLength, f.read(arraySize*4)) # Reading the actual array
    print 'Received array1'
    
    if not useDualImages:
        f.close()
    
    #imageSize1 = 360 
    #imageSize2 = 480 
    
    #isViSP = True
    isAfma = True
    isKinova = True
    '''if not isAfma:
        imageSize1 = 320 
        imageSize2 = 420 
    else:
        # Image format sent by ViSP
        imageSize1 = 240 
        imageSize2 = 320 '''
    if isKinova:
        imageSize1 = 640 
        imageSize2 = 480 
        
    image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
    #print image.shape
        
    
        
    # Unpacking the array from the vector shape
    imageTmp = np.reshape(array, (imageSize1, imageSize2), order='C')/255.0
    
    image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
    
    if not useDualImages:
        image[:, :, 0] = imageTmp
        image[:, :, 1] = imageTmp
        image[:, :, 2] = imageTmp   
        
    else: # Getting the 2nd image
        #rawInput = f.read(3*4)
        #print 'rawInput = '+`rawInput`
    
        #arraySize = struct.unpack('i', f.read(4))[0] # Reading array size
        #strLength = `arraySize`+'i'
        array = struct.unpack(strLength, f.read(arraySize*4)) # Reading the actual array
        print 'Received array2'
        f.close()

        # Unpacking the array from the vector shape
        imageTmp2 = np.reshape(array, (imageSize1, imageSize2), order='C')/255.0
        image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
    
    image[:, :, 0] = imageTmp
    image[:, :, 1] = imageTmp2
    image[:, :, 2] = 0

    ## For display only
    '''print 'Displaying Debug figures' 
    plt.figure()
    plt.imshow(image[:, :, 0])
    plt.show()
    plt.figure()
    plt.imshow(image[:, :, 1])
    plt.show()
    time.sleep(100000)'''
    start2 = time.clock()
    #desc = getCNNDescriptor(image, 'conv3')
    desc = getRelativePose(image)
    end2 = time.clock()
    print "Descriptor shape = ", desc.shape
    
    #desc = np.zeros((64896,1), float, 'C') # Mock result
    #desc = np.zeros((6,1), float, 'C') # Mock result
    #desc[0] = 123.5
    #desc[1] = 0.666
    arraySize = desc.shape[0]
    #print "Array Size = ", arraySize
    
    # Sending back the results
    #print 'Sending data back ; ArraySize'
    
    # Stuck here when client is killed while in 'getRelativePose()'...
    print 'Sending results to Client...'
    wp = open(file2Path, 'w')
    wp.write(struct.pack('>i',arraySize))
    #wp.close()        
    print 'Array Size sent'
    #wp = open(file2Path, 'w')
    packed = struct.pack('<'+`arraySize`+'f', *desc)
    #print 'sending = ' + `packed`
    wp.write(packed)
    wp.close()
    print 'Array Sent'
    
    end1 = time.clock()
    elapsed1 = (end1-start1)*1000
    elapsed2 = (end2-start2)*1000
    print 'Elapsed server time total = ' + `elapsed1` + 'ms'
    print 'Elapsed time descriptor computation = ' + `elapsed2` + 'ms\n'


if __name__ == '__main__':

    #from trainAlexNetOnVSDataset_Simple import VS_net
    
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    import sys
    caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    import os
    import math
    import cPickle
    import struct
    import signal

    usePoseNet = False # CANNOT WORK WITH THE STANDARD CAFFE BUILD !!!!!
    useDualImages = True # FlowNet-style : need to receive 2 images, desired current one.

    # Preparing the network for further feature extraction
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
    print 'Initializing network'
    net = initializeNetwork(usePoseNet)
    if useDualImages:
        print 'Server running, using DualImages'

    
    file1Path = "/dev/shm/fifo_server"
    file2Path = "/dev/shm/fifo_client"
    #file1Path = '/dev/shm/fifo'
    #file2Path = '/dev/shm/fifo'
    
    #while(True):
    #time.sleep(1)
    start1 = time.clock()
    #communicate with another process through named pipe
    print "Server Waiting for Client to connect "
        
    ## ~Handshake
        #print 'Sending data back'
        #wp = open(file2Path, 'w')
        #wp.write(struct.pack('>i',123456789))
        #wp.close()
    print 'Handshake...'
    f = open(file1Path, 'r')
    #rawInput = f.read(4)
    #print 'rawInput = '+`rawInput`
    arraySize = struct.unpack('i', f.read(4))[0] # Reading array size
    print 'Received int = ', arraySize
    strLength = `arraySize`+'i'
    
    array = struct.unpack(strLength, f.read(arraySize*4)) # Reading the actual array
    #receivedData = f.read()
    #print 'Received data = ', receivedData
    #receivedData = f.read(arraySize*4)
    #print 'Received Raw Data size = ', len(receivedData)
    
    #fmt = "<%dI" % (arraySize // 4)
    #array = struct.unpack(fmt, receivedData) # Reading the actual array
    print 'Received array size = ', len(array)
    f.close()
        
        
    #imageSize1 = 360 
        #imageSize2 = 480 
        
        #isViSP = True
    '''isAfma = True
    
    if not isAfma:
        imageSize1 = 320 
        imageSize2 = 420 
    else:
        # Image format sent by ViSP
        imageSize1 = 240 
        imageSize2 = 320 '''
        
    # For KINOVA arm
    imageSize1 = 640 
    imageSize2 = 480 
            
    image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
    #print image.shape
        
        # Unpacking the array from the vector shape
    imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')/255.0
    image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
    image[:, :, 0] = imageTmp
    image[:, :, 1] = imageTmp
    image[:, :, 2] = imageTmp   
        
    ## For display only
        #plt.figure()
        #plt.imshow(image)
        #plt.show()
        
    if usePoseNet:
        image = cv2.resize(image, (455,256))    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
        
    start2 = time.clock()
    #desc = getCNNDescriptor(image, 'conv3')
    desc = getRelativePose(image)
    end2 = time.clock()
    #print "Descriptor shape = ", desc.shape
        
        #desc = np.zeros((64896,1), float, 'C') # Mock result
        #desc = np.zeros((6,1), float, 'C') # Mock result
        #desc[0] = 123.5
        #desc[1] = 0.666
    arraySize = desc.shape[0]
    #print "Array Size = ", arraySize

        # Sending back the results
    print 'Sending data back ; ArraySize'
    wp = open(file2Path, 'w')
    wp.write(struct.pack('>i',arraySize))
    #wp.close()        

    # Duplication for debug
        #wp = open(wfPath, 'w')
        #print 'sending = '+ `struct.pack('>i',arraySize)`
        #wp.write(struct.pack('>1i',arraySize))
        #wp.close()
        
        #wp = open(file2Path, 'w')
        ##wp.write(struct.pack('>'+`arraySize`+'f', *array))
        #wp.write(struct.pack('>'+`arraySize`+'f', *desc))
        #wp.close()
        
    print 'Sending data back ; FLOAT_Array'
    #wp = open(file2Path, 'w')
    packed = struct.pack('<'+`arraySize`+'f', *desc)
    #print 'sending = ' + `packed`
    wp.write(packed)
    wp.close()
        
    print 'Array Sent'

    end1 = time.clock()
    elapsed1 = (end1-start1)*1000
    elapsed2 = (end2-start2)*1000
    #print 'Elapsed server time total = ' + `elapsed1` + 'ms'
        #print 'Elapsed time descriptor computation = ' + `elapsed2` + 'ms\n'
    print 'Ending handshake.' 
    while(True):
        try:
            mainCommunication(useDualImages)
        except Exception, exc:
            print exc
