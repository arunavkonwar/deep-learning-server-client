caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import tempfile
import numpy as np
from pylab import *
import time

isGantry = False
isRemovingPool = False # TODO True do not work for now: net too wide, exceeds memory...
if(isGantry):
    # Needed to activate the GPU mode
    caffe.set_device(0)
    caffe.set_mode_gpu()


if(isGantry):
    # For GANTRY  computer
    
    # For 6DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrxryrz_noZeros/'
    
    #hdf5DataFolderPath = '/local/qbateux/testSet_10k_txtytzrxryrz_noZeros/'
    
    # 6DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_4DOFs/'
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood_2DOFs/'
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_Hollywood/'
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_DegradedTest_Hollywood_6DOFs_SinglePos/'
    hdf5DataFolderPath = '/local/qbateux/Afma_Hollywood/'
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_DegradedTest_Hollywood_6DOFs/'
    # For 4DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrz_noZeros/'
    
    # For 2DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txty_noZeros/'
    
    # For 1DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_tx_noZeros/'
        
else:
    # For Gerty computer
    # For 6DOF
    hdf5DataFolderPath = '/media/quentin/OSDisk/testSet_10k_txtytzrxryrz_noZeros/'
    
    #hdf5DataFolderPath = '/media/quentin/OSDisk/trainingSet_10k_txty_noZeros/'
    
    #hdf5DataFolderPath = '/local/qbateux/testSet_10k_txtytzrxryrz_noZeros/'

    # For 4DOF
    #hdf5DataFolderPath = '/local/qbateux/trainingSet_10k_txtytzrz_noZeros/'

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image


with open( hdf5DataFolderPath+'DatasetIndex', 'r' ) as T :
    lines = T.readlines()
print 'Size lines = ', len(lines)
l = lines[0]
l = l.strip()
sp = l.split(';') # Take out the extra whitespaces, newlines...etc
numClassesVS = len(sp)-1
print "Nb Classes = ", numClassesVS
#time.sleep(1000)

# # Downloading the dataset associated with the current tutorial
# Download just a small subset of the data for this exercise.
# (2000 of 80K images, 5 of 20 labels.)
# To download the entire dataset, set `full_dataset = True`.
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5
# This downloads the ilsvrc auxiliary data (mean file, etc),
# and a subset of 2000 images for the style recognition task.
import os

#os.chdir(caffe_root)  # run scripts from caffe root
#os.system("./data/ilsvrc12/get_ilsvrc_aux.sh")
#os.system("./scripts/download_model_binary.py models/bvlc_reference_caffenet")
##os.system("python examples/finetune_flickr_style/assemble_data.py --workers=-1  --seed=1701 --images=$NUM_STYLE_IMAGES  --label=$NUM_STYLE_LABELS")
#os.system("python examples/finetune_flickr_style/assemble_data.py --workers=-1  --seed=1701 --images=2000  --label=5")
# back to examples
#os.chdir('examples/VS')

# # Getting the weights from the already trained AlexNet
import os
learnFromAlexNet = False

if learnFromAlexNet:
    weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
else:
    # Lena converged with 6DOFs
    #weights = 'Converged_txtytzrxryrz/weightsVS.pretrained, end-to-end.caffemodel' 
    
    # Hollywood converged with 6DOFs
    weights = 'Hollywood_Converged_6DOFs/weightsVS.pretrained, end-to-end.caffemodel'
    
    
    
assert os.path.exists(weights)

# Load ImageNet labels to imagenet_labels
#imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
#imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
#assert len(imagenet_labels) == 1000
#print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# Load style labels to style_labels
#style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
#style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
#if NUM_STYLE_LABELS > 0:
#    style_labels = style_labels[:NUM_STYLE_LABELS]
#print '\nLoaded style labels:\n', ', '.join(style_labels)

# # Defining the fine-tune network
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def caffenet(data, label=None, train=True, num_classes=1000, classifier_name='fc8', learn_all=False, deploy=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    if not deploy:
        n.data = data
    else:
        n.data =  L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
        label = None
                
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    if(not isRemovingPool):
        n.pool1 = max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
        n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    else:
        n.conv2, n.relu2 = conv_relu(n.relu1, 5, 256, pad=2, group=2, param=param)
    if(not isRemovingPool):
        n.pool2 = max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    else : 
        n.norm2 = L.LRN(n.relu2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    if(not isRemovingPool):
        n.pool5 = max_pool(n.relu5, 3, stride=2)
        n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    else : 
        n.fc6, n.relu6 = fc_relu(n.relu5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    #fc8 = L.InnerProduct(fc8input, num_output=numClassesVS, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    #if not train:
    #    n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        #n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.loss = L.EuclideanLoss(fc8, n.label)
        #n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name


# # Dummy data to check the output of the already trained network
#dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
#imagenet_net_filename = caffenet(data=dummy_data, train=False)
#imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

# # Replace the last fc layer in order not to load the one associated with the trained AlexNet's one.
def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + 'data/flickr_style/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227, mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(transform_param=transform_param, source=source,batch_size=50, new_height=256, new_width=256, ntop=2) 
    
    # HDF5 data source
    style_data, style_label = L.HDF5Data(source=hdf5DataFolderPath+'train_h5_list.txt',batch_size=50, ntop=2) 
    
    # test
    print type(style_label)
    return caffenet(data=style_data, label=style_label, train=train, num_classes=NUM_STYLE_LABELS, classifier_name='fc8_flickr', learn_all=learn_all)

def VS_net(train=True, learn_all=False, subset=None, deploy=False, numClassesVS=numClassesVS):
    if subset is None:
        subset = 'train' if train else 'test'
    #source = caffe_root + 'data/flickr_style/%s.txt' % subset
    #transform_param = dict(mirror=train, crop_size=227, mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    #style_data, style_label = L.ImageData(transform_param=transform_param, source=source,batch_size=50, new_height=256, new_width=256, ntop=2) 
    
    if train:
        batch_size = 50
    else:
        batch_size = 1

        
    # HDF5 data source
    VS_data, VS_label = L.HDF5Data(source=hdf5DataFolderPath+'train_h5_list.txt',batch_size=batch_size, ntop=2) 
    
    # test
    print type(VS_label)
    return caffenet(data=VS_data, label=VS_label, train=train, num_classes=numClassesVS, classifier_name='fc8_VS', learn_all=learn_all, deploy=deploy)




# # Load the new network and run the first batch of training data.
#untrained_style_net = caffe.Net(style_net(train=False, subset='train'), weights, caffe.TEST)
#untrained_style_net = caffe.Net(VS_net(train=False, subset='train'), weights, caffe.TEST)
#untrained_style_net.forward()
#style_data_batch = untrained_style_net.blobs['data'].data.copy()
#style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.float64)

def disp_pred_VS(net, image, k=5, name='VS'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    print 'probs = ', probs

def disp_VS_preds(net, image):
    disp_pred_VS(net, image, name='VSNet')


def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')


#batch_index = 8
#image = style_data_batch[batch_index]

#image = image.copy()              # don't modify destructively
#image = image.transpose(1, 2, 0)  # CHW -> HWC
#plt.imshow(image)
#plt.show()

#time.sleep(10)

#print 'actual label =', style_labels[style_label_batch[batch_index]]
#print 'actual label =', style_label_batch[batch_index]
# Show the prediction of the trained alexnet
#disp_imagenet_preds(imagenet_net, image)

# Show the prediction of the untrained style net (useless)
#disp_style_preds(untrained_style_net, image)
#disp_VS_preds(untrained_style_net, image)

# # Check the network activation
#diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
#error = (diff ** 2).sum()
#assert error < 1e-8

# # Unload the untrained style network
#del untrained_style_net


# # Defining the learning process of the style network
from caffe.proto import caffe_pb2
def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 2000
    s.snapshot_prefix = hdf5DataFolderPath
    
    # Train on the CPU.
    if(isGantry):
        s.solver_mode = caffe_pb2.SolverParameter.GPU
    else:
        s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weightsVS = {}
    for name, s in solvers:
        filename = 'weightsVS.%s.caffemodel' % name
        weightsVS[name] = os.path.join(weight_dir, filename)
        s.net.save(weightsVS[name])
    return loss, acc, weightsVS

def run_solvers_VS(niter, solvers, disp_interval=1): # Adapation: no accuracy layer
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    #blobs = ('loss')
    loss = ({name: np.zeros(niter) for name, _ in solvers})
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it]= (s.net.blobs['loss'].data.copy())
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%f' % (n, loss[n][it]) for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    #weight_dir = tempfile.mkdtemp()
    weight_dir = hdf5DataFolderPath
    weightsVS = {}
    for name, s in solvers:
        filename = 'weightsVS.%s.caffemodel' % name
        weightsVS[name] = os.path.join(weight_dir, filename)
        s.net.save(weightsVS[name])
        print 'Saving ', weightsVS[name]
    return loss, weightsVS

#
####################################################################################
# # Actually running the training

if __name__ == '__main__':
    niter = 10000  # number of iterations to train
    # For DOF1, 2
    base_lr = 0.0001
    
    # if 'True' we train the intialized net and a net initialized from scratch for comparison purposes. If 'False', then only the initialized one.
    trainBothNets = False
    # If True, then we will try to learn the last layer alone, then the complete end-to-end network. If False, then the learning starts directly with the end-to-end learning
    trainEndToEndOnly = True
    if(not trainEndToEndOnly):
        # Reset style_solver as before.
    #style_solver_filename = solver(style_net(train=True))
        VS_solver_filename = solver(VS_net(train=True, learn_all=False), base_lr=base_lr) # Using the VS Net
        VS_solver = caffe.get_solver(VS_solver_filename)
        if(not isRemovingPool): # If we removed the pooling layers, we cannot copy the weights directly from AlexNet
            VS_solver.net.copy_from(weights)

        # For reference, we also create a solver that isn't initialized from
        # the pretrained ImageNet weights.
        if(trainBothNets):
            scratch_VS_solver_filename = solver(VS_net(train=True, learn_all=False), base_lr=base_lr) # Using the VS Net
            scratch_VS_solver = caffe.get_solver(scratch_VS_solver_filename)
 

        print 'Running solvers for %d iterations...' % niter
        if(trainBothNets):
            solvers = [('pretrained', VS_solver), ('scratch', scratch_VS_solver)]
        else:
            solvers = [('pretrained', VS_solver)]

        loss, weightsVS = run_solvers_VS(niter, solvers)
        print 'Done.'

        if(trainBothNets):
            train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
            VS_weights, scratch_VS_weights = weightsVS['pretrained'], weightsVS['scratch']
        else:
            train_loss = loss['pretrained']
            VS_weights = weightsVS['pretrained']


        # Delete solvers to save memory.
        if(trainBothNets):
            del VS_solver, scratch_VS_solver, solvers
        else:
            del VS_solver, solvers



    ################################
    # Starting end-to-end learning
    
    # For DOF1,2, 6 End-To-End Training.
    # For Lena, 1,2,4,6 DOFs OK
    base_lr = 0.00001
    
    # For hollywood triangle
    
    # For DOF2 OK
    base_lr = 0.00001
    #For 4DOFS OK
    base_lr = 0.00009
    if learnFromAlexNet :
        #For 6DOFS ?
        base_lr = 0.00085
    else:
        base_lr = 0.00001


    end_to_end_net = VS_net(train=True, learn_all=True)

    VS_solver_filename = solver(end_to_end_net, base_lr=base_lr)
    VS_solver = caffe.get_solver(VS_solver_filename)
    if(not trainEndToEndOnly):
        VS_solver.net.copy_from(VS_weights)
    else:
        VS_solver.net.copy_from(weights)

    if(trainBothNets):
        scratch_VS_solver_filename = VS_solver(end_to_end_net, base_lr=base_lr)
        scratch_VS_solver = caffe.get_solver(scratch_VS_solver_filename)
        scratch_VS_solver.net.copy_from(scratch_VS_weights)

    print 'Running solvers for %d iterations...' % niter
    if(trainBothNets):
        solvers = [('pretrained, end-to-end', VS_solver), ('scratch, end-to-end', scratch_VS_solver)]
    else:
        solvers = [('pretrained, end-to-end', VS_solver)]
    loss, weightsVS = run_solvers_VS(niter, solvers)
    print 'Done.'

    #if(trainBothNets):
    #    VS_weights_ft = finetuned_weights['pretrained, end-to-end']
    #    scratch_VS_weights_ft = finetuned_weights['scratch, end-to-end']
    #else:
    #    VS_weights_ft = finetuned_weights['pretrained, end-to-end']

    # Delete solvers to save memory.
    if(trainBothNets):
        del VS_solver, scratch_VS_solver, solvers
    else:
        del VS_solver, solvers

    if(trainBothNets):
        train_loss_ete, scratch_train_loss_ete = loss['pretrained, end-to-end'], loss['scratch, end-to-end']
        #VS_weights, scratch_VS_weights = weights['pretrained, end-to-end'], weights['scratch, end-to-end']
    else:
        train_loss_ete = loss['pretrained, end-to-end']
        #VS_weights = weights['pretrained, end-to-end']
        
    ################################################
    # # Displaying the loss evolution
    
    if(not trainEndToEndOnly):
        if(trainBothNets):
            plot(np.vstack([train_loss, scratch_train_loss]).T)
        else:
            plot(np.vstack([train_loss]).T)
        xlabel('Iteration #')
        ylabel('Loss')
        show()
        print 'Learning last layer finished'
        
    if(trainBothNets):
        plot(np.vstack([train_loss_ete, scratch_train_loss_ete]).T)
    else:
        plot(np.vstack([train_loss_ete]).T)
    xlabel('Iteration #')
    ylabel('Loss')
    show()
