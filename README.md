# Deep Learning Server Client

![alt text](https://www.inria.fr/var/inria/storage/images/medias/inria/images-corps/logo-inria-institutionnel-couleur/422541-6-eng-GB/logo-inria-institutionnel-couleur_large.png)

This repository was created at INRIA, France as part of the efforts to make communication between the C++ code for the visual servoing using ViSP and the deep learning model in python easier. The code leverages the power of the pipe functionality in the Unix computers and enables codes written in different languages to communicate with each other.

Files:
- callPythonServer.cpp: Contains the functions with which I communicate in C ++ with the python code of the cnn.

- createPipes.cpp: Code that creates pipes for communication between c ++ and python

- createNoisyDataset.py: takes a folder of images and adds noises.

- createHDF5Dataset.py: take an image folder and create a caffe readable HDF5 database

- runServer_TrainedVSNet_Simple.py: takes a weight file and a network architecture and waits for the c ++ code to send an image, pass it in the network and return the pose estimate.

- trainAlexNetOnVSDataset.py: Takes an HDF5 base, and unage network architecture and drives the network.

- config.prototxt: architecture configuration file of a VGG network
