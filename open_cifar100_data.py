__author__ = 'mangate'

from six.moves import cPickle as pickle
import numpy as np
import os
import fnmatch
import sys
import matplotlib.pyplot as plt
from pylearn2.datasets.cifar100 import CIFAR100

"""
This file opens the CIFAR100 data after whitening and ZCA made by 'process_cifar_100_data' script
which uses pylearn2 library
This file also re-arragne the data so it can enter a nueral net properly
"""

image_size = 32
num_channels = 3
num_classes = 100
pixel_depth = 255.0

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def rearrange_data(data,labels):
    #data = np.cast['float32'](data)
    #data = data / 255.
    #data = data - data.mean(axis=0)
    images = np.ndarray(shape=(len(labels),image_size,image_size,num_channels), dtype=np.float32)
    labels_out = np.zeros(shape=(len(labels),num_classes),dtype=np.float32)
    max = len(labels)
    for i in range(max):
        images[i] = np.reshape(data[i],(3,32,32)).transpose(1,2,0)
        #plt.imshow(images[i])
        #plt.show()
        labels_out[i][labels[i]]=1.0
    return images,labels_out

def process_data(file_name):
    data = unpickle(file_name)
    # images = data['data']
    # labels = data['fine_labels']
    images = np.asarray(data.X)
    labels = np.asarray(data.y)
    images,labels = rearrange_data(images,labels)
    return images,labels

def get_data():
    ROOT_FOLDER = '/cs/img/mangate/pylearn2_datasets/cifar100/pylearn2_gcn_whitened/'
    train_images, train_labels = process_data(ROOT_FOLDER+'train.pkl')
    test_images,test_labels = process_data(ROOT_FOLDER+'test.pkl')
    # ROOT_FOLDER = '/cs/img/mangate/thesis/Cifar-100/cifar-100-python/'
    #
    # train_images, train_labels = process_data(ROOT_FOLDER+'train')
    # test_images,test_labels = process_data(ROOT_FOLDER+'test')
    print 'Train Date shape is',train_images.shape, 'and labels is',train_labels.shape
    print 'Test Date shape is',test_images.shape, 'and labels is',test_labels.shape
    return  train_images, train_labels, test_images,test_labels

get_data()