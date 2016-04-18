import os
import numpy as np
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,InverseLayer
from lasagne.layers import Conv2DLayer 
from lasagne.layers import Pool2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer

from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify

import file_dir as file_dir

pathToImagesPickle = file_dir.pathToImagesPickle

def buildNetwork( inputWidth, inputHeight, input_var=None ):
    net = {}

    net['input'] = InputLayer((None, 3, inputWidth, inputHeight), input_var=input_var)

    # conv1
    net['conv1'] = Conv2DLayer( net['input'],num_filters=96,filter_size=(11, 11),stride = 4,nonlinearity=rectify)
    net['conv1'].add_param(net['conv1'].W, net['conv1'].W.get_value().shape, trainable=False)
    net['conv1'].add_param(net['conv1'].b, net['conv1'].b.get_value().shape, trainable=False)
    print "conv1: {}".format(net['conv1'].output_shape[1:])

    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)
    print "pool1: {}".format(net['pool1'].output_shape[1:])
    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],n=5,alpha=0.0001/5.0,beta = 0.75,k=1)
    print "norm1: {}".format(net['norm1'].output_shape[1:])

    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)
    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],num_filters=128,filter_size=(5, 5),pad = 2)
    net['conv2_part1'].add_param(net['conv2_part1'].W, net['conv2_part1'].W.get_value().shape, trainable=False)
    net['conv2_part1'].add_param(net['conv2_part1'].b, net['conv2_part1'].b.get_value().shape, trainable=False)

    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],num_filters=128,filter_size=(5, 5),pad = 2)
    net['conv2_part2'].add_param(net['conv2_part2'].W, net['conv2_part2'].W.get_value().shape, trainable=False)
    net['conv2_part2'].add_param(net['conv2_part2'].b, net['conv2_part2'].b.get_value().shape, trainable=False)
    # now combine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
    print "conv2: {}".format(net['conv2'].output_shape[1:]) 

    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:]) 
    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],n=5,alpha=0.0001/5.0,beta = 0.75,k=1)

    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],num_filters=384,filter_size=(3, 3),pad = 1)
    net['conv3'].add_param(net['conv3'].W, net['conv3'].W.get_value().shape, trainable=False)
    net['conv3'].add_param(net['conv3'].b, net['conv3'].b.get_value().shape, trainable=False)
    print "conv3: {}".format(net['conv3'].output_shape[1:]) 

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)

    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],num_filters=192,filter_size=(3, 3),pad = 1)
    net['conv4_part1'].add_param(net['conv4_part1'].W, net['conv4_part1'].W.get_value().shape, trainable=False)
    net['conv4_part1'].add_param(net['conv4_part1'].b, net['conv4_part1'].b.get_value().shape, trainable=False)

    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],num_filters=192,filter_size=(3, 3),pad = 1)
    net['conv4_part2'].add_param(net['conv4_part2'].W, net['conv4_part2'].W.get_value().shape, trainable=False)
    net['conv4_part2'].add_param(net['conv4_part2'].b, net['conv4_part2'].b.get_value().shape, trainable=False)

    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)
    print "conv4: {}".format(net['conv4'].output_shape[1:]) 

    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)

    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],num_filters=128,filter_size=(3, 3),pad = 1)
    net['conv5_part1'].add_param(net['conv5_part1'].W, net['conv5_part1'].W.get_value().shape, trainable=False)
    net['conv5_part1'].add_param(net['conv5_part1'].b, net['conv5_part1'].b.get_value().shape, trainable=False)

    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],num_filters=128,filter_size=(3, 3),pad = 1)
    net['conv5_part2'].add_param(net['conv5_part2'].W, net['conv5_part2'].W.get_value().shape, trainable=False)
    net['conv5_part2'].add_param(net['conv5_part2'].b, net['conv5_part2'].b.get_value().shape, trainable=False)

    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)
    print "conv5: {}".format(net['conv5'].output_shape[1:]) 

    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)
    print "pool5: {}".format(net['pool5'].output_shape[1:]) 

    #Adaptive stage
    net['apool5'] = InverseLayer(net['pool5'],net['pool5'])
    print "upool5: {}".format(net['apool5'].output_shape[1:])

    net['aconv2'] = Conv2DLayer(net['apool5'],num_filters=256,filter_size=(2, 2),pad = 1,stride = 2) #(6x8x256)
    print "aconv2: {}".format(net['aconv2'].output_shape[1:])

    '''
    net['uconv5'] = Conv2DLayer(net['upool5'],num_filters=256,filter_size=(3, 3),pad = 1)
    print "uconv5: {}".format(net['uconv5'].output_shape[1:])

    net['uconv4'] = Conv2DLayer(net['uconv5'],num_filters=384,filter_size=(3, 3),pad = 1)
    print "uconv4: {}".format(net['uconv4'].output_shape[1:])

    net['uconv3'] = Conv2DLayer(net['uconv4'],num_filters=384,filter_size=(3, 3),pad = 1)
    print "uconv3: {}".format(net['uconv3'].output_shape[1:])
    
    net['upool2'] = InverseLayer(net['pool2'],net['uconv3'])
    print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2'] = Conv2DLayer(net['upool2'],num_filters=256,filter_size=(5, 5),pad = 2)
    print "uconv2: {}".format(net['uconv2'].output_shape[1:])

    net['upool1'] = InverseLayer(net['pool1'],net['uconv2'])
    print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1'] = InverseLayer(net['conv1'],net['upool1'])
    print "uconv1: {}".format(net['uconv1'].output_shape[1:])
    '''

    net['upool5'] = Upscale2DLayer(net['aconv2'], scale_factor=2)
    print "upool5: {}".format(net['upool5'].output_shape[1:])

    net['uconv5_3'] = ConvLayer(net['upool5'], 512, 3, pad=1)
    print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['upool4'] = Upscale2DLayer(net['uconv5_1'], scale_factor=2)
    print "upool4: {}".format(net['upool4'].output_shape[1:])

    net['uconv4_3'] = ConvLayer(net['upool4'], 512, 3, pad=1)
    print "uconv4_3: {}".format(net['uconv4_3'].output_shape[1:])

    net['uconv4_2'] = ConvLayer(net['uconv4_3'], 512, 3, pad=1)
    print "uconv4_2: {}".format(net['uconv4_2'].output_shape[1:])

    net['uconv4_1'] = ConvLayer(net['uconv4_2'], 512, 3, pad=1)
    print "uconv4_1: {}".format(net['uconv4_1'].output_shape[1:])

    net['upool3'] = Upscale2DLayer(net['uconv4_1'], scale_factor=2)
    print "upool3: {}".format(net['upool3'].output_shape[1:])

    net['uconv3_3'] = ConvLayer(net['upool3'], 256, 3, pad=1)
    print "uconv3_3: {}".format(net['uconv3_3'].output_shape[1:])

    net['uconv3_2'] = ConvLayer(net['uconv3_3'], 256, 3, pad=1)
    print "uconv3_2: {}".format(net['uconv3_2'].output_shape[1:])

    net['uconv3_1'] = ConvLayer(net['uconv3_2'], 256, 3, pad=1)
    print "uconv3_1: {}".format(net['uconv3_1'].output_shape[1:])

    net['upool2'] = Upscale2DLayer(net['uconv3_1'], scale_factor=2)
    print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2_2'] = ConvLayer(net['upool2'], 128, 3, pad=1)
    print "uconv2_2: {}".format(net['uconv2_2'].output_shape[1:])

    net['uconv2_1'] = ConvLayer(net['uconv2_2'], 128, 3, pad=1)
    print "uconv2_1: {}".format(net['uconv2_1'].output_shape[1:])

    net['upool1'] = Upscale2DLayer(net['uconv2_1'], scale_factor=2)
    print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1_2'] = ConvLayer(net['upool1'], 64, 3, pad=1)
    print "uconv1_2: {}".format(net['uconv1_2'].output_shape[1:])

    net['uconv1_1'] = ConvLayer(net['uconv1_2'], 64, 3, pad=1)
    print "uconv1_1: {}".format(net['uconv1_1'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv1_1'], 1, 1, pad=0)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):

        yield l[i:i+n]

if __name__ == "__main__":

    # Load data

    #with open( './salicon_data/validationData.pickle', 'rb') as f:
        #validationData = pickle.load( f )       

    # with open( 'testData.pickle', 'rb') as f:
        # testData = pickle.load( f )   

    #valData = validationData[0:2000]

    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    width = 256
    height = 192

    net = buildNetwork(height,width,inputImage)

    '''
    d = pickle.load(open('vgg16.pkl'))
    numElementsToSet = 26 # Number of W and b elements for the first convolutional layers
    lasagne.layers.set_all_param_values(net['pool5'], d['param values'][:numElementsToSet])
    '''
    d = pickle.load(open('caffe_places.pkl'))
    numElementsToSet = 16
    lasagne.layers.set_all_param_values(net['pool5'], d[:numElementsToSet])


    prediction = lasagne.layers.get_output(net['output'])
    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
    loss = lasagne.objectives.squared_error(prediction, outputSaliency)
    #the_costum_loss = costum_loss()
    #loss = the_costum_loss.kl_loss(prediction, outputSaliency)
    loss = loss.mean()

    init_learningrate = 0.01
    momentum = 0.0 # start momentum at 0.0
    max_momentum = 0.9
    min_learningrate = 0.00001
    lr = theano.shared(np.array(init_learningrate, dtype=theano.config.floatX))
    mm = theano.shared(np.array(momentum, dtype=theano.config.floatX))

    # Let's only train the trainable layers (i.e. the deconvolutional part, check buildNetwork() function)
    params = lasagne.layers.get_all_params(net['output'], trainable=True)

    updates_sgd = lasagne.updates.sgd(loss, params, learning_rate = lr)
    updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum = mm) 

    train_fn = theano.function([inputImage, outputSaliency], loss, updates=updates, allow_input_downcast=True)

    val_fn = theano.function([inputImage, outputSaliency], loss)


    predict_fn = theano.function([inputImage], test_prediction)

    #imageMean = d['mean value'][:, np.newaxis, np.newaxis]

    batchSize = 64
    numEpochs = 50


    batchIn = np.zeros((batchSize, 3, height, width ), theano.config.floatX )
    batchOut = np.zeros((batchSize, 1, height, width ), theano.config.floatX )

    print 'Loading training data...'
    with open( pathToImagesPickle, 'rb') as f:
        trainData = pickle.load( f )    
    print '-->done!'

    for currEpoch in tqdm(range(numEpochs)):

        random.shuffle( trainData )
        #random.shuffle( valData )

        train_err = 0.
        #val_err = 0.

        for currChunk in chunks(trainData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range( batchSize ):
                batchIn[k,...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2,0,1))/255. #-imageMean)/255.
                batchOut[k,...] = (currChunk[k].saliency.data.astype(theano.config.floatX))/255.
            train_err += train_fn( batchIn, batchOut )
        '''
        for currChunk in chunks(valData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range( batchSize ):
                batchIn[k,...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2,0,1)-imageMean)/255.
                batchOut[k,...] = (currChunk[k].saliency.data.astype(theano.config.floatX))/255.
            val_err += val_fn( batchIn, batchOut )
        '''
        print 'Epoch:', currEpoch, ' ->', train_err #(train_err, val_err )

        if currEpoch % 10 == 0:
            np.savez( "modelWights{:04d}.npz".format(currEpoch), *lasagne.layers.get_all_param_values(net['output']))
