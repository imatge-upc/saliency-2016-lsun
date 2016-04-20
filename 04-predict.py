import os
import numpy as np
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,InverseLayer
from lasagne.layers import Conv2DLayer, Upscale2DLayer, DropoutLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat

from lasagne.nonlinearities import softmax, rectify
from lasagne.utils import floatX

import scipy.io
import cPickle as pickle
from scipy import misc
'''
def buildNetwork( inputWidth, inputHeight, input_var=None ):
    net = {}
    net['input'] = InputLayer((None, 3, inputWidth, inputHeight), input_var=input_var)
    #print "Input: {}".format(net['input'].output_shape[1:])

    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
    net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv1_1'].output_shape[1:])

    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
    net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv1_2'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    #print "Input: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
    net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
    net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    #print "Input: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
    net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
    net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv3_2'].output_shape[1:])

    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
    net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv3_3'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    #print "Input: {}".format(net['pool3'].output_shape[1:])

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape, trainable=False)
    net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv4_1'].output_shape[1:])

    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape, trainable=False)
    net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv4_2'].output_shape[1:])

    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape, trainable=False)
    net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv4_3'].output_shape[1:])

    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    #print "Input: {}".format(net['pool4'].output_shape[1:])

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape, trainable=False)
    net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv5_1'].output_shape[1:])

    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape, trainable=False)
    net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv5_2'].output_shape[1:])

    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape, trainable=False)
    net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape, trainable=False)
    #print "Input: {}".format(net['conv5_3'].output_shape[1:])

    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    #print "Input: {}".format(net['output'].output_shape[1:])

    net['upool5'] = Upscale2DLayer(net['pool5'], scale_factor=2)
    #print "upool5: {}".format(net['upool5'].output_shape[1:])

    net['uconv5_3'] = ConvLayer(net['upool5'], 512, 3, pad=1)
    #print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    #print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    #print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['upool4'] = Upscale2DLayer(net['uconv5_1'], scale_factor=2)
    #print "upool4: {}".format(net['upool4'].output_shape[1:])

    net['uconv4_3'] = ConvLayer(net['upool4'], 512, 3, pad=1)
    #print "uconv4_3: {}".format(net['uconv4_3'].output_shape[1:])

    net['uconv4_2'] = ConvLayer(net['uconv4_3'], 512, 3, pad=1)
    #print "uconv4_2: {}".format(net['uconv4_2'].output_shape[1:])

    net['uconv4_1'] = ConvLayer(net['uconv4_2'], 512, 3, pad=1)
    #print "uconv4_1: {}".format(net['uconv4_1'].output_shape[1:])

    net['upool3'] = Upscale2DLayer(net['uconv4_1'], scale_factor=2)
    #print "upool3: {}".format(net['upool3'].output_shape[1:])

    net['uconv3_3'] = ConvLayer(net['upool3'], 256, 3, pad=1)
    #print "uconv3_3: {}".format(net['uconv3_3'].output_shape[1:])

    net['uconv3_2'] = ConvLayer(net['uconv3_3'], 256, 3, pad=1)
    #print "uconv3_2: {}".format(net['uconv3_2'].output_shape[1:])

    net['uconv3_1'] = ConvLayer(net['uconv3_2'], 256, 3, pad=1)
    #print "uconv3_1: {}".format(net['uconv3_1'].output_shape[1:])

    net['upool2'] = Upscale2DLayer(net['uconv3_1'], scale_factor=2)
    #print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2_2'] = ConvLayer(net['upool2'], 128, 3, pad=1)
    #print "uconv2_2: {}".format(net['uconv2_2'].output_shape[1:])

    net['uconv2_1'] = ConvLayer(net['uconv2_2'], 128, 3, pad=1)
    #print "uconv2_1: {}".format(net['uconv2_1'].output_shape[1:])

    net['upool1'] = Upscale2DLayer(net['uconv2_1'], scale_factor=2)
    #print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1_2'] = ConvLayer(net['upool1'], 64, 3, pad=1)
    #print "uconv1_2: {}".format(net['uconv1_2'].output_shape[1:])

    net['uconv1_1'] = ConvLayer(net['uconv1_2'], 64, 3, pad=1)
    #print "uconv1_1: {}".format(net['uconv1_1'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv1_1'], 1, 1, pad=0)
    #print "output: {}".format(net['output'].output_shape[1:])

    return net
'''
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

    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    width = 256
    height = 192

    net = buildNetwork(height,width,inputImage)

    epochToLoad = 40

    with np.load("modelWights{:04d}.npz".format(epochToLoad)) as f:
    	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['output'], param_values)

    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
    predict_fn = theano.function([inputImage], test_prediction)

    imageMean = np.array([[[103.939]],[[116.779]],[[123.68]]])

    batchSize = 200

    batch = np.zeros((batchSize, 3, height, width ), theano.config.floatX )
    y_pred = np.zeros((5000, 1, height, width ), theano.config.floatX )
    paths = []
    idx = 0

    # Load data
    print 'Loading validation data...'
    with open( './salicon_data/validationData.pickle', 'rb') as f:
        validationData = pickle.load( f )       
    print '-->done!'

    # with open( 'testData.pickle', 'rb') as f:
        # testData = pickle.load( f )   

    listImages = [ k.image.filePath.split('/')[-1].split('.')[0] for k in validationData ] 
        
    for currChunk in chunks(validationData, batchSize):

        for k in range( batchSize ):
            batch[k,...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2,0,1)-imageMean)/255.
            
        #result = np.squeeze( predict_fn( batch ) )
        print idx
        y_pred[idx:idx+batchSize] = predict_fn( batch )
        idx=idx+batchSize

    for res, name in zip(y_pred,listImages):
        saliencyMap = np.squeeze( res )
        #blured= ndimage.gaussian_filter(tmp, sigma=3)
        #y = misc.imresize(blured,(480,640))/255.
        #cv2.resize(saliencyMap,(640,480)
        saliencyMap = misc.imresize(saliencyMap,(480,640))/255.
        saliencyMapMatlab = {'I':saliencyMap}
        scipy.io.savemat('/imatge/jpan/work/lsun2016/results/18-04-2016/'+name,saliencyMapMatlab )