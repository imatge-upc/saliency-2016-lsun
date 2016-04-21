import os
import numpy as np
import cv2
import sys
import cPickle as pickle
from collections import OrderedDict
import glob
import random
import time
from tqdm import tqdm
from eliaLib import dataRepresentation
import matplotlib.pyplot as plt
from models import ConvVGGPlaces, ConvVGGFaces, ConvVGGImageNet

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,InverseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import softmax


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):

        yield l[i:i+n]

if __name__ == "__main__":
	
	# Load data
	print 'Loading training data...'
	with open( 'trainData.pickle', 'rb') as f:
		trainData = pickle.load( f )    
	print '-->done!'

	# with open( 'validationData.pickle', 'rb') as f:
		# validationData = pickle.load( f )       

	# with open( 'testData.pickle', 'rb') as f:
		# testData = pickle.load( f )   
		
	
	# Variables 
	inputWidth = 256
	inputHeight = 192
	batchSize = 64
	numEpochs = 50

	
	# Create network PLACES
	inputImagePlaces = T.tensor4()
	networkPlaces = OrderedDict()
	inputPlaces = ConvVGGPlaces.ConvVGGPlaces(inputWidth,inputHeight)
	inputPlaces.build( networkPlaces, inputImagePlaces )
	inputPlaces.load( 'ConvVGG_Places.pkl', networkPlaces)
	
	# Create network FACES
	inputImageFace = T.tensor4()
	networkFaces = OrderedDict()
	inputFaces = ConvVGGFaces.ConvVGGFaces(inputWidth,inputHeight)
	inputFaces.build( networkFaces, inputImageFace )
	inputFaces.load( 'ConvVGG_Faces.pkl', networkFaces)
	
	# Create network ImageNet
	inputImageImageNet = T.tensor4()
	networkImageNet = OrderedDict()
	inputImageNet = ConvVGGImageNet.ConvVGGImageNet(inputWidth,inputHeight)
	inputImageNet.build( networkImageNet, inputImageImageNet )
	inputImageNet.load( 'ConvVGG_ImageNet.pkl', networkImageNet )
	
	# Merge networks in a common OrderedDict
	inputNetwork = networkPlaces.copy()
	inputNetwork.update( networkFaces )
	inputNetwork.update( networkImageNet )
	del networkPlaces
	del networkFaces
	del networkImageNet
	
	# Concat inputs
	inputNetwork['concat'] = lasagne.layers.ConcatLayer([inputNetwork[inputPlaces.outputLayerName],\
                                                     inputNetwork[inputFaces.outputLayerName],\
                                                     inputNetwork[inputImageNet.outputLayerName]], axis=1)
	
	# Deconvolutional part
	inputNetwork['upool5'] = Upscale2DLayer(inputNetwork['concat'], scale_factor=2)
	print "upool5: {}".format(inputNetwork['upool5'].output_shape[1:])

	inputNetwork['uconv5_3'] = ConvLayer(inputNetwork['upool5'], 512, 3, pad=1)
	print "uconv5_3: {}".format(inputNetwork['uconv5_3'].output_shape[1:])

	inputNetwork['uconv5_2'] = ConvLayer(inputNetwork['uconv5_3'], 512, 3, pad=1)
	print "uconv5_2: {}".format(inputNetwork['uconv5_2'].output_shape[1:])

	inputNetwork['uconv5_1'] = ConvLayer(inputNetwork['uconv5_2'], 512, 3, pad=1)
	print "uconv5_1: {}".format(inputNetwork['uconv5_1'].output_shape[1:])

	inputNetwork['upool4'] = Upscale2DLayer(inputNetwork['uconv5_1'], scale_factor=2)
	print "upool4: {}".format(inputNetwork['upool4'].output_shape[1:])

	inputNetwork['uconv4_3'] = ConvLayer(inputNetwork['upool4'], 512, 3, pad=1)
	print "uconv4_3: {}".format(inputNetwork['uconv4_3'].output_shape[1:])

	inputNetwork['uconv4_2'] = ConvLayer(inputNetwork['uconv4_3'], 512, 3, pad=1)
	print "uconv4_2: {}".format(inputNetwork['uconv4_2'].output_shape[1:])

	inputNetwork['uconv4_1'] = ConvLayer(inputNetwork['uconv4_2'], 512, 3, pad=1)
	print "uconv4_1: {}".format(inputNetwork['uconv4_1'].output_shape[1:])

	inputNetwork['upool3'] = Upscale2DLayer(inputNetwork['uconv4_1'], scale_factor=2)
	print "upool3: {}".format(inputNetwork['upool3'].output_shape[1:])

	inputNetwork['uconv3_3'] = ConvLayer(inputNetwork['upool3'], 256, 3, pad=1)
	print "uconv3_3: {}".format(inputNetwork['uconv3_3'].output_shape[1:])

	inputNetwork['uconv3_2'] = ConvLayer(inputNetwork['uconv3_3'], 256, 3, pad=1)
	print "uconv3_2: {}".format(inputNetwork['uconv3_2'].output_shape[1:])

	inputNetwork['uconv3_1'] = ConvLayer(inputNetwork['uconv3_2'], 256, 3, pad=1)
	print "uconv3_1: {}".format(inputNetwork['uconv3_1'].output_shape[1:])

	inputNetwork['upool2'] = Upscale2DLayer(inputNetwork['uconv3_1'], scale_factor=2)
	print "upool2: {}".format(inputNetwork['upool2'].output_shape[1:])

	inputNetwork['uconv2_2'] = ConvLayer(inputNetwork['upool2'], 128, 3, pad=1)
	print "uconv2_2: {}".format(inputNetwork['uconv2_2'].output_shape[1:])

	inputNetwork['uconv2_1'] = ConvLayer(inputNetwork['uconv2_2'], 128, 3, pad=1)
	print "uconv2_1: {}".format(inputNetwork['uconv2_1'].output_shape[1:])

	inputNetwork['upool1'] = Upscale2DLayer(inputNetwork['uconv2_1'], scale_factor=2)
	print "upool1: {}".format(inputNetwork['upool1'].output_shape[1:])

	inputNetwork['uconv1_2'] = ConvLayer(inputNetwork['upool1'], 64, 3, pad=1)
	print "uconv1_2: {}".format(inputNetwork['uconv1_2'].output_shape[1:])

	inputNetwork['uconv1_1'] = ConvLayer(inputNetwork['uconv1_2'], 64, 3, pad=1)
	print "uconv1_1: {}".format(inputNetwork['uconv1_1'].output_shape[1:])

	inputNetwork['output'] = ConvLayer(inputNetwork['uconv1_1'], 1, 1, pad=0)
	print "output: {}".format(inputNetwork['output'].output_shape[1:])    
	
	# Compile functions
	outputSaliency = T.tensor4()

	initialLearningRate = 0.01
	initialMomentum = 0.0 # start momentum at 0.0
	maxMomentum = 0.9
	minLearningRate = 0.00001

	currentLearningRate = theano.shared( np.array( initialLearningRate, dtype = theano.config.floatX ) )
	currentMomentum = theano.shared( np.array( initialMomentum, dtype = theano.config.floatX ) )
	 
	prediction = lasagne.layers.get_output(inputNetwork['output'])
	test_prediction = lasagne.layers.get_output(inputNetwork['output'], deterministic=True)

	loss = lasagne.objectives.squared_error( prediction, outputSaliency )
	loss = loss.mean()

	params = lasagne.layers.get_all_params(inputNetwork['output'], trainable=True)

	updates_sgd = lasagne.updates.sgd(loss, params, learning_rate = currentLearningRate )
	updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum = currentMomentum ) 

	trainFunction = theano.function([inputImagePlaces, inputImageFace, inputImageImageNet, outputSaliency],\
									loss, updates=updates, allow_input_downcast=True)

	predictFunction = theano.function([inputImagePlaces,inputImageFace,inputImageImageNet], test_prediction)

	# Data structures to hold batches
	batchInPlaces = np.zeros((batchSize, 3, inputHeight, inputWidth ), theano.config.floatX )
	batchInFaces = np.zeros((batchSize, 3, inputHeight, inputWidth ), theano.config.floatX )
	batchInImageNet = np.zeros((batchSize, 3, inputHeight, inputWidth ), theano.config.floatX )
	batchOut = np.zeros((batchSize, 1, inputHeight, inputWidth ), theano.config.floatX )

	for currEpoch in tqdm(range(numEpochs), ncols=20):
		
		random.shuffle( trainData )

		err = 0.
		numBatches = 0.
		
		for currChunk in chunks(trainData, batchSize):

			if len(currChunk) != batchSize:
				continue

			for k in range( batchSize ):
				batchInPlaces[k,...] = ( trainData[k].image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputPlaces.meanImage )
				batchInFaces[k,...] = ( trainData[k].image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputFaces.meanImage )
				batchInImageNet[k,...] = ( trainData[k].image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputImageNet.meanImage )
				batchOut[k,...] = (trainData[k].saliency.data.astype(theano.config.floatX))/255.
			err += trainFunction( batchInPlaces, batchInFaces, batchInImageNet, batchOut )
			
		print 'Epoch:', currEpoch, ' ->', err 
		np.savez( "modelWights{:04d}.npz".format(currEpoch), *lasagne.layers.get_all_param_values(model.net['output']))
