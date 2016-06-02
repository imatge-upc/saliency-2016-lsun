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
from models import ConvVGGPlacesUpsample, ConvVGGFacesUpsample, ConvVGGImageNetUpsample

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
	inputImageImageNet = T.tensor4()
	networkImageNet = OrderedDict()
	inputImageNet = ConvVGGImageNetUpsample.ConvVGGImageNetUpsample(inputWidth,inputHeight)
	inputImageNet.build( networkImageNet, 'ConvVGG_ImageNet.pkl', inputImageImageNet )
	
	# Create network FACES
	inputImageFace = T.tensor4()
	networkFaces = OrderedDict()
	inputFaces = ConvVGGFacesUpsample.ConvVGGFacesUpsample(inputWidth,inputHeight)
	inputFaces.build( networkFaces,'ConvVGG_Faces.pkl', inputImageFace )	

	# Create network ImageNet
	inputImagePlaces = T.tensor4()
	networkPlaces = OrderedDict()
	inputPlaces = ConvVGGPlacesUpsample.ConvVGGPlacesUpsample(inputWidth,inputHeight)
	inputPlaces.build( networkPlaces, 'ConvVGG_Places.pkl', inputImagePlaces )	
	
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
	print "Output: {}".format(inputNetwork['concat'].output_shape[1:])

	
	# Deconvolutional part
	inputNetwork['output'] = Upscale2DLayer( ConvLayer(inputNetwork['concat'], 1, 3, pad = 1), scale_factor=4 )
	print "Output: {}".format(inputNetwork['output'].output_shape[1:]) 
	
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
		
		startTime = time.time()
		
		random.shuffle( trainData )

		err = 0.
		numBatches = 0.
		
		for currChunk in tqdm(chunks(trainData, batchSize), ncols=30, total=len(trainData)/batchSize):

			if len(currChunk) != batchSize:
				continue

			for k, dataElement in enumerate( currChunk ):
				batchInPlaces[k,...] = ( dataElement.image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputPlaces.meanImage )
				batchInFaces[k,...] = ( dataElement.image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputFaces.meanImage )
				batchInImageNet[k,...] = ( dataElement.image.data.astype( theano.config.floatX ).transpose(2,0,1) - inputImageNet.meanImage )
				batchOut[k,...] = (dataElement.saliency.data.astype(theano.config.floatX))/255.
			err += trainFunction( batchInPlaces, batchInFaces, batchInImageNet, batchOut )
			
		print 'Epoch:', currEpoch, ' ->', err 
		print '-- Took ', time.time() - startTime, ' seconds'
		np.savez( "modelWights{:04d}.npz".format(currEpoch), *lasagne.layers.get_all_param_values(inputNetwork['output']))
