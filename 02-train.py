import os
import numpy as np
import cv2
import sys
import cPickle as pickle
import glob
import random
import time
from tqdm import tqdm

from eliaLib import dataRepresentation

from models import Conv_ImageNet as dnnModel

import theano
import theano.tensor as T
import lasagne

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


	# Create network
	inputImage = T.tensor4()
	outputSaliency = T.tensor4()

	model = dnnModel.Model(1000)
	model.build( inputImage, outputSaliency )

	batchSize = 64
	numEpochs = 50

	
	batchIn = np.zeros((batchSize, 3, model.inputHeight, model.inputWidth ), theano.config.floatX )
	batchOut = np.zeros((batchSize, 1, model.inputHeight, model.inputWidth ), theano.config.floatX )

	for currEpoch in tqdm(range(numEpochs), ncols=20):
		
		random.shuffle( trainData )

		err = 0.
		
		numBatches = 0.
		
		for currChunk in chunks(trainData, batchSize):

			if len(currChunk) != batchSize:
				continue

			for k in range( batchSize ):
				batchIn[k,...] = (trainData[k].image.data.astype(theano.config.floatX).transpose(2,0,1)-model.meanImage_VGG_ImageNet)/255.
				batchOut[k,...] = (trainData[k].saliency.data.astype(theano.config.floatX))/255.
			err += model.trainFunction( batchIn, batchOut)
			
		print 'Epoch:', currEpoch, ' ->', err 
		np.savez( "modelWights{:04d}.npz".format(currEpoch), *lasagne.layers.get_all_param_values(model.net['output']))
