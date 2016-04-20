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

import scipy.io
import cPickle as pickle
from scipy import misc

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

    model = dnnModel.Model(1000)
    model.build( inputImage, outputSaliency )

    epochToLoad = 49

    with np.load("./model/modelWeights{:04d}.npz".format(epochToLoad)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

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
        y_pred[idx:idx+batchSize] = model.predictFunction( batch )
        idx=idx+batchSize

    for res, name in zip(y_pred,listImages):
        saliencyMap = np.squeeze( res )
        #blured= ndimage.gaussian_filter(tmp, sigma=3)
        #y = misc.imresize(blured,(480,640))/255.
        #cv2.resize(saliencyMap,(640,480)
        saliencyMap = misc.imresize(saliencyMap,(480,640))/255.
        saliencyMapMatlab = {'I':saliencyMap}
        scipy.io.savemat('/imatge/jpan/work/lsun2016/results/19-04-2016/'+name,saliencyMapMatlab )