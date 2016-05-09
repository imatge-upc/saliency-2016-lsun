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
from scipy import ndimage

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, InverseLayer
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

from models import Conv_ImageNet as dnnModel


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


if __name__ == "__main__":

    # Load data
    print 'Loading validation data...'
    with open('quickData.pickle', 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'
    # with open( 'testData.pickle', 'rb') as f:
    # testData = pickle.load( f )

    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    width = 256
    height = 192

    model = dnnModel.Model(1000)
    model.build(inputImage, outputSaliency)

    epochToLoad = 40

    with np.load("./model/modelWeights{:04d}.npz".format(epochToLoad)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

    imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])

    # Let's pick a random image and process it!

    numRandom = random.choice(range(len(validationData)))

    #numRandom = 8
    cv2.imwrite('validationRandomImage.png', cv2.cvtColor(validationData[numRandom].image.data, cv2.cv.CV_RGB2BGR))
    cv2.imwrite('validationRandomSaliencyGT.png', validationData[numRandom].saliency.data)

    blob = np.zeros((1, 3, height, width), theano.config.floatX)
    blob[0, ...] = (validationData[numRandom].image.data.astype(theano.config.floatX).transpose(2, 0,
                                                                                                1) - imageMean) / 255.

    result = np.squeeze(model.predictFunction(blob))

    saliencyMap = (result * 255).astype(np.uint8)
    cv2.imwrite('validationRandomSaliencyPred.png', saliencyMap)
    saliencyMap = np.clip(saliencyMap, 0, 255)
    # resize back to original size
    h, w = (480, 640)
    saliencyMap = cv2.resize(saliencyMap, (w, h), interpolation=cv2.INTER_CUBIC)
    # blur
    blur_size = 5
    saliencyMap = cv2.GaussianBlur(saliencyMap, (blur_size, blur_size), 0)
    # clip again
    saliencyMap = np.clip(saliencyMap, 0, 255)
    # stretch
    if saliencyMap.max() > 0:
        saliencyMap = (saliencyMap / float(saliencyMap.max())) * 255.0
    saliencyMap = saliencyMap.astype(np.uint8)
    #cv2.imwrite('validationRandomSaliencyPred_p.png', saliencyMap)
    '''
    filtered = lasagne.layers.get_output(net['conv5_3'], deterministic=True)
    f_filter = theano.function([inputImage], filtered)

    im = np.squeeze(f_filter(blob))
    print(im.shape)
    print im[0].shape
    for i in range(512):
        image = cv2.resize(im[i], (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./test_images/' + str(i) + '.png', image)
        i += 1
    '''
