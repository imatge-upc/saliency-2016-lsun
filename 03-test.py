import os
import numpy as np
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
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


model_path = "model/img_places_conv45_old"
epochToLoad = 180


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def predict(model, validationData, numEpoch, dir='test'):
    width = model.inputWidth
    height = model.inputHeight

    blob = np.zeros((1, 3, height, width), theano.config.floatX)
    # blob2 = np.zeros((1, 1, height, width), theano.config.floatX)
    # imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])

    blob[0, ...] = (validationData.image.data.astype(theano.config.floatX).transpose(2, 0, 1))  # - imageMean) / 255.
    # blob2[0, ...] = (validationData.saliency.data.astype(theano.config.floatX))

    result = np.squeeze(model.predictFunction(blob))  # , blob2))

    saliencyMap = (result * 255).astype(np.uint8)

    '''
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
    '''

    # saliencyMap = (result).astype(np.uint8)
    # print saliencyMap.shape
    # cv2.imwrite('./'+dir+'/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch),  cv2.cvtColor(saliencyMap.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    cv2.imwrite('./' + dir + '/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch), saliencyMap)

    # cv2.imwrite('./results/validationRandomImage_'+str(numEpoch)+'.png',
    #            cv2.cvtColor(validationData.image.data, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./results/validationRandomSaliencyGT_'+str(numEpoch)+'.png', validationData.saliency.data)


def test():
    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    model = dnnModel.Model()
    model.build_generator(inputImage, outputSaliency)
    # model.build_generator(inputImage, outputSaliency)

    print 'Loading validation data...'
    with open('valSample.pkl', 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'

    with np.load(
            "/home/users/jpang/scratch-local/lsun2016/saliency-2016-lsun/"+model_path+"/modelWeights{:04d}.npz".format(
                epochToLoad)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

    [predict(model=model, validationData=validationData[currEpoch], numEpoch=currEpoch, dir='results') for currEpoch in
     range(10)]


if __name__ == "__main__":
    test()