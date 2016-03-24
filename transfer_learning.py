'''
    Transfer learning
    Transfer the weights of vgg16(lasagne) to vgg16(nolearn).
    vgg16(nolearn) is an odered dicctionary. ---> vgg_weights.pickle
'''
import numpy as np
import theano
import theano.tensor as T
import lasagne
import skimage.transform
import sklearn.cross_validation
import pickle
import os
import matplotlib.pyplot as plt
import cPickle 
import glob

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
from lasagne.updates import adam
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import BatchIterator

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))

    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

# Load model weights and metadata
d = pickle.load(open('../vgg16.pkl'))

# Build the network and fill with pretrained weights
net = build_model()
lasagne.layers.set_all_param_values(net['prob'], d['param values'])


layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 3, 224, 224)}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad':1}),
    (MaxPool2DLayer,{'pool_size': (2,2)}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad':1}),
    (MaxPool2DLayer,{'pool_size': (2,2)}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (MaxPool2DLayer,{'pool_size': (2,2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (MaxPool2DLayer,{'pool_size': (2,2)}),

    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (MaxPool2DLayer,{'pool_size': (2,2)}),

    (DenseLayer,{'num_units': 4096}),
    (DenseLayer,{'num_units': 4096}),
    (DenseLayer,{'num_units': 1000, 'nonlinearity': None}),
    (NonlinearityLayer,{'name': 'prob', 'nonlinearity': softmax}),
]

net0 = NeuralNet(
    layers=layers0,
    update_learning_rate=0.01,
    verbose=1,
)
net0.initialize()
dest_layers = net0.get_all_layers()
source_layers = lasagne.layers.get_all_layers(net['prob'])

success = "Loaded parameters to layer '{}' (shape {})."
failure = ("Could not load parameters to layer '{}' because "
                   "shapes did not match: {} vs {}.")
'''
for l_d,l_s in zip(dest_layers,source_layers):
    for p1 ,p2 in zip(l_d.get_params(),l_s.get_params()):
        shape1 = p1.get_value().shape
        shape2 = p2.get_value().shape
        shape1s = 'x'.join(map(str, shape1))
        shape2s = 'x'.join(map(str, shape2))
        if shape1 == shape2:
            p1.set_value(p2.get_value())
            print(success.format(
                'key', shape1s, shape2s))

        else:
            print(failure.format(
                key, shape1s, shape2s))

layer_list = lasagne.layers.get_all_layers(net['prob'])
print len(layer_list)
conv1_1_params = lasagne.layers.get_all_params(layer_list[1])
print conv1_1_params == [net['conv1_1'].W, net['conv1_1'].b]

'''
d = net0.layers_
for key,l_s in zip(d,source_layers):
    l_d = d[key]
    for p1 ,p2 in zip(l_d.get_params(),l_s.get_params()):
        shape1 = p1.get_value().shape
        shape2 = p2.get_value().shape
        shape1s = 'x'.join(map(str, shape1))
        shape2s = 'x'.join(map(str, shape2))
        if shape1 == shape2:
            p1.set_value(p2.get_value())
            print(success.format(
                key, shape1s, shape2s))
        else:
            print(failure.format(
                key, shape1s, shape2s))

net0.save_params_to('vgg_weights.pickle')
