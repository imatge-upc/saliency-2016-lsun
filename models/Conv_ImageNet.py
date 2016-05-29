import theano
import theano.tensor as T
import lasagne
import numpy as np
from theano.tensor.nnet import abstract_conv

from lasagne.layers import Layer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, InverseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX

from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import cPickle as pickle
from collections import OrderedDict


def sgd(loss_or_grads, params, learning_rate, num_ft_params):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    num_ft_params : The number of parameters to finetuning,
        (W,b) of each conv-layer corresponds to 2 params.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    flag = 0

    for param, grad in zip(params, grads):
        if flag < num_ft_params:
            updates[param] = param - 0.08 * learning_rate * grad
        else:
            updates[param] = param - learning_rate * grad
        flag += 1
    return updates


class Model:
    def __init__(self, batchesPerEpoch):
        self.inputWidth = 256
        self.inputHeight = 192

        self.initialLearningRate = 0.01
        self.initialMomentum = 0.0  # start momentum at 0.0
        self.maxMomentum = 0.9
        self.minLearningRate = 0.00001

        self.currentLearningRate = theano.shared(np.array(self.initialLearningRate, dtype=theano.config.floatX))
        self.currentMomentum = theano.shared(np.array(self.initialMomentum, dtype=theano.config.floatX))

        # self.lrDecay = np.array(0.995, dtype=theano.config.floatX)

        self.net = None

        self.trainFunction = None
        self.predictFunction = None
        self.meanImage_VGG_ImageNet = None

        # LearningRate, Momentum scheduling
        self.numBatchesPerEpoch = batchesPerEpoch
        self.currBatchNum = 0

    def buildInputNetwork_VGG_ImageNet(self, input_layer, input_var=None):
        net = {}

        net['input'] = input_layer
        print "Input: {}".format(net['input'].output_shape[1:])

        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
        net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_1'].output_shape[1:])

        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
        net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_2'].output_shape[1:])

        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        print "Input: {}".format(net['pool1'].output_shape[1:])

        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
        net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_1'].output_shape[1:])

        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
        net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_2'].output_shape[1:])

        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        print "Input: {}".format(net['pool2'].output_shape[1:])

        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
        net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_1'].output_shape[1:])

        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
        net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_2'].output_shape[1:])

        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
        net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_3'].output_shape[1:])

        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        print "Input: {}".format(net['pool3'].output_shape[1:])

        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape)
        net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape)
        print "Input: {}".format(net['conv5_3'].output_shape[1:])

        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        print "Input: {}".format(net['pool5'].output_shape[1:])

        # Set out weights
        d = pickle.load(open('/scratch/local/jpang/vgg16.pkl'))
        numElementsToSet = 26  # Number of W and b elements for the first convolutional layers
        lasagne.layers.set_all_param_values(net['pool5'], d['param values'][:numElementsToSet])

        self.meanImage_VGG_ImageNet = d['mean value'][:, np.newaxis, np.newaxis]

        return net

    def buildInputNetwork_VGG_Faces(self, input_layer, input_var=None):
        net = {}

        net['input'] = input_layer
        print "Input: {}".format(net['input'].output_shape[1:])

        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
        net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_1'].output_shape[1:])

        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
        net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_2'].output_shape[1:])

        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        print "Input: {}".format(net['pool1'].output_shape[1:])

        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
        net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_1'].output_shape[1:])

        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
        net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_2'].output_shape[1:])

        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        print "Input: {}".format(net['pool2'].output_shape[1:])

        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
        net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_1'].output_shape[1:])

        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
        net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_2'].output_shape[1:])

        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
        net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_3'].output_shape[1:])

        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        print "Input: {}".format(net['pool3'].output_shape[1:])

        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape)
        net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape)
        print "Input: {}".format(net['conv5_3'].output_shape[1:])

        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        print "Input: {}".format(net['pool5'].output_shape[1:])

        # Set out weights
        d = pickle.load(open('/scratch/local/jpang/ConvVGG_Faces.pkl'))
        numElementsToSet = 26  # Number of W and b elements for the first convolutional layers
        lasagne.layers.set_all_param_values(net['pool5'], d['params'][:numElementsToSet])

        return net

    def buildInputNetwork_VGG_Places(self, input_layer, input_var=None):
        net = {}

        net['input'] = input_layer
        print "Input: {}".format(net['input'].output_shape[1:])

        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
        net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_1'].output_shape[1:])

        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
        net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_2'].output_shape[1:])

        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        print "Input: {}".format(net['pool1'].output_shape[1:])

        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
        net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_1'].output_shape[1:])

        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
        net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_2'].output_shape[1:])

        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        print "Input: {}".format(net['pool2'].output_shape[1:])

        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
        net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_1'].output_shape[1:])

        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
        net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_2'].output_shape[1:])

        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
        net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_3'].output_shape[1:])

        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        print "Input: {}".format(net['pool3'].output_shape[1:])

        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape)
        net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape)
        print "Input: {}".format(net['conv5_3'].output_shape[1:])

        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        print "Input: {}".format(net['pool5'].output_shape[1:])

        # Set out weights
        d = pickle.load(open('/scratch/local/jpang/ConvVGG_Places.pkl'))
        numElementsToSet = 26  # Number of W and b elements for the first convolutional layers
        lasagne.layers.set_all_param_values(net['pool5'], d['params'][:numElementsToSet])

        return net

    def buildOutputNetwork(self, input_stage):
        inputNetwork = {}

        inputNetwork['uconv5_3'] = ConvLayer(input_stage, 512, 3, pad=1)
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

        return inputNetwork

    def buildOutputNetwork_M(self, input_network):
        outputNetwork = {}

        outputNetwork['upool1'] = Upscale2DLayer(input_network['conv5_3'], scale_factor=4)
        print "upool1: {}".format(outputNetwork['upool1'].output_shape[1:])

        outputNetwork['upool2'] = Upscale2DLayer(input_network['conv4_3'], scale_factor=2)
        print "upool2: {}".format(outputNetwork['upool2'].output_shape[1:])

        outputNetwork['concat'] = concat((outputNetwork['upool1'], outputNetwork['upool2'],
                                          input_network['conv3_3']), axis=1)
        print "concat: {}".format(outputNetwork['concat'].output_shape[1:])

        outputNetwork['conv1_1'] = ConvLayer(outputNetwork['concat'], 512, 3, pad=1)
        print "conv1_1: {}".format(outputNetwork['conv1_1'].output_shape[1:])

        # outputNetwork['conv1_2'] = ConvLayer(outputNetwork['conv1_1'], 512, 3, pad=1)
        # print "conv1_2: {}".format(outputNetwork['conv1_2'].output_shape[1:])

        outputNetwork['conv2_1'] = ConvLayer(outputNetwork['conv1_1'], 256, 3, pad=1)
        print "conv2_1: {}".format(outputNetwork['conv2_1'].output_shape[1:])

        # outputNetwork['conv2_2'] = ConvLayer(outputNetwork['conv2_1'], 256, 3, pad=1)
        # print "conv2_2: {}".format(outputNetwork['conv2_2'].output_shape[1:])

        outputNetwork['conv3_1'] = ConvLayer(outputNetwork['conv2_1'], 128, 3, pad=1)
        print "conv3_1: {}".format(outputNetwork['conv3_1'].output_shape[1:])

        # outputNetwork['conv3_2'] = ConvLayer(outputNetwork['conv3_1'], 128, 3, pad=1)
        # print "conv3_2: {}".format(outputNetwork['conv3_2'].output_shape[1:])

        outputNetwork['upool3'] = Upscale2DLayer(outputNetwork['conv3_1'], scale_factor=4)
        print "upool3: {}".format(outputNetwork['upool3'].output_shape[1:])

        outputNetwork['conv4'] = ConvLayer(outputNetwork['upool3'], 64, 3, pad=1)
        print "conv4: {}".format(outputNetwork['conv4'].output_shape[1:])

        outputNetwork['conv5'] = ConvLayer(outputNetwork['conv4'], 32, 3, pad=1)
        print "conv5: {}".format(outputNetwork['conv5'].output_shape[1:])

        outputNetwork['output'] = ConvLayer(outputNetwork['conv5'], 1, 1, pad=0)
        print "output: {}".format(outputNetwork['output'].output_shape[1:])

        return outputNetwork

    def build(self, input_var, output_var):
        input_layer = InputLayer((None, 3, self.inputHeight, self.inputWidth), input_var=input_var);

        vggNet = self.buildInputNetwork_VGG_ImageNet(input_layer, input_var)

        vggNet_faces = self.buildInputNetwork_VGG_Faces(input_layer, input_var)

        vggNet_places = self.buildInputNetwork_VGG_Places(input_layer, input_var)

        # input_stage = vggNet['conv5_3']

        # input_stage = concat((vggNet['conv5_3'], vggNet_faces['conv5_3']), axis=1)

        # input_stage = concat((vggNet['conv5_3'], vggNet_places['conv5_3']),axis=1)

        input_stage = concat((vggNet['conv5_3'], vggNet_places['conv5_3'], vggNet_faces['conv5_3']), axis=1)

        self.net = self.buildOutputNetwork(input_stage)

        outputLayerName = 'output'

        # epochToLoad = 49
        # with np.load("modelWeights{:04d}.npz".format(epochToLoad)) as f:
        #    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(self.net['output'], param_values)

        prediction = lasagne.layers.get_output(self.net[outputLayerName])

        test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)

        loss = lasagne.objectives.squared_error(prediction, output_var)

        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)

        # Here to change the number of parameters to finetuning.
        updates_sgd = sgd(loss, params, learning_rate=self.currentLearningRate, num_ft_params=36)

        # updates_sgd = lasagne.updates.sgd(loss, params, learning_rate=self.currentLearningRate)

        updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum=self.currentMomentum)

        self.trainFunction = theano.function([input_var, output_var], loss, updates=updates, allow_input_downcast=True)
        self.predictFunction = theano.function([input_var], test_prediction)
