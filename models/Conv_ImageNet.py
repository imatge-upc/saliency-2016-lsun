import theano
import theano.tensor as T
import lasagne
import numpy as np
from theano.tensor.nnet import abstract_conv
import cv2
from lasagne.layers import Layer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, InverseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX

from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import TransposedConv2DLayer, DilatedConv2DLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import cPickle as pickle
from collections import OrderedDict
import nn


class RGBtoBGRLayer(lasagne.layers.Layer):
    def __init__(self, l_in, bgr_mean=np.array([103.939, 116.779, 123.68]),
                 data_format='bc01', **kwargs):
        """A Layer to normalize and convert images from RGB to BGR
        This layer converts images from RGB to BGR to adapt to Caffe
        that uses OpenCV, which uses BGR. It also subtracts the
        per-pixel mean.
        Parameters
        ----------
        l_in : :class:``lasagne.layers.Layer``
            The incoming layer, typically an
            :class:``lasagne.layers.InputLayer``
        bgr_mean : iterable of 3 ints
            The mean of each channel. By default, the ImageNet
            mean values are used.
        data_format : str
            The format of l_in, either `b01c` (batch, rows, cols,
            channels) or `bc01` (batch, channels, rows, cols)
        """
        super(RGBtoBGRLayer, self).__init__(l_in, **kwargs)
        assert data_format in ['bc01', 'b01c']
        self.l_in = l_in
        floatX = theano.config.floatX
        self.bgr_mean = bgr_mean.astype(floatX)
        self.data_format = data_format

    def get_output_for(self, input_im, **kwargs):
        if self.data_format == 'bc01':
            input_im = input_im[:, ::-1, :, :]
            input_im -= self.bgr_mean[:, np.newaxis, np.newaxis]
        else:
            input_im = input_im[:, :, :, ::-1]
            input_im -= self.bgr_mean
        return input_im


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
    def __init__(self):
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
        self.discriminator = None

        self.trainFunction = None
        self.D_trainFunction = None
        self.G_trainFunction = None
        self.predictFunction = None
        self.featureFunction = None
        self.meanImage_VGG_ImageNet = None

        # LearningRate, Momentum scheduling
        # self.numBatchesPerEpoch = batchesPerEpoch
        self.currBatchNum = 0

    def buildInputNetwork_VGG_ImageNet(self, input_layer, input_var=None):
        net = {'input': input_layer}

        print "Input: {}".format(net['input'].output_shape[1:])

        net['bgr'] = RGBtoBGRLayer(net['input'])

        net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
        net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
        net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_1'].output_shape[1:])

        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
        net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_2'].output_shape[1:])

        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        print "Input: {}".format(net['pool1'].output_shape[1:])

        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
        net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_1'].output_shape[1:])

        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
        net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_2'].output_shape[1:])

        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        print "Input: {}".format(net['pool2'].output_shape[1:])

        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
        net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_1'].output_shape[1:])

        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
        net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_2'].output_shape[1:])

        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
        net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_3'].output_shape[1:])

        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        print "Input: {}".format(net['pool3'].output_shape[1:])

        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv4_3'].W.get_value().shape)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape)
        net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape)
        print "Input: {}".format(net['conv5_3'].output_shape[1:])

        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        # print "Input: {}".format(net['pool5'].output_shape[1:])

        # Set out weights
        d = pickle.load(open('/scratch/local/jpang/vgg16.pkl'))
        numElementsToSet = 26  # Number of W and b elements for the first convolutional layers
        lasagne.layers.set_all_param_values(net['pool5'], d['param values'][:numElementsToSet])

        self.meanImage_VGG_ImageNet = d['mean value'][:, np.newaxis, np.newaxis]

        return net

    def buildInputNetwork_VGG_Places(self, input_layer, input_var=None):
        net = {'input': input_layer}

        print "Input: {}".format(net['input'].output_shape[1:])

        net['bgr'] = RGBtoBGRLayer(net['input'], bgr_mean=np.array([105.487823486, 113.741088867, 116.060394287]))

        net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
        net['conv1_1'].add_param(net['conv1_1'].W, net['conv1_1'].W.get_value().shape, trainable=False)
        net['conv1_1'].add_param(net['conv1_1'].b, net['conv1_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_1'].output_shape[1:])

        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'].add_param(net['conv1_2'].W, net['conv1_2'].W.get_value().shape, trainable=False)
        net['conv1_2'].add_param(net['conv1_2'].b, net['conv1_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv1_2'].output_shape[1:])

        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        print "Input: {}".format(net['pool1'].output_shape[1:])

        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_1'].add_param(net['conv2_1'].W, net['conv2_1'].W.get_value().shape, trainable=False)
        net['conv2_1'].add_param(net['conv2_1'].b, net['conv2_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_1'].output_shape[1:])

        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'].add_param(net['conv2_2'].W, net['conv2_2'].W.get_value().shape, trainable=False)
        net['conv2_2'].add_param(net['conv2_2'].b, net['conv2_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv2_2'].output_shape[1:])

        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        print "Input: {}".format(net['pool2'].output_shape[1:])

        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_1'].add_param(net['conv3_1'].W, net['conv3_1'].W.get_value().shape, trainable=False)
        net['conv3_1'].add_param(net['conv3_1'].b, net['conv3_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_1'].output_shape[1:])

        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'].add_param(net['conv3_2'].W, net['conv3_2'].W.get_value().shape, trainable=False)
        net['conv3_2'].add_param(net['conv3_2'].b, net['conv3_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_2'].output_shape[1:])

        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'].add_param(net['conv3_3'].W, net['conv3_3'].W.get_value().shape, trainable=False)
        net['conv3_3'].add_param(net['conv3_3'].b, net['conv3_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv3_3'].output_shape[1:])

        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        print "Input: {}".format(net['pool3'].output_shape[1:])

        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
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
        inputNetwork = {'uconv5_3': ConvLayer(input_stage, 512, 3, pad=1)}

        print "uconv5_3: {}".format(inputNetwork['uconv5_3'].output_shape[1:])

        inputNetwork['uconv5_2'] = ConvLayer(inputNetwork['uconv5_3'], 512, 3, pad=1)
        print "uconv5_2: {}".format(inputNetwork['uconv5_2'].output_shape[1:])

        inputNetwork['uconv5_1'] = ConvLayer(inputNetwork['uconv5_2'], 512, 3, pad=1)
        print "uconv5_1: {}".format(inputNetwork['uconv5_1'].output_shape[1:])

        inputNetwork['upool4'] = Upscale2DLayer(inputNetwork['uconv5_1'], scale_factor=2)
        # inputNetwork['upool4'] = TransposedConv2DLayer(inputNetwork['uconv5_1'], 512, 2, stride=2)
        # inputNetwork['upool4'] = nn.Deconv2DLayer(inputNetwork['uconv5_1'], (None,512,24,32), (5,5), W=Normal(0.05),nonlinearity=nn.relu)
        print "upool4: {}".format(inputNetwork['upool4'].output_shape[1:])

        inputNetwork['uconv4_3'] = ConvLayer(inputNetwork['upool4'], 512, 3, pad=1)
        print "uconv4_3: {}".format(inputNetwork['uconv4_3'].output_shape[1:])

        inputNetwork['uconv4_2'] = ConvLayer(inputNetwork['uconv4_3'], 512, 3, pad=1)
        print "uconv4_2: {}".format(inputNetwork['uconv4_2'].output_shape[1:])

        inputNetwork['uconv4_1'] = ConvLayer(inputNetwork['uconv4_2'], 512, 3, pad=1)
        print "uconv4_1: {}".format(inputNetwork['uconv4_1'].output_shape[1:])

        inputNetwork['upool3'] = Upscale2DLayer(inputNetwork['uconv4_1'], scale_factor=2)
        # inputNetwork['upool3'] = TransposedConv2DLayer(inputNetwork['uconv4_1'], 256, 2, stride=2)
        # inputNetwork['upool3'] = nn.Deconv2DLayer(inputNetwork['uconv4_1'], (None,256,48,64), (5,5), W=Normal(0.05),nonlinearity=nn.relu)
        print "upool3: {}".format(inputNetwork['upool3'].output_shape[1:])

        inputNetwork['uconv3_3'] = ConvLayer(inputNetwork['upool3'], 256, 3, pad=1)
        print "uconv3_3: {}".format(inputNetwork['uconv3_3'].output_shape[1:])

        inputNetwork['uconv3_2'] = ConvLayer(inputNetwork['uconv3_3'], 256, 3, pad=1)
        print "uconv3_2: {}".format(inputNetwork['uconv3_2'].output_shape[1:])

        inputNetwork['uconv3_1'] = ConvLayer(inputNetwork['uconv3_2'], 256, 3, pad=1)
        print "uconv3_1: {}".format(inputNetwork['uconv3_1'].output_shape[1:])

        inputNetwork['upool2'] = Upscale2DLayer(inputNetwork['uconv3_1'], scale_factor=2)
        # inputNetwork['upool2'] = TransposedConv2DLayer(inputNetwork['uconv3_1'], 128, 2, stride=2)
        # inputNetwork['upool2'] = nn.Deconv2DLayer(inputNetwork['uconv3_1'], (None,128,96,128), (5,5), W=Normal(0.05),nonlinearity=nn.relu)
        print "upool2: {}".format(inputNetwork['upool2'].output_shape[1:])

        inputNetwork['uconv2_2'] = ConvLayer(inputNetwork['upool2'], 128, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.rectify)
        print "uconv2_2: {}".format(inputNetwork['uconv2_2'].output_shape[1:])

        inputNetwork['uconv2_1'] = ConvLayer(inputNetwork['uconv2_2'], 128, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.rectify)
        print "uconv2_1: {}".format(inputNetwork['uconv2_1'].output_shape[1:])

        inputNetwork['upool1'] = Upscale2DLayer(inputNetwork['uconv2_1'], scale_factor=2)
        # inputNetwork['upool1'] = TransposedConv2DLayer(inputNetwork['uconv2_1'], 64, 2, stride=2)
        # inputNetwork['upool1'] = nn.Deconv2DLayer(inputNetwork['uconv2_1'], (None,64,192,256), (5,5), W=Normal(0.05),nonlinearity=nn.relu)
        print "upool1: {}".format(inputNetwork['upool1'].output_shape[1:])

        inputNetwork['uconv1_2'] = ConvLayer(inputNetwork['upool1'], 64, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.rectify)
        print "uconv1_2: {}".format(inputNetwork['uconv1_2'].output_shape[1:])

        inputNetwork['uconv1_1'] = ConvLayer(inputNetwork['uconv1_2'], 64, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.rectify)
        print "uconv1_1: {}".format(inputNetwork['uconv1_1'].output_shape[1:])

        inputNetwork['output'] = ConvLayer(inputNetwork['uconv1_1'], 1, 1, pad=0,
                                           nonlinearity=lasagne.nonlinearities.sigmoid)
        print "output: {}".format(inputNetwork['output'].output_shape[1:])

        return inputNetwork

    def buildOutputNetwork_S(self, input_stage):

        inputNetwork = {'upool4': Upscale2DLayer(input_stage, scale_factor=2)}

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

        inputNetwork['uconv2_2'] = DilatedConv2DLayer(inputNetwork['upool2'], 256, 3, dilation=(2, 2))
        print "uconv2_2: {}".format(inputNetwork['uconv2_2'].output_shape[1:])

        inputNetwork['uconv2_1'] = TransposedConv2DLayer(inputNetwork['uconv2_2'], 256, 5)
        print "uconv2_1: {}".format(inputNetwork['uconv2_1'].output_shape[1:])

        inputNetwork['upool1'] = Upscale2DLayer(inputNetwork['uconv2_1'], scale_factor=2)
        print "upool1: {}".format(inputNetwork['upool1'].output_shape[1:])

        inputNetwork['uconv1_2'] = ConvLayer(inputNetwork['upool1'], 64, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.tanh)
        print "uconv1_2: {}".format(inputNetwork['uconv1_2'].output_shape[1:])

        inputNetwork['uconv1_1'] = ConvLayer(inputNetwork['uconv1_2'], 64, 3, pad=1,
                                             nonlinearity=lasagne.nonlinearities.tanh)
        print "uconv1_1: {}".format(inputNetwork['uconv1_1'].output_shape[1:])

        inputNetwork['output'] = ConvLayer(inputNetwork['uconv1_1'], 1, 1, pad=0,
                                           nonlinearity=lasagne.nonlinearities.sigmoid)
        print "output: {}".format(inputNetwork['output'].output_shape[1:])

        return inputNetwork

    def build_discriminator0(self, concat_var):
        # Discriminator 0 with conventional conv layers
        net1 = {'input': InputLayer((None, 4, self.inputHeight, self.inputWidth), input_var=concat_var)}

        print "Input: {}".format(net1['input'].output_shape[1:])

        net1['merge'] = ConvLayer(net1['input'], 3, 1, pad=0, flip_filters=False)
        print "merge: {}".format(net1['merge'].output_shape[1:])

        net1['conv1'] = ConvLayer(net1['merge'], 32, 3, pad=1)
        print "Input: {}".format(net1['conv1'].output_shape[1:])

        net1['pool1'] = PoolLayer(net1['conv1'], 4)
        print "Input: {}".format(net1['pool1'].output_shape[1:])

        net1['conv2_1'] = ConvLayer(net1['pool1'], 64, 3, pad=1)
        print "Input: {}".format(net1['conv2_1'].output_shape[1:])

        net1['conv2_2'] = ConvLayer(net1['conv2_1'], 64, 3, pad=1)
        print "Input: {}".format(net1['conv2_2'].output_shape[1:])

        net1['pool2'] = PoolLayer(net1['conv2_2'], 2)
        print "Input: {}".format(net1['pool2'].output_shape[1:])

        net1['conv3_1'] = nn.weight_norm(ConvLayer(net1['pool2'], 64, 3, pad=1))
        print "Input: {}".format(net1['conv3_1'].output_shape[1:])

        net1['conv3_2'] = nn.weight_norm(ConvLayer(net1['conv3_1'], 64, 3, pad=1))
        print "Input: {}".format(net1['conv3_2'].output_shape[1:])

        net1['pool3'] = PoolLayer(net1['conv3_2'], 2)
        print "Input: {}".format(net1['pool3'].output_shape[1:])

        net1['fc3'] = DenseLayer(net1['pool3'], num_units=60, nonlinearity=lasagne.nonlinearities.tanh)

        net1['fc4'] = DenseLayer(net1['fc3'], num_units=2, nonlinearity=lasagne.nonlinearities.tanh)

        net1['fc5'] = DenseLayer(net1['fc4'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print "Input: {}".format(net1['fc5'].output_shape[1:])

        return net1

    def build_discriminator1(self, concat_var):
        # Discriminator 1 with atrous conv layers
        net1 = {'input': InputLayer((None, 4, self.inputHeight, self.inputWidth), input_var=concat_var)}

        print "Input: {}".format(net1['input'].output_shape[1:])

        net1['merge'] = DilatedConv2DLayer(net1['input'], 3, 1, dilation=(3, 3))
        print "merge: {}".format(net1['merge'].output_shape[1:])

        net1['conv1'] = ConvLayer(net1['merge'], 32, 3, pad=1)
        print "conv1: {}".format(net1['conv1'].output_shape[1:])

        net1['pool1'] = PoolLayer(net1['conv1'], 2)
        print "pool1: {}".format(net1['pool1'].output_shape[1:])

        net1['conv2_1'] = ConvLayer(net1['pool1'], 64, 3, pad=1)
        print "conv2_1: {}".format(net1['conv2_1'].output_shape[1:])

        net1['conv2_2'] = ConvLayer(net1['conv2_1'], 64, 3, pad=1)
        print "conv2_2: {}".format(net1['conv2_2'].output_shape[1:])

        net1['pool2'] = PoolLayer(net1['conv2_2'], 4)
        print "pool2: {}".format(net1['pool2'].output_shape[1:])

        net1['conv3_1'] = nn.weight_norm(ConvLayer(net1['pool2'], 64, 3, pad=1))
        print "conv3_1: {}".format(net1['conv3_1'].output_shape[1:])

        net1['conv3_2'] = nn.weight_norm(ConvLayer(net1['conv3_1'], 64, 3, pad=1))
        print "conv3_2: {}".format(net1['conv3_2'].output_shape[1:])

        net1['pool3'] = PoolLayer(net1['conv3_2'], 2)
        print "pool3: {}".format(net1['pool3'].output_shape[1:])

        net1['fc3'] = DenseLayer(net1['pool3'], num_units=60, nonlinearity=lasagne.nonlinearities.tanh)

        net1['fc4'] = DenseLayer(net1['fc3'], num_units=2, nonlinearity=lasagne.nonlinearities.tanh)

        net1['fc5'] = DenseLayer(net1['fc4'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print "fc5: {}".format(net1['fc5'].output_shape[1:])

        return net1

    def build_generator(self, input_var, output_var):
        input_layer = InputLayer((None, 3, self.inputHeight, self.inputWidth), input_var=input_var)

        vggNet = self.buildInputNetwork_VGG_ImageNet(input_layer, input_var)
        input_stage = vggNet['conv5_3']
        # vggNet_places = self.buildInputNetwork_VGG_Places(input_layer, input_var)
        # input_stage = concat((vggNet['conv5_3'], vggNet_places['conv5_3']), axis=1)

        self.net = self.buildOutputNetwork(input_stage)

        outputLayerName = 'output'

        prediction = lasagne.layers.get_output(self.net[outputLayerName])

        test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)

        self.predictFunction = theano.function([input_var], test_prediction)

        # train_err = lasagne.objectives.binary_crossentropy(prediction, output_var).mean()
        output_var_pooled = T.signal.pool.pool_2d(output_var, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)
        train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_pooled).mean()
        G_obj = train_err

        # parameters update and training

        G_params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)
        G_lr = theano.shared(np.array(0.005, dtype=theano.config.floatX))
        G_updates = lasagne.updates.nesterov_momentum(G_obj, G_params, learning_rate=G_lr, momentum=0.5)

        G_train = theano.function(inputs=[input_var, output_var], outputs=G_obj, updates=G_updates,
                                  allow_input_downcast=True)

        self.G_trainFunction = G_train

    def build(self, input_var, output_var, fake_output_var):
        # Build Generator
        input_layer = InputLayer((None, 3, self.inputHeight, self.inputWidth), input_var=input_var)
        vggNet = self.buildInputNetwork_VGG_ImageNet(input_layer, input_var)
        input_stage = vggNet['conv5_3']
        # vggNet_places = self.buildInputNetwork_VGG_Places(input_layer, input_var)
        # input_stage = concat((vggNet['conv5_3'], vggNet_places['conv5_3']), axis=1)
        self.net = self.buildOutputNetwork(input_stage)

        # Build Discriminator
        self.discriminator = self.build_discriminator0(T.concatenate([output_var, input_var], axis=1))

        outputLayerName = 'output'

        prediction = lasagne.layers.get_output(self.net[outputLayerName])
        test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)
        self.predictFunction = theano.function([input_var], test_prediction)

        feature = lasagne.layers.get_output(self.discriminator['merge'], T.concatenate([output_var, input_var], axis=1),
                                            deterministic=True)
        self.featureFunction = theano.function([input_var, output_var], feature)

        disc_lab = lasagne.layers.get_output(self.discriminator['fc5'], T.concatenate([output_var, input_var], axis=1))
        # disc_fake = lasagne.layers.get_output(self.discriminator['fc5'], T.concatenate([fake_output_var, input_var], axis=1))
        disc_gen = lasagne.layers.get_output(self.discriminator['fc5'], T.concatenate([prediction, input_var], axis=1))

        # Downscale the saliency maps
        output_var_pooled = T.signal.pool.pool_2d(output_var, (4, 4), mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)
        train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_pooled).mean()

        # l_unl = disc_lab
        # l_gen = disc_gen

        # D_obj_real = 0.5 * T.mean(T.nnet.softplus(l_unl)) - 0.5 * T.mean(T.nnet.softplus(l_gen))
        # G_obj_d = T.mean(T.nnet.softplus(l_gen))

        # D_obj_real = lasagne.objectives.binary_crossentropy(disc_lab, T.ones(disc_lab.shape)).mean()
        # D_obj_fake = lasagne.objectives.binary_crossentropy(disc_fake, T.zeros(disc_fake.shape)).mean()
        # D_obj_gen = lasagne.objectives.binary_crossentropy(disc_gen, T.zeros(disc_lab.shape)).mean()
        # D_obj_real = 0.5 * D_obj_real + 0.25 * D_obj_gen + 0.25 * D_obj_fake + 1e-4 * lasagne.regularization.regularize_network_params(self.discriminator['fc5'], lasagne.regularization.l2)

        # Define loss funtion and input data
        ones = T.ones(disc_lab.shape)
        zeros = T.zeros(disc_lab.shape)
        D_obj = lasagne.objectives.binary_crossentropy(T.concatenate([disc_lab, disc_gen], axis=0),T.concatenate([ones, zeros], axis=0)).mean() \
                + 1e-4 * lasagne.regularization.regularize_network_params(self.discriminator['fc5'],lasagne.regularization.l2)
        G_obj_d = lasagne.objectives.binary_crossentropy(disc_gen, T.ones(disc_lab.shape)).mean() \
                  + 1e-4 * lasagne.regularization.regularize_network_params(self.net[outputLayerName],lasagne.regularization.l2)

        G_obj = G_obj_d + train_err / 20
        cost = [G_obj, D_obj, train_err]

        # parameters update and training of Generator
        G_params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)
        G_lr = theano.shared(np.array(4e-4, dtype=theano.config.floatX))
        G_updates = lasagne.updates.adagrad(G_obj, G_params, learning_rate=G_lr)
        G_train = theano.function(inputs=[input_var, output_var], outputs=cost, updates=G_updates,
                                  allow_input_downcast=True)

        # parameters update and training of Discriminator
        D_params = lasagne.layers.get_all_params(self.discriminator['fc5'], trainable=True)
        D_lr = theano.shared(np.array(4e-4, dtype=theano.config.floatX))
        D_updates = lasagne.updates.adagrad(D_obj, D_params, learning_rate=D_lr)
        D_train = theano.function([input_var, output_var], cost, updates=D_updates,
                                  allow_input_downcast=True)

        self.D_trainFunction = D_train
        self.G_trainFunction = G_train
