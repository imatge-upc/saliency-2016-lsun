import theano
import theano.tensor as T
import lasagne
import numpy as np
from theano.tensor.nnet import abstract_conv

from lasagne.layers import Layer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, InverseLayer, TransposedConv2DLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.utils import floatX
from lasagne.init import Normal

from lasagne.layers import DropoutLayer, GlobalPoolLayer, NINLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import TransposedConv2DLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import cPickle as pickle
from collections import OrderedDict

import os
import sys
import glob
import random
import time
from tqdm import tqdm
import cv2

import nn


def sgd(loss_or_grads, params, learning_rate, num_ft_params):
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

        self.net = None

        self.D_trainFunction = None
        self.G_trainFunction = None
        self.predictFunction = None
        self.meanImage_VGG_ImageNet = None

        # LearningRate, Momentum scheduling
        self.numBatchesPerEpoch = batchesPerEpoch
        self.currBatchNum = 0

    def buildInputNetwork_VGG_ImageNet(self, input_layer, input_var=None):
        net = {}

        net['input'] = input_layer
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
        net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape, trainable=False)
        net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv4_1'].output_shape[1:])

        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape, trainable=False)
        net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv4_2'].output_shape[1:])

        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'].add_param(net['conv4_3'].W, net['conv4_3'].W.get_value().shape, trainable=False)
        net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv4_3'].output_shape[1:])

        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        print "Input: {}".format(net['pool4'].output_shape[1:])

        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape, trainable=False)
        net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv5_1'].output_shape[1:])

        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape, trainable=False)
        net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv5_2'].output_shape[1:])

        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape, trainable=False)
        net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape, trainable=False)
        print "Input: {}".format(net['conv5_3'].output_shape[1:])

        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        print "Input: {}".format(net['pool5'].output_shape[1:])

        # Set out weights
        d = pickle.load(open('/scratch/local/jpang/vgg16.pkl'))
        numElementsToSet = 26  # Number of W and b elements for the first convolutional layers
        lasagne.layers.set_all_param_values(net['pool5'], d['param values'][:numElementsToSet])

        self.meanImage_VGG_ImageNet = d['mean value'][:, np.newaxis, np.newaxis]

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

        inputNetwork['uconv2_2'] = ConvLayer(inputNetwork['upool2'], 128, 3, pad=1, nonlinearity=nn.relu)
        print "uconv2_2: {}".format(inputNetwork['uconv2_2'].output_shape[1:])

        inputNetwork['uconv2_1'] = ConvLayer(inputNetwork['uconv2_2'], 128, 3, pad=1, nonlinearity=nn.relu)
        print "uconv2_1: {}".format(inputNetwork['uconv2_1'].output_shape[1:])

        inputNetwork['upool1'] = Upscale2DLayer(inputNetwork['uconv2_1'], scale_factor=2)
        # inputNetwork['upool1'] = TransposedConv2DLayer(inputNetwork['uconv2_1'], 64, 2, stride=2)
        # inputNetwork['upool1'] = nn.Deconv2DLayer(inputNetwork['uconv2_1'], (None,64,192,256), (5,5), W=Normal(0.05),nonlinearity=nn.relu)
        print "upool1: {}".format(inputNetwork['upool1'].output_shape[1:])

        inputNetwork['uconv1_2'] = ConvLayer(inputNetwork['upool1'], 64, 3, pad=1, nonlinearity=nn.relu)
        print "uconv1_2: {}".format(inputNetwork['uconv1_2'].output_shape[1:])

        inputNetwork['uconv1_1'] = ConvLayer(inputNetwork['uconv1_2'], 64, 3, pad=1, nonlinearity=nn.relu)
        print "uconv1_1: {}".format(inputNetwork['uconv1_1'].output_shape[1:])

        inputNetwork['output'] = ConvLayer(inputNetwork['uconv1_1'], 1, 1, pad=0, nonlinearity=lasagne.nonlinearities.sigmoid)
        print "output: {}".format(inputNetwork['output'].output_shape[1:])

        return inputNetwork

    def build_discrim(self, input_stage):
        # Discriminator 1
        net1 = {}
        outputSaliency = T.tensor4()

        # input_var = lasagne.layers.get_output(G_out)

        net1['input'] = InputLayer((None, 1, self.inputHeight, self.inputWidth), outputSaliency)
        print "Input: {}".format(net1['input'].output_shape[1:])

        net1['conv1'] = ConvLayer(net1['input'], 32, 3, pad=1)
        print "Input: {}".format(net1['conv1'].output_shape[1:])

        net1['pool1'] = PoolLayer(net1['conv1'], 4)
        print "Input: {}".format(net1['pool1'].output_shape[1:])

        net1['conv2'] = ConvLayer(net1['pool1'], 64, 3, pad=1)
        print "Input: {}".format(net1['conv2'].output_shape[1:])

        net1['pool2'] = PoolLayer(net1['conv2'], 2)
        print "Input: {}".format(net1['pool2'].output_shape[1:])

        net1['conv3'] = ConvLayer(net1['pool2'], 128, 3, pad=1)
        print "Input: {}".format(net1['conv3'].output_shape[1:])

        net1['pool3'] = PoolLayer(net1['conv3'], 2)
        print "Input: {}".format(net1['pool3'].output_shape[1:])

        net1['fc4'] = DenseLayer(net1['pool3'], num_units=30, nonlinearity=lasagne.nonlinearities.tanh)

        net1['fc5'] = DenseLayer(net1['fc4'], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        print "Input: {}".format(net1['fc5'].output_shape[1:])

        return net1

    def build(self, input_var, output_var):
        input_layer = InputLayer((None, 3, self.inputHeight, self.inputWidth), input_var=input_var);

        vggNet = self.buildInputNetwork_VGG_ImageNet(input_layer, input_var)

        input_stage = vggNet['conv5_3']

        self.net = self.buildOutputNetwork(input_stage)

        outputLayerName = 'output'

        prediction = lasagne.layers.get_output(self.net[outputLayerName])

        test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)

        self.predictFunction = theano.function([input_var], test_prediction)

        disc = self.build_discrim(prediction)

        disc_lab = lasagne.layers.get_output(disc['fc5'], output_var)
        disc_gen = lasagne.layers.get_output(disc['fc5'], prediction)

        # loss = lasagne.objectives.squared_error(prediction, output_var)
        loss = lasagne.objectives.binary_crossentropy(prediction, output_var)
        train_err = loss.mean()

        l_unl = disc_lab
        l_gen = disc_gen

        # D_obj_real = 0.5 * T.mean(T.nnet.softplus(l_unl)) - 0.5 * T.mean(T.nnet.softplus(l_gen))
        # G_obj_d = T.mean(T.nnet.softplus(l_gen))

        D_obj_real = lasagne.objectives.binary_crossentropy(disc_lab, T.ones(disc_lab.shape)).mean()
        D_obj_gen = lasagne.objectives.binary_crossentropy(disc_gen, T.zeros(disc_lab.shape)).mean()
        D_obj_real = 0.5*D_obj_real + 0.5*D_obj_gen
        G_obj_d = lasagne.objectives.binary_crossentropy(disc_gen, T.ones(disc_lab.shape)).mean()

        D_obj = D_obj_real  # + D_obj_gen
        G_obj = G_obj_d + train_err * 10

        cost = [G_obj, D_obj, train_err]
        # cost = train_err

        # parameters update and training
        # G = self.net[outputLayerName]
        G_params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)
        G_lr = theano.shared(np.array(0.003, dtype=theano.config.floatX))
        G_updates = lasagne.updates.nesterov_momentum(G_obj, G_params, learning_rate=G_lr, momentum=0.5)

        # G_train = theano.function([input_var, output_var], G_obj, updates=G_updates, allow_input_downcast=True)
        G_train = theano.function(inputs=[input_var, output_var], outputs=cost, updates=G_updates,
                                  allow_input_downcast=True)

        D_params = lasagne.layers.get_all_params(disc['fc5'], trainable=True)
        D_lr = theano.shared(np.array(0.0003, dtype=theano.config.floatX))
        D_updates = lasagne.updates.nesterov_momentum(D_obj, D_params, learning_rate=D_lr, momentum=0.5)

        # D_train = theano.function([input_var, output_var], D_obj, updates=D_updates, allow_input_downcast=True)
        D_train = theano.function([input_var, output_var], cost, updates=D_updates, allow_input_downcast=True)

        self.D_trainFunction = D_train
        self.G_trainFunction = G_train


def predict(model, validationData, numEpoch, dir='test'):

    width = 256
    height = 192

    blob = np.zeros((1, 3, height, width), theano.config.floatX)
    # imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])

    blob[0, ...] = (validationData.image.data.astype(theano.config.floatX).transpose(2, 0, 1))  # - imageMean) / 255.
    result = np.squeeze(model.predictFunction(blob))

    saliencyMap = (result * 255).astype(np.uint8)
    cv2.imwrite('./'+dir+'/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch), saliencyMap)

    # cv2.imwrite('./results/validationRandomImage_'+str(numEpoch)+'.png',
    #            cv2.cvtColor(validationData.image.data, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./results/validationRandomSaliencyGT_'+str(numEpoch)+'.png', validationData.saliency.data)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def test():

    # Create network

    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    model = Model(1000)
    model.build(inputImage, outputSaliency)

    print 'Loading validation data...'
    with open('valSample.pkl', 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'

    with np.load(
            "/home/users/jpang/scratch-local/lsun2016/saliency-2016-lsun/test/modelWeights{:04d}.npz".format(
                150)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

    [predict(model=model, validationData=validationData[currEpoch], numEpoch=currEpoch, dir='results') for currEpoch in range(10)]


def train():
    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    model = Model(1000)
    model.build(inputImage, outputSaliency)

    batchSize = 32
    numEpochs = 181

    batchIn = np.zeros((batchSize, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    batchOut = np.zeros((batchSize, 1, model.inputHeight, model.inputWidth), theano.config.floatX)
    # batchOut = np.zeros((batchSize, 1, 48, 64), theano.config.floatX)
    # Load data

    print 'Loading training data...'
    with open('/home/users/jpang/scratch-local/salicon_data/trainData.pickle', 'rb') as f:
        trainData = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open('valSample.pkl', 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'

    nr_batches_train = int(10000 / batchSize)

    # trainData = trainData[0:3000]

    # validationData = validationData[0:20]
    # pickle.dump(validationData, open("valSample.pkl", 'w'))
    # pickle.dump(trainData, open("trainSample.pkl", 'w'))

    width = 256
    height = 192

    # imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])
    # blob = np.zeros((1, 3, height, width), theano.config.floatX)

    numRandom = random.choice(range(len(validationData)))

    cv2.imwrite('./test/validationRandomImage.png', cv2.cvtColor(validationData[numRandom].image.data,
                                                                    cv2.COLOR_RGB2BGR))
    cv2.imwrite('./test/validationRandomSaliencyGT.png', validationData[numRandom].saliency.data)
    # blob[0, ...] = (validationData[numRandom].image.data.astype(theano.config.floatX).transpose(2, 0, 1)
    #                - imageMean) / 255.
    # result = np.squeeze(model.predictFunction(blob))

    # saliencyMap = (result * 255).astype(np.uint8)
    # cv2.imwrite('./test/validationRandomSaliencyPred.png', saliencyMap)

    n_updates = 1

    for currEpoch in tqdm(range(numEpochs),ncols=20):

        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(trainData)

        for currChunk in chunks(trainData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range(batchSize):
                batchIn[k, ...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2, 0, 1))
                                   # - model.meanImage_VGG_ImageNet) / 255.
                batchOut[k, ...] = (currChunk[k].saliency.data.astype(theano.config.floatX)) / 255.

            if n_updates % 2 == 1:
                G_obj, D_obj, G_cost = model.G_trainFunction(batchIn, batchOut)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batchIn, batchOut)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost

        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train

        print 'Epoch:', currEpoch, ' ->', (g_cost, d_cost, e_cost)

        n_updates += 1

        if currEpoch % 3 == 0:
            np.savez("./test/modelWeights{:04d}.npz".format(currEpoch),
                    *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, validationData=validationData[numRandom], numEpoch=currEpoch)

if __name__ == "__main__":
    test()