from Model import ModelInput
import theano
import theano.tensor as T
import lasagne
import numpy as np
import cPickle as pickle

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

import cPickle as pickle

class ConvVGGImageNet( ModelInput ):

	def __init__( self, iInputWidht, iInputHeight ):
		super(ConvVGGImageNet, self).__init__( 'ConvVGGImageNet', iInputWidht, iInputHeight, None, 1.0 )
		
		
	def build( self, sharedNet, input_var = None ):
			
		self.inputLayerName = self.netName+'input'
		sharedNet[self.netName+'input'] = InputLayer((None, 3, self.inputHeight, self.inputWidth ), input_var=input_var)
		print "Input: {}".format(sharedNet[self.netName+'input'].output_shape[1:])
		
		sharedNet[self.netName+'conv1_1'] = ConvLayer(sharedNet[self.netName+'input'], 64, 3, pad=1)
		sharedNet[self.netName+'conv1_1'].add_param(sharedNet[self.netName+'conv1_1'].W, sharedNet[self.netName+'conv1_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv1_1'].add_param(sharedNet[self.netName+'conv1_1'].b, sharedNet[self.netName+'conv1_1'].b.get_value().shape, trainable=False)
		print "conv1_1: {}".format(sharedNet[self.netName+'conv1_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv1_2'] = ConvLayer(sharedNet[self.netName+'conv1_1'], 64, 3, pad=1)
		sharedNet[self.netName+'conv1_2'].add_param(sharedNet[self.netName+'conv1_2'].W, sharedNet[self.netName+'conv1_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv1_2'].add_param(sharedNet[self.netName+'conv1_2'].b, sharedNet[self.netName+'conv1_2'].b.get_value().shape, trainable=False)
		print "conv1_2: {}".format(sharedNet[self.netName+'conv1_2'].output_shape[1:])
		
		sharedNet[self.netName+'pool1'] = PoolLayer(sharedNet[self.netName+'conv1_2'], 2)
		print "pool1: {}".format(sharedNet[self.netName+'pool1'].output_shape[1:])
		
		sharedNet[self.netName+'conv2_1'] = ConvLayer(sharedNet[self.netName+'pool1'], 128, 3, pad=1)
		sharedNet[self.netName+'conv2_1'].add_param(sharedNet[self.netName+'conv2_1'].W, sharedNet[self.netName+'conv2_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv2_1'].add_param(sharedNet[self.netName+'conv2_1'].b, sharedNet[self.netName+'conv2_1'].b.get_value().shape, trainable=False)
		print "conv2_1: {}".format(sharedNet[self.netName+'conv2_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv2_2'] = ConvLayer(sharedNet[self.netName+'conv2_1'], 128, 3, pad=1)
		sharedNet[self.netName+'conv2_2'].add_param(sharedNet[self.netName+'conv2_2'].W, sharedNet[self.netName+'conv2_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv2_2'].add_param(sharedNet[self.netName+'conv2_2'].b, sharedNet[self.netName+'conv2_2'].b.get_value().shape, trainable=False)
		print "conv2_2: {}".format(sharedNet[self.netName+'conv2_2'].output_shape[1:])
		
		sharedNet[self.netName+'pool2'] = PoolLayer(sharedNet[self.netName+'conv2_2'], 2)
		print "pool2: {}".format(sharedNet[self.netName+'pool2'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_1'] = ConvLayer(sharedNet[self.netName+'pool2'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_1'].add_param(sharedNet[self.netName+'conv3_1'].W, sharedNet[self.netName+'conv3_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_1'].add_param(sharedNet[self.netName+'conv3_1'].b, sharedNet[self.netName+'conv3_1'].b.get_value().shape, trainable=False)
		print "conv3_1: {}".format(sharedNet[self.netName+'conv3_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_2'] = ConvLayer(sharedNet[self.netName+'conv3_1'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_2'].add_param(sharedNet[self.netName+'conv3_2'].W, sharedNet[self.netName+'conv3_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_2'].add_param(sharedNet[self.netName+'conv3_2'].b, sharedNet[self.netName+'conv3_2'].b.get_value().shape, trainable=False)
		print "conv3_2: {}".format(sharedNet[self.netName+'conv3_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_3'] = ConvLayer(sharedNet[self.netName+'conv3_2'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_3'].add_param(sharedNet[self.netName+'conv3_3'].W, sharedNet[self.netName+'conv3_3'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_3'].add_param(sharedNet[self.netName+'conv3_3'].b, sharedNet[self.netName+'conv3_3'].b.get_value().shape, trainable=False)
		print "conv3_3: {}".format(sharedNet[self.netName+'conv3_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool3'] = PoolLayer(sharedNet[self.netName+'conv3_3'], 2)
		print "pool3: {}".format(sharedNet[self.netName+'pool3'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_1'] = ConvLayer(sharedNet[self.netName+'pool3'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_1'].add_param(sharedNet[self.netName+'conv4_1'].W, sharedNet[self.netName+'conv4_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_1'].add_param(sharedNet[self.netName+'conv4_1'].b, sharedNet[self.netName+'conv4_1'].b.get_value().shape, trainable=False)
		print "conv4_1: {}".format(sharedNet[self.netName+'conv4_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_2'] = ConvLayer(sharedNet[self.netName+'conv4_1'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_2'].add_param(sharedNet[self.netName+'conv4_2'].W, sharedNet[self.netName+'conv4_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_2'].add_param(sharedNet[self.netName+'conv4_2'].b, sharedNet[self.netName+'conv4_2'].b.get_value().shape, trainable=False)
		print "conv4_2: {}".format(sharedNet[self.netName+'conv4_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_3'] = ConvLayer(sharedNet[self.netName+'conv4_2'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_3'].add_param(sharedNet[self.netName+'conv4_3'].W, sharedNet[self.netName+'conv3_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_3'].add_param(sharedNet[self.netName+'conv4_3'].b, sharedNet[self.netName+'conv4_3'].b.get_value().shape, trainable=False)
		print "conv4_3: {}".format(sharedNet[self.netName+'conv4_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool4'] = PoolLayer(sharedNet[self.netName+'conv4_3'], 2)
		print "pool4: {}".format(sharedNet[self.netName+'pool4'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_1'] = ConvLayer(sharedNet[self.netName+'pool4'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_1'].add_param(sharedNet[self.netName+'conv5_1'].W, sharedNet[self.netName+'conv5_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_1'].add_param(sharedNet[self.netName+'conv5_1'].b, sharedNet[self.netName+'conv5_1'].b.get_value().shape, trainable=False)
		print "conv5_1: {}".format(sharedNet[self.netName+'conv5_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_2'] = ConvLayer(sharedNet[self.netName+'conv5_1'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_2'].add_param(sharedNet[self.netName+'conv5_2'].W, sharedNet[self.netName+'conv5_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_2'].add_param(sharedNet[self.netName+'conv5_2'].b, sharedNet[self.netName+'conv5_2'].b.get_value().shape, trainable=False)
		print "conv5_2: {}".format(sharedNet[self.netName+'conv5_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_3'] = ConvLayer(sharedNet[self.netName+'conv5_2'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_3'].add_param(sharedNet[self.netName+'conv5_3'].W, sharedNet[self.netName+'conv5_3'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_3'].add_param(sharedNet[self.netName+'conv5_3'].b, sharedNet[self.netName+'conv5_3'].b.get_value().shape, trainable=False)
		print "conv5_3: {}".format(sharedNet[self.netName+'conv5_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool5'] = PoolLayer(sharedNet[self.netName+'conv5_3'], 2)
		print "pool5: {}".format(sharedNet[self.netName+'pool5'].output_shape[1:])
		
		sharedNet[self.netName+'output'] = sharedNet[self.netName+'pool5']
		print "output: {}".format(sharedNet[self.netName+'output'].output_shape[1:])
		
		self.outputLayerName = self.netName+'output'
		
	def load( self, fileName, sharedNet ):
		
		with open( fileName, 'rb' ) as f:
			data = pickle.load( f )
		
		lasagne.layers.set_all_param_values(sharedNet[self.netName+'output'], data['params'] )
		
		self.meanImage = data['meanImg']
		self.scaleFactor = data['scaleFactor']
		
	def prepareImage( self, img ):
	
		return img.astype( theano.config.floatX ).transpose( 2,0,1 ) - self.meanImage

	# def buildOutputNetwork( self, inputNetwork ):

		# inputNetwork['upool5'] = Upscale2DLayer(inputNetwork['pool5'], scale_factor=2)
		# print "upool5: {}".format(inputNetwork['upool5'].output_shape[1:])

		# inputNetwork['uconv5_3'] = ConvLayer(inputNetwork['upool5'], 512, 3, pad=1)
		# print "uconv5_3: {}".format(inputNetwork['uconv5_3'].output_shape[1:])

		# inputNetwork['uconv5_2'] = ConvLayer(inputNetwork['uconv5_3'], 512, 3, pad=1)
		# print "uconv5_2: {}".format(inputNetwork['uconv5_2'].output_shape[1:])

		# inputNetwork['uconv5_1'] = ConvLayer(inputNetwork['uconv5_2'], 512, 3, pad=1)
		# print "uconv5_1: {}".format(inputNetwork['uconv5_1'].output_shape[1:])

		# inputNetwork['upool4'] = Upscale2DLayer(inputNetwork['uconv5_1'], scale_factor=2)
		# print "upool4: {}".format(inputNetwork['upool4'].output_shape[1:])

		# inputNetwork['uconv4_3'] = ConvLayer(inputNetwork['upool4'], 512, 3, pad=1)
		# print "uconv4_3: {}".format(inputNetwork['uconv4_3'].output_shape[1:])

		# inputNetwork['uconv4_2'] = ConvLayer(inputNetwork['uconv4_3'], 512, 3, pad=1)
		# print "uconv4_2: {}".format(inputNetwork['uconv4_2'].output_shape[1:])

		# inputNetwork['uconv4_1'] = ConvLayer(inputNetwork['uconv4_2'], 512, 3, pad=1)
		# print "uconv4_1: {}".format(inputNetwork['uconv4_1'].output_shape[1:])

		# inputNetwork['upool3'] = Upscale2DLayer(inputNetwork['uconv4_1'], scale_factor=2)
		# print "upool3: {}".format(inputNetwork['upool3'].output_shape[1:])

		# inputNetwork['uconv3_3'] = ConvLayer(inputNetwork['upool3'], 256, 3, pad=1)
		# print "uconv3_3: {}".format(inputNetwork['uconv3_3'].output_shape[1:])

		# inputNetwork['uconv3_2'] = ConvLayer(inputNetwork['uconv3_3'], 256, 3, pad=1)
		# print "uconv3_2: {}".format(inputNetwork['uconv3_2'].output_shape[1:])

		# inputNetwork['uconv3_1'] = ConvLayer(inputNetwork['uconv3_2'], 256, 3, pad=1)
		# print "uconv3_1: {}".format(inputNetwork['uconv3_1'].output_shape[1:])

		# inputNetwork['upool2'] = Upscale2DLayer(inputNetwork['uconv3_1'], scale_factor=2)
		# print "upool2: {}".format(inputNetwork['upool2'].output_shape[1:])

		# inputNetwork['uconv2_2'] = ConvLayer(inputNetwork['upool2'], 128, 3, pad=1)
		# print "uconv2_2: {}".format(inputNetwork['uconv2_2'].output_shape[1:])

		# inputNetwork['uconv2_1'] = ConvLayer(inputNetwork['uconv2_2'], 128, 3, pad=1)
		# print "uconv2_1: {}".format(inputNetwork['uconv2_1'].output_shape[1:])

		# inputNetwork['upool1'] = Upscale2DLayer(inputNetwork['uconv2_1'], scale_factor=2)
		# print "upool1: {}".format(inputNetwork['upool1'].output_shape[1:])

		# inputNetwork['uconv1_2'] = ConvLayer(inputNetwork['upool1'], 64, 3, pad=1)
		# print "uconv1_2: {}".format(inputNetwork['uconv1_2'].output_shape[1:])

		# inputNetwork['uconv1_1'] = ConvLayer(inputNetwork['uconv1_2'], 64, 3, pad=1)
		# print "uconv1_1: {}".format(inputNetwork['uconv1_1'].output_shape[1:])

		# inputNetwork['output'] = ConvLayer(inputNetwork['uconv1_1'], 1, 1, pad=0)
		# print "output: {}".format(inputNetwork['output'].output_shape[1:])
		
		# return inputNetwork
		
	# def build( self, input_var, output_var ):

		# inputNet = self.buildInputNetwork_VGG_ImageNet( self.inputHeight, self.inputWidth, input_var )
		# self.net = self.buildOutputNetwork( inputNet )
		
		# outputLayerName = 'output'
		
		# prediction = lasagne.layers.get_output(self.net[outputLayerName])
		
		# test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)
		
		# loss = lasagne.objectives.squared_error( prediction, output_var )
		# loss = loss.mean()
		
		# params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)

		# updates_sgd = lasagne.updates.sgd(loss, params, learning_rate = self.currentLearningRate )
		# updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum = self.currentMomentum ) 

		# self.trainFunction = theano.function([input_var, output_var], loss, updates=updates, allow_input_downcast=True)
		# self.predictFunction = theano.function([input_var], test_prediction)
		