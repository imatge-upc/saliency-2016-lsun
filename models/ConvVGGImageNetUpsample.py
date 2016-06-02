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

class ConvVGGImageNetUpsample( ModelInput ):

	def __init__( self, iInputWidht, iInputHeight ):
		super(ConvVGGImageNetUpsample, self).__init__( 'ConvVGGImageNetUpsample', iInputWidht, iInputHeight, None, 1.0 )
		
		
	def build( self, sharedNet, fileName, input_var = None ):
			
		self.inputLayerName = self.netName+'input'
		sharedNet[self.netName+'input'] = InputLayer((None, 3, self.inputHeight, self.inputWidth ), input_var=input_var)
		print "Input: {}".format(sharedNet[self.netName+'input'].output_shape[1:])
		
		sharedNet[self.netName+'conv1_1'] = ConvLayer(sharedNet[self.netName+'input'], 64, 3, pad=1)
		sharedNet[self.netName+'conv1_1'].add_param(sharedNet[self.netName+'conv1_1'].W, sharedNet[self.netName+'conv1_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv1_1'].add_param(sharedNet[self.netName+'conv1_1'].b, sharedNet[self.netName+'conv1_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv1_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv1_2'] = ConvLayer(sharedNet[self.netName+'conv1_1'], 64, 3, pad=1)
		sharedNet[self.netName+'conv1_2'].add_param(sharedNet[self.netName+'conv1_2'].W, sharedNet[self.netName+'conv1_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv1_2'].add_param(sharedNet[self.netName+'conv1_2'].b, sharedNet[self.netName+'conv1_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv1_2'].output_shape[1:])
		
		sharedNet[self.netName+'pool1'] = PoolLayer(sharedNet[self.netName+'conv1_2'], 2)
		print "Input: {}".format(sharedNet[self.netName+'pool1'].output_shape[1:])
		
		sharedNet[self.netName+'conv2_1'] = ConvLayer(sharedNet[self.netName+'pool1'], 128, 3, pad=1)
		sharedNet[self.netName+'conv2_1'].add_param(sharedNet[self.netName+'conv2_1'].W, sharedNet[self.netName+'conv2_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv2_1'].add_param(sharedNet[self.netName+'conv2_1'].b, sharedNet[self.netName+'conv2_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv2_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv2_2'] = ConvLayer(sharedNet[self.netName+'conv2_1'], 128, 3, pad=1)
		sharedNet[self.netName+'conv2_2'].add_param(sharedNet[self.netName+'conv2_2'].W, sharedNet[self.netName+'conv2_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv2_2'].add_param(sharedNet[self.netName+'conv2_2'].b, sharedNet[self.netName+'conv2_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv2_2'].output_shape[1:])
		
		sharedNet[self.netName+'pool2'] = PoolLayer(sharedNet[self.netName+'conv2_2'], 2)
		print "Input: {}".format(sharedNet[self.netName+'pool2'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_1'] = ConvLayer(sharedNet[self.netName+'pool2'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_1'].add_param(sharedNet[self.netName+'conv3_1'].W, sharedNet[self.netName+'conv3_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_1'].add_param(sharedNet[self.netName+'conv3_1'].b, sharedNet[self.netName+'conv3_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv3_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_2'] = ConvLayer(sharedNet[self.netName+'conv3_1'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_2'].add_param(sharedNet[self.netName+'conv3_2'].W, sharedNet[self.netName+'conv3_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_2'].add_param(sharedNet[self.netName+'conv3_2'].b, sharedNet[self.netName+'conv3_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv3_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv3_3'] = ConvLayer(sharedNet[self.netName+'conv3_2'], 256, 3, pad=1)
		sharedNet[self.netName+'conv3_3'].add_param(sharedNet[self.netName+'conv3_3'].W, sharedNet[self.netName+'conv3_3'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv3_3'].add_param(sharedNet[self.netName+'conv3_3'].b, sharedNet[self.netName+'conv3_3'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv3_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool3'] = PoolLayer(sharedNet[self.netName+'conv3_3'], 2)
		print "Input: {}".format(sharedNet[self.netName+'pool3'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_1'] = ConvLayer(sharedNet[self.netName+'pool3'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_1'].add_param(sharedNet[self.netName+'conv4_1'].W, sharedNet[self.netName+'conv4_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_1'].add_param(sharedNet[self.netName+'conv4_1'].b, sharedNet[self.netName+'conv4_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv4_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_2'] = ConvLayer(sharedNet[self.netName+'conv4_1'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_2'].add_param(sharedNet[self.netName+'conv4_2'].W, sharedNet[self.netName+'conv4_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_2'].add_param(sharedNet[self.netName+'conv4_2'].b, sharedNet[self.netName+'conv4_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv4_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv4_3'] = ConvLayer(sharedNet[self.netName+'conv4_2'], 512, 3, pad=1)
		sharedNet[self.netName+'conv4_3'].add_param(sharedNet[self.netName+'conv4_3'].W, sharedNet[self.netName+'conv3_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv4_3'].add_param(sharedNet[self.netName+'conv4_3'].b, sharedNet[self.netName+'conv4_3'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv4_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool4'] = PoolLayer(sharedNet[self.netName+'conv4_3'], 2)
		print "Input: {}".format(sharedNet[self.netName+'pool4'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_1'] = ConvLayer(sharedNet[self.netName+'pool4'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_1'].add_param(sharedNet[self.netName+'conv5_1'].W, sharedNet[self.netName+'conv5_1'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_1'].add_param(sharedNet[self.netName+'conv5_1'].b, sharedNet[self.netName+'conv5_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv5_1'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_2'] = ConvLayer(sharedNet[self.netName+'conv5_1'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_2'].add_param(sharedNet[self.netName+'conv5_2'].W, sharedNet[self.netName+'conv5_2'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_2'].add_param(sharedNet[self.netName+'conv5_2'].b, sharedNet[self.netName+'conv5_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv5_2'].output_shape[1:])
		
		sharedNet[self.netName+'conv5_3'] = ConvLayer(sharedNet[self.netName+'conv5_2'], 512, 3, pad=1)
		sharedNet[self.netName+'conv5_3'].add_param(sharedNet[self.netName+'conv5_3'].W, sharedNet[self.netName+'conv5_3'].W.get_value().shape, trainable=False)
		sharedNet[self.netName+'conv5_3'].add_param(sharedNet[self.netName+'conv5_3'].b, sharedNet[self.netName+'conv5_3'].b.get_value().shape, trainable=False)
		print "Input: {}".format(sharedNet[self.netName+'conv5_3'].output_shape[1:])
		
		sharedNet[self.netName+'pool5'] = PoolLayer(sharedNet[self.netName+'conv5_3'], 2)
		print "Input: {}".format(sharedNet[self.netName+'pool5'].output_shape[1:])
		
		sharedNet[self.netName+'output'] = sharedNet[self.netName+'pool5']
		print "Input: {}".format(sharedNet[self.netName+'output'].output_shape[1:])
		
		self.outputLayerName = self.netName+'output'
		
		self.load( fileName, sharedNet )
		
		sharedNet[ self.netName + 'output1' ] = Upscale2DLayer( sharedNet[self.netName+'conv5_3'], scale_factor=4 )
		sharedNet[ self.netName + 'output2' ] = Upscale2DLayer( sharedNet[self.netName+'conv4_3'], scale_factor=2 )
		
		sharedNet[self.netName+'output'] = lasagne.layers.ConcatLayer([ sharedNet[ self.netName + 'output1' ],\
																	    sharedNet[ self.netName + 'output2' ]], axis=1)
		
		print "Input: {}".format(sharedNet[self.netName+'output'].output_shape[1:])
		
		
		
	def load( self, fileName, sharedNet ):
		
		with open( fileName, 'rb' ) as f:
			data = pickle.load( f )
		
		lasagne.layers.set_all_param_values(sharedNet[self.outputLayerName], data['params'] )
		
		self.meanImage = data['meanImg']
		self.scaleFactor = data['scaleFactor']
		
	def prepareImage( self, img ):
	
		return img.astype( theano.config.floatX ).transpose( 2,0,1 ) - self.meanImage