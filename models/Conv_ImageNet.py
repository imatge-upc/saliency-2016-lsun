import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,InverseLayer
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

class Model:

	def __init__( self, batchesPerEpoch ):
		self.inputWidth = 256
		self.inputHeight = 192

		self.initialLearningRate = 0.01
		self.initialMomentum = 0.0 # start momentum at 0.0
		self.maxMomentum = 0.9
		self.minLearningRate = 0.00001
		
		self.currentLearningRate = theano.shared( np.array( self.initialLearningRate, dtype = theano.config.floatX ) )
		self.currentMomentum = theano.shared( np.array( self.initialMomentum, dtype = theano.config.floatX ) )
		
		self.net = None
		
		self.trainFunction = None
		self.predictFunction = None
		self.meanImage_VGG_ImageNet = None
		
		# LearningRate, Momentum scheduling
		self.numBatchesPerEpoch = batchesPerEpoch
		self.currBatchNum = 0
			
	#def buildInputNetwork_VGG_ImageNet( self, inputHeight, inputWidth, input_var=None ):
	def buildInputNetwork_VGG_ImageNet( self, input_layer, input_var=None ):
		net = {}
		#net['input'] = InputLayer((None, 3, inputHeight, inputWidth ), input_var=input_var)
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
		net['conv4_1'].add_param(net['conv4_1'].W, net['conv4_1'].W.get_value().shape, trainable=False)
		net['conv4_1'].add_param(net['conv4_1'].b, net['conv4_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv4_1'].output_shape[1:])
		
		net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
		net['conv4_2'].add_param(net['conv4_2'].W, net['conv4_2'].W.get_value().shape, trainable=False)
		net['conv4_2'].add_param(net['conv4_2'].b, net['conv4_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv4_2'].output_shape[1:])
		
		net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
		net['conv4_3'].add_param(net['conv4_3'].W, net['conv3_1'].W.get_value().shape, trainable=False)
		net['conv4_3'].add_param(net['conv4_3'].b, net['conv4_3'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv4_3'].output_shape[1:])
		
		net['pool4'] = PoolLayer(net['conv4_3'], 2)
		print "Input: {}".format(net['pool4'].output_shape[1:])
		
		net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
		net['conv5_1'].add_param(net['conv5_1'].W, net['conv5_1'].W.get_value().shape, trainable=False)
		net['conv5_1'].add_param(net['conv5_1'].b, net['conv5_1'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv5_1'].output_shape[1:])
		
		net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
		net['conv5_2'].add_param(net['conv5_2'].W, net['conv5_2'].W.get_value().shape, trainable=False)
		net['conv5_2'].add_param(net['conv5_2'].b, net['conv5_2'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv5_2'].output_shape[1:])
		
		net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
		net['conv5_3'].add_param(net['conv5_3'].W, net['conv5_3'].W.get_value().shape, trainable=False)
		net['conv5_3'].add_param(net['conv5_3'].b, net['conv5_3'].b.get_value().shape, trainable=False)
		print "Input: {}".format(net['conv5_3'].output_shape[1:])
		
		net['pool5'] = PoolLayer(net['conv5_3'], 2)
		print "Input: {}".format(net['pool5'].output_shape[1:])
		
		# Set out weights 
		d = pickle.load(open('/imatge/jpan/lsun2016/saliency-2016-lsun/vgg16.pkl'))
		numElementsToSet = 26 # Number of W and b elements for the first convolutional layers
		lasagne.layers.set_all_param_values(net['pool5'], d['param values'][:numElementsToSet])
		
		self.meanImage_VGG_ImageNet = d['mean value'][:, np.newaxis, np.newaxis]
		
		return net

	def buildInputNetwork_AlexNet_Places( self, input_layer, input_var=None ):
		net = {}

		net['input'] = input_layer
		print "Input: {}".format(net['input'].output_shape[1:])
		# conv1
		net['conv1'] = Conv2DLayer( net['input'],num_filters=96,filter_size=(11, 11),stride = 4,nonlinearity=rectify)
		net['conv1'].add_param(net['conv1'].W, net['conv1'].W.get_value().shape, trainable=False)
		net['conv1'].add_param(net['conv1'].b, net['conv1'].b.get_value().shape, trainable=False)
		print "conv1: {}".format(net['conv1'].output_shape[1:])

		# pool1
		net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)
		print "pool1: {}".format(net['pool1'].output_shape[1:])
		# norm1
		net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],n=5,alpha=0.0001/5.0,beta = 0.75,k=1)
		print "norm1: {}".format(net['norm1'].output_shape[1:])

		# before conv2 split the data
		net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
		net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)
		# now do the convolutions
		net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],num_filters=128,filter_size=(5, 5),pad = 2)
		net['conv2_part1'].add_param(net['conv2_part1'].W, net['conv2_part1'].W.get_value().shape, trainable=False)
		net['conv2_part1'].add_param(net['conv2_part1'].b, net['conv2_part1'].b.get_value().shape, trainable=False)

		net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],num_filters=128,filter_size=(5, 5),pad = 2)
		net['conv2_part2'].add_param(net['conv2_part2'].W, net['conv2_part2'].W.get_value().shape, trainable=False)
		net['conv2_part2'].add_param(net['conv2_part2'].b, net['conv2_part2'].b.get_value().shape, trainable=False)
		# now combine
		net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
		print "conv2: {}".format(net['conv2'].output_shape[1:]) 

		# pool2
		net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)
		print "pool2: {}".format(net['pool2'].output_shape[1:]) 
		# norm2
		net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],n=5,alpha=0.0001/5.0,beta = 0.75,k=1)

		# conv3
		# no group
		net['conv3'] = Conv2DLayer(net['norm2'],num_filters=384,filter_size=(3, 3),pad = 1)
		net['conv3'].add_param(net['conv3'].W, net['conv3'].W.get_value().shape, trainable=False)
		net['conv3'].add_param(net['conv3'].b, net['conv3'].b.get_value().shape, trainable=False)
		print "conv3: {}".format(net['conv3'].output_shape[1:]) 

		# conv4
		# group = 2
		net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
		net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)

		net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],num_filters=192,filter_size=(3, 3),pad = 1)
		net['conv4_part1'].add_param(net['conv4_part1'].W, net['conv4_part1'].W.get_value().shape, trainable=False)
		net['conv4_part1'].add_param(net['conv4_part1'].b, net['conv4_part1'].b.get_value().shape, trainable=False)

		net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],num_filters=192,filter_size=(3, 3),pad = 1)
		net['conv4_part2'].add_param(net['conv4_part2'].W, net['conv4_part2'].W.get_value().shape, trainable=False)
		net['conv4_part2'].add_param(net['conv4_part2'].b, net['conv4_part2'].b.get_value().shape, trainable=False)

		net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)
		print "conv4: {}".format(net['conv4'].output_shape[1:]) 

		# conv5
		# group 2
		net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
		net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)

		net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],num_filters=128,filter_size=(3, 3),pad = 1)
		net['conv5_part1'].add_param(net['conv5_part1'].W, net['conv5_part1'].W.get_value().shape, trainable=False)
		net['conv5_part1'].add_param(net['conv5_part1'].b, net['conv5_part1'].b.get_value().shape, trainable=False)

		net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],num_filters=128,filter_size=(3, 3),pad = 1)
		net['conv5_part2'].add_param(net['conv5_part2'].W, net['conv5_part2'].W.get_value().shape, trainable=False)
		net['conv5_part2'].add_param(net['conv5_part2'].b, net['conv5_part2'].b.get_value().shape, trainable=False)

		net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)
		print "conv5: {}".format(net['conv5'].output_shape[1:]) 

		# pool 5
		net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)
		print "pool5: {}".format(net['pool5'].output_shape[1:]) 

		#Adaptive stage
		net['apool5'] = InverseLayer(net['pool5'],net['pool5'])
		print "upool5: {}".format(net['apool5'].output_shape[1:])

		net['aconv2'] = Conv2DLayer(net['apool5'],num_filters=256,filter_size=(2, 2),pad = 1,stride = 2) #(6x8x256)
		print "aconv2: {}".format(net['aconv2'].output_shape[1:])

		d = pickle.load(open('/imatge/jpan/lsun2016/saliency-2016-lsun/caffe_places.pkl'))
		numElementsToSet = 16
		lasagne.layers.set_all_param_values(net['pool5'], d[:numElementsToSet])

		return net

	def buildOutputNetwork( self, input_stage ):

		inputNetwork = {}

		inputNetwork['upool5'] = Upscale2DLayer(input_stage, scale_factor=2)
		print "upool5: {}".format(inputNetwork['upool5'].output_shape[1:])

		inputNetwork['uconv5_3'] = ConvLayer(inputNetwork['upool5'], 512, 3, pad=1)
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
		
	def build( self, input_var, output_var ):

		######################################################

		input_layer = InputLayer((None, 3, self.inputHeight, self.inputWidth ), input_var=input_var);

		vggNet = self.buildInputNetwork_VGG_ImageNet( input_layer, input_var )

		alexNet = self.buildInputNetwork_AlexNet_Places( input_layer,input_var )

		input_stage = concat((vggNet['pool5'],alexNet['aconv2']),axis=1)

		self.net = self.buildOutputNetwork( input_stage )

		######################################################
		
		outputLayerName = 'output'
		
		prediction = lasagne.layers.get_output(self.net[outputLayerName])
		
		test_prediction = lasagne.layers.get_output(self.net[outputLayerName], deterministic=True)
		
		loss = lasagne.objectives.squared_error( prediction, output_var )
		loss = loss.mean()
		
		params = lasagne.layers.get_all_params(self.net[outputLayerName], trainable=True)

		updates_sgd = lasagne.updates.sgd(loss, params, learning_rate = self.currentLearningRate )
		updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum = self.currentMomentum ) 

		self.trainFunction = theano.function([input_var, output_var], loss, updates=updates, allow_input_downcast=True)
		self.predictFunction = theano.function([input_var], test_prediction)
		
