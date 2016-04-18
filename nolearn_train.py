'''
    Traning fcnn, with nolearn code. 
    Model is initialized with vgg16
    A deconvolutional stage is added at the end.
    The loss function is the L2 distance 
'''
import os
import numpy as np
import theano.tensor as T
import theano
import glob
import cPickle
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
import file_dir as file_dir

pathToImagesPickle = file_dir.pathToImagesPickle
layers0 = [
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

    #Deconvolutional Stage
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Upscale2DLayer, {'scale_factor': 2}),
    
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 512, 'filter_size': 3, 'pad':1}),
    (Upscale2DLayer, {'scale_factor': 2}),

    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 256, 'filter_size': 3, 'pad':1}),
    (Upscale2DLayer, {'scale_factor': 2}),

    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad':1}),
    (Upscale2DLayer, {'scale_factor': 2}),

    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad':1}),
    (Upscale2DLayer, {'scale_factor': 2}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, 'pad':1}),
    (Conv2DLayer, {'num_filters': 32, 'filter_size': 3, 'pad':1}),
    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 1, 'filter_size': 3, 'pad':1}),
]

def float32(k):
    return np.cast['float32'](k)

# Flip half of the images in this batch at random:
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        yb[indices] = yb[indices, :, :, ::-1]
        return Xb, yb

#Changing with respect to the training epoch the learning rate 
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net0 = NeuralNet(
    layers=layers0,
    max_epochs=70,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
    AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    AdjustVariable('update_momentum', start=0.9, stop=0.999),
    ],

    regression=True,
    y_tensor_type = T.tensor4,

    #Batch size of 16 is set.(Memory limitation)
    batch_iterator_train=FlipBatchIterator(batch_size=16),
    batch_iterator_test=BatchIterator(batch_size=16),

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)


#Load weights from the vgg16
net0.load_params_from('vgg_weights.pickle')
# Load data

print 'Loading training data...'
with open( pathToImagesPickle, 'rb') as f:
    trainData = cPickle.load( f )    
print '-->done!'

width = 256
height = 192

X = np.zeros((10000, 3, height, width ), theano.config.floatX )
y = np.zeros((10000, 1, height, width ), theano.config.floatX )

for k in range( 10000 ):
    X[k,...] = (trainData[k].image.data.astype(theano.config.floatX).transpose(2,0,1)-127)/255.
    y[k,...] = (trainData[k].saliency.data.astype(theano.config.floatX))/255.

#Training 
net0.fit(X, y)

#Save all parameters
net0.save_params_to('net0_weights.pickle')
#Save the network object
with open('net0.pickle', 'w') as f:
   cPickle.dump(net0, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
