'''
    Training JuntingNet with new nolearn code.
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
from lasagne.layers import FeaturePoolLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import objective
from nolearn.lasagne import BatchIterator


def float32(k):
    return np.cast['float32'](k)

# Flip half of the images in this batch at random:
class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        tmp =  yb[indices].reshape(bs/2,1,48,48)
        mirror = tmp[ :,:,:, ::-1]
        yb[indices] =  mirror.reshape(bs/2,48*48)
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


layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 3, 96, 96)}),

    (Conv2DLayer, {'num_filters': 32, 'filter_size': 5}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DLayer, {'num_filters': 64, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    (DenseLayer, {'num_units': 48*48*2}),
    (FeaturePoolLayer,{'pool_size': 2}),
    (DenseLayer, {'num_units': 48*48}),
]

net0 = NeuralNet(
    layers=layers0,
    max_epochs=10,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,

    on_epoch_finished=[
    AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
    AdjustVariable('update_momentum', start=0.9, stop=0.999),
    ],

    batch_iterator_train=FlipBatchIterator(batch_size=16),

    verbose=1,
)

f = file('/imatge/jpan/lasagne/examples/data_Salicon_TV.cPickle','rb')
loaded_obj = cPickle.load(f)
X,y=loaded_obj

X = X.astype(np.float32)
y = y.astype(np.float32)

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))

net0.fit(X, y)
