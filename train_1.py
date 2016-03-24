'''
    Test: traing a network form scratch.
    Lasage code only.
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


DATA_PATH = './train_data/'

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,InverseLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['conv2_1'] = ConvLayer(net['conv1_2'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['conv3_1'] = ConvLayer(net['conv2_2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv4_1'] = ConvLayer(net['conv3_3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv5_1'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['output_layer'] = ConvLayer(net['conv5_3'],1,1)
    return net


# Build the network and fill with pretrained weights
net = build_model()

# The network expects input in a particular format and size.
# We define a preprocessing function to load a file and apply the necessary transformations

# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 16

# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == N:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def train_batch():
    ix = range(len(y_tr))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return train_fn(X_tr[ix], y_tr[ix])

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])


class costum_loss():
    def kl_loss(fm,sm):
        epsilon = 0.000001
        Sm = sm / (T.sum(sm,axis=[0,1]) + epsilon)
        Fm = fm / (T.sum(fm,axis=[0,1]) + epsilon)
        kl =  T.sum(T.log(Fm / (Sm + epsilon)+epsilon),axis =[0,1])
        return kl

if __name__ == '__main__':

    #Here we load the data    
    #f = file('data_Salicon_T_img.cPickle', 'rb')
    #loaded_obj = cPickle.load(f)
    #f.close()
    #X,y = loaded_obj
    flag = 0
    for files in glob.glob(DATA_PATH+"*.cPickle"):
        f = file(files, 'rb')
        loaded_obj = cPickle.load(f)
        f.close()
        if flag == 0:
            X,y = loaded_obj
            flag=1
        else:
            X2,y2 = loaded_obj
            X = np.append(X,X2,axis=0)
            y = np.append(y,y2,axis=0)


    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))

    train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(y)))
    train_ix, val_ix = sklearn.cross_validation.train_test_split(range(len(train_ix)))

    X_tr = X[train_ix]
    y_tr = y[train_ix]

    X_val = X[val_ix]
    y_val = y[val_ix]

    X_te = X[test_ix]
    y_te = y[test_ix]

    print 'here the output shape'
    for l in lasagne.layers.get_all_layers(net['output_layer']):
        print l.output_shape

    # Define loss function and metrics, and get an updates dictionary
    X_sym = T.tensor4()
    #y_sym = T.ivector()
    y_sym = T.tensor4()

    prediction = lasagne.layers.get_output(net['output_layer'], X_sym)

    loss = lasagne.objectives.squared_error(prediction, y_sym)
    #loss = costum_loss.kl_loss(prediction,y_sym)
    loss = loss.mean()

    acc = T.mean(T.eq(prediction, y_sym),
                      dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(net['output_layer'], trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.001, momentum=0.9)

    # Compile functions for training, validation and prediction
    train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
    val_fn = theano.function([X_sym, y_sym], [loss, acc])
    pred_fn = theano.function([X_sym], prediction)

    #traning loop
    for epoch in range(5):
        for batch in range(25):
            loss = train_batch()

        ix = range(len(y_val))
        np.random.shuffle(ix)

        loss_tot = 0.
        acc_tot = 0.
        for chunk in batches(ix, BATCH_SIZE):
            loss, acc = val_fn(X_val[chunk], y_val[chunk])
            loss_tot += loss * len(chunk)
            acc_tot += acc * len(chunk)

        loss_tot /= len(ix)
        acc_tot /= len(ix)
        print(epoch, loss_tot, acc_tot * 100) 
        
        
