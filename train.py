# Model definition for VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8

# More pretrained models are available from
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/

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

CLASSES = ['pancakes', 'waffles']
LABELS = {cls: i for i, cls in enumerate(CLASSES)}

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
d = pickle.load(open('vgg16.pkl'))

# Build the network and fill with pretrained weights
net = build_model()
lasagne.layers.set_all_param_values(net['prob'], d['param values'])

# The network expects input in a particular format and size.
# We define a preprocessing function to load a file and apply the necessary transformations
IMAGE_MEAN = d['mean value'][:, np.newaxis, np.newaxis]

# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 16

def prep_image(fn, ext='jpg'):
    im = plt.imread(fn, ext)

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # discard alpha channel if present
    im = im[:3]

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - IMAGE_MEAN
    return rawim, floatX(im[np.newaxis])

#Load and preprocess the entire dataset into numpy arrays

def prep_data():
    X = []
    y = []

    for cls in CLASSES:
        for fn in os.listdir('./images/{}'.format(cls)):
            _, im = prep_image('./images/{}/{}'.format(cls, fn))
            X.append(im)
            y.append(LABELS[cls])
            
    X = np.concatenate(X)
    y = np.array(y).astype('int32')
    return X,y

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

def deprocess(im):
    im = im[::-1, :, :]
    im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    im = (im - im.min())
    im = im / im.max()
    return im

class costum_loss():
    def kl_loss(fm,sm):
        epsilon = 0.000001
        Sm = sm / (T.sum(sm,axis=[0,1]) + epsilon)
        Fm = fm / (T.sum(fm,axis=[0,1]) + epsilon)
        kl =  T.sum(T.log(Fm / (Sm + epsilon)+epsilon),axis =[0,1])
        return kl

if __name__ == '__main__':
    #main()
    #X,y=prep_data()
    #data_to_save = (X, y)
    #pickle.dump(data_to_save, open('data.pkl', 'w'))
    #X,y = cPickle .load(open('data_Salicon_T.cPickle'))
    # Split into train, validation and test sets

    f = file('data_Salicon_T.cPickle', 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    X, y = loaded_obj

    train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(y)))
    train_ix, val_ix = sklearn.cross_validation.train_test_split(range(len(train_ix)))

    X_tr = X[train_ix]
    y_tr = y[train_ix]

    X_val = X[val_ix]
    y_val = y[val_ix]

    X_te = X[test_ix]
    y_te = y[test_ix]


    
    # We'll connect our output classifier to the last fully connected layer of the network
    net2={}
    net2['fc_6']=ConvLayer(net['pool5'], 1024, 7)
    net2['deconv_7'] = InverseLayer(net2['fc_6'], net2['fc_6'])
    net2['unpool_8'] = InverseLayer(net2['deconv_7'],net['pool5'])
    net2['deconv_9'] = InverseLayer(net2['unpool_8'],net['conv5_1'])
    net2['unpool_10'] = InverseLayer(net2['deconv_9'],net['pool4'])
    net2['deconv_11'] = InverseLayer(net2['unpool_10'],net['conv4_1'])
    net2['unpool_12'] = InverseLayer(net2['deconv_11'],net['pool3'])
    net2['deconv_13'] = InverseLayer(net2['unpool_12'],net['conv3_1'])
    net2['unpool_14'] = InverseLayer(net2['deconv_13'],net['pool2'])
    net2['deconv_15'] = InverseLayer(net2['unpool_14'],net['conv2_1'])
    net2['unpool_16'] = InverseLayer(net2['deconv_15'],net['pool1'])
    net2['deconv_17'] = InverseLayer(net2['unpool_16'],net['conv1_1'])
    net2['output_layer'] = ConvLayer(net2['deconv_17'],1,1)
    print 'here the output shape'
    for l in lasagne.layers.get_all_layers(net2['output_layer']):
        print l.output_shape

    # Define loss function and metrics, and get an updates dictionary
    X_sym = T.tensor4()
    #y_sym = T.ivector()
    y_sym = T.tensor4()

    prediction = lasagne.layers.get_output(net2['output_layer'], X_sym)

    loss = lasagne.objectives.squared_error(prediction, y_sym)
    #loss = costum_loss.kl_loss(prediction,y_sym)
    loss = loss.mean()

    acc = T.mean(T.eq(prediction, y_sym),
                      dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(net2['output_layer'], trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)

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