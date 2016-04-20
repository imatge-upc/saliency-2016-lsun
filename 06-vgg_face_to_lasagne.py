import os
import sys
caffe_root = '/usr/local/opt/caffe-2015-10/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
    
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

import cPickle as pickle
import lasagne
import numpy as np
from lasagne.utils import floatX

def build_model():

    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))

    net['conv1_1'] = ConvLayer( net['input'], 64, 3, pad=1, flip_filters=False )
    net['conv1_2'] = ConvLayer( net['conv1_1'], 64, 3, pad=1, flip_filters=False )

    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    net['conv2_1'] = ConvLayer( net['pool1'], 128, 3, pad=1, flip_filters=False )
    net['conv2_2'] = ConvLayer( net['conv2_1'], 128, 3, pad=1, flip_filters=False)

    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    net['conv3_1'] = ConvLayer( net['pool2'], 256, 3, pad=1, flip_filters=False )
    net['conv3_2'] = ConvLayer( net['conv3_1'], 256, 3, pad=1, flip_filters=False )
    net['conv3_3'] = ConvLayer( net['conv3_2'], 256, 3, pad=1, flip_filters=False )

    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    net['conv4_1'] = ConvLayer( net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer( net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer( net['conv4_2'], 512, 3, pad=1, flip_filters=False)

    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    net['conv5_1'] = ConvLayer( net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer( net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer( net['conv5_2'], 512, 3, pad=1, flip_filters=False)

    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    net['fc6'] = DenseLayer( net['pool5'], num_units=4096)

    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)

    net['fc7'] = DenseLayer( net['fc6_dropout'], num_units=4096)

    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    
    net['fc8'] = DenseLayer( net['fc7_dropout'], num_units=2622, nonlinearity=None)
    
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

def load_caffe():

    net = build_model()

    specfile = '/imatge/jpan/lsun2016/saliency-2016-lsun/cnn_models/vgg_face_caffe/VGG_FACE_deploy.prototxt'
    modelfile = '/imatge/jpan/lsun2016/saliency-2016-lsun/cnn_models/vgg_face_caffe/VGG_FACE.caffemodel'

    caffe_net=caffe.Net(specfile, modelfile, caffe.TEST)

    layers_caffe = dict(zip(list(caffe_net._layer_names), caffe_net.layers))

    for name, layer in net.items():
        try:
            if name == 'fc6' or name == 'fc7' or name == 'fc8':
                    # no need to flip for fully connected layers
                    layer.W.set_value(np.transpose(layers_caffe[name].blobs[0].data)) 
                    layer.b.set_value(layers_caffe[name].blobs[1].data)
            else:
                # need to flip to get the same answer in convolution
                layer.W.set_value(layers_caffe[name].blobs[0].data[:,:,::-1,::-1]) 
                layer.b.set_value(layers_caffe[name].blobs[1].data)     
        except AttributeError:
            continue


    values = lasagne.layers.get_all_param_values(net['prob'])
    pickle.dump(values, open('vgg16_faces.pkl', 'w'),protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    load_caffe()