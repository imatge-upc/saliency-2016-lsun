import os
import numpy as np
import sys
import cPickle as pickle
import glob
import random
import time
from tqdm import tqdm
import cv2
from models import Conv_ImageNet as dnnModel
import theano
import theano.tensor as T
import lasagne
import pysaliency
from constant import SALICON_API_DIR, SALICON_VAL_DATA_DIR, TRAIN_DATA_DIR, HOME_DIR, SAMPLE_VAL_DATA_DIR
from constant import SAMPLE_TRAIN_DATA_DIR

sys.path.insert(0, SALICON_API_DIR)

# dir_to_save = 'test_img_places'
# dir_to_save = 'model/img_places_conv45'
# dir_to_save = 'test_img'
# dir_to_save = 'test_transposed'   #diltated g and d
dir_to_save = 'test_crazy'


class MySaliencyMapModel(pysaliency.SaliencyMapModel):
    def __init__(self, model):
        """
        Model class needed to compute saliency maps in order to evalute matrics like: AUC_Judd and AUC_Shuffled
        :param model:
        """
        super(MySaliencyMapModel, self).__init__()
        self.model = model

    def _saliency_map(self, stimulus):
        h, w = (stimulus.shape[0], stimulus.shape[1])
        blur_size = 5
        stimulus = cv2.resize(stimulus, (self.model.inputWidth, self.model.inputHeight), interpolation=cv2.INTER_CUBIC)
        blob = np.zeros((1, 3, stimulus.shape[0], stimulus.shape[1]), theano.config.floatX)
        blob[0, ...] = stimulus.astype(theano.config.floatX).transpose(2, 0, 1)
        result = np.squeeze(self.model.predictFunction(blob))
        salmap = (result * 255.).astype(np.uint8)
        salmap = np.clip(salmap, 0, 255)
        # resize back to original size
        salmap = cv2.resize(salmap, (w, h), interpolation=cv2.INTER_CUBIC)
        # blur
        salmap = cv2.GaussianBlur(salmap, (blur_size, blur_size), 0)
        # clip again
        salmap = np.clip(salmap, 0, 255)
        return salmap


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def load_weights(net, path, epochtoload):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    with np.load(HOME_DIR+path+"modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def predict(model, validationData, numEpoch, dir=None):
    width = model.inputWidth
    height = model.inputHeight

    blob = np.zeros((1, 3, height, width), theano.config.floatX)

    blob[0, ...] = (validationData.image.data.astype(theano.config.floatX).transpose(2, 0, 1))  # - imageMean) / 255.

    result = np.squeeze(model.predictFunction(blob))

    saliencyMap = (result * 255).astype(np.uint8)

    cv2.imwrite('./' + dir + '/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch), saliencyMap)

    cv2.imwrite('./results/validationRandomImage_'+str(numEpoch)+'.png',
                cv2.cvtColor(validationData.image.data, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./results/validationRandomSaliencyGT_'+str(numEpoch)+'.png', validationData.saliency.data)


def feature_extraction(model, validationData, numEpoch, dir='test'):
    width = model.inputWidth
    height = model.inputHeight

    blob_img = np.zeros((1, 3, height, width), theano.config.floatX)
    blob_salmap = np.zeros((1, 1, height, width), theano.config.floatX)
    # imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])

    blob_img[0, ...] = (
        validationData.image.data.astype(theano.config.floatX).transpose(2, 0, 1))  # - imageMean) / 255.
    blob_salmap[0, ...] = (validationData.saliency.data.astype(theano.config.floatX)) / 255.

    result = np.squeeze(model.featureFunction(blob_img, blob_salmap))

    featureMap = (result * 255.).astype(np.uint8)
    # print saliencyMap.shape
    cv2.imwrite('./' + dir + '/validationRandomSaliencyPred_{:04d}.png'.format(numEpoch),
                cv2.cvtColor(featureMap.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

    # cv2.imwrite('./results/validationRandomImage_'+str(numEpoch)+'.png',
    #            cv2.cvtColor(validationData.image.data, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./results/validationRandomSaliencyGT_'+str(numEpoch)+'.png', validationData.saliency.data)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def train_gan():
    """
    Train both generator and discriminator
    :return:
    """
    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()
    wrongSaliency = T.tensor4()

    model = dnnModel.Model()
    model.build(inputImage, outputSaliency, wrongSaliency)

    dataset_location = SALICON_VAL_DATA_DIR
    stimuli_salicon_val, fixations_salicon_val = pysaliency.get_SALICON_val(location=dataset_location)

    # Load a pre-trained model
    load_weights(net=model.net['output'], path="test/gen_", epochtoload=15)
    # load_weights(net=model.discriminator['fc5'], path="tests/disrim", epochtoload=21)

    batchSize = 32
    numEpochs = 301

    # batchIn = np.zeros((batchSize, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    # batchOut = np.zeros((batchSize, 1, model.inputHeight, model.inputWidth), theano.config.floatX)
    # batchFake = np.zeros((batchSize, 1, model.inputHeight, model.inputWidth), theano.config.floatX)

    # Load data
    print 'Loading training data...'
    with open(TRAIN_DATA_DIR, 'rb') as f:
        # with open(SAMPLE_VAL_DATA_DIR, 'rb') as f:
        # with open(SAMPLE_TRAIN_DATA_DIR, 'rb') as f:
        trainData = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open(SAMPLE_VAL_DATA_DIR, 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'

    nr_batches_train = int(len(trainData) / batchSize)

    numRandom = random.choice(range(len(validationData)))

    cv2.imwrite('./' + dir_to_save + '/validationRandomSaliencyGT.png', validationData[numRandom].saliency.data)
    cv2.imwrite('./' + dir_to_save + '/validationRandomImage.png', cv2.cvtColor(validationData[numRandom].image.data,
                                                                                cv2.COLOR_RGB2BGR))

    n_updates = 1

    batchIn_val = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in validationData],
                             dtype=theano.config.floatX)
    batchOut_val = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in validationData],
                              dtype=theano.config.floatX)
    batchOut_val = np.expand_dims(batchOut_val, axis=1)

    for currEpoch in tqdm(range(numEpochs), ncols=20):

        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(trainData)

        for currChunk in chunks(trainData, batchSize):

            if len(currChunk) != batchSize:
                continue

            # fakeChunk = currChunk
            # random.shuffle(fakeChunk)

            batchIn = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                 dtype=theano.config.floatX)
            batchOut = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                  dtype=theano.config.floatX)
            batchOut = np.expand_dims(batchOut, axis=1)

            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost = model.G_trainFunction(batchIn, batchOut)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batchIn, batchOut)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost

            n_updates += 1

        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train

        # Save weights every 3 epoch
        if currEpoch % 3 == 0:
            np.savez('./' + dir_to_save + '/gen_modelWeights{:04d}.npz'.format(currEpoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez('./' + dir_to_save + '/disrim_modelWeights{:04d}.npz'.format(currEpoch),
                     *lasagne.layers.get_all_param_values(model.discriminator['fc5']))

            # Check the auc_judd , auc_shuffled while training the network (this may slow down the training process)
            #
            predict(model=model, validationData=validationData[numRandom], numEpoch=currEpoch, dir=dir_to_save)
            _, _, val_cost = model.G_trainFunction(batchIn_val, batchOut_val)

            my_model = MySaliencyMapModel(model)
            auc_judd = my_model.AUC(stimuli_salicon_val[0:100], fixations_salicon_val[fixations_salicon_val.n < 100],
                                    nonfixations='uniform')
            auc_shuffled = my_model.AUC(stimuli_salicon_val[0:20], fixations_salicon_val[fixations_salicon_val.n < 20],
                                        nonfixations='shuffled')

            print('Epoch:' + str(currEpoch) + ' val_loss->' + '\x1b[0;32;40m' + str(val_cost) + ' ' + str(
                auc_judd) + ' ' + str(auc_shuffled) + '\x1b[0m')
            #
            #
            print 'Epoch:', currEpoch, ' train_loss->', (g_cost, d_cost, e_cost)
        else:
            print 'Epoch:', currEpoch, ' train_loss->', (g_cost, d_cost, e_cost)


def train_generator():
    """
    Only trains the generator (without GAN)
    :return:
    """
    # Create network
    inputImage = T.tensor4()
    outputSaliency = T.tensor4()

    model = dnnModel.Model()
    model.build_generator(inputImage, outputSaliency)

    # Load a pretrained model
    load_weights(net=model.net['output'], path="model/img_places_conv45/", epochtoload=90)

    batchSize = 32
    numEpochs = 301

    batchIn = np.zeros((batchSize, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    batchOut = np.zeros((batchSize, 1, model.inputHeight, model.inputWidth), theano.config.floatX)

    # Load data
    print 'Loading training data...'
    # with open(TRAIN_DATA_DIR, 'rb') as f:
    with open(SAMPLE_TRAIN_DATA_DIR, 'rb') as f:
        trainData = pickle.load(f)
    print '-->done!'

    print 'Loading validation data...'
    with open(SAMPLE_VAL_DATA_DIR, 'rb') as f:
        validationData = pickle.load(f)
    print '-->done!'

    nr_batches_train = int(10000 / batchSize)

    # trainData = trainData[0:1000]
    # validationData = validationData[0:20]
    # pickle.dump(validationData, open("valSample64x48.pkl", 'w'))
    # pickle.dump(trainData[0:3000], open("trainSample64x48.pkl", 'w'))

    # imageMean = np.array([[[103.939]], [[116.779]], [[123.68]]])
    # blob = np.zeros((1, 3, height, width), theano.config.floatX)

    numRandom = random.choice(range(len(validationData)))

    cv2.imwrite('./' + dir_to_save + '/validationRandomImage.png', cv2.cvtColor(validationData[numRandom].image.data,
                                                                                cv2.COLOR_RGB2BGR))
    cv2.imwrite('./' + dir_to_save + '/validationRandomSaliencyGT.png', validationData[numRandom].saliency.data)

    n_updates = 1

    for currEpoch in tqdm(range(numEpochs), ncols=20):

        e_cost = 0.
        random.shuffle(trainData)

        for currChunk in chunks(trainData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range(batchSize):
                batchIn[k, ...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2, 0, 1))
                # - model.meanImage_VGG_ImageNet) / 255.
                batchOut[k, ...] = (currChunk[k].saliency.data.astype(theano.config.floatX)) / 255.

            G_cost = model.G_trainFunction(batchIn, batchOut)
            e_cost += G_cost

        e_cost /= nr_batches_train

        print 'Epoch:', currEpoch, ' ->', e_cost

        n_updates += 1

        if currEpoch % 5 == 0:
            np.savez('./' + dir_to_save + '/gen_modelWeights{:04d}.npz'.format(currEpoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, validationData=validationData[numRandom], numEpoch=currEpoch, dir=dir_to_save)


if __name__ == "__main__":
    train_gan()
