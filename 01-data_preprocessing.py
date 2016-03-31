import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import Image
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation

pathToImages = '/home/crcanton/data/Datasets/Saliency/salicon/images'
pathToMaps = '/home/crcanton/data/Datasets/Saliency/salicon/saliency'

pathOutputImages = '/home/crcanton/data/Datasets/Saliency/salicon/images256x192'
pathOutputMaps = '/home/crcanton/data/Datasets/Saliency/salicon/saliency256x192'

## Resize train/validation files

listMapFiles = [ k.split('/')[-1].split('.')[0] for k in glob.glob( os.path.join(pathToMaps, '*' ) ) ]

for currFile in tqdm(listMapFiles):
    tt = dataRepresentation.Target( os.path.join( pathToImages, currFile +'.jpg'), 
									os.path.join( pathToMaps, currFile +'.mat'),                                    
									dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.saliencyMapMatlab )
    
    if tt.image.getImage().shape[:2] != (480,640): # Security check
        print 'Error:', currFile
    
    imageResized = cv2.cvtColor( cv2.resize(tt.image.getImage(),(256,192),interpolation=cv2.cv.CV_INTER_AREA), cv2.cv.CV_RGB2BGR )
    saliencyResized = cv2.resize(tt.saliency.getImage(),(256,192),interpolation=cv2.cv.CV_INTER_AREA)
    
    cv2.imwrite( os.path.join( pathOutputImages, currFile +'.png'), imageResized )
    cv2.imwrite( os.path.join( pathOutputMaps, currFile +'.png'), saliencyResized )
    
## Resize test files

listTestImages = [ k.split('/')[-1].split('.')[0] for k in glob.glob( os.path.join( pathToImages, '*test*') ) ] 

for currFile in tqdm(listTestImages):
    tt = dataRepresentation.Target( os.path.join( pathToImages, currFile+'.jpg' ),                                    os.path.join( pathToMaps, currFile +'.mat'),                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                    dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty )
    
    imageResized = cv2.cvtColor( cv2.resize(tt.image.getImage(),(256,192),interpolation=cv2.cv.CV_INTER_AREA), cv2.cv.CV_RGB2BGR )
    cv2.imwrite( os.path.join( pathOutputImages, currFile +'.png'), imageResized )


## LOAD DATA

## Train

listFilesTrain = [ k for k in listMapFiles if 'train' in k ]
trainData = []
for currFile in tqdm(listFilesTrain):
    trainData.append( dataRepresentation.Target( os.path.join( pathOutputImages, currFile +'.png'),                                    os.path.join( pathOutputMaps, currFile +'.png'),                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale ) )
    
with open( 'trainData.pickle', 'wb') as f:
    pickle.dump( trainData, f )    

## Validation

listFilesValidation = [ k for k in listMapFiles if 'val' in k ]
validationData = []
for currFile in tqdm(listFilesValidation):
    validationData.append( dataRepresentation.Target( os.path.join( pathOutputImages, currFile +'.png'),                                    os.path.join( pathOutputMaps, currFile +'.png'),                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale ) )
    
with open( 'validationData.pickle', 'wb') as f:
    pickle.dump( validationData, f )       


## Test

testData = []

for currFile in tqdm(listTestImages):
    testData.append( dataRepresentation.Target( os.path.join( pathOutputImages, currFile +'.png'),                                    os.path.join( pathOutputMaps, currFile +'.png'),                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                    dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty ) )
    
with open( 'testData.pickle', 'wb') as f:
    pickle.dump( testData, f )     