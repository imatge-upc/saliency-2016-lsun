#Prepare data with '.cPickle' format, fast loading to memory
import os
import numpy as np
import cPickle as pickle
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import glob

MAT_TRAIN = '/imatge/jpan/work/training_copy.mat'
MAT_VAL = '/imatge/jpan/work/validation.mat'
MAT_TEST = '/imatge/jpan/work/testing.mat'

def loadNameList(MATFILE,numImg,typee): #numImg:5000 for validation 10.000 for training
  mat = scipy.io.loadmat(MATFILE)
  xd = mat[typee] 
  imageList=range(numImg) 
  for i in range(numImg):
    imageList[i] = xd[0,i][0][0]
  return imageList

def loadSaliencyMap(FILENAME):
    saliency = scipy.io.loadmat('/imatge/jpan/work/saliency/'+FILENAME+'.mat')
    mapp =saliency['I']
    return mapp

def loadnewSliencyMap(FILENAME):
    saliency = io.imread('/imatge/jpan/work/new_salicon/'+FILENAME+'.png')
    return saliency

def loadSalicon():
    NumSample = 99;
    X1 = np.zeros((NumSample, 3, 224, 224), dtype='float32')
    y1 = np.zeros((NumSample, 1, 224,224), dtype='float32')
    names = loadNameList(MAT_TRAIN,NumSample,'training')
    for i in range(NumSample):
        img = Image.open('/imatge/jpan/work/images/'+names[i]+'.jpg')
        img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') / 255.
        #img = misc.imresize(img,(96,96)) / 255.
        img = img.transpose(2,0,1).reshape(3, 224, 224)
        X1[i] = img
        label = loadnewSliencyMap(names[i])
        label = misc.imresize(label,(224,224)) / 127.5
        label = label -1.
        y1[i] = label#label.reshape(1,48*48)

    data_to_save = (X1,y1)
    f = file('data_Salicon_T_img.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()  
    '''  
    data_to_save = y1
    f = file('data_Salicon_T_sm.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close() 
 
    
    NumSample = 5000
    names = loadNameList(MAT_VAL,NumSample,'validation')
    X2 = np.zeros((NumSample, 3, 224, 224), dtype='float32')
    y2 = np.zeros((NumSample,224,224), dtype='float32')
    for i in range(NumSample):
        img = Image.open('/imatge/jpan/work/images/'+names[i]+'.jpg')
        img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
        img = np.asarray(img, dtype = 'float32') / 255.
        #img = Image.open('/imatge/jpan/work/images/'+names[i]+'.jpg')
        #img = np.asarray(img, dtype = 'float32') 
        #img = misc.imresize(img,(96,96)) / 255.
        img = img.transpose(2,0,1).reshape(3, 224, 224)
        X2[i] = img
        label = loadnewSliencyMap(names[i])
        label = misc.imresize(label,(224,224)) / 127.5
        label = label -1.
        y2[i] = label#label.reshape(1,48*48)

    X=np.append(X1,X2,axis=0)
    y=np.append(y1,y2,axis=0)   
     
    data_to_save = (X, y)
    f = file('data_Salicon_TV.cPickle', 'wb')
    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    '''
def  main():
    loadSalicon()
if __name__ == '__main__':
    main()