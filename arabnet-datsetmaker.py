from __future__ import print_function

'''
Arabic Handwritten Digit Recognition Using DNN 

Akm Ashiquzzaman
Fall 2016
13101002@uap-bd.edu
zamanashiq3@gmail.com

The Simple Convnet model classifing

'''
#Numpy and Scipy Import 
import numpy as np
np.random.seed(1337)
'''
In 1000 epoch

Test score: 0.196029246261
Test accuracy: 0.974
'''

#OS CV and PIL for preprossing
from os import listdir
from PIL import Image as img
from PIL import ImageOps as ops
from os.path import isfile, join
import cv2 as cv

hog = cv.HOGDescriptor('hog-properties.xml')
#Main data parser from BMP files
def dataProcess(dirname):
	Names = []
	for filename in listdir(dirname):
		if(filename.endswith('.bmp')):
			Names.append(dirname+'/'+filename)
	Names.sort()
	X1 = np.array([np.array(ops.invert(img.open(name).convert('L'))) for name in Names]).astype('float32')
	X2 = np.array([np.array(hog.compute(cv.imread(name,0))) for name in Names ])
	num = len(Names)	
	Y = np.array([(x%10) for x in range(0,num)]).astype('int')
	return X1 , X2, Y

#Now main spllitiing

def load_data(dirname):
	dataX1,dataX2,dataY = dataProcess(dirname)
	(train_X1,train_X2,train_y), (test_X1,test_X2,test_y) = (dataX1[0:2000,:,:],dataX2[0:2000,:],dataY[0:2000]),(dataX1[2000:,:,:],dataX2[2000:,:],dataY[2000:])
	return (train_X1,train_X2,train_y), (test_X1,test_X2,test_y)


#Now it's just a function call to load and test data.
(X1_train,X2_train,y_train), (X1_test,X2_test,y_test) = load_data('ArabicBMP')

# data Normalization, getting all values in 0~1
X1_train /= 255
X1_test /= 255

X1_train = X1_train.astype('float32')
X1_test = X1_test.astype('float32')


#again trasforming into floating poing value
X2_train=X2_train.reshape(X2_train.shape[0],X2_train.shape[1]).astype('float32')
X2_test=X2_test.reshape(X2_test.shape[0],X2_test.shape[1]).astype('float32')


#Keras Import
from keras.utils import np_utils
from keras import backend as K

nb_classes = 10

# input image dimensions
img_rows, img_cols = 32, 32


#first used the tensorflow in code dev, but theano will be used in training, so need this part for dim ordering
if K.image_dim_ordering() == 'th':
    X1_train = X1_train.reshape(X1_train.shape[0], 1, img_rows, img_cols)
    X1_test = X1_test.reshape(X1_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X1_train = X1_train.reshape(X1_train.shape[0], img_rows, img_cols, 1)
    X1_test = X1_test.reshape(X1_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

Y_train = Y_train.astype('float32') 
Y_test = Y_test.astype('float32')

print('Training Tensor Label:'+str(Y_train.shape))
print('Testing Tensor Label:'+str(Y_test.shape))

print('Training Tensor 1:'+str(X1_train.shape))
print('Testing Tensor 1:'+str(X1_test.shape))

print('Training Tensor 2:'+str(X2_train.shape))
print('Testing Tensor 2:'+str(X2_test.shape))

np.savez_compressed('hybrid_model-data.npz',Y_train,Y_test,X1_train,X1_test,X2_train,X2_test)


