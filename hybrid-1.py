from __future__ import print_function,division

"""
Akm Ashiquzzaman
13101002@uap-bd.edu

applying hybrid model (CNN+ANN to recognise arabic digits)

CNN ==> gets the data in 32x32 image scale. 
ANN ==> gets the data in 600 HOG 1D vectors

"""
batch_size = 20
epochs = 10

#Numpy Import 
import numpy as np
np.random.seed(1337)

#main data load
data= np.load('hybrid_model-data.npz')

#data load for both models
Y_train,Y_test,X1_train,X1_test,X2_train,X2_test = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3'],data['arr_4'],data['arr_5']

print('Training Tensor Label:'+str(Y_train.shape))
print('Testing Tensor Label:'+str(Y_test.shape))

print('Training Tensor 1:'+str(X1_train.shape))
print('Testing Tensor 1:'+str(X1_test.shape))

print('Training Tensor 2:'+str(X2_train.shape))
print('Testing Tensor 2:'+str(X2_test.shape))

#keras import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

#model 1
model1 = Sequential()

model1.add(Convolution2D(32,3,3,
                        border_mode='valid',
                        input_shape=(1,32,32)))
model1.add(Activation('relu'))

model1.add(Dropout(0.25))

model1.add(Convolution2D(16,3,3))
model1.add(Activation('relu'))

model1.add(MaxPooling2D(pool_size=(2,2)))

model1.add(Dropout(0.25))

model1.add(Flatten())

#model 2
model2 = Sequential()

model2.add(Dense(64, input_shape=(600,)))
model2.add(Activation('relu'))

model2.add(Dropout(0.50))

model2.add(Dense(32))
model2.add(Activation('relu'))

#merge models
from keras.layers import Merge
merged = Merge([model1,model2],mode='concat') 

final_model = Sequential()

final_model.add(merged)
final_model.add(Activation('relu'))

final_model.add(Dropout(0.25))

final_model.add(Dense(128))
final_model.add(Activation('relu'))

final_model.add(Dense(10))
final_model.add(Activation('softmax'))

#ConvNET weight flepath saving, we're usung the callback loss to save the best weight model.

from os.path import isfile
  
weight_path="hybrid-1_weights.hdf5"

if isfile(weight_path):
	final_model.load_weights(weight_path)

final_model.compile(optimizer='adadelta',loss='categorical_crossentropy'
,metrics=['accuracy'])


hist = checkpoint = ModelCheckpoint(weight_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

#Some other parameter will be added later for finetune
callbacks_list = [checkpoint]

hist = final_model.fit([X1_train,X2_train],Y_train,batch_size=batch_size, nb_epoch=epochs,validation_data=([X1_test,X2_test], Y_test),callbacks=callbacks_list)


acc = np.array(hist.history['acc']).astype('float32')
np.savetxt('acc-arabnet.txt',acc,delimiter=',')

loss = np.array(hist.history['loss']).astype('float32')
np.savetxt('loss-arabnet.txt',acc,delimiter=',')

val_acc = np.array(hist.history['val_acc']).astype('float32')
np.savetxt('val-acc-arabnet.txt',acc,delimiter=',')

val_loss = np.array(hist.history['val_loss']).astype('float32')
np.savetxt('val-loss-arabnet.txt',acc,delimiter=',')

score = final_model.evaluate([X1_test,X2_test], Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

