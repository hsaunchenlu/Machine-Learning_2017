
# coding: utf-8

# In[ ]:

# coding: utf-8


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D,ZeroPadding2D, MaxPooling2D,BatchNormalization  ,Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import h5py
import sys
data_path = sys.argv[1]




def data_split(s1,s2,data):
    return np.vstack([data[:s1],data[s2:]]) ,data[s1:s2]

train = pd.read_csv(data_path)
train = np.array(train)
label = train[:,0]
feat = np.zeros([len(train[:,1]),48,48],'float32')

for i in range(len(train[:,1])):           
    f = train[:,1][i].split(" ")
    f = np.reshape(f,(48,48))
    feat[i] = f

feat= feat.reshape(len(train[:,1]),48,48,1)
feat = feat/255
label = np_utils.to_categorical(label, 7)


(feat_tr,feat_va) = data_split(500,3500,feat)
(label_tr,label_va) = data_split(500,3500,label)








model = Sequential()

# CNN part (you can repeat this part several times)
model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Convolution2D(32,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

'''
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(BatchNormalization())
'''
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))
          
          
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

'''
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(BatchNormalization())
'''
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))




model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

'''
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(Activation('relu'))
model.add(BatchNormalization())
'''
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))



# Fully connected part
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.35))


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.35))



model.add(Dense(7))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = Adam(lr=0.0005, decay=5e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.summary() # show the whole model in terminal


model.fit(feat_tr,label_tr,batch_size=100,validation_split=0.05,epochs=100)

model.save("model.h5")

