# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D,ZeroPadding2D, MaxPooling2D,BatchNormalization  ,Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import h5py 

# In[2]:
import sys
data_path = sys.argv[1]
outcome = sys.argv[2]

#if data_path[-1] != '/' :
#    data_path = data_path + '/'



model_name = "model.h5"


if outcome[-4:] != '.csv':
    outcome = outcome + '.csv'

filepath = model_name


file_read_test = open(data_path, 'r')
file_write_result = open(outcome, 'w')
file_read_test.readline()

test_data=np.asarray([ [0.0 for i in range(2304)] for j in range(7178)])
for i in range(7178):
	line = file_read_test.readline()
	line = line.split(',')
	test_data[i] = (np.asarray(line[1].split(' ')).astype(float)) / 255.0
	#print(i)
all_test_data = test_data.reshape(7178, 48, 48, 1)




# In[ ]:

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


model.load_weights(model_name)


adam = Adam(lr=0.0005, decay=5e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.summary() # show the whole model in terminal
#model.fit(feat_tr,label_tr,batch_size=100,validation_split=0.05,epochs=100)
#model.save("model_pass.h5")



# In[ ]:



prediction = model.predict(all_test_data)
predict_result = np.argmax(prediction, axis = 1)
file_write_result.write('id,label\n')
for i in range(7178):
	string = str(i) + ',' + str(predict_result[i]) + '\n'
	file_write_result.write(string)

