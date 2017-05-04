
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils


# In[2]:

def data_split(s1,s2,data):
    return np.vstack([data[:s1],data[s2:]]) ,data[s1:s2]

train = pd.read_csv('train.csv')
train = np.array(train)
label = train[:,0]
feat = np.zeros([len(train[:,1]),48,48],'float32')

for i in range(len(train[:,1])):           
    f = train[:,1][i].split(" ")
    f = np.reshape(f,(48,48))
    feat[i] = f

feat= feat.reshape(len(train[:,1]),48*48)
feat = feat/255
label = np_utils.to_categorical(label, 7)


(feat_tr,feat_va) = data_split(500,3500,feat)
(label_tr,label_va) = data_split(500,3500,label)


# In[3]:


# In[ ]:

model = Sequential()

# CNN part (you can repeat this part several times)
model.add(Dense(input_dim=48*48,output_dim=32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.35))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.35))
          
          
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.35))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.35))


model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.35))


model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.35))



model.add(Dense(7))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = Adam(lr=0.0005, decay=5e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary() # show the whole model in terminal


# In[4]:

his = model.fit(feat_tr,label_tr,batch_size=100,validation_split=0.05,epochs=100)


# In[16]:

mdhis = pd.DataFrame(his.history).to_csv("DNN_his.csv")


# In[15]:

Mdhis = np.array(mdhis)

Mdhis = Mdhis.transpose()

import matplotlib.pyplot as plt

epoch = np.linspace(0,100,100)
plt.figure()
plt.plot(epoch,Mdhis[0])
plt.plot(epoch,Mdhis[2])
plt.title("Training Process")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend(["Training set","Validation set"])
plt.axis([0,100,0,1])
plt.savefig("DNN.png")


# In[14]:

# In[ ]:
test = pd.read_csv('test.csv')
test = np.array(test)
file_write_result = open('DNN.csv', 'w')
test_f = np.zeros([len(test[:,1]),48,48],'float32')

for i in range(len(test[:,1])):           
    f = test[:,1][i].split(" ")
    f = np.reshape(f,(48,48))
    test_f[i] = f
    
test_f= test_f.reshape(len(test[:,1]),48*48)
test_f = test_f/255


# In[15]:


prediction = model.predict(test_f)
predict_result = np.argmax(prediction, axis = 1)
file_write_result.write('id,label\n')
for i in range(7178):
	string = str(i) + ',' + str(predict_result[i]) + '\n'
	file_write_result.write(string)


# In[8]:

import keras 
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


# In[16]:

tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,  
          write_graph=True, write_images=True)


# In[18]:

tbCallBack.set_model(model)


# In[19]:

tensorboard  --logdir Graph/


# In[ ]:



