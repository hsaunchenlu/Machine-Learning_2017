
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from keras.utils import np_utils

def data_split(s1,s2,data):
    return np.vstack([data[:s1],data[s2:]]) ,data[s1:s2]

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

train = pd.read_csv('train.csv')
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
    
te_label = pd.read_csv('train.csv')
te_label = np.array(te_label)
te_label = te_label[:,0]
te_label = te_label[500:3500]
te_label = te_label.astype(int)


model = load_model("model.h5")
prediction = model.predict_classes(feat_va)

conf_mat = confusion_matrix(te_label,prediction)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig("confus matrix.png")


# In[ ]:



