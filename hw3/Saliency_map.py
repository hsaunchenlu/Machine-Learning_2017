
# coding: utf-8

# In[141]:

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pydot
import graphviz
import pandas as pd
import numpy as np


# In[142]:

emotion_classifier = load_model("model.h5")


# In[143]:

from termcolor import colored,cprint
import keras.backend as K


# In[177]:

test = pd.read_csv('train.csv')
test = np.array(test)
feat = np.zeros([len(test[:,1]),1,48,48,1],'float32')


# In[178]:

for i in range(len(test[:,1])):           
    f = test[:,1][i].split(" ")
    f = np.reshape(f,(1,48,48,1))
    feat[i] = f
    



# In[179]:

input_img = emotion_classifier.input
img_ids = [210]
idd = 210


# In[160]:

for idx in img_ids:
        val_proba = emotion_classifier.predict(feat[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])


# In[163]:

heatmap = fn([feat[idd],0])
Heatmap = np.reshape(heatmap,(48,48))
Heatmap_avg = Heatmap.mean()
Heatmap_nor = (Heatmap-Heatmap.min())/(Heatmap.max()-Heatmap.min())
heatmap_nor = np.zeros_like(Heatmap_nor)
for i in range(len(Heatmap_nor)):
    for j in range(len(Heatmap_nor[0])):
        heatmap_nor[i,j] = Heatmap_nor[i,j]*((Heatmap_nor[i,j]-Heatmap_nor.min())/(Heatmap_nor.max()-Heatmap_nor.min()))
heat = (heatmap_nor-heatmap_nor.min())/(heatmap_nor.max()-heatmap_nor.min())


# In[ ]:

plt.imshow(feat[idd].reshape(48,48),cmap='gray')
plt.show()


# In[164]:

plt.figure()
plt.imshow(heat, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
plt.show()


# In[193]:

thres = 0.38
see = feat[idd].reshape(48,48)
see[np.where(Heatmap_nor <= thres)] = np.mean(see)


# In[194]:




# In[195]:

plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
plt.show()


# In[ ]:




# In[ ]:



