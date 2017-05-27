

# coding: utf-8

# In[2]:

#Import packages
import pandas as pd
import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
import sys

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(1, activation='linear'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]


# In[2]:

#Define constants
TESTING_CSV_FILE = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
MODEL_WEIGHTS_FILE = 'weights.h5'
K_FACTORS = 120

max_userid = 6040
max_movieid  = 3952

# In[1]:

#Import data


test = pd.read_csv(TESTING_CSV_FILE, usecols=['TestDataID', 'UserID', 'MovieID'])
print (len(test), 'descriptions of', max_movieid, 'movies loaded.')


# In[11]:

#Predict
trained_model = CFModel(max_userid, max_movieid, K_FACTORS)

trained_model.load_weights(MODEL_WEIGHTS_FILE)


prediction = trained_model.predict([test['UserID'].values,test['MovieID'].values])

ids = test['TestDataID'].values

with open(OUTPUT_PATH,'w') as sm:
    print('\"TestDataID\",\"Rating\"',file=sm)
    for i in range(len(ids )):
        print('\"%d\",\"%.1f\"'%(ids[i],prediction[i]),file=sm)


# In[ ]:
