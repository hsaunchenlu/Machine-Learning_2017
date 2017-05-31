
# coding: utf-8

# In[ ]:

import math
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from itertools import product
from keras import backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import add, dot, concatenate
from keras.models import Model
from keras.optimizers import Adam
import sys

def MF(n_U,n_M,F):    
    U_input = Input(shape=(1,), dtype='int64', name='users')
    M_input = Input(shape=(1,), dtype='int64', name='movies')

    U_embedding = Embedding(input_dim=max_userid, output_dim=F)(U_input)
    M_embedding = Embedding(input_dim=max_movieid, output_dim=F)(M_input)

    predicted_preference = dot(inputs=[U_embedding, M_embedding], axes=2)
    predicted_preference = Flatten()(predicted_preference)
    
    model = Model(inputs=[U_input, M_input],outputs=predicted_preference)
    return model

def IMF(n_U,n_M,F):    
    U_input = Input(shape=(1,), dtype='int64', name='users')
    M_input = Input(shape=(1,), dtype='int64', name='movies')

    U_embedding = Embedding(input_dim=max_userid, output_dim=F)(U_input)
    M_embedding = Embedding(input_dim=max_movieid, output_dim=F)(M_input)

    predicted_preference = dot(inputs=[U_embedding, M_embedding], axes=2)

    U_bias = Embedding(input_dim=max_userid, output_dim=1, name='user_bias', input_length=1)(U_input)
    M_bias = Embedding(input_dim=max_movieid, output_dim=1, name='movie_bias', input_length=1)(M_input)

    predicted_preference = add(inputs=[predicted_preference, M_bias, U_bias])
    predicted_preference = Flatten()(predicted_preference)
    
    model = Model(inputs=[U_input, M_input],outputs=predicted_preference)
    return model



def DeepMF(n_U,n_M,F):    
    U_input = Input(shape=(1,), dtype='int64', name='users')
    M_input = Input(shape=(1,), dtype='int64', name='movies')

    U_embedding = Embedding(input_dim=max_userid, output_dim=F)(U_input)
    M_embedding = Embedding(input_dim=max_movieid, output_dim=F)(M_input)
    
    #U_bias = Embedding(input_dim=max_userid, output_dim=1, name='user_bias', input_length=1)(U_input)
    #M_bias = Embedding(input_dim=max_movieid, output_dim=1, name='movie_bias', input_length=1)(M_input)

    concatenation = concatenate(inputs=[U_embedding, M_embedding])

    dropout = Dropout(.1)(concatenation)
    dense_layer = Dense(activation='relu', units=120)( dropout)
    dropout = Dropout(.1)(dense_layer)
    
    predicted_preference = Dense(activation='linear',units=1, name='predicted_preference')(dropout)
    predicted_preference = Flatten()(predicted_preference)
    
    #predicted_preference = add(inputs=[predicted_preference, M_bias, U_bias])
    #predicted_preference = Flatten()(predicted_preference)
    
    model = Model(inputs=[U_input, M_input],outputs=predicted_preference)
    return model

#Define constants
TESTING_CSV_DIR = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
MODEL_WEIGHTS_FILE = 'MF120.h5'
K_FACTORS = 120
max_userid = 6040
max_movieid  = 3952

#Import data


test = pd.read_csv(TESTING_CSV_DIR+'train.csv', usecols=['TestDataID', 'UserID', 'MovieID'])
print (len(test), 'descriptions of', max_movieid, 'movies loaded.')


#Predict
trained_model = MF(max_userid, max_movieid, K_FACTORS)

trained_model.load_weights(MODEL_WEIGHTS_FILE)


prediction = trained_model.predict([test['UserID'].values,test['MovieID'].values])

ids = test['TestDataID'].values

with open(OUTPUT_PATH,'w') as sm:
    print('\"TestDataID\",\"Rating\"',file=sm)
    for i in range(len(ids )):
        print('\"%d\",\"%.1f\"'%(ids[i],prediction[i]),file=sm)

