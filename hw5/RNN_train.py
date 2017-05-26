
# coding: utf-8

# In[9]:

import numpy as np
import string
import sys
import h5py 
import keras.backend as K 
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.layers import GRU,LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint



train_path = 'train_data.csv'
test_path = 'test_data.csv'
output_path = 'result.csv'

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding =  'ISO-8859-1') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r',encoding = 'utf8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

def create_model(num_words,embedding_dim,embedding_matrix,max_article_length):
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(256,activation='tanh', 
                   recurrent_initializer = 'orthogonal',
                   bias_initializer='ones',
                   recurrent_dropout=0.1,
                   dropout=0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()
    return model

    
###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.5
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################


### read training and testing data
(Y_data,X_data,tag_list) = read_data(train_path,True)
(_, X_test,_) = read_data(test_path,False)
all_corpus = X_data + X_test
print ('Find %d articles.' %(len(all_corpus)))
    
### tokenizer for all data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
word_index = tokenizer.word_index

### convert word sequences to index sequence
print ('Convert to index sequences.')
train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

### padding to equal length
print ('Padding sequences.')
train_sequences = pad_sequences(train_sequences)
max_article_length = train_sequences.shape[1]
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
###
train_tag = to_multi_categorical(Y_data,tag_list) 
    
### split data into training set and validation set
(X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
### get mebedding matrix from glove
print ('Get embedding dict from glove.')
embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
print ('Found %s word vectors.' % len(embedding_dict))
num_words = len(word_index) + 1
print ('Create embedding matrix.')
embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

### build model
print ('Building model.')
model = create_model(num_words,embedding_dim,embedding_matrix,max_article_length)

adam = Adam(lr=0.001,decay=1e-6)#,clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
   
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 5, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            monitor='val_f1_score',
                            mode='max')
hist = model.fit(X_train, Y_train, 
                validation_data=(X_val, Y_val),
                epochs=nb_epoch, 
                batch_size=batch_size,
                callbacks=[earlystopping,checkpoint])

model2 = create_model(num_words,embedding_dim,embedding_matrix,max_article_length)
model2.load_weights('best.hdf5')
Y_pred = model2.predict(test_sequences)
    
thresh = 0.5
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)



# In[10]:

import pandas as pd


# In[11]:

log = np.array(pd.DataFrame(hist.history))


# In[12]:

import matplotlib.pyplot as plt


# In[13]:

epoch = hist.epoch
tr_f1 = log[:,0]
val_f1 = log[:,2]


# In[14]:

plt.plot(epoch,tr_f1)
plt.plot(epoch,val_f1)
plt.show()


# In[15]:

Y_pred[0]


# In[33]:

thresh = 0.5
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)


# In[34]:

for i in range(len(Y_pred_thresh)):
    N = 0
    for j in range(38):
        if Y_pred_thresh[i][j] ==1:
            N +=1
    print(N)


# In[ ]:



