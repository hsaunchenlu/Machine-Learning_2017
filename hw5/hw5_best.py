
# coding: utf-8

# In[3]:

import pickle
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


test_path = sys.argv[1]
output_path = sys.argv[2]

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
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

(_, X_test,_) = read_data(test_path,False)

with open('max_article_length.pkl','rb') as mal:
    max_article_length = pickle.load(mal)
with open('label_mapping.pkl','rb') as lm:
    tag_list = pickle.load(lm)
with open('word_index.pkl','rb') as wi:
    word_index = pickle.load(wi)

tokenizer = Tokenizer()
### convert word sequences to index sequence
print ('Convert to index sequences.')
tokenizer.word_index = word_index
test_sequences = tokenizer.texts_to_sequences(X_test)  

### padding to equal length
print ('Padding sequences.')
#train_sequences = pad_sequences(train_sequences)
#max_article_length = train_sequences.shape[1]
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
print ('Get embedding dict from glove.')
embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
print ('Found %s word vectors.' % len(embedding_dict))
num_words = len(word_index) + 1
print ('Create embedding matrix.')
embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

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


# In[ ]:



