
# coding: utf-8

# In[86]:

import word2vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from adjustText import adjust_text
from nltk.corpus import stopwords
import string
import random

stop = set(stopwords.words('english'))
english_punctuations = set(string.punctuation)

# merge all book into one txt
s = ["Book 1 - The Philosopher's Stone_djvu",
     "Book 2 - The Chamber of Secrets_djvu",
     "Book 3 - The Prisoner of Azkaban_djvu",
     "Book 4 - The Goblet of Fire_djvu",
     "Book 5 - The Order of the Phoenix_djvu",
     "Book 6 - The Half Blood Prince_djvu",
     "Book 7 - The Deathly Hallows_djvu"]
ALL = ""
new = open("Harry Potter 1-7.txt",'w',encoding = 'utf-8')
for txt in s:
    f = open(txt+".txt",'r',encoding = 'utf-8')
    ALL =ALL+ f.read()
new.write(ALL)
#train word2vec model
word2vec.word2vec("Harry Potter 1-7.txt","Harry Potter 1-7.bin", 
             size=2000, window=10, sample='1e-4', hs=0,
             negative=5, threads= 4, iter_=1000, min_count=10, alpha=0.025,
             debug=2, binary=1, cbow=5, verbose=True)
#load model
model = word2vec.load("Harry Potter 1-7.bin")
x_voc = model.vocab
x_vec = model.vectors

#sample top 1000 frequentest word and do TSNE
N=1000
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(x_vec[0:N])
pos = pos = nltk.pos_tag(model.vocab)

#plot data
we_need = ['NNP','NN','JJ','NNS']
texts = []
colorset = ['r','c','g','y','m','b']
rcolour = np.random.randint(6,size = len(X_tsne))
for i , txt in enumerate(x_voc[0:N]):
    
    if ((pos[i][1] in we_need) and (str(pos[i][0]).isalpha())) and (not str(pos[i][0]).lower() in stop) :
        print(str(pos[i]))
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], 70,color=colorset[rcolour[i]])
        texts.append(plt.text(X_tsne[i, 0], X_tsne[i, 1], txt ,size = 15 ))
plt.gcf().set_size_inches(37, 21)    
plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))+' iterations')
plt.savefig("vis.png",dpi = 500)
plt.close()

