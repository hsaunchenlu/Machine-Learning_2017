
# coding: utf-8

# In[210]:

import pandas as pd
import numpy as np
import scipy as cp
from matplotlib import pyplot as plt
import sys
from math import floor 

Train = sys.argv[1]
Test = sys.argv[2]
Train_X = sys.argv[3]
Train_Y = sys.argv[4]
Test_X = sys.argv[5]
Res_path = sys.argv[6]


# In[211]:

def data_split(s1,s2,data):
    return np.vstack([data[:s1],data[s2:]]) ,data[s1:s2]


# In[212]:

def Accuracy(Xva,Yva,b,w):
    result = b+np.dot(Xva,w)
    result = np.clip(1/(1+np.exp(-result)),0.000000000001,0.99999999999)
    correct = 0
    for i in range(len(Xva)):
        if result[i] >= 0.5:
            if Yva[i] ==1:
                correct = correct+1
        else:
            if Yva[i] ==0:
                correct = correct+1
                
    accuracy = correct/len(Yva)
    return accuracy
    


# In[213]:

def fscaling(data):
    N = data.T
    f =  1/(np.max(N[0])-np.min(N[0]))
    for i in range(len(data)):
        data[i,0]= (data[i,0]-np.min(N[0]))*f
        
    f =  1/(np.max(N[1])-np.min(N[1]))
    for i in range(len(data)):
        data[i,1]= (data[i,1]-np.min(N[1]))*f
    
    f =  1/(np.max(N[3])-np.min(N[3]))
    for i in range(len(data)):
        data[i,3]= (data[i,3]-np.min(N[3]))*f
    
    f =  1/(np.max(N[4])-np.min(N[4]))
    for i in range(len(data)):
        data[i,4]= (data[i,4]-np.min(N[4]))*f
    
    f =  1/(np.max(N[5])-np.min(N[5]))
    for i in range(len(data)):
        data[i,5]= (data[i,5]-np.min(N[5]))*f
    return data


# In[214]:

def shuffle(X,Y):
    rand = np.arange(len(X))
    np.random.shuffle(rand)
    return (X[rand],Y[rand])


# In[215]:

def feature_filter(Xtr,col):
    return np.delete(Xtr,col,axis = 1)


# In[216]:

def train(Xtr,Ytr):
    datalength = len(Xtr)
    fnumber = len(Xtr[0])
    b = 0
    w = np.zeros(fnumber)
    lr = 10
    b_lr = 0
    w_lr = np.zeros(fnumber)
    lamda = 0.001
    iteration = 10000
    it_his = []
    ac_his = []
    
    batch_size = datalength
    batch_num = int(floor(datalength / batch_size))
    display_num = 1000

    for it in range(iteration):
        Xtr, Ytr = shuffle(Xtr, Ytr)
        for bn in range(batch_num):
            
            b_grad = 0
            w_grad = np.zeros(fnumber)
            X = Xtr[bn*batch_size:(bn+1)*batch_size]
            Y = Ytr[bn*batch_size:(bn+1)*batch_size]
        
            #z = b+np.dot(Xtr,w)
            z = b+np.dot(X,w)
            sigmoid = np.clip(1/(1+np.exp(-z)),0.000000000001,0.99999999999)
            
            w_grad = -np.dot(X.T,(Y-sigmoid.T)) + 2*lamda*w+0.0000000000000000000001
            b_grad = -np.sum((Y-sigmoid.T))
            
            
            
            #adagrad
            b_lr = b_lr+b_grad**2
            w_lr = w_lr+w_grad**2
            b = b - lr/np.sqrt(b_lr) * b_grad
            w = w - lr/np.sqrt(w_lr) * w_grad
            
        if it%display_num == 0:
            ac = Accuracy(Xtr,Ytr,b,w)
            print('iter:',it)
            print('Accuracy:',ac)
            it_his.append(it)
            ac_his.append(ac)
        #if Accuracy(Xtr,Ytr,b,w)> 0.858:
            #return b,w
        
    return b,w,it_his,ac_his


# In[279]:

s1 = 2000
s2 = 8500
#col = [i for i in range(32,38)]  #drop marry situation
col = [i for i in range(15,21)] #from
#col = 2
#batch_num = 10


# In[280]:

X = pd.read_csv(Train_X,encoding = 'big5')
Y = pd.read_csv(Train_Y,encoding = 'big5',names = ['haha'])
X = np.array(X,float)
Y = np.array(Y,float)
[X,Y] = shuffle(X,Y)
[Xtr,Xva] = data_split(s1,s2,X)
[Ytr,Yva] = data_split(s1,s2,Y)
Ytr = np.reshape(Ytr,(len(Ytr)))
Yva = np.reshape(Yva,(len(Yva)))



# In[281]:

Xtr = fscaling(Xtr)
Xtr = feature_filter(Xtr,col)

Xva = fscaling(Xva)
Xva = feature_filter(Xva,col)


# In[282]:


[b,w,it,Ac] = train(Xtr,Ytr)
    


# In[283]:

print('Bias :',b)
print('Weight:',w)


# In[285]:

print('Training Accuracy:',Accuracy(Xtr,Ytr,b,w))


# In[286]:

print('Training Accuracy:',Accuracy(Xva,Yva,b,w))



Xte = pd.read_csv(Test_X,encoding ='big5')
Xte = np.array(Xte,float)
Xte = fscaling(Xte)
Xte = feature_filter(Xte,col)




result = b + np.dot(Xte,w)
result = np.clip(1/(1+np.exp(-result)),0.000000000001,0.99999999999)
Result = np.zeros(len(Xte),int)
for i in range(len(Xte)):
        if result[i] >= 0.5:
            Result[i]=1      
        else:
            Result[i]=0  




final = []
num=1
while num <= len(Xte):
    final.append([str(num),Result[num-1]])
    num = num+1




Final = pd.DataFrame(final,columns = ['id','label'] )
print(Final)
Final.to_csv(Res_path,index = False)




