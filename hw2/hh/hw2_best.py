
# coding: utf-8

# In[123]:

import pandas as pd
import numpy as np
import scipy as cp
import sys

Train = sys.argv[1]
Test = sys.argv[2]
Train_X = sys.argv[3]
Train_Y = sys.argv[4]
Test_X = sys.argv[5]
Res_path = sys.argv[6]


# In[124]:

#data cleaning


# In[125]:

def data_split(s1,s2,data):
    return np.vstack([data[:s1],data[s2:]]) ,data[s1:s2]


# In[126]:

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
    


# In[127]:

def data_clean(X,Y,b,w):
    result = b+np.dot(Xva,w)
    for i in range(len(Xva)):
        if result[i] >= 0.5:
            if Y[i] ==0:
                Y[i] = 1
        else:
            if Yva[i] ==1:
                Y[i]=0
    return Y
                


# In[128]:

def data_boot(X,Y,b,w):
    result = b+np.dot(Xva,w)
    for i in range(len(Xva)):
        if result[i] >= 0.5:
            if Y[i] ==0:
                X = np.delete(X,[i],axis=0)
                Y = np.delete(Y,[i],axis=0)
        else:
            if Yva[i] ==1:
                X = np.delete(X,[i],axis=0)
                Y = np.delete(Y,[i],axis=0)
    return X,Y


# In[129]:

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

# In[130]:

def feature_filter(Xtr,col):
    return np.delete(Xtr,col,axis = 1)


# In[131]:


# In[132]:

s1 = 2000
s2 = 8500
col = [i for i in range(32,38)]  #drop marry situation
#col2 = [i for i in range(7,15)]
#col = np.hstack([col2,col1])

#batch_num = 10


# In[133]:

X = pd.read_csv(Train_X,encoding = 'big5')
Y = pd.read_csv(Train_Y,encoding = 'big5',names = ['haha'])
X = np.array(X,float)
Y = np.array(Y,float)
[Xtr,Xva] = data_split(s1,s2,X)
[Ytr,Yva] = data_split(s1,s2,Y)
Ytr = np.reshape(Ytr,(len(Ytr)))
Yva = np.reshape(Yva,(len(Yva)))



# In[134]:

Xtr = fscaling(Xtr)
Xtr = feature_filter(Xtr,col)

Xva = fscaling(Xva)
Xva = feature_filter(Xva,col)

datalength = len(Xtr)
fnumber = len(Xtr[0])


# In[135]:

#Xtr = batch(Xtr,batch_num)
#Ytr = batch(Ytr,batch_num)
#print(Xtr)


# In[136]:

#len(Ytr[3])


# In[137]:


b = 10
w = np.zeros(fnumber)
lr = 10
b_lr = 0
w_lr = np.zeros(fnumber)
lamda = 0.01
iteration = 20000

for it in range(iteration):
    #for bn in range(1,batch_num-1):
            
        b_grad = 0
        w_grad = np.zeros(fnumber)
        
        
        z = b+np.dot(Xtr,w)

        
        sigmoi = np.clip(1/(1+np.exp(-z)),0.000000000001,0.99999999999)
        
        w_grad = -np.dot(Xtr.T,(Ytr-sigmoi.T)) + 2*lamda*w
        b_grad = -np.sum((Ytr-sigmoi.T))
    
        b_lr = b_lr+b_grad**2
        w_lr = w_lr+w_grad**2
    
    
        b = b - lr/np.sqrt(b_lr) * b_grad
        w = w - lr/np.sqrt(w_lr) * w_grad
        
        if it%1000 == 0:
            print('iter:',it)
            print('Accuracy:',Accuracy(Xtr,Ytr,b,w))
    
        


    


# In[138]:

#print('Bias :',b)
#print('Weight:',w)




print("Training Accuracy:",Accuracy(Xtr,Ytr,b,w))
print("Training Accuracy:",Accuracy(Xva,Yva,b,w))




Xte = pd.read_csv(Test_X,encoding ='big5')
Xte = np.array(Xte,float)
Xte = fscaling(Xte)
Xte = feature_filter(Xte,col)


# In[ ]:

result = b + np.dot(Xte,w)
result = np.clip(1/(1+np.exp(-result)),0.000000000001,0.99999999999)
Result = np.zeros(len(Xte),int)
for i in range(len(Xte)):
        if result[i] >= 0.5:
            Result[i]=1      
        else:
            Result[i]=0  


# In[ ]:

final = []
num=1
while num <= len(Xte):
    final.append([str(num),Result[num-1]])
    num = num+1


# In[ ]:

Final = pd.DataFrame(final,columns = ['id','label'] )
#print(Final)
Final.to_csv(Res_path,index = False)


# In[ ]:



