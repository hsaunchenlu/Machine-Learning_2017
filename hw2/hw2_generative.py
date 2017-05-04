
# coding: utf-8


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


def Accuracy(Xva,Yva, mu1, mu2, shared_sigma, N1, N2):
    result = predict(Xva, mu1, mu2, shared_sigma, N1, N2)
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



def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)



def feature_filter(Xtr,col):
    return np.delete(Xtr,col,axis = 1)



def predict(Xte, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = Xte.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y



def train(Xtr,Ytr):
    datalength = len(Xtr)
    fnumber = len(Xtr[0])
    
    cnt1 = 0
    cnt2 = 0
    mu1 = np.zeros((fnumber,))
    mu2 = np.zeros((fnumber,))
    for i in range(datalength):
        if Ytr[i] == 1:
            mu1 += Xtr[i]
            cnt1 += 1
        else:
            mu2 += Xtr[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2
    
    sigma1 = np.zeros((fnumber,fnumber))
    sigma2 = np.zeros((fnumber,fnumber))
    for i in range(datalength):
        if Ytr[i] == 1:
            sigma1 += np.dot(np.transpose([Xtr[i] - mu1]), [(Xtr[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([Xtr[i] - mu2]), [(Xtr[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / datalength) * sigma1 + (float(cnt2) / datalength) * sigma2
    
    return mu1, mu2, shared_sigma, cnt1, cnt2

############
#   main   #
############

#parameter setting

#s1 = 2900
#s2 = 2999
#col = [i for i in range(32,38)]  #drop marry situation


#data processing

X = pd.read_csv(Train_X,encoding = 'big5')
Y = pd.read_csv(Train_Y,encoding = 'big5',names = ['haha'])
Xtr = np.array(X,float)
Ytr = np.array(Y,float)
Ytr = np.reshape(Ytr,(len(Ytr)))


#Xtr = fscaling(Xtr)
#Xtr = feature_filter(Xtr,col)
#Xva = fscaling(Xva)
#Xva = feature_filter(Xva,col)


#training

[mu1, mu2, shared_sigma, N1, N2] = train(Xtr,Ytr)
print(Accuracy(Xtr,Ytr, mu1, mu2, shared_sigma, N1, N2))


#predicting

Xte = pd.read_csv(Test_X,encoding ='big5')
Xte = np.array(Xte,float)
#Xte = fscaling(Xte)
#Xte = feature_filter(Xte,col)

result = predict(Xte, mu1, mu2, shared_sigma, N1, N2)
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
#print(Final)
Final.to_csv(Res_path,index = False)





