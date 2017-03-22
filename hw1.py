
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import csv

Train_D = sys.argv[1]
Test_D = sys.argv[2]
Res_path = sys.argv[3]
#rp=os.path.split(Res_path)


data = pd.read_csv(Train_D,encoding = "big5")
variables = 18



def RMSE(Weight,Bias,X,Y,TDataMonth,TDataLeng):
    cost = 0.0
    for i in range(TDataMonth):
        for j in range(TDataLeng):
            cost = cost + (Y[i*TDataLeng+j] - Bias-np.dot(Weight,X[i,j,0:variables]))**2.0
    error = (cost/(TDataMonth*TDataLeng))**(0.5)
    return error


def fscaling(V):
    FS = (V-np.mean(V))/(np.max(V)-np.min(V))
    return FS


def Tfscaling(V,TR):
    FS = (V-np.mean(TR))/(np.max(TR)-np.min(TR))
    return FS



def TFprocess(data,S):
    V = data[data["測項"]==S]		#讀取所選資料
    V = V.drop(['日期','測站','測項'],axis = 1)
    V = np.array(V,float)		
    v = np.reshape(V,(12,480))		#將資料照月份分好
    datarow = len(v)			#月份
    datacol = len(v[0])			#資料筆數
    DSV = np.zeros((datarow,datacol-9,9),float) 
    for i in range(0,datarow):		#將輸入特徵每九小時一筆分好
        for j in range(0,datacol-9):
            for k in range(0,9):
                DSV[i,j,k] = v[i,j+k]
    return DSV		#回傳feature陣列


def TLprocess(data,S):
    V = data[data["測項"]==S]
    V = V.drop(['日期','測站','測項'],axis = 1)
    V = np.array(V,float)
    v = np.reshape(V,(12,480))
    datarow = len(v)
    datacol = len(v[0])
    DSV = np.zeros(datarow*(datacol-9),float)
    for i in range(0,datarow):
        for j in range(0,datacol-9):
            DSV[i*(datacol-9)+j] = v[i,j+9]      #target
    return DSV



def StackData(X,S,data):
    if len(X) == 0:
        X = TFprocess(data,S)
        return fscaling(X)
    else:
        V = TFprocess(data,S)
        X = np.dstack([X,fscaling(V)])
        return X


def TStackData(Xt,S,tdata):
    Vt = tdata[tdata['測項']==S]
    Vt=Vt.drop(['1','測項'],axis = 1)
    Vt = np.array(Vt,float)
    TR = data[data["測項"]==S]
    TR = TR.drop(['日期','測站','測項'],axis = 1)
    TR = np.array(TR,float)
    if len(Xt) == 0:
        return Tfscaling(Vt,TR)
    else:
        Xt = np.hstack([Xt,Tfscaling(Vt,TR)])
        return Xt




X = []
X = StackData(X,"PM2.5",data)
X = StackData(X,"PM10",data)
#print(X)
Y = TLprocess(data,"PM2.5")
TDataDimen = variables#len(X[0,0])
TDataLeng = len(X[0])
TDataMonth = len(X)
#print(Y)



# train model y = b+wx+Lw^2
Bias = 10.0
#Xk is {pm_25 if k = 0~8} {pm_10 if k = 9~17} {so2 if k = 18~26} {no2 if k = 27~35}
#{nox if k = 36~44} {o3 if k = 45~53}
Weight = np.zeros(TDataDimen,float)

lr = 1
itera = 10000
lamda = 0.00001

B_lr = 0.0
W_lr = np.zeros(TDataDimen,float)
error_his = []
times = []

for it in range(itera):
    B_grad = 0.0
    W_grad = np.zeros(TDataDimen,float)
    for i in range(TDataMonth):
        for j in range(0,TDataLeng):
            L = Y[i*TDataLeng+j]-(Bias+np.dot(X[i,j,0:variables],Weight))
            B_grad = B_grad - 2.0*L*1.0
            W_grad = W_grad - 2.0*L*X[i,j,0:variables] + lamda*2.0*Weight

    B_lr = B_lr+B_grad**2
    W_lr = W_lr+W_grad**2
    

    # Update parameters
    Bias = Bias - lr/np.sqrt(B_lr) * B_grad
    Weight = Weight - lr/np.sqrt(W_lr)*W_grad
    #error=RMSE(Weight,Bias,X,Y,TDataMonth,TDataLeng)
    #error_his.append(error)
    #times.append(it)
#plt.scatter(times[2:],error_his[2:])
#plt.show()



print('Bias :',Bias)
print('Weight:',Weight)



print("Root Mean Square Error:",RMSE(Weight,Bias,X,Y,TDataMonth,TDataLeng))


tdata = pd.read_csv(Test_D,encoding = "big5",names = ['1','測項','3','4','5','6','7','8','9','10','11'])

PM_25t = tdata[tdata['測項']=="PM2.5"]
index_temp = np.array(PM_25t)
index = index_temp[:,0]

Xt = []
Xt = TStackData(Xt,"PM2.5",tdata)
Xt = TStackData(Xt,"PM10",tdata)


TrDataDimen = len(Xt[0])
TrDataLeng = len(Xt)



result = np.zeros((TrDataLeng),float)
for i in range(TrDataLeng):
    result[i] = Bias + np.dot(Weight,Xt[i,0:variables])
    



final = [index,result]
final = np.transpose(final)
Final = pd.DataFrame(final,columns = ['id','value'] )
print(Final)
Final.to_csv(Res_path,index = False)





