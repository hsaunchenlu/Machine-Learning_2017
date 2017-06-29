
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
from math import floor
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib


# In[2]:

def get_data(path):
    data = pd.read_csv(path)
    
    if 'week_start_date' in np.array(data.columns):
        data = data.drop('week_start_date',axis = 1)
    if 'weekofyear' in np.array(data.columns):
        data = data.drop('weekofyear',axis = 1)    
    if 'year' in np.array(data.columns):
        data = data.drop('year',axis = 1)
    #reanalysis
    if 'reanalysis_avg_temp_k' in np.array(data.columns):
        data = data.drop('reanalysis_avg_temp_k',axis = 1)
    if 'reanalysis_precip_amt_kg_per_m2' in np.array(data.columns):
        data = data.drop('reanalysis_precip_amt_kg_per_m2',axis = 1)
    if 'reanalysis_sat_precip_amt_mm' in np.array(data.columns):
        data = data.drop('reanalysis_sat_precip_amt_mm',axis = 1)
    if 'reanalysis_air_temp_k' in np.array(data.columns):
        data = data.drop('reanalysis_air_temp_k',axis = 1)
    #if 'reanalysis_dew_point_temp_k' in np.array(data.columns):
    #    data = data.drop('reanalysis_dew_point_temp_k',axis = 1)
    #if 'reanalysis_max_air_temp_k' in np.array(data.columns):
    #    data = data.drop('reanalysis_max_air_temp_k',axis = 1)
    #if 'reanalysis_min_air_temp_k' in np.array(data.columns):
    #    data = data.drop('reanalysis_min_air_temp_k',axis = 1)
        #
    if 'reanalysis_relative_humidity_percent' in np.array(data.columns):
        data = data.drop('reanalysis_relative_humidity_percent',axis = 1)
    #if 'reanalysis_specific_humidity_g_per_kg' in np.array(data.columns):
    #    data = data.drop('reanalysis_specific_humidity_g_per_kg',axis = 1)
    if 'reanalysis_tdtr_k' in np.array(data.columns):
        data = data.drop('reanalysis_tdtr_k',axis = 1)
    # station
    
    if 'station_diur_temp_rng_c' in np.array(data.columns):
        data = data.drop('station_diur_temp_rng_c',axis = 1)
    if 'station_precip_mm' in np.array(data.columns):
        data = data.drop('station_precip_mm',axis = 1)
    if 'station_max_temp_c' in np.array(data.columns):
        data = data.drop('station_max_temp_c',axis = 1)
    if 'station_min_temp_c' in np.array(data.columns):
        data = data.drop('station_min_temp_c',axis = 1)
    if 'station_precip_mm' in np.array(data.columns):
        data = data.drop('station_precip_mm',axis = 1)
        #
    if 'station_avg_temp_c' in np.array(data.columns):
        data = data.drop('station_avg_temp_c',axis = 1)
    
    # ndvi

    if 'ndvi_ne' in np.array(data.columns):
        data = data.drop('ndvi_ne',axis = 1)
    if 'ndvi_nw' in np.array(data.columns):
        data = data.drop('ndvi_nw',axis = 1)
    if 'ndvi_se' in np.array(data.columns):
        data = data.drop('ndvi_se',axis = 1)
    if 'ndvi_sw' in np.array(data.columns):
        data = data.drop('ndvi_sw',axis = 1)

    #precipitation
    #
    if 'precipitation_amt_mm' in np.array(data.columns):
        data = data.drop('precipitation_amt_mm',axis = 1)
    
    dmax = data.max()
    dmin = data.min()
    #data = data.fillna(method = 'ffill')
    data = data.fillna(0)
    ndata = np.array(data)
    return ndata,dmax,dmin 

#clean data that have too much missing data
def clean_train(feature,label):
    thresh = len(feature[0])-1
    row2del =[]
    nonzero = np.count_nonzero(feature,axis = 1)
    for i in range(len(feature)):
        if nonzero[i]<thresh:
            row2del.append(i)
    newfeature = np.delete(feature,row2del,axis=0)
    newlabel = np.delete(label,row2del,axis=0)
    return newfeature,newlabel


def feature_scaling(pre,dmax,dmin):
    for i in range(len(pre)):
        for j in range(len(pre[0])):
            if np.isreal(pre[i][j]):
                if pre[i][j] >= dmin[j]:
                    pre[i][j] = (pre[i][j]-dmin[j])/(dmax[j]-dmin[j])
     
    return pre



def split_city(data):
    sj_data = []
    iq_data = []
    for i in range(len(data)):
        if data[i][0] == 'sj':
            sj_data.append(data[i,1:])
        else:
            iq_data.append(data[i,1:])
    return np.array(sj_data,dtype = 'float32'),np.array(iq_data,dtype = 'float32')

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.seed(1)
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

def shift_trfeature(data,n):
    return data[:-n-1]
def shift_tefeature(data,n):
    temp = np.zeros((n,len(data[0])),dtype = 'float32')    
    temp = data[-n-1:-1]
    data[n:-1] = data[:-n-1]
    data[0:n] = temp
    return data
def delete_label(data,n):
    return data[n:-1]


# In[3]:

trainpath = 'dengue_features_train.csv'
testpath = 'dengue_features_test.csv'
trainlabelpath = 'dengue_labels_train.csv'



[tr_data,tr_max,tr_min] = get_data(trainpath)
[te_data,te_max,te_min] = get_data(testpath)
[label_data,_,_] = get_data(trainlabelpath)

#feature scaling
realmax = np.maximum(tr_max,te_max)
realmin = np.minimum(tr_min,te_min)
feature_tr = feature_scaling(tr_data,realmax,realmin)
feature_te = feature_scaling(te_data,realmax,realmin)

#city split
[sj_trfeat,iq_trfeat] = split_city(feature_tr)
[sj_tefeat,iq_tefeat] = split_city(feature_te)
[sj_label,iq_label] = split_city(label_data)
sj_trlab = sj_label[:,-1].reshape(len(sj_label),1)
iq_trlab = iq_label[:,-1].reshape(len(iq_label),1)

#shift data by N weeks
N =7
sjshift_tr = shift_trfeature(sj_trfeat,N)
sjshift_te = shift_tefeature(sj_tefeat,N)
sjshift_la = delete_label(sj_trlab,N)

N =0
iqshift_tr = shift_trfeature(iq_trfeat,N)
iqshift_te = shift_tefeature(iq_tefeat,N)
iqshift_la = delete_label(iq_trlab,N)



# In[4]:

[sjnf,sjnl] = clean_train(sjshift_tr,sjshift_la)
[iqnf,iqnl] = clean_train(iqshift_tr,iqshift_la)
polydegree = 1
poly = PolynomialFeatures(polydegree,include_bias=False)
sjpoly = poly.fit_transform(sjnf)
sjpoly_te = poly.fit_transform(sjshift_te)

polydegree = 1
poly = PolynomialFeatures(polydegree,include_bias=False)
iqpoly = poly.fit_transform(iqnf)
iqpoly_te = poly.fit_transform(iqshift_te)


# In[5]:

###############
# Train model #
###############
#sj train

(sjX_train,sjY_train),(sjX_val,sjY_val) = split_data(sjpoly,sjnl,0.1)

sjlr = linear_model.LinearRegression()
sjlr.fit(sjX_train,sjY_train)

sjpre_val = sjlr.predict(sjX_val)
# if result is small than zero I set it zero
for i in range(len(sjpre_val)):
    if sjpre_val[i]<0:
        sjpre_val[i] = 0

print('MAE of sj city:',mean_absolute_error(sjY_val,sjpre_val))



#iq train

(iqX_train,iqY_train),(iqX_val,iqY_val) = split_data(iqpoly,iqnl,0.1)

iqlr = linear_model.LinearRegression()
iqlr.fit(iqX_train,iqY_train)

iqpre_val = iqlr.predict(iqX_val)
# if result is small than zero I set it zero
for i in range(len(iqpre_val)):
    if iqpre_val[i]<0:
        iqpre_val[i] = 0

print('MAE of iq city:',mean_absolute_error(iqY_val,iqpre_val))



#total 
true = np.vstack([sjY_val,iqY_val])
pred = np.vstack([sjpre_val,iqpre_val])

print('MAE of two city:',mean_absolute_error(true,pred))


# In[6]:

'''
sjpre_self = sjlr.predict(sjpoly)
iqpre_self = iqlr.predict(iqpoly)

fig1 = plt.figure(figsize=(20,4))
plt.plot(sjpre_self)
plt.plot(sjshift_la)
plt.savefig('sjresult',dpi = 300)
plt.show()

fig2 = plt.figure(figsize=(20,4))
plt.plot(iqpre_self)
plt.plot(iqshift_la)
plt.savefig('iqresult',dpi = 300)
plt.show()
'''


# In[60]:

#predict

sjpredict = np.array(sjlr.predict(sjpoly_te))
sjpredict = np.int32(sjpredict).reshape(len(sjpredict),1)
for i in range(len(sjpredict)):
    if sjpredict[i]<0:
        sjpredict[i] = 0

iqpredict = np.array(iqlr.predict(iqpoly_te))
iqpredict = np.int32(iqpredict).reshape(len(iqpredict),1)
for i in range(len(iqpredict)):
    if iqpredict[i]<0:
        iqpredict[i] = 0
        
total_predict = np.vstack([sjpredict,iqpredict])

teL = pd.read_csv(testpath)

teL = np.array(teL)
teL = teL[:,0:3]

final = np.hstack([teL,total_predict])

col = ['city','year','weekofyear','total_cases']
Final = pd.DataFrame(final,columns=col)
Final.to_csv('predict.csv',index = False)


# In[ ]:

'''
joblib.dump(sjlr, 'sj_model.pkl')
joblib.dump(iqlr, 'iq_model.pkl')
'''

