
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Import Image
IMG = np.zeros([100,64,64])

for c in range(10):
    for i in range(10):
        #print( chr(65+c)+"0"+str(i)+".bmp" )
        img = Image.open( chr(65+c)+"0"+str(i)+".bmp" )
        Img = np.array([img.getpixel((i, j)) for j in range(64) for i in range(64)])
        Img=Img.reshape(64,64)
        IMG[10*c+i] = Img



# Implement PCA

def pca(dataMat,n):  
    meanVal= np.mean(dataMat,axis=0)     #find the mean  
    newData= dataMat-meanVal 
    covMat=np.cov(newData,rowvar=0)    #convariance matrix
      
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#find eigenvalue and eigenvevtor
    eigValIndice=np.argsort(eigVals)            # sort eigenvalue 
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   # Top n value index
    n_eigVect=eigVects[:,n_eigValIndice]        # Top n eigenvector 
    lowDDataMat=newData*n_eigVect               # Eigenface
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  # Reconstruction 
    return lowDDataMat,reconMat  

# Image average
img_avg = np.zeros([64,64])

for i in range(10):
    for j in range(10):
        img_avg += IMG[10*i+j]

img_avg = img_avg/100
plt.imshow(img_avg,cmap = 'gray')
plt.axis('off')
plt.savefig("avgface.png",dpi = 1000)
plt.show()

Data = IMG.reshape(100,64*64).T  # Image in row

#plot top 9 eigenfaces
[lo,re] = pca(Data,9)

for i in range(9):
    plt.subplot(331+i)
    plt.imshow(lo[:,i].reshape(64,64),cmap = 'gray')
    plt.axis('off')
plt.savefig("eigenface.png",dpi = 1000)    
plt.show()



# Recnostruction by top 5 eigenfaces
[lo_5,re_5] = pca(Data,5)

for i in range(100):
    ax = plt.subplot(10,10,i+1)
    ax.imshow(Data.T[i].reshape(64,64),cmap = 'gray')
    ax.axis('off')
plt.savefig("allface.png",dpi = 1000)    
plt.show()





for i in range(100):
    ax = plt.subplot(10,10,i+1)
    ax.imshow(re_5.T[i].reshape(64,64),cmap = 'gray')
    ax.axis('off')
plt.savefig("reconface.png",dpi = 1000)    
plt.show()

# find how many eigenfaces are enough
for i in range(100):   
    [lo_k,re_k] = pca(Data,i+1)
    err = np.sqrt(np.abs(np.mean(((Data.T-re_k.T))*((Data.T-re_k.T)).T)))/255*100
    print("top",i+1,"eigenface:",err,"%")

