
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

def get_eigenvalues(data):
    SAMPLE = 300 # sample some points to estimate
    NEIGHBOR = 400 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

# Train a linear SVR

npzfile = np.load('large_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1)
svr.fit(X, y)
joblib.dump(svr,'model.sav')


# In[ ]:



