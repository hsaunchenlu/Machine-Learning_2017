
# coding: utf-8

# In[8]:

import numpy as np
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import sys

inputpath = sys.argv[1]
outputpath = sys.argv[2]


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


# predict
model = joblib.load('model.sav')
testdata = np.load(inputpath)




test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = model.predict(test_X)

with open(outputpath, 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        string = str(i) + ',' + str(np.log(d)) + '\n'
        f.write(string)

