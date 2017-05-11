
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
from sklearn.externals import joblib
# Train a linear SVR

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1)
svr.fit(X, y)
joblib.dump(svr,'model.sav')


# In[ ]:



