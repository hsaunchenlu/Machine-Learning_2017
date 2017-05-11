
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
from sklearn.externals import joblib
import sys
# predict
inputpath = sys.argv[1]
outputpath = sys.argv[2]


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

