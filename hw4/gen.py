
# coding: utf-8

# In[ ]:

import numpy as np
from sklearn.neighbors import NearestNeighbors


def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


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


# generate some data for training
X = []
y = []
for i in range(60):
    dim = i + 1
    for N in [10000, 20000, 50000, 80000, 100000]:
        layer_dims = [np.random.randint(60, 80), 100]
        data = gen_data(dim, layer_dims, N).astype('float32')
        eigenvalues = get_eigenvalues(data)
        X.append(eigenvalues)
        y.append(dim)

X = np.array(X)
y = np.array(y)

np.savez('large_data.npz', X=X, y=y)

