# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# to make this notebook's output stable across runs
np.random.seed(42)

### Build 3D Model
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
X_centered[0]

U, s, V = np.linalg.svd(X_centered)
V.T[0]

c1 = V.T[:,0]
c1
c2 = V.T[:,1]
c2

m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

W2 = V.T[:,:2]
W2
X2D = X_centered.dot(W2)
X2D

plt.plot(X[:,0], X[:,1],X[:,2], 'b.')
plt.plot(X2D[:,0], X2D[:,1], 'b.')

Axes3D.plot(xs=X[:,0], ys=X[:,1], zs=X[:,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1],X[:,2], marker = m)
plt.show()

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

xs.shape
X[:,0].shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in X:
    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    ax.scatter(xs, ys, zs, marker=m)


### PCA for dimension reduction 
from sklearn.decomposition import PCA

pca = PCA (n_components=2)
X2D = pca.fit_transform(X)

pca.components_

pca.explained_variance_


### Incremental PCA
from six.moves import urllib
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)

X_train_reduced = inc_pca.transform(X_train)
X_train_reduced.shape

X_train_reduced[0]

### Kernel PCA
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=.04)
X_reduced = rbf_pca.fit_transform(X)