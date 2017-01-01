import numpy as np 
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons

iris = datasets.load_iris()
X = iris['data'][:, (2,3)]
y = (iris['target']==2).astype(np.float64)

X.shape

def plot_dataset(X, y):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'g^')
    plt.xlabel(r"$x_1$", fontsize = 10)
    plt.ylabel(r"$x_2$", fontsize = 10, rotation = 0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    y0s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, y0s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap = plt.cm.brg, alpha=.2)
    plt.contour(x0, x1, y_decision, cmap = plt.cm.brg, alpha  = .1)


svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=10, loss = 'hinge'))]
)

svm_clf.fit(X, y)
svm_clf.predict([[1, 0]])

axes = [1,7,0,2.5]
plot_predictions(svm_clf, axes)
plot_dataset(X,y)

svc_poly_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc_clf", SVC(kernel = "poly", degree = 1, coef0 = 10, C = 10))
])

svc_poly_clf.fit(X,y)
axes = [1,7,0,2.5]
plot_predictions(svc_poly_clf, axes)
plot_dataset(X,y)


### RBF
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel = "rbf", gamma = .01, C = 1))
])
rbf_kernel_svm_clf.fit(X,y)
plot_predictions(rbf_kernel_svm_clf, axes)
plot_dataset(X,y)

### SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd_clf', SGDClassifier(alpha=1 ))
])
sgd_clf.fit(X,y)
plot_predictions(sgd_clf, axes)
plot_dataset(X,y)



### Nonlinear SVM Classification
import matplotlib.pyplot as plt


polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree = 3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss = "hinge"))
])

X, y = make_moons(n_samples = 1000, noise = .15, random_state = 42)
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'g^')
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize = 10)
    plt.ylabel(r"$x_2$", fontsize = 10, rotation = 0)

plot_dataset(X, y, [-2, 2, -1, 1.5])

polynomial_svm_clf.fit(X, y)
polynomial_svm_clf.decision_function()

axes = [-2, 2, -1, 1.5]

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    y0s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, y0s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap = plt.cm.brg, alpha=.2)
    plt.contour(x0, x1, y_decision, cmap = plt.cm.brg, alpha  = .1)

plot_predictions(polynomial_svm_clf, axes)
plot_dataset(X, y, [-2, 2, -1, 1.5])


### Polynomial Kernel

from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel = "poly", degree = 10, coef0 = 10, C = 5))
))
poly_kernel_svm_clf.fit(X,y)

plot_predictions(poly_kernel_svm_clf, axes)
plot_dataset(X, y, axes)

### Gaussian RBF Kernel
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel = "rbf", gamma = 5, C = 1000))
])
rbf_kernel_svm_clf.fit(X,y)

plot_predictions(rbf_kernel_svm_clf, axes)
plot_dataset(X, y, axes)

### SVM Regression
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X,y)