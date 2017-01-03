import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

tf1.reset_default_graph()

x = tf1.Variable(3, name = 'x')
y = tf1.Variable(4, name = 'y')
f = x*x*y + y + 2

root_logdir = "/Users/catalinmates/Desktop/Python_Testing/TF/tf_logs"
tf1.summary.FileWriter(root_logdir, tf1.get_default_graph())



tf1.__version__

with tf1.Session() as ses:
    # Evaluate the tensor `f`.
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

init = tf1.global_variables_initializer()

with tf1.Session() as sess:
    init.run()
    result = f.eval()
result


### Managing Graphs
x1 = tf1.Variable(1)
x1.graph is tf1.get_default_graph()

graph = tf1.Graph()
with graph.as_default():
    x2 = tf1.Variable(2)

x2.graph is graph
x2.graph is tf1.get_default_graph


w = tf1.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf1.Session() as sess:
    print(y.eval())
    print(z.eval())

### Linear Regression with Tensorflow
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf1.constant(housing_data_plus_bias, dtype = tf1.float32, name = "X" )
y = tf1.constant(housing.target.reshape(-1,1), dtype = tf1.float32, name = "y")
XT = tf1.transpose(X)
theta = tf1.matmul(tf1.matmul(tf1.matrix_inverse( tf1.matmul( XT, X )), XT ), y)

with tf1.Session() as sess:
    theta_value = theta.eval()

print(theta_value)

### Gradient Descent 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing.data)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing_scaled]
housing_data_plus_bias[0]

n_epochs = 1000
learning_rate = .01

X = tf1.constant(housing_data_plus_bias, dtype = tf1.float32, name = "X")
y = tf1.constant(housing.target.reshape(-1,1), dtype = tf1.float32, name = "y")
theta = tf1.Variable(tf1.random_uniform([n+1,1],-1,1), name = 'theta')
y_pred = tf1.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf1.reduce_mean(tf1.square(error), name = "mse")
gradients = 2/m * tf1.matmul(tf1.transpose(X), error)
training_op = tf1.assign(theta, theta - learning_rate * gradients)
init = tf1.global_variables_initializer()
saver = tf1.train.Saver()


with tf1.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            save_path = saver.save(sess,"/Users/catalinmates/Desktop/Python_Testing/TF/tmp/my_model.ckpt")
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()
    save_path = saver.save(sess, "/Users/catalinmates/Desktop/Python_Testing/TF/tmp/my_model_final.ckpt")

best_theta
save_path

### Feeding data to the training algo (mini-batches)
A = tf1.placeholder(tf1.float32, shape = (None, 3))
B = A + 5

with tf1.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A: [[4,5,6], [7,8,9]]})

print(B_val_2)


X = tf1.placeholder(tf1.float32, shape = (None, n+1), name = "X")
y = tf1.placeholder(tf1.float32, shape = (None, 1), name = "y")

batch_size = 100
n_batches = int(np.ceil(m / batch_size))


theta = tf1.Variable(tf1.random_uniform([n+1,1],-1,1), name = 'theta')
y_pred = tf1.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf1.reduce_mean(tf1.square(error), name = "mse")
gradients = 2/m * tf1.matmul(tf1.transpose(X), error)
training_op = tf1.assign(theta, theta - learning_rate * gradients)
init = tf1.global_variables_initializer()
saver = tf1.train.Saver()



def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size = batch_size)
    X_batch = housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1,1)[indices]

    return X_batch, y_batch

with tf1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict = {X: X_batch, y:y_batch})

    best_theta = theta.eval()

best_theta

### Visualizing the Graph and Training using TensorBoard
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs/"
logdir = "{}/run-{}/"

mse_summary = tf1.summary.scalar('MSE', mse)
file_writer = tf1.summary.FileWriter('/Users/catalinmates/Desktop/Python_Testing/TF/tf_logs', tf1.get_default_graph())


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))


with tf1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()
best_theta



### Names Scopes
tf1.reset_default_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "/Users/catalinmates/Desktop/Python_Testing/TF/tf_logs"
logdir = "{}/run-{}/"

n_epochs = 1000
learning_rate = 0.01

X = tf1.placeholder(tf1.float32, shape = (None, n+1), name = "X")
y = tf1.placeholder(tf1.float32, shape = (None, 1), name = "y")
theta = tf1.Variable(tf1.random_uniform([n+1,1],-1,1), name = 'theta')
y_pred = tf1.matmul(X, theta, name = "predictions")

with tf1.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf1.reduce_mean(tf1.square(error), name = "mse")

optimizer = tf1.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

init = tf1.global_variables_initializer()

mse_summary = tf1.summary.scalar('MSE', mse)
file_writer = tf1.summary.FileWriter(root_logdir, tf1.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()
best_theta


### Modularity
tf1.reset_default_graph()

def relu(X):
    with tf1.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]),1)
        w = tf1.Variable(tf1.random_normal(w_shape), name = "weights", dtype = tf1.float32) 
        b = tf1.Variable(0, name = "bias", dtype = tf1.float32)
        z = tf1.add(tf1.matmul(X,w), b, name = "z")
        return tf1.maximum(z, 0, name = "relu")

n_features = 3
X = tf1.placeholder(tf1.float32, shape = (None, n_features), name = "X")
relus = [relu(X) for i in range(5)]
output = tf1.add_n(relus, name = "output")
file_writer = tf1.summary.FileWriter(root_logdir, tf1.get_default_graph())

