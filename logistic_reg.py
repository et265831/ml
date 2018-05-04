import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

random.seed(1)
dataset=  datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset["data"],dataset["target"],test_size=0.3, random_state=0)
x_train = x_train.transpose()
x_test = x_test.transpose()
y_train = y_train.reshape(1,-1)
y_test = y_test.reshape(1,-1)
feature_count = x_train.shape[0]
theta = np.zeros((feature_count,1))
bias = random.uniform(0,1)

print("X.shape = {}, Y.shape = {}, theta.shape = {}".format(x_train.shape,y_train.shape,theta.shape))
def sigmoid(theta, x, bias):
    z = np.matmul(theta.transpose(),x)+bias
    y_hat = 1/(1+np.exp(-1*z))
    return y_hat
def loss_function(y, y_hat):
    num_of_samples = y.shape[1]
    loss = ( y * np.log(y_hat) + (1-y) * np.log(1-y_hat) )*-1
    loss = np.sum(loss) / num_of_samples
    return loss
y_hat = sigmoid(theta,x_train,bias)
loss_function(y_train, y_hat)
# theta.shape = n, 1
# X.shape = n, m -> each column is an instance. each row is a feature
# theta.T * X shape = 1, m
# LOSS FUNCTION = J = ( 1/y * log(y_hat) + 1/(1-y) * log(1-y_hat) ) * -1
# y_hat = sigmoid(z) = 1/(1+e^-z)
# z = theta.transpose dot x + b

# dJ/dy_hat = -y/y_hat + (1-y)/(1-y_hat)
# dy_hat/dz = y_hat * (1-y_hat)
# dz/dtheta = x
# dJ/dtheta = (y_hat-y) * x




