#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:17:35 2017

@author: wty
"""

import numpy as np 
import scipy  
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def sigmoid(inX):  
    return 1.0 / (1 + np.exp(-inX))

def h(w,X):
    return sigmoid(X.dot(w))

def L(w,X,Y,lamda=0.0):
    num_records,num_features  = np.shape(X)  
    
    hx = h(w,X)
    regulation_loss = 1.0/2 * lamda * w.transpose().dot(w)
    loss = 1.0/2 * 1.0/(1 - -1) * 1.0/num_records * (-np.log(hx).transpose().dot(1+Y) - np.log(1-hx).transpose().dot(1-Y))\
            + regulation_loss
            
    #loss = 
    return loss[0][0]

def g(w,X,Y,lamda = 0.0):
    num_records,num_features  = np.shape(X)  
    
    # L2 norm
    return 1.0/num_records * X.transpose().dot(2*h(w,X)-(Y+1)) \
            + lamda * w

X_train, Y_train = load_svmlight_file("./resources/a9a")
X_test, Y_test = load_svmlight_file("./resources/a9a.t")

X_train = scipy.sparse.hstack(\
    (scipy.sparse.csr_matrix(np.ones((len(Y_train),1))),X_train))
X_test = scipy.sparse.hstack(\
    (scipy.sparse.csr_matrix(np.ones((len(Y_test),1))),X_test))
#Something wrong with this dataset
X_test = scipy.sparse.hstack(\
    (X_test, scipy.sparse.csr_matrix(np.zeros((len(Y_test),1)))))

X_train = X_train.tocsr()
X_test = X_test.tocsr()

Y_train = Y_train.reshape((len(Y_train),1))
Y_test = Y_test.reshape((len(Y_test),1))

train_size,num_features  = np.shape(X_train)

# initialize w
w = np.random.normal(size=(num_features,1))

lamda = 0.1
eta = 0.1
threshold=0.5
max_iterate = 50
batch_size = 5000

loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []

for epoch in range(max_iterate):
    loss_train.append(L(w,X_train,Y_train,lamda))
    loss_test.append(L(w,X_test,Y_test,lamda))
    starts = [i*batch_size for i in range(math.ceil(train_size/batch_size))]
    ends = [i*batch_size for i in range(1,math.ceil(train_size/batch_size))]
    ends.append(train_size)
    for start, end in zip(starts, ends):
        w = w - eta * g(w,X_train[start:end,:],Y_train[start:end,:],lamda)
    
fig, ax = plt.subplots()
train_loss_line = ax.plot(range(max_iterate),loss_train,label='train loss')
test_loss_line = ax.plot(range(max_iterate),loss_test,label='test loss')
plt.legend()
ax.set(xlabel='Epoch', ylabel='Loss with l2 norm')
plt.show()
    








