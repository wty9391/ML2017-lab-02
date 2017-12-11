#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:58:35 2017

@author: wty
"""
import numpy as np 
import scipy  
import matplotlib.pyplot as plt
import math
import sys

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import optimizer as op

def h(w,X):
    return X.dot(w)

def hinge_loss(w,X,Y,C=1.0):
    num_records,num_features  = np.shape(X)  
    zero = np.zeros((num_records,1))
    margin = 1 - C * Y * h(w,X)
    return np.max([zero,margin],axis=0)

def L(w,X,Y,lamda=0.0,C=1.0):
    num_records,num_features  = np.shape(X)  
    e = hinge_loss(w,X,Y,C)
    regulation_loss = 1.0/2 * lamda * w.transpose().dot(w)
    loss = 1.0/float(num_records) * e.sum()  + regulation_loss
    return loss[0][0]

def g(w,X,Y,lamda=0.0,C=1.0):
    num_records,num_features  = np.shape(X)  
    e = hinge_loss(w,X,Y,C)
    indicator = np.zeros((num_records,1))
    indicator[np.nonzero(e)] = 1
    
    return - 1.0/float(num_records) * C \
        * X.transpose().dot(Y * indicator).sum(axis=1).reshape((num_features,1)) \
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

lamda = 0.1
eta = 0.05
C = 1.0
gamma = 0.9
threshold=0.5
max_iterate = 100
batch_size = 10000
epsilon=1e-8#sys.float_info.epsilon
Adadelta_last_E_delta_2_init = 1e-4
Adam_beta1 = 0.9
Adam_beta2 = 0.999

GD_loss_test = []
w = np.random.normal(size=(num_features,1))


epoch = 0
for counter in range(max_iterate):
    starts = [i*batch_size for i in range(math.ceil(train_size/batch_size))]
    ends = [i*batch_size for i in range(1,math.ceil(train_size/batch_size))]
    ends.append(train_size)
    for start, end in zip(starts, ends):
        # mini-batch gradient decent
        GD_loss_test.append(L(w,X_test,Y_test,lamda,C))
        w = op.gradient_decent(w,
            g(w,X_train[start:end,:],Y_train[start:end,:],lamda,C),
            eta)
              
        
        
        
        
        epoch += 1 
        
fig, ax = plt.subplots()
test_loss_line = ax.plot(range(epoch),GD_loss_test,label=r'GD loss, $\eta$='+str(eta))
#NAG_test_loss_line = ax.plot(range(epoch),NAG_loss_test,label=r'NAG loss, $\eta$='+str(eta)+r', $\gamma$='+str(gamma))
#Adadelta_test_loss_line = ax.plot(range(epoch),Adadelta_loss_test,label=r'Adadelta, with init $\Delta^2$='+str(Adadelta_last_E_delta_2_init))
#RMSprop_test_loss_line = ax.plot(range(epoch),RMSprop_loss_test,label=r'RMSprop loss, $\eta$='+str(eta)+r', $\gamma$='+str(gamma))
#Adam_test_loss_line = ax.plot(range(epoch),Adam_loss_test,label=r'Adam loss, $\eta$='+str(eta)+r', $\beta_1$='+str(Adam_beta1)+r', $\beta_2$='+str(Adam_beta2))
plt.legend()
ax.set(xlabel='Epoch', ylabel='Loss in testset with l2 norm')
plt.show()        
