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
import sys

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
            
def gradient_decent(w,X,Y,lamda,eta):
    return w - eta * g(w,X,Y,lamda)

def NAG(w,X,Y,last_delta,lamda,eta,gamma):
    delta = gamma * last_delta + eta * g(w-gamma * last_delta,X,Y,lamda)
    return w-delta,delta

def Adadelta(w,X,Y,last_E_g_2,last_E_delta_2,lamda):
    gradient = g(w,X,Y,lamda)
    E_g_2 = gamma * last_E_g_2 + (1-gamma) * (gradient*gradient)
    RMS_g = np.sqrt(E_g_2+epsilon)
    RMS_last_delta = np.sqrt(last_E_delta_2+epsilon)
    delta = RMS_last_delta / RMS_g * gradient
    return w - delta,\
            E_g_2,\
            gamma * last_E_delta_2 + (1-gamma) * (delta*delta)

def RMSprop(w,X,Y,last_E_g_2,lamda,eta):
    gradient = g(w,X,Y,lamda)
    E_g_2 = gamma * last_E_g_2 + (1-gamma) * (gradient*gradient)
    return w - (eta/np.sqrt(E_g_2+epsilon))*gradient,\
            E_g_2
            
def Adam(w,X,Y,last_m,last_v,lamda,eta,beta1,beta2,epoch,counteract_bias=True):
    gradient = g(w,X,Y,lamda)
    m = beta1 * last_m + (1-beta1) * gradient
    v = beta2 * last_v + (1-beta2) * (gradient*gradient)
    if counteract_bias:
        m = m/(1-beta1**epoch)
        v = v/(1-beta2**epoch)
        
    return w - (eta/(np.sqrt(v)+epsilon))*m,\
            m*(1-beta1**epoch),v*(1-beta2**epoch)

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
gamma = 0.9
threshold=0.5
max_iterate = 100
batch_size = 10000
epsilon=1e-8#sys.float_info.epsilon
Adadelta_last_E_delta_2_init = 1e-4
Adam_beta1 = 0.9
Adam_beta2 = 0.99

GD_loss_test = []
w = np.random.normal(size=(num_features,1))

NAG_loss_test = []
NAG_w = w
NAG_last_delta = np.zeros((num_features,1))

Adadelta_loss_test = []
Adadelta_w = w
Adadelta_last_E_g_2 = np.zeros((num_features,1))
Adadelta_last_E_delta_2 = np.zeros((num_features,1)) + Adadelta_last_E_delta_2_init

RMSprop_loss_test = []
RMSprop_w = w
RMSprop_last_E_g_2 = np.zeros((num_features,1))

Adam_loss_test = []
Adam_w = w
Adam_last_m = np.zeros((num_features,1))
Adam_last_v = np.zeros((num_features,1))

epoch = 0
for counter in range(max_iterate):
    starts = [i*batch_size for i in range(math.ceil(train_size/batch_size))]
    ends = [i*batch_size for i in range(1,math.ceil(train_size/batch_size))]
    ends.append(train_size)
    for start, end in zip(starts, ends):
        # mini-batch gradient decent
        GD_loss_test.append(L(w,X_test,Y_test,lamda))
        w = gradient_decent(w,X_train[start:end,:],Y_train[start:end,:],lamda,eta)
        
        # Nesterov accelerated gradient decent
        NAG_loss_test.append(L(NAG_w,X_test,Y_test,lamda))
        NAG_w,NAG_last_delta = NAG(NAG_w,
                                   X_train[start:end,:],
                                   Y_train[start:end,:],
                                   NAG_last_delta,lamda,eta,gamma)
        
        # Adadelta gradient decent
        Adadelta_loss_test.append(L(Adadelta_w,X_test,Y_test,lamda))        
        Adadelta_w, Adadelta_last_E_g_2, Adadelta_last_E_delta_2 =\
            Adadelta(Adadelta_w,
                     X_train[start:end,:],
                     Y_train[start:end,:],
                     Adadelta_last_E_g_2,
                     Adadelta_last_E_delta_2,
                     lamda)
        
        # RMSprop gradient decent
        RMSprop_loss_test.append(L(RMSprop_w,X_test,Y_test,lamda))
        RMSprop_w, RMSprop_last_E_g_2 =\
            RMSprop(RMSprop_w,
                    X_train[start:end,:],
                    Y_train[start:end,:],
                    RMSprop_last_E_g_2,
                    lamda,eta)
        
        # Adaptive Moment Estimation
        Adam_loss_test.append(L(Adam_w,X_test,Y_test,lamda))
        Adam_w, Adam_last_m, Adam_last_v =\
            Adam(Adam_w,
                 X_train[start:end,:],
                 Y_train[start:end,:],
                 Adam_last_m,
                 Adam_last_v,
                 lamda,eta,Adam_beta1,Adam_beta2,epoch+1)
        
        epoch += 1 
    
fig, ax = plt.subplots()
test_loss_line = ax.plot(range(epoch),GD_loss_test,label=r'GD loss, $\eta$='+str(eta))
NAG_test_loss_line = ax.plot(range(epoch),NAG_loss_test,label=r'NAG loss, $\eta$='+str(eta)+r', $\gamma$='+str(gamma))
Adadelta_test_loss_line = ax.plot(range(epoch),Adadelta_loss_test,label=r'Adadelta, with init $\Delta^2$='+str(Adadelta_last_E_delta_2_init))
RMSprop_test_loss_line = ax.plot(range(epoch),RMSprop_loss_test,label=r'RMSprop loss, $\eta$='+str(eta)+r', $\gamma$='+str(gamma))
Adam_test_loss_line = ax.plot(range(epoch),Adam_loss_test,label=r'Adam loss, $\eta$='+str(eta)+r', $\beta_1$='+str(Adam_beta1)+r', $\beta_2$='+str(Adam_beta2))
plt.legend()
ax.set(xlabel='Epoch', ylabel='Loss in testset with l2 norm')
plt.show()
    








