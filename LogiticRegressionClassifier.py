#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:40:41 2017

@author: wty
"""


import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
import math

import optimizer as op

class Classifier(BaseEstimator,ClassifierMixin):  
    """A Logistic regression Classifier for ML2017-lab-02"""
    
    def __init__(self, w=0, lamda=0.1, eta=0.05, gamma=0.9,\
                 threshold=0.5, max_iterate=100, batch_size=10000,\
                 Adam_beta1=0.9, Adam_beta2=0.999,\
                 Adadelta_last_E_delta_2_init=1e-4, optimizer='GD'):
        """
        Called when initializing the classifier,
        optimizer expectes {'NAG','Adadelta','RMSprop','Adam','GD'}
        """
        self.w = w
        self.lamda = lamda
        self.eta = eta
        self.gamma = gamma
        self.threshold = threshold
        self.max_iterate = max_iterate
        self.batch_size = batch_size
        self.Adam_beta1 = Adam_beta1
        self.Adam_beta2 = Adam_beta2
        self.Adadelta_last_E_delta_2_init = Adadelta_last_E_delta_2_init
        self.optimizer = optimizer
        
        self.w_history = []
        
    def sigmoid(self,inX):  
        return 1.0 / (1 + np.exp(-inX))
    
    def __h(self,w,X):
        return self.sigmoid(X.dot(w))
    
    def h(self,X):
        return self.__h(self.w,X)
    
    def L(self,X,Y):
        return self.__L(self.w,X,Y)
    
    def __L(self,w,X,Y):
        num_records,num_features  = np.shape(X)  
        lamda = self.lamda
        
        hx = self.__h(w,X)
        regulation_loss = 1.0/2 * lamda * w.transpose().dot(w)
        loss = 1.0/2 * 1.0/(1 - -1) * 1.0/num_records * (-np.log(hx).transpose().dot(1+Y) - np.log(1-hx).transpose().dot(1-Y))\
                + regulation_loss
                
        return loss[0][0]
        
    def g(self,X,Y):
        return self.__g(self.w,X,Y)
    
    def __g(self,w,X,Y):
        num_records,num_features  = np.shape(X)    
        lamda = self.lamda
    
        # L2 norm
        return 1.0/num_records * X.transpose().dot(2*self.__h(w,X)-(Y+1)) \
                + lamda * w
            
    
    def fit(self, X, Y):
        """
        A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_, _ = np.unique(Y, return_inverse=True)
        train_size,num_features  = np.shape(X)
        batch_size = self.batch_size
        optimizer = self.optimizer
        eta = self.eta
        gamma = self.gamma
        
        NAG_last_delta = np.zeros((num_features,1))
        
        Adadelta_last_E_g_2 = np.zeros((num_features,1))
        Adadelta_last_E_delta_2 = np.zeros((num_features,1)) + self.Adadelta_last_E_delta_2_init
        
        RMSprop_last_E_g_2 = np.zeros((num_features,1))
        
        Adam_last_m = np.zeros((num_features,1))
        Adam_last_v = np.zeros((num_features,1))
        
        epoch = 0 
        self.w_history.append(self.w)
        for counter in range(self.max_iterate):
            starts = [i*batch_size for i in range(math.ceil(train_size/batch_size))]
            ends = [i*batch_size for i in range(1,math.ceil(train_size/batch_size))]
            ends.append(train_size)
            for start, end in zip(starts, ends):
                
                if optimizer == 'NAG':
                    # Nesterov accelerated gradient decent
                    self.w,NAG_last_delta =\
                        op.NAG(self.w,
                            self.g(X[start:end,:],Y[start:end,:]),
                            NAG_last_delta,eta,gamma)
                    self.w_history.append(self.w)
                elif optimizer == 'Adadelta':
                    # Adadelta gradient decent
                    self.w, Adadelta_last_E_g_2, Adadelta_last_E_delta_2 =\
                        op.Adadelta(self.w,
                                self.g(X[start:end,:],Y[start:end,:]),
                                 Adadelta_last_E_g_2,
                                 Adadelta_last_E_delta_2,
                                 gamma)       
                    self.w_history.append(self.w)             
                elif optimizer == 'RMSprop':
                    # RMSprop gradient decent
                    self.w, RMSprop_last_E_g_2 =\
                        op.RMSprop(self.w,
                                self.g(X[start:end,:],Y[start:end,:]),
                                RMSprop_last_E_g_2,
                                eta,gamma)      
                    self.w_history.append(self.w)              
                elif optimizer == 'Adam':
                    # Adaptive Moment Estimation
                    self.w, Adam_last_m, Adam_last_v =\
                        op.Adam(self.w,
                             self.g(X[start:end,:],Y[start:end,:]),
                             Adam_last_m,
                             Adam_last_v,
                             eta,self.Adam_beta1,self.Adam_beta2,epoch+1)       
                    self.w_history.append(self.w)             
                elif optimizer == 'GD':
                    # mini-batch gradient decent
                    self.w = op.gradient_decent(self.w,
                        self.g(X[start:end,:],Y[start:end,:]),
                        eta)
                    self.w_history.append(self.w)
                    
                else:
                    raise ValueError("Optimizer error, expected {'NAG','Adadelta','RMSprop','Adam','GD'}, got %s" % optimizer)
        
            epoch += 1 
        
        return self    
    
    def __predict(self,w,X):
        threshold = self.threshold
        raw = self.__h(w,X)
        raw[raw<=threshold] = self.classes_[0]
        raw[raw>threshold] = self.classes_[1]
        return raw
    
    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        return self.__predict(self.w,X)
    
    def __score(self,w,X,Y):
        num_records,num_features  = np.shape(X)
        P = self.__predict(w,X)
        
        is_right = P * Y
        is_right[is_right < 0] = 0
        
        return 1.0/num_records * np.count_nonzero(is_right)
    
    def score(self, X, Y):
        # RMSE
        return self.__score(self.w,X,Y)
    
    def getLossHistory(self,X,Y):
        return [self.__L(w,X,Y) for w in self.w_history]
    
    def getScoreHistory(self,X,Y):
        return [self.__score(w,X,Y) for w in self.w_history]