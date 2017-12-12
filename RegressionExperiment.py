#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:17:35 2017

@author: wty
"""

import numpy as np 
import scipy  
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV

import LogiticRegressionClassifier as LRC

def plotFigure(GD,NAG,Adadelta,RMSprop,Adam,tuned='tuned'):
    GD_loss_test = GD.getLossHistory(X_test,Y_test)
    NAG_loss_test = NAG.getLossHistory(X_test,Y_test)
    Adadelta_loss_test = Adadelta.getLossHistory(X_test,Y_test)
    RMSprop_loss_test = RMSprop.getLossHistory(X_test,Y_test)
    Adam_loss_test = Adam.getLossHistory(X_test,Y_test)
    
    _, ax = plt.subplots()
    ax.plot(range(len(GD_loss_test)),GD_loss_test,label=\
            r'GD,$\lambda$=%.2f,$\eta$=%.2f'\
            %(GD.get_params()['lamda'],GD.get_params()['eta']))
    ax.plot(range(len(NAG_loss_test)),NAG_loss_test,label=\
            r'NAG,$\lambda$=%.2f,$\eta$=%.2f,$\gamma$=%.2f'\
            %(NAG.get_params()['lamda'],NAG.get_params()['eta'],NAG.get_params()['gamma']))
    ax.plot(range(len(Adadelta_loss_test)),Adadelta_loss_test,label=\
            r'Adadelta,$\lambda$=%.2f,$\gamma$=%.2f'\
            %(Adadelta.get_params()['lamda'],Adadelta.get_params()['gamma']))
    ax.plot(range(len(RMSprop_loss_test)),RMSprop_loss_test,label=\
            r'RMSprop,$\lambda$=%.2f,$\eta$=%.2f,$\gamma$=%.2f'\
            %(RMSprop.get_params()['lamda'],RMSprop.get_params()['eta'],RMSprop.get_params()['gamma']))
    ax.plot(range(len(Adam_loss_test)),Adam_loss_test,label=\
            r'Adam,$\lambda$=%.2f,$\eta$=%.2f,$\beta_1$=%.2f,$\beta_2$=%.3f'\
            %(Adam.get_params()['lamda'],Adam.get_params()['eta'],Adam.get_params()['Adam_beta1'],Adam.get_params()['Adam_beta2']))
    
    plt.legend()
    plt.title('Different %s estimators\' performance'%tuned)
    ax.set(xlabel='Epoch', ylabel='Loss in testset with l2 norm')
    plt.show()
    plt.close('all')
    
result_path = './results/regression/grid_search'

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

max_iterate = 50
batch_size = 8000

figure_num = 2

init_w = np.random.normal(size=(num_features,1))

optimizers = ['NAG','Adadelta','RMSprop','Adam','GD']
o = {}

param_grid = {
        'NAG': {'lamda': [0.01, 0.1], 
               'eta': [0.01, 0.05],
               'gamma': [0.8, 0.9, 0.95],
               'threshold': [0.5,0.6]},
        'Adadelta' : {'lamda': [0.01, 0.1],
                      'gamma': [0.8, 0.9, 0.95],
                      'threshold': [0.5,0.6]},
        'RMSprop' : {'lamda': [0.01, 0.1], 
                   'eta': [0.01, 0.05],
                   'gamma': [0.8, 0.9, 0.95],
                   'threshold': [0.5,0.6]},
        'Adam' : {'lamda': [0.01, 0.1], 
                   'eta': [0.01, 0.05],
                   'Adam_beta1': [0.9, 0.95],
                   'Adam_beta2' : [0.99, 0.999],
                   'threshold': [0.5,0.6]},
        'GD' : {'lamda': [0.01, 0.1, 0.5], 
               'eta': [0.1, 0.2, 0.3, 0.4, 0.5],
               'threshold': [0.4,0.5,0.6]}}

print ("===========================")
print ("Start to execute exhaustive grid search")
for i in range(len(optimizers)):
    optimizer_name = optimizers[i]
    
    cls = GridSearchCV(LRC.Classifier(init_w,max_iterate=max_iterate,batch_size=batch_size,optimizer=optimizer_name), param_grid[optimizer_name],return_train_score=True,n_jobs=4)
    cls.fit(X_train,Y_train)
    result = pd.DataFrame(cls.cv_results_)
    result.sort_values('rank_test_score',inplace=True)
    result = result.reset_index(drop = True)
    
    # Best optimizer
    o[optimizer_name] = cls.best_estimator_
    
    print ("Exhaustive Grid Search Result of %s"%optimizer_name)
    print ("The best estimator's parameter is",cls.best_params_)
    print (result.loc[0:5,['rank_test_score','mean_test_score','mean_train_score','mean_fit_time','params']])
    path = result_path+'_'+optimizer_name+'.csv'
    result.to_csv(path)
    print ("Result has been saved in",path)
    
    
    print ("Printing the best %d models loss curves"%figure_num)
    for j in range(figure_num):
        params = result.loc[i,'params']
        print ("Figure of",params)
        cls = LRC.Classifier(init_w,max_iterate=max_iterate,batch_size=batch_size,optimizer=optimizer_name,**params)
        cls.fit(X_train,Y_train)
        loss_train = cls.getLossHistory(X_train,Y_train)
        loss_test = cls.getLossHistory(X_test,Y_test)
        accuracy_train = cls.getScoreHistory(X_train,Y_train)
        accuracy_test = cls.getScoreHistory(X_test,Y_test)
        
        plt.figure(j)
        _, ax = plt.subplots()
        ax_e = ax.twinx()
        ax.plot(range(len(loss_train)),loss_train,label='train loss')
        ax.plot(range(len(loss_test)),loss_test,label='test loss')
        ax_e.plot(range(len(accuracy_train)),accuracy_train,'r',label='train accuracy')
        ax_e.plot(range(len(accuracy_test)),accuracy_test,'g',label='test accuracy')
            
        ax.set(xlabel='Epoch', ylabel='Loss with l2 norm')
        ax_e.set_ylabel('Accuracy with threshold=%s'%str(cls.get_params()['threshold']))
        
        ax.legend(loc=4)
        ax_e.legend(loc=1)
        plt.show()    
        
    plt.close('all')
    
    
print ("===========================")
print ("Start to figure the accuracy and loss curves of\
       estimators of different optimized algorithms with tuned hyperparameter")
for i in range(len(optimizers)):
    cls_name = optimizers[i]
    cls = o[cls_name]
    
    print("Optimizer %s, parameters:"%cls_name)
    params = cls.get_params()
    params.pop('w')
    print(params)
    
    loss_train = cls.getLossHistory(X_train,Y_train)
    loss_test = cls.getLossHistory(X_test,Y_test)
    accuracy_train = cls.getScoreHistory(X_train,Y_train)
    accuracy_test = cls.getScoreHistory(X_test,Y_test)
    
    plt.figure(i)
    _, ax = plt.subplots()
    ax_e = ax.twinx()
    ax.plot(range(len(loss_train)),loss_train,label='train loss')
    ax.plot(range(len(loss_test)),loss_test,label='test loss')
    ax_e.plot(range(len(accuracy_train)),accuracy_train,'r',label='train accuracy')
    ax_e.plot(range(len(accuracy_test)),accuracy_test,'g',label='test accuracy')
    
    ax.set(xlabel='Epoch', ylabel='Loss with l2 norm')
    ax_e.set_ylabel('Accuracy with threshold=%s'%str(cls.get_params()['threshold']))
    
    ax.legend(loc=4)
    ax_e.legend(loc=1)
    plt.title(cls_name+' Gradient Decent')
    plt.show()
    plt.close('all')

print ("===========================")
print ("Start to figure the loss curves of\
       tuned estimators in one figure")

GD = o['GD']
NAG = o['NAG']
Adadelta = o['Adadelta']
RMSprop = o['RMSprop']
Adam = o['Adam']

plotFigure(GD,NAG,Adadelta,RMSprop,Adam,'tuned')

init_w = np.random.normal(size=(num_features,1))
GD = LRC.Classifier(w=init_w,optimizer='GD')
NAG = LRC.Classifier(w=init_w,optimizer='NAG')
Adadelta = LRC.Classifier(w=init_w,optimizer='Adadelta')
RMSprop = LRC.Classifier(w=init_w,optimizer='RMSprop')
Adam = LRC.Classifier(w=init_w,optimizer='Adam')

GD.fit(X_train,Y_train)
NAG.fit(X_train,Y_train)
Adadelta.fit(X_train,Y_train)
RMSprop.fit(X_train,Y_train)
Adam.fit(X_train,Y_train)

plotFigure(GD,NAG,Adadelta,RMSprop,Adam,'untuned')