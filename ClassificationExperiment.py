#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:58:35 2017

@author: wty
"""
import numpy as np 
import scipy  
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV

import LinearClassifier as LC
import PlotFigure as plt

result_path = './results/classification/grid_search'

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
print ("Start to figure the loss curves of\
       untuned estimators in one figure")

GD = LC.Classifier(w=init_w,optimizer='GD')
NAG = LC.Classifier(w=init_w,optimizer='NAG')
Adadelta = LC.Classifier(w=init_w,optimizer='Adadelta')
RMSprop = LC.Classifier(w=init_w,optimizer='RMSprop')
Adam = LC.Classifier(w=init_w,optimizer='Adam')

GD.fit(X_train,Y_train)
NAG.fit(X_train,Y_train)
Adadelta.fit(X_train,Y_train)
RMSprop.fit(X_train,Y_train)
Adam.fit(X_train,Y_train)

plt.plotFigureAllInOne(GD,NAG,Adadelta,RMSprop,Adam,X_test,Y_test,'untuned',path='./results/classification/SVM_untuned.pdf')


print ("===========================")
print ("Start to execute exhaustive grid search")
for i in range(len(optimizers)):
    optimizer_name = optimizers[i]
    
    cls = GridSearchCV(LC.Classifier(init_w,max_iterate=max_iterate,batch_size=batch_size,optimizer=optimizer_name), param_grid[optimizer_name],return_train_score=True,n_jobs=4)
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
        params = result.loc[j,'params']
        print ("Figure of",params)
        cls = LC.Classifier(init_w,max_iterate=max_iterate,batch_size=batch_size,optimizer=optimizer_name,**params)
        cls.fit(X_train,Y_train)
        plt.plotFigureTrainTest(cls,X_train,Y_train,X_test,Y_test,path='./results/classification/SVM_%s_%d.pdf'%(optimizer_name,j))
    
    
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
    
    plt.plotFigureTrainTest(cls,X_train,Y_train,X_test,Y_test,path='./results/classification/SVM_%s.pdf'%cls_name)

print ("===========================")
print ("Start to figure the loss curves of\
       tuned estimators in one figure")


plt.plotFigureAllInOne(o['GD'],o['NAG'],o['Adadelta'],o['RMSprop'],o['Adam'],X_test,Y_test,'tuned',path='./results/classification/SVM_tuned.pdf')