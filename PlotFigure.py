#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:50:06 2017

@author: wty
"""

import matplotlib.pyplot as plt

def plotFigureAllInOne(GD,NAG,Adadelta,RMSprop,Adam,X,Y,tuned='tuned',every=20,size=6,path=None):
    GD_loss_test = GD.getLossHistory(X,Y)
    NAG_loss_test = NAG.getLossHistory(X,Y)
    Adadelta_loss_test = Adadelta.getLossHistory(X,Y)
    RMSprop_loss_test = RMSprop.getLossHistory(X,Y)
    Adam_loss_test = Adam.getLossHistory(X,Y)
    
    _, ax = plt.subplots()
    ax.plot(range(len(GD_loss_test)),GD_loss_test,'.-',markersize=size,markevery=every,label=\
            r'GD,$\lambda$=%.2f,$\eta$=%.2f'\
            %(GD.get_params()['lamda'],GD.get_params()['eta']))
    ax.plot(range(len(NAG_loss_test)),NAG_loss_test,'s-',markersize=size,markevery=every,label=\
            r'NAG,$\lambda$=%.2f,$\eta$=%.2f,$\gamma$=%.2f'\
            %(NAG.get_params()['lamda'],NAG.get_params()['eta'],NAG.get_params()['gamma']))
    ax.plot(range(len(Adadelta_loss_test)),Adadelta_loss_test,'*-',markersize=size,markevery=every,label=\
            r'Adadelta,$\lambda$=%.2f,$\gamma$=%.2f'\
            %(Adadelta.get_params()['lamda'],Adadelta.get_params()['gamma']))
    ax.plot(range(len(RMSprop_loss_test)),RMSprop_loss_test,'v-',markersize=size,markevery=every,label=\
            r'RMSprop,$\lambda$=%.2f,$\eta$=%.2f,$\gamma$=%.2f'\
            %(RMSprop.get_params()['lamda'],RMSprop.get_params()['eta'],RMSprop.get_params()['gamma']))
    ax.plot(range(len(Adam_loss_test)),Adam_loss_test,'d-',markersize=size,markevery=every,label=\
            r'Adam,$\lambda$=%.2f,$\eta$=%.2f,$\beta_1$=%.2f,$\beta_2$=%.3f'\
            %(Adam.get_params()['lamda'],Adam.get_params()['eta'],Adam.get_params()['Adam_beta1'],Adam.get_params()['Adam_beta2']))
    
    plt.legend()
    plt.title('Different %s estimators\' performance'%tuned)
    ax.set(xlabel='Epoch', ylabel='Loss in testset with l2 norm')
    
    if path!=None:
        plt.savefig(path,format='pdf')
    
    plt.show()
    plt.close('all')
    
def plotFigureTrainTest(cls,X_train,Y_train,X_test,Y_test,every=20,size=6,path=None):
    loss_train = cls.getLossHistory(X_train,Y_train)
    loss_test = cls.getLossHistory(X_test,Y_test)
    accuracy_train = cls.getScoreHistory(X_train,Y_train)
    accuracy_test = cls.getScoreHistory(X_test,Y_test)
    
    _, ax = plt.subplots()
    ax_e = ax.twinx()
    ax.plot(range(len(loss_train)),loss_train,'*-b',markersize=size,markevery=every,label='train loss')
    ax.plot(range(len(loss_test)),loss_test,'v-g',markersize=size,markevery=every,label='test loss')
    ax_e.plot(range(len(accuracy_train)),accuracy_train,'*-r',markersize=size,markevery=every,label='train accuracy')
    ax_e.plot(range(len(accuracy_test)),accuracy_test,'v-m',markersize=size,markevery=every,label='test accuracy')
        
    ax.set(xlabel='Epoch', ylabel='Loss with l2 norm')
    ax_e.set_ylabel('Accuracy with threshold=%s'%str(cls.get_params()['threshold']))
    
    ax.legend(loc=4)
    ax_e.legend(loc=1)
    
    if path!=None:
        plt.savefig(path,format='pdf')
        
    plt.show()
    plt.close('all')