#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:08:36 2017

@author: wty
"""
import numpy as np 

epsilon = 1e-8

def gradient_decent(w,gradient,eta):
    return w - eta * gradient

def NAG(w,gradient,last_delta,eta,gamma):
    delta = gamma * last_delta + eta * gradient
    return w-delta,delta

def Adadelta(w,gradient,last_E_g_2,last_E_delta_2,gamma):
    E_g_2 = gamma * last_E_g_2 + (1-gamma) * (gradient*gradient)
    RMS_g = np.sqrt(E_g_2+epsilon)
    RMS_last_delta = np.sqrt(last_E_delta_2+epsilon)
    delta = RMS_last_delta / RMS_g * gradient
    return w - delta,\
            E_g_2,\
            gamma * last_E_delta_2 + (1-gamma) * (delta*delta)

def RMSprop(w,gradient,last_E_g_2,eta,gamma):
    E_g_2 = gamma * last_E_g_2 + (1-gamma) * (gradient*gradient)
    return w - (eta/np.sqrt(E_g_2+epsilon))*gradient,\
            E_g_2
            
def Adam(w,gradient,last_m,last_v,eta,beta1,beta2,epoch,counteract_bias=True):
    m = beta1 * last_m + (1-beta1) * gradient
    v = beta2 * last_v + (1-beta2) * (gradient*gradient)
    if counteract_bias:
        m = m/(1-beta1**epoch)
        v = v/(1-beta2**epoch)
        
    return w - (eta/(np.sqrt(v)+epsilon))*m,\
            m*(1-beta1**epoch),v*(1-beta2**epoch)
