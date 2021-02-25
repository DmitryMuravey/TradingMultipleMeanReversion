#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:36:42 2019

@author: dm
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

import OInOU

def intPsi(k, T, gamma) : 
    sq_delta = np.sqrt(1 / (1- gamma))
    e_arg = k * T * sq_delta 
    omega = (1 - sq_delta) / (1 + sq_delta)
    return 0.5 * (e_arg * (1 + sq_delta) - np.log((1+ omega)/(np.exp(2 * e_arg) + omega)))

def abs_rho(k1, k2, T, gamma) : 
    return intPsi(k2, T, gamma) / (intPsi(k2, T, gamma)+ intPsi(k1, T, gamma))

k1 = 1;
T = 1;
k2log = np.linspace(-1, 2, 20)
rho = k2log.copy()
gamma_range = np.linspace(-10, -1, 2)

k2exp = np.exp(k2log)
plt.figure()
plt.grid(True)
for i in range(len(k2log)) : 
    for j in range (100) :
        k1 = np.random.uniform(0, 10)
        T = np.random.uniform(0, 5)
        gamma = np.random.uniform(-10, 1)
        k2 = k1 * k2exp[i]
        tr_rho = abs_rho(k1, k2, T, gamma)
        est_rho = k2 / (k1+ k2)
        err = (est_rho - tr_rho) / tr_rho
        plt.scatter(k2log[i], err)
        
#%%


for j in range(1) :
    k1 = np.random.uniform(0, 10)
    T = np.random.uniform(1, 1)
    gamma = np.random.uniform(-1000, 1)
    for i in range(len(k2log)) :
        rho[i] = abs_rho(k1, k1 * k2exp[i], T, gamma)
#    plt.plot(k2log, rho)
    plt.plot(k2exp, rho)
plt.plot(k2exp, 1/ (1 + 1/k2exp))

#plt.plot(k2log, 1 / (1 + np.exp(-k2log)) )   
