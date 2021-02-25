#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:04:47 2019

@author: dm
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')


def OU_sim(T, dt, x0) : 
    t = np.arange(0, T, dt)
    sample_sz = len(t)
    X = np.zeros(sample_sz)
    X[0] = x0
    for i in range(0, sample_sz -1) : 
        X[i + 1] = X[i] -X[i] * dt+ np.sqrt(dt) * np.random.normal(0, 1, 1) 
    return X    

def C(t, T, gamma):
    tau = T - t
    nu = np.sqrt(1/ (1- gamma))
    return np.cosh(nu * tau) + nu * np.sinh(nu * tau)  

def D(t,T, gamma) : 
    tau = T -t
    nu = np.sqrt(1/ (1- gamma))
    return (nu * np.sinh(nu * tau) + nu** 2 * np.cosh(nu * tau)) / C(t, T, gamma)

def Wealth(w0, X, dt, T, gamma) : 
    Wt = np.zeros(X.shape)
    Wt[0] = w0
    for i in range(0, len(Wt) - 1) : 
        t = dt * i
        Wt[i + 1] = Wt[i] + (X[i + 1] - X[i]) * (-X[i] * Wt[i] * D(t, T, gamma))
    return Wt    

def Wealth2(w0, X, dt, T, gamma) : 
    delta = 1/(1-gamma)
    Wt = np.zeros(X.shape)
    Wt[0] = w0
    for i in range(0, len(Wt) - 1) : 
        Wt[i + 1] = Wt[i] *  np.exp(-0.5 * delta * (X[i] ** 2) * dt)
    for i in range(0, len(Wt)) : 
        t = i * dt
        Wt[i] = Wt[i] * np.sqrt(C(0, T, gamma) / C(t, T, gamma)) * np.exp(-0.5 * D(t, T, gamma) * X[i] ** 2 + 0.5 * D(0, T, gamma) * X[0] ** 2)  
    return Wt    




T = 1
dt = 1/252/4
X = OU_sim(T, dt, 0)
gamma = -3

rho = np.linspace(0, 0.99, 100)
answer = rho.copy()
for i in range(len(rho)) :
    rho_arg = (1- rho[i] **2) / 2
    answer[i] = ((np.exp(T) * C(T, 0, gamma) ** (gamma - 1)) ** rho_arg) /gamma
#%%

W = Wealth(1, X, dt, T , gamma)
W2 = Wealth2(1, X, dt, T, gamma)
t= np.arange(0, T, dt)


plt.figure()
plt.grid(True)
plt.plot(t,X)
plt.plot(t, W)
plt.plot(t, W2)