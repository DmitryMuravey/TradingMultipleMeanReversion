#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:37:55 2019

@author: dm
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import OInOU

sys.path.append('../')

#%%
# second branch
kappa = np.zeros(2)
sigma = np.zeros(2) + 1
theta = np.zeros(2)
x0 = theta.copy()
kappa[0] = 0.2
rho = 0.9
corr_matrix = np.zeros((2,2)) + 1
corr_matrix[0,1] = rho
corr_matrix[1,0] = rho
T = 5
dt = 1/252 / 4
t = np.arange(0,T,dt)
xi = 1 / (1- rho ** 2)
gamma = 1- rho ** 2 -2
 
delta = 1/ (1-gamma)
k = kappa[0]
OU = OInOU.OU_process(kappa, sigma, corr_matrix, theta)
Asol = OInOU.Asolver(OU, gamma, T, dt)
A = Asol.solve()
A = A[:,0]
if gamma > 1 - rho ** 2:
    beta = np.sqrt(-delta * (xi + delta -delta * xi))
    F = -beta * k * (beta * np.sin(beta * k * t) - delta * np.cos(beta * k * t)) \
    / (beta * np.cos(beta * k * t) + delta * np.sin(beta * k * t))  

elif gamma < 1 - rho ** 2 :
    beta = np.sqrt(delta * (xi + delta -delta * xi))
    F = beta * k * (beta * np.sinh(beta * k * t) + delta * np.cosh(beta * k * t)) \
     / (beta * np.cosh(beta * k * t) + delta * np.sinh(beta * k * t))  
else : 
    F = 1 / (1 / delta / k + t)

A_analyt  = -0.5 * (F - delta * k)    
A_analyt = np.flip(A_analyt)
plt.figure()
plt.grid(True)
plt.plot(t, delta * xi * k  - 2 * A)
plt.plot(t, delta * xi * k  - 2 * A_analyt)
   

#%%
gamma = 0.2
T = 2
dt = 1/252 / 4
Asol = OInOU.Asolver(OU, gamma, T, dt)
A = Asol.solve()
A = A[:,0]
t = np.arange(0,T,dt)

xi = 1 / (1- rho ** 2)
delta = 1/ (1-gamma)
k = kappa[0]
Dnum = -2* A + delta * k 

plt.figure()
plt.grid(True)
plt.plot(t, Dnum)
plt.plot(t, Danalyt)
