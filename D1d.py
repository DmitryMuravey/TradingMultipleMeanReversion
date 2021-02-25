#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:59:50 2019

@author: dm
"""
import numpy as np 
import matplotlib.pyplot as plt

def D1d(kappa, gamma, T) : 
    dt = 1/252 /4
    delta = 1 / (1 - gamma)
    sq_delta = np.sqrt(delta)
    t = np.arange(0, T, dt)
    sh_t = np.sinh(kappa * sq_delta * t)
    ch_t = np.cosh(kappa * sq_delta * t)   
    answer = kappa * sq_delta * (ch_t * sq_delta + sh_t) / (sh_t * sq_delta + ch_t)
    return [t, np.flip(answer)]
    

kappa_range = [1]
gamma_range = [-16, -2, -0.1, 0, 0.1, 0.3, 0.7]
gamma_range = np.flip(gamma_range)
legend_list = []
for gamma in gamma_range :
   legend_list.append(r'$\gamma = ' + str(gamma) + '$')


plt.figure()
plt.grid(True)
plt.title(r'$ \kappa = 1, \quad T =5 $')
plt.xlabel('$t$')
plt.ylabel('$D(T - t)$')
for kappa in kappa_range :
    for gamma in gamma_range : 
        [t, D] = D1d(kappa, gamma, 5)
        plt.plot(t, D, linewidth = 2)    
        
        
plt.legend(legend_list, loc = 'best')
