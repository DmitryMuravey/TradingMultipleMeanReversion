#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:04:56 2019

@author: dm
"""
import OInOU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
kappa = np.zeros(2)
kappa[0] = 1
kappa[1] = 0.2
#rho = 0.99
gamma = -4
kappalg = np.linspace(-4, 0, 200)
rho_range = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
legend_list = []
for rho in rho_range :
   legend_list.append(r'$\rho = ' + str(rho) + '$')




font = {'weight' : 'bold', 'size'   : 16}

matplotlib.rc('font', **font)
   
sigma = 1 + np.zeros(2)
theta = np.zeros(2)
x0 = np.zeros(2)
T = 3
dt = 1/252
plt.figure()
plt.grid(True)
plt.xlabel(r'$\log{\kappa_2 / \kappa_1}$')
plt.ylabel(r'$J$')
plt.title(r'$ \kappa_1 = 1, \quad \sigma_1 = 1,\quad \sigma_2 = 1, \quad \gamma = -4, \quad T =3, \quad x_0 = 0, \quad \theta = 0$')
for rho in rho_range : 
    corr_matrix = np.zeros((2,2)) + 1
    corr_matrix[0, 1] = rho
    corr_matrix[1, 0] = rho
    Jkappa = kappalg.copy()
    for i in range(len(kappalg)) :
        kappa[1] = np.exp(kappalg[i]) * kappa[0]
        OU = OInOU.OU_process(kappa, sigma, corr_matrix, theta)
        W = OInOU.W_t(gamma, OU, T, dt)
        Jkappa[i] = W.J_value(1, x0)
    plt.plot(kappalg, Jkappa, linewidth=2.5)

plt.legend(legend_list)
