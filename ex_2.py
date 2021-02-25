#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:53:28 2019

@author: dm
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

import OInOU

NN = 2
kappa = np.random.uniform(1, 1.4, NN)
sigma = np.random.uniform(0.5, 2, NN)
theta = np.random.uniform(0, 2,  NN)
x0 = np.random.uniform(0, 2,  NN)
x0 = theta.copy()
corr_matrix = OInOU.generate_random_correlation_matrix(int(3 *NN/ 4), NN)
kappa[0] = 3
kappa[1] = 0.14
rho = 0.8
corr_matrix[0, 1] = rho
corr_matrix[1, 0] = rho 
gamma = -8
kappalg = np.linspace(-0.5, 0, 100)
Jkappa = kappalg.copy()
JkappaNC = kappalg.copy()
corr_matrix_zero = np.diag([1, 1])
rho_range = np.array([0, 0.5, 0.8, 0.9, 0.92])
dt = 1/252
T = 3
plt.figure()
plt.grid(True)
plt.xlabel("log(k2/k1)")
plt.ylabel("J")
for j in range(len(rho_range)) : 
    corr_matrix[0, 1] = rho_range[j]
    corr_matrix[1, 0] = rho_range[j]
    for i in range(len(kappalg)) :
        kappa[1] = np.exp(kappalg[i]) * kappa[0]
        OU = OInOU.OU_process(kappa, sigma, corr_matrix, theta)
        W = OInOU.W_t(gamma, OU, T, dt)
#        OUnc = OInOU.OU_process(kappa, sigma, corr_matrix_zero, theta)
#        Wnc = OInOU.W_t(gamma, OUnc, T,dt)
        Jkappa[i] = W.J_value(1, x0)
#        JkappaNC[i] = Wnc.J_value(1, x0)
    plt.plot(kappalg, Jkappa)

plt.legend(['0', '0.5', '0.8', '0.9', '0.92'])

#plt.plot(kappalg, JkappaNC)
#%%
kappa= np.array([0.1, 0.15])
#gamma_range = np.array([-3, 0.1])
rho_range = np.array([0,  0.5, 0.95])
#gamma_range = np.array([-3,])
#rho_range = np.array([0.9])

kappa_err = 0.92
OInOU.kappa_misspecified_test2D(gamma_range, kappa, rho_range, kappa_err, 30, 1, 1)

#Wms = OInOU.ms_W_t(gamma, OU, OU_ms, T, dt)
#Wms.plot_simulation(1, x0)
