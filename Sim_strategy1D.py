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


NN = 1
kappa = [1]
sigma = [1]
theta = [0] 
x0 = theta.copy()
#kappa[1] = 0
#kappa[0] = 1
corr_matrix = OInOU.generate_random_correlation_matrix(int(3 *NN/ 4), NN)
#rho = 0.
#corr_matrix[0, 1] = rho
#corr_matrix[1, 0] = rho
OU = OInOU.OU_process(kappa, sigma, corr_matrix, theta)
gamma =-4
T = 3
dt = 1/252/4

W = OInOU.W_t(gamma, OU, T, dt)
t_wealth = W.EWT(1, x0)
var_WT = W.varWT(1, x0)
W.plot_simulation1D(1, x0)


