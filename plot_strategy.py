#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:53:28 2019

@author: dm
"""

import sys

import numpy as np
sys.path.append('../')


import OInOU
# Number of Assets
NN = 2
kappa = np.random.uniform(0., 0.7, NN)
sigma = np.random.uniform(0.5, 1, NN)
theta = np.random.uniform(0, 2,  NN)
x0 = 2 * theta.copy()
corr_matrix = OInOU.generate_random_correlation_matrix(int(3 *NN/ 4), NN)
OU = OInOU.OU_process(kappa, sigma, corr_matrix, theta)
# Risk aversion
gamma =  -5
# time horizon (5 Years)
T = 5
# time step 4 step for a day.
dt = 1/252/4

# Create Wealth process
W = OInOU.W_t(gamma, OU, T, dt)
# Visualize optimal strategy 
W.plot_simulation(1, x0)

#%%