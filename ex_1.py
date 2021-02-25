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



gamma_range = np.array([-2, -0.03,0.3])
kappa= np.array([0.1, 0.2])
rho_range = np.array([0, 0.5,  0.9])
kappa_err = 0.92
OInOU.kappa_misspecified_test2D(gamma_range, kappa, rho_range, kappa_err, 30, 1, 1)

#Wms = OInOU.ms_W_t(gamma, OU, OU_ms, T, dt)
#Wms.plot_simulation(1, x0)
