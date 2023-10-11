#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:33:33 2023

@author: evanshapiro
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

U = np.array([[1,.999,0.99],[0,1,.95],[0,0,.01]])

C = np.dot(U.T,U)
mu = [0,0,0]
beta = np.array([1,1,-1])

mv_norm = sp.stats.multivariate_normal( mean = mu, cov = C)
N  = 100
n = 1000
beta_h = np.zeros([n,3])
for i in range(0,n):
    X = mv_norm.rvs(size = N)
    y = np.dot(X,beta) + np.random.normal(0,1, size = N )
    beta_h[i,:] =  np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T,y))
    print(f'beta_h_{i} = ', np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T,y)))

beta_0_max = np.round(np.max(beta_h[:,0]))
beta_0_min = np.round(np.min(beta_h[:,0]) -1)
beta_1_max = np.round(np.max(beta_h[:,1]))
beta_1_min = np.round(np.min(beta_h[:,1]) -1)
beta_2_max = np.round(np.max(beta_h[:,2]))
beta_2_min = np.round(np.min(beta_h[:,2]) -1)

#plt.hist2d(beta_h[:,0], beta_h[:,1])
#Create histograms of coefficients here   
plt.figure(0)
plt.hist(beta_h[:,0])
plt.xlabel('beta_0_h')

plt.figure(1)
plt.hist(beta_h[:,1])
plt.xlabel('beta_1_h')

plt.figure(2)
plt.hist(beta_h[:,2])
plt.xlabel('beta_2_h')
# Show that bootstrap estimate of L1 distance of estimator from true coefficients
# aligns with theoretical E[L1] value.

# Try to create 3-D histograms

f
    