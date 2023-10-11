#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:38:59 2023

@author: evanshapiro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:33:33 2023

@author: evanshapiro
"""
#Ridge regression example
#Compute the trace of the ridge regressor
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

U = np.array([[1,.999,0.99],[0,1,.95],[0,0,.01]])

C = np.dot(U.T,U)
mu = [0,0,0]
beta = np.array([1,0,0])
mv_norm = sp.stats.multivariate_normal( mean = mu, cov = C)
N  = 100
n = 1000

#Implement ridge trace here keeping track of the trace
X = mv_norm.rvs(size = N)
y = np.dot(X,beta) + np.random.normal(0,1, size = N )
 
M = 1000
k = np.linspace(0,1,M)
beta_r = np.zeros([M,3])
for i in range(0,M):
    beta_r[i,:] = np.dot(np.linalg.inv(np.dot(X.T, X) + k[i]*np.identity(C.shape[0])),np.dot(X.T,y))
    
    
plt.figure(1)
plt.plot(k, beta_r[:,0], 'r', label = ' \beta_1')
plt.plot(k, beta_r[:,1], 'b', label = '\beta_2')
plt.plot(k, beta_r[:,2], 'g', label = '\beta_3')
plt.legend(["beta_1", "beta_2", "beta_3"])

plt.title('Ridge Trace for 3 Variable Regression Problem ')
beta_h = np.zeros([n,3])
for i in range(0,n):
    X = mv_norm.rvs(size = N)
    y = np.dot(X,beta) + np.random.normal(0,1, size = N )
    beta_h[i,:] =  np.dot(np.linalg.inv(np.dot(X.T, X) + 0.4*np.identity(C.shape[0])),np.dot(X.T,y))
    print(f'beta_h_{i} = ', np.dot(np.linalg.inv(np.dot(X.T, X) + 0.4*np.identity(C.shape[0])),np.dot(X.T,y)))



beta_0_max = np.round(np.max(beta_h[:,0]))
beta_0_min = np.round(np.min(beta_h[:,0]) -1)
beta_1_max = np.round(np.max(beta_h[:,1]))
beta_1_min = np.round(np.min(beta_h[:,1]) -1)
beta_2_max = np.round(np.max(beta_h[:,2]))
beta_2_min = np.round(np.min(beta_h[:,2]) -1)

#plt.hist2d(beta_h[:,0], beta_h[:,1])
#Create histograms of coefficients here   
plt.figure(2)
plt.hist(beta_h[:,0])
plt.xlabel('beta_0_h')

plt.figure(3)
plt.hist(beta_h[:,1])
plt.xlabel('beta_1_h')


plt.figure(4)
plt.hist(beta_h[:,2])
plt.xlabel('beta_2_h')
# # Show that bootstrap estimate of L1 distance of estimator from true coefficients
# # aligns with theoretical E[L1] value.

# # Try to create 3-D histograms

# fig = plt.figure(2)
# ax = fig.add_subplot(projection='3d')
# x = beta_h[:,0]
# y = beta_h[:,1]
# hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[beta_0_min, beta_0_max], [beta_1_min, beta_1_max]])

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()
    