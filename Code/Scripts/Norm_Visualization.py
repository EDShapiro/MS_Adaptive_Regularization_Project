#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 00:36:38 2023

@author: evanshapiro
"""
import matplotlib.pyplot as plt
import numpy as np

#x_0 = np.linspace(0,1,100)
#x_1 = np.linspace(-1,0, 100)
x_0 = np.linspace(-1,1,1000)
y_0 = 1-np.abs(x)
y_1 = np.abs(x) -1
#y_2 = x_1 +1 
#y_3 = -1 - x_1 

x_1 = np.linspace(-np.sqrt(2),np.sqrt(2),10000)
y_2 = np.sqrt(1 - x_1**2/2)
y_3 = -np.sqrt(1 - x_1**2/2)

x_2 = np.linspace(-np.sqrt(1.5),np.sqrt(1.5) ,10000)
y_4 = 1/2*np.sqrt(1 - x_2**2/1.5)
y_5 = -1/2*np.sqrt(1 - x_2**2/1.5)


#y_6 = 3/2 + np.sqrt(1/4 - x[0:100]**2/4)
#y_7 = 3/2 - np.sqrt(1/4 - x[0:100]**2/4)
#Apply rotation to the ellipse using the rotation matrix
#Also use an SVD
#Rotation Matrix
#[[cos,-sin] [sin, cos ] ]
#Standard Rotation
#R = np.array([[np.cos(np.pi/4), - np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
#Affine Translation
R_aff = np.array([[1, 0, -1 ], [0, 1, -2], [0,0,1]])
#Affine Rotation
R_rot = np.array([[np.cos(np.pi/4), - np.sin(np.pi/4), 0], [np.sin(np.pi/4), np.cos(np.pi/4), 0], [0,0,1]])

hom = np.ones(len(x_0))
X_1 = np.array([[x_0, y_0, hom]])
X_2 = np.array([[x_0, y_1, hom]])
X_1_t = np.dot(R_aff,X_1)
X_2_t = np.dot(R_aff,X_2)

hom = np.ones(len(x_1))
X_3 = np.array([x_1, y_2, hom])
X_4 = np.array([x_1, y_3, hom])
X_3_r = np.dot(R_rot,X_3)
X_4_r = np.dot(R_rot,X_4)

hom = np.ones(len(x_2))
X_5 = np.array([x_2, y_4, hom])
X_6 = np.array([x_2, y_5, hom])
X_5_r = np.dot(R_rot,X_5)
X_6_r = np.dot(R_rot,X_6)

# X_3 = np.array([x, y_4])
# X_4 = np.array([x, y_5])
# X_3_t = np.dot(R,X_3)
# X_4_t = np.dot(R,X_4)
# #np.mult(x)
# #Replace x_0 x_1 with x
# plt.figure(0)
# plt.plot(x, y_0,'b', x, y_1, 'b', x, y_2,'b', x, y_3, 'b')
# plt.figure(1)
# plt.plot(x, y_0,'b', x, y_1, 'b', X_1_t[0,:], X_1_t[1,:],'r', X_2_t[0,:], X_2_t[1,:],'r')
# plt.figure(2)
# plt.plot(x, y_0,'b', x, y_1, 'b', X_3_t[0,:], X_3_t[1,:],'b', X_4_t[0,:], X_4_t[1,:],'b')
plt.figure(3)
plt.scatter(X_1_t[0,:],X_1_t[1,:],s =1/6, color = 'b')
plt.scatter( X_2_t[0,:],X_2_t[1,:], s =1/6, color = 'b')
plt.scatter( X_3_r[0,:], X_3_r[1,:],s =1/6, color = 'g')
plt.scatter( X_4_r[0,:], X_4_r[1,:],s =1/6, color = 'g')
plt.scatter( X_5_r[0,:], X_5_r[1,:],s =1/6, color = 'g')
plt.scatter( X_6_r[0,:], X_6_r[1,:],s =1/6, color = 'g')
plt.title('Visualization of Possible LASSO Solution')
