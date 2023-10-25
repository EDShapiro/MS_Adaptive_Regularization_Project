#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:02:14 2023

@author: evanshapiro
"""
plt.figure(0)
Y_pred_enet = np.dot(X[0:n,:], beta_arr_t)
Y_pred_aenet = np.dot(X[0:n,:],beta_v_t[91,:])
plt.scatter(Y[:,0],Y_pred_enet, color = 'r')
plt.scatter(Y[:,0],Y_pred_aenet, color = 'b' )
plt.ylabel('Y_predicted')
plt.xlabel('Y_true')
plt.title('Overlay of Predictions from ENete and AENet vs True')

for i in range(0,51): 
    if np.abs(beta_arr[0,i]) <0.5:
        beta_arr_t[i] = 0 
    else: 
        beta_arr_t[i] = beta_arr[0,i]