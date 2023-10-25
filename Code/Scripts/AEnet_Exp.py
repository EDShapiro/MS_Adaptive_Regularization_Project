#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:50:00 2023

@author: evanshapiro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:22:09 2023

@author: evanshapiro
"""
#pn = number of parameters
import cvxpy as cp
import random
import numpy as np

def tts(N, cv_n):
    cv_idx = np.zeros((cv_n, int(N/cv_n)))
    idx_list = np.arange(0,N)
    for i in range(0, cv_n):
        cv_idx[i,:] = random.sample(sorted(idx_list),int(N/cv_n))
        idx_list = np.setdiff1d(idx_list, cv_idx[i,:])
    return cv_idx.astype(int)
        
    

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2


#aenet objective function takes weights from a lasso and uses them to weight
# the coefficients in the regularization function 

#Weights \hat{w}_j :=(|\hat{beta}_j(enet)| +1/n)^{-\gamma}; gamma \ge 0
#lambda_2:= lambda_2 from original elastic-net 
#lambda_1*:= New hyperparameter thatis determined via CV, or some such nonsense.

def aenet_objective_fn(X, Y, beta, lambd_1, lambd_2, weights):
    return loss_fn(X, Y, beta) + lambd_1 * cp.norm1(cp.multiply(weights,beta)) + lambd_2*cp.norm2(beta)

def alasso_objective_fn(X, Y, beta, lambd_1, lambd_2, weights):
    return loss_fn(X, Y, beta) + lambd_1 * cp.norm1(cp.multiply(weights,beta))


def mse(X, Y, beta):
    return loss_fn(X, Y, beta).value

def ada_meth_script(N, cv_n, n_lmbda, pn, X, Y, lambda_2, ad_w_arr, meth = 'aenet'):
    beta = cp.Variable(pn)
    lambda_1 = cp.Parameter(nonneg=True)
    lambda_values = np.logspace(-8, 0, n_lmbda)
    
    train_errors = np.zeros((cv_n, n_lmbda, N))
    test_errors = np.zeros((cv_n, n_lmbda, N))
    beta_values = np.zeros((cv_n, n_lmbda, N))
    
    train_f = np.zeros((N, n_lmbda))
    test_f = np.zeros((N, n_lmbda))
    beta_f = np.zeros((N, n_lmbda))
    
    #Cross-Validation to prevent overtraining the model
    for i in range(0, N):
        
        idx_arr = tts(N, cv_n)
        X_init = X[i*n:(i+1)*n,:]
        Y_init = Y[:,i]
        
        for j in range(0, cv_n):
            test_idx = idx_arr[j,:]
            
            idx_set  = np.arange(0, cv_n)
            idx_set =  np.setdiff1d(idx_set, j)
            train_idx = idx_arr[idx_set,:].flatten()
            
            X_train = X_init[train_idx,:]
            Y_train = Y_init[train_idx]
            
            X_test  = X_init[test_idx,:]
            Y_test = Y_init[test_idx]
            
            if meth == 'aenet':
                problem = cp.Problem(cp.Minimize(aenet_objective_fn(X_train, Y_train, beta, lambda_1, lambda_2[i], ad_w_arr[i,:])))
            if meth == 'alasso':
                problem = cp.Problem(cp.Minimize(aenet_objective_fn(X_train, Y_train, beta, lambda_1, lambda_2[i], ad_w_arr[i,:])))
                
                
            #Compare solution to solution where where weights for coefficients that were 0 
            #in ENet are also zero. 
            train_err_int = []
            test_err_int = []
            beta_val_int = []
            
            for v in lambda_values:
                lambda_1.value = v
                problem.solve()
                train_err_int.append(mse(X_train, Y_train, beta))
                test_err_int.append(mse(X_test, Y_test, beta))
                beta_val_int.append(beta.value)

            
            
            train_errors[j,:,i] = train_err_int
            test_errors[j,:,i] = test_err_int
            #beta_values[j,:,i] = beta_val_int
    
    for i in range(0, N):
        test_f[i,:] = 1/X_init.shape[0]*np.sum(test_errors[:,:,i], axis = 0)
    return test_f

cv_n = 5
n_lmbda = 100
test_f = ada_meth_script(N, cv_n, n_lmbda, pn, X, Y, lambda_2, ad_w_arr[:,:,0], meth = 'aenet')
    
            
            # beta_v_t = np.zeros((n_lmbda, pn))
            # beta_err_count = np.zeros(n_lmbda)
            # beta_count = np.zeros(n_lmbda)
            
            # for i in range(0,n_lmbda):
            #     for j in range(0,pn):
            #         if beta_values[i][j] <= 0.00001:
            #             beta_v_t[i,j] = 0
            #         else:
            #             beta_v_t[i,j] = beta_values[i][j]
                        
            #     beta_err_count[i] = np.abs(np.sum(beta_h.astype(bool).astype(int) - beta_v_t[i,:].astype(bool).astype(int)))
            #     beta_count[i] = np.abs(np.sum(beta_v_t[i,:].astype(bool).astype(int)))
                
            # plt.figure(0)
            # plt.scatter(lambda_values, beta_err_count)
            # plt.title('Number of Incorrect Nonzero Coefficients vs. L1 Regularization Coefficient')
            # plt.xlabel('L1 Coefficient')
            # plt.ylabel('# of Coefficients')
            # #Make sure to connect the number of incorrect nonzero coefficients to the 
            # #number of nonzero coefficients from the elastic net
            
            # #Create loop to feed in all results from unit test to determine whether AEnet
            # #controls variance in parameters 
            
            
            # plt.figure(1)
            # plt.scatter(lambda_values, beta_count)
            # plt.title('Number of Non-Zero Coefficients vs. L1 Regularization Coefficient')
            # plt.xlabel('L1 Coefficient')
            # plt.ylabel('# of Coefficients')
            
            # #Identify index of first lambda_1 value where set of nonzero predictor variables is correct
            # lmbda_idx = np.min(np.where(beta_err_count ==0))
            # lmbda_1_sol = lambda_values[lmbda_idx]

#Figure out what to return.
#1) Solutions for all lambda_1 values
#2) Number of nonzero coefficients for each lambda_1 value
#3) Prediction error for each lambda_1
 

#Create convergence condition to find approximate first index correspnding to 0
#incorrect nonzero coefficients

#Return vector


#Multiprocessing template for cross validation with y, lambda_2, lambda_1
# from multiprocessing import Pool

# # Assign a value to gamma and find the optimal x.
# def get_x(gamma_value):
#     gamma.value = gamma_value
#     result = prob.solve()
#     return x.value

# # Parallel computation (set to 1 process here).
# pool = Pool(processes = 1)
# x_values = pool.map(get_x, gamma_vals)


#Calculate adaptive weights from coefficients

#Plot the identity line and histograms of coefficients

#Compare number of incorrect nonzero terms for adaptive Lasso, Lasso and Eleastic NEt
