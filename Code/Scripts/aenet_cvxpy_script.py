#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:22:09 2023

@author: evanshapiro
"""
#pn = number of parameters
import cvxpy as cp

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

#aenet objective function takes weights from a lasso and uses them to weight
# the coefficients in the regularization function 

#Weights \hat{w}_j :=(|\hat{beta}_j(enet)| +1/n)^{-\gamma}; gamma \ge 0
#lambda_2:= lambda_2 from original elastic-net 
#lambda_1*:= New hyperparameter thatis determined via CV, or some such nonsense.

def aenet_objective_fn(X, Y, beta, lambd_1, lambd_2, weights):
    return loss_fn(X, Y, beta) + lambd_1 * cp.norm1(cp.multiply(weights,beta)) + lambd_2*cp.norm2(beta)


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


n_lmbda = 100  # number of lambda_1 values
beta = cp.Variable(pn)
lambda_1 = cp.Parameter(nonneg=True)
#lambd_2 = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(aenet_objective_fn(X[0:n,:], Y[:,0], beta, lambda_1, lambda_2[0], ad_w_arr[0,:])))

#Compare solution to solution where where weights for coefficients that were 0 
#in ENet are also zero. 

lambda_values = np.logspace(-8, 0, n_lmbda)
train_errors = []
test_errors = []
beta_values = []
for v in lambda_values:
    lambda_1.value = v
    #lambd_2.value = v
    problem.solve()
    train_errors.append(mse(X[0:n,:], Y[:,0], beta))
    test_errors.append(mse(X[0:n,:], Y[:,0], beta))
    beta_values.append(beta.value)
    


beta_v_t = np.zeros((n_lmbda, pn))
beta_err_count = np.zeros(n_lmbda)
beta_count = np.zeros(n_lmbda)

for i in range(0,n_lmbda):
    for j in range(0,pn):
        if beta_values[i][j] <= 0.00001:
            beta_v_t[i,j] = 0
        else:
            beta_v_t[i,j] = beta_values[i][j]
            
    beta_err_count[i] = np.abs(np.sum(beta_h.astype(bool).astype(int) - beta_v_t[i,:].astype(bool).astype(int)))
    beta_count[i] = np.abs(np.sum(beta_v_t[i,:].astype(bool).astype(int)))
    
plt.figure(0)
plt.scatter(lambda_values, beta_err_count)
plt.title('Number of Incorrect Nonzero Coefficients vs. L1 Regularization Coefficient')
plt.xlabel('L1 Coefficient')
plt.ylabel('# of Coefficients')
#Make sure to connect the number of incorrect nonzero coefficients to the 
#number of nonzero coefficients from the elastic net

#Create loop to feed in all results from unit test to determine whether AEnet
#controls variance in parameters 


plt.figure(1)
plt.scatter(lambda_values, beta_count)
plt.title('Number of Non-Zero Coefficients vs. L1 Regularization Coefficient')
plt.xlabel('L1 Coefficient')
plt.ylabel('# of Coefficients')

#Identify index of first lambda_1 value where set of nonzero predictor variables is correct
lmbda_idx = np.min(np.where(beta_err_count ==0))
lmbda_1_sol = lambda_values[lmbda_idx]

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