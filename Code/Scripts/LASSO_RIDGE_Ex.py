#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:45:22 2023

@author: evanshapiro
"""
# beta := set of coefficients defined for model. 
# Cov  := Covariance matrix

# Generate synthetic example problem to apply adaptive elastic net to. 
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

def linear_f_AENET(X, beta, mu, sigma):
    y = np.dot(X, beta) + np.random.normal(mu,sigma)
    #X_sim = 
    return y
 
def cor_mat(X):
    Cor = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        for j in range(0,X.shape[0]):
            Cor[i,j] = X[i,j]/(np.sqrt(X[i,i]*X[j,j]))
    return Cor
            
np.random.seed(42)
Cov = np.array([[1, 0.85, -.35, 0.1], [0.85, 1, 0,0 ], [-0.35, 0,1, 0], [0.1,0,0,1]])
Cov_s = np.dot(Cov.T, Cov)
Cor = cor_mat(Cov_s)
# Cor = np.identity(4)

mean = np.zeros(Cov.shape[0])
X  = np.random.multivariate_normal(mean,Cor, (5000))
mu = 0
sigma = 1
beta =np.array([1,-1,2,0])
Y = np.dot(X, beta) + np.random.normal(0,1, 5000)

alpha = np.linspace(0.001,1,1000)
grid = {'alpha':alpha}

# reg = linear_model.Lasso()
# clf = GridSearchCV(reg, grid)
# clf.fit(X,Y)
# print('Best model coefficients:', clf.best_estimator_.coef_)
#Coeffs are accurate up to sign
#print(reg.coef_)
#Control for covariates in research. You are blindly applying analysis to data
#without controlling for covariation. Jesus fucking christ dude. Are you a noob.

#Perform sequence of regressions to find partial correlation coefficients
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(X[:,0].reshape(-1,1), Y)
Y_res = Y-reg.predict(X[:,0].reshape(-1,1))
reg_2 = linear_model.LinearRegression(fit_intercept=False)
reg_2.fit(X[:,1].reshape(-1,1), Y_res)
Y_res_2 = Y_res - reg_2.predict(X[:,1].reshape(-1,1))
reg_3 = linear_model.LinearRegression(fit_intercept=False)
reg_3.fit(X[:,2].reshape(-1,1), Y_res_2)

Y_res_3 = Y_res_2 - reg_3.predict(X[:,2].reshape(-1,1))

reg = linear_model.LinearRegression(fit_intercept=False).fit(X[:,3].reshape(-1,1), Y_res_3)

