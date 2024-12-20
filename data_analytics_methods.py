# -*- coding: utf-8 -*-
"""
data_analytics_methods.py
Version: 0.5.0
Date: 2024/02/25
Author: Fabian Mohr fmohr@mit.edu

# GNU General Public License version 3 (GPL-3.0) ------------------------------

Smart Process Analytics for Process Monitoring
Copyright (C) 2024 Fabian Mohr & Elia Arnese-Feffin

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/gpl-3.0.html.

-------------------------------------------------------------------------------

To attribute credit to the author of the software, please refer to the
companion Journal Paper:
F. Mohr, E. Arnese-Feffin, M. Barolo, and R. D. Braatz (2025):
    Smart Process Analytics for Process Monitoring.
    *Computers and Chemical Engineering*, **194**, 108918.
    DOI: https://doi.org/10.1016/j.compchemeng.2024.108918.

"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import f, chi2
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy.linalg import fractional_matrix_power
from sklearn.metrics.pairwise import pairwise_kernels
from copy import deepcopy
from SVDD import BaseSVDD
import statsmodels.api as sm
import pickle

def getControlLimits(TSquared,Q,alpha, Tmethod, Qmethod, method, Components = None, latent = None, S = None, TrSquared = None, lagp = None):
    if latent is None:
        Qmethod = 'chi2_distribution'
    if Tmethod == 'F_distribution':
        N = np.shape(TSquared)[0]
        thresholdTsquared = Components*(N**2-1)/(N*(N-Components))*f.ppf(1-alpha,Components,N-Components)
    elif Tmethod == 'chi2_distribution':
        a = np.mean(TSquared)
        b = np.var(TSquared)
        thresholdTsquared = b/(2*a)*chi2.ppf(1 - alpha,2*a**2/b)
    if Qmethod == 'Jackson_Mudholkar':
        lambda_minus = latent[:Components];
        theta_1 = S.trace() - sum(lambda_minus);
        theta_2 = (S@S).trace() - sum(lambda_minus**2)
        theta_3 = (S@S@S).trace() - sum(lambda_minus**3)
        h_0 = 1-(2*theta_1*theta_3)/(3*theta_2**2)
        #alpha/2 is the correct value if you follow the example in Jackson and Mudholkar
        c_alpha = norm.ppf(1-alpha/2) 
        thresholdQ = theta_1*(c_alpha*h_0*(2*theta_2)**(0.5)/theta_1+1+theta_2*h_0*(h_0-1)/theta_1**2)**(1/h_0)
    elif Qmethod == 'chi2_distribution':
        a = np.mean(Q)
        b = np.var(Q)
        thresholdQ = b/(2*a)*chi2.ppf(1 - alpha,2*a**2/b)
    if method in ['CVA', 'KDE-CVA']:
        if Tmethod == 'F_distribution':
            N = np.shape(TrSquared)[0]
            thresholdTrsquared = (lagp - Components)*(N**2-1)/(N*(N-(lagp - Components)))*f.ppf(1-alpha,(lagp - Components),N-(lagp - Components))
        elif Tmethod == 'chi2_distribution':
            a = np.mean(TrSquared)
            b = np.var(TrSquared)
            thresholdTrsquared = b/(2*a)*chi2.ppf(1 - alpha,2*a**2/b)
        return (thresholdTsquared,thresholdTrsquared,thresholdQ)
    
        
    return (thresholdTsquared,thresholdQ)
        
def CVpartition(X, y, Type = 'Re_KFold', K = 5, Nr = 10, random_state = None, if_have_output = 0):
    if if_have_output:
        if Type == 'Re_KFold':
            CV = RepeatedKFold(n_splits= int(K), n_repeats= Nr, random_state =random_state)
            for train_index, val_index in CV.split(X,y):
                yield (X[train_index], y[train_index], X[val_index], y[val_index])
                
        elif Type == 'Timeseries':
            TS = TimeSeriesSplit(n_splits=int(K))
            for train_index, val_index in TS.split(X,y):
                yield (X[train_index], y[train_index], X[val_index], y[val_index])           
      
        else:
            print('Wrong type specified for data partition')
    else:
        if Type == 'Re_KFold':
            CV = RepeatedKFold(n_splits= int(K), n_repeats= Nr, random_state =random_state)
            for train_index, val_index in CV.split(X):
                yield (X[train_index], X[val_index])
                
        elif Type == 'Timeseries':
            TS = TimeSeriesSplit(n_splits=int(K))
            for train_index, val_index in TS.split(X):
                yield (X[train_index], X[val_index])           
      
        else:
            print('Wrong type specified for data partition')
            
        
def PlotStatistics(tSquared,thresholdTsquared,Q,thresholdQ,model_index,data_case):
    #Plot Tsqared statistics
    N = np.shape(Q)[0]
    plt.figure()
    plt.plot(range(0,N),tSquared,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*thresholdTsquared,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('$T^2$')
    plt.title('$T^2$ statistic of ' + model_index + ' on ' + data_case + ' data')
    
    #Plot Q statistics
    plt.figure()
    plt.plot(range(0,N),Q,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*thresholdQ,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('Q')
    plt.title('Q statistic of ' + model_index + ' on ' + data_case + ' data')
    
def PlotStatisticsCVA(tSquared,thresholdTsquared,trSquared,thresholdTrsquared,Q,thresholdQ,model_index,data_case):
    #Plot Tsqared statistics
    N = np.shape(Q)[0]
    plt.figure()
    plt.plot(range(0,N),tSquared,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*thresholdTsquared,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('$T^2$')
    plt.title('$T^2$ statistic of ' + model_index + ' on ' + data_case + ' data')
    
    #Plot Q statistics
    plt.figure()
    plt.plot(range(0,N),Q,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*thresholdQ,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('Q')
    plt.title('Q statistic of ' + model_index + ' on ' + data_case + ' data')
    
    #Plot Trsqared statistics
    N = np.shape(Q)[0]
    plt.figure()
    plt.plot(range(0,N),trSquared,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*thresholdTrsquared,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('$Tr^2$')
    plt.title('$Tr^2$ statistic of ' + model_index + ' on ' + data_case + ' data')

def PlotStatisticsSVDD(distance,radius,model_index,data_case):
    N = np.shape(distance)[0]
    plt.figure()
    plt.plot(range(0,N),distance,
             color = 'black',           
            linewidth=1)
    plt.plot(range(0,N),np.ones((N,1))*radius,color = 'red',linestyle='--')
    plt.xlabel('Sample number')
    plt.ylabel('Distance')
    plt.title('D statistic of ' + model_index + ' on ' + data_case + ' data')
    
def ApplyAlgorithm(X,X_test,method,alpha = 0.01,y = None,y_test = None,K_fold = 5,Nr = 10,if_have_output = 0, Tmethod = 'chi2_distribution', Qmethod = 'chi2_distribution', if_have_hyper_params = 0, hyper_params_file_name = None, plot_training = False, plot_testing = False):
    #%%% PCA
    if method == 'PCA':
        num_iter = K_fold*Nr
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        Components = np.array(range(1,min(N,p,10)+1))
        Violations_result = np.zeros((min(N,p,10),K_fold*Nr))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, X_val in CVpartition(X, y, K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                pca = PCA(n_components = Components[-1]).fit(X_train)
                i_idx = []
                for i in range(len(Components)):
                    i_idx.append(i)
                    # pca = PCA(n_components = Components[i]).fit(X_train)
                    latent = pca.explained_variance_
                    P = pca.components_[i_idx,:].T #obtain PCA loadings
                    S = np.cov(X_train,rowvar = False)
                    
                    #Calculate train statistics
                    tSquared = np.zeros((np.shape(X_train)[0],1))
                    Q = np.zeros((np.shape(X_train)[0],1))
                    for k in range(np.shape(X_train)[0]):
                        tSquared[k] = X_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_train[k,:].T
                        Q[k] = X_train[k,:]@(np.identity(p)-P@P.T)@X_train[k,:].T
                        
                    #Calculate val statistics
                    tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                    QVal = np.zeros((np.shape(X_val)[0],1))
                    for k in range(np.shape(X_val)[0]):
                        tSquaredVal[k] = X_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_val[k,:].T
                        QVal[k] = X_val[k,:]@(np.identity(p)-P@P.T)@X_val[k,:].T
                    
                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)
                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                    Violations_result[i,counter-1] += np.abs(Violations-alpha)
                    
            Violations_mean = np.sum(Violations_result,axis = 1)/counter
                 
            Violations_std = np.std(Violations_result, axis = 1)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0]]
            Violations_bar = Violations_min + Violations_std[idx[0]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            # print('PCA: A_min = {0:2d}, A_oster = {1:2d}'.format(Components[idx[0]], Components[idx_final[0]])) # REMOVEME
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        
        #fit the final model using opt hyper_params
        pcaFinal = PCA(n_components = hyper_params['num_components']).fit(X)
        latent = pcaFinal.explained_variance_
        P = pcaFinal.components_.T #obtain PCA loadings
        S = np.cov(X,rowvar = False)
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = X[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X[i,:].T
            Q[i] = X[i,:]@(np.identity(p)-P@P.T)@X[i,:].T
        
        if X_test is not None:
            #Calculate test statistics
            X_test = scaler.transform(X_test)
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = X_test[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_test[i,:].T
                QTest[i] = X_test[i,:]@(np.identity(p)-P@P.T)@X_test[i,:].T  
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method,Components = hyper_params['num_components'], latent = latent, S = S)
    
    #%%% PLS
    elif method == 'PLS':
        num_iter = K_fold*Nr
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        Components = np.array(range(1,min(N,p,10)+1))
        Violations_result = np.zeros((min(N,p,10),K_fold*Nr))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                scaler.fit(y_train)
                y_train = scaler.transform(y_train)
                y_val = scaler.transform(y_val)
                # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(X_train,y_train)
                i_idx = []
                #Check if P and R are correctly defined in reference paper
                for i in range(len(Components)):
                    i_idx.append(i)
                    PLS_model = PLSRegression(n_components = Components[i], scale = False).fit(X_train,y_train)
                    P = PLS_model.x_loadings_
                    R = PLS_model.x_rotations_
                    # R = PLS.x_loadings_[:,i_idx]
                    # P = (np.linalg.pinv(R)).transpose()
                    T = X_train@R
                    # T_val = X_val@R
                    
                    #Calculate train statistics
                    Scm = np.linalg.pinv(T.T@T/(np.shape(X_train)[0] - 1))
                    tSquared = np.zeros((np.shape(X_train)[0],1))
                    Q = np.zeros((np.shape(X_train)[0],1))
                    for k in range(np.shape(X_train)[0]):
                        tSquared[k] = T[k,:]@Scm@T[k,:].T
                        e = X_train[k,:]@(np.identity(p) - R@P.T)
                        Q[k] = e@e.T
                        # tSquared[k] = X_train[k,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_train[k,:].T
                        # Q[k] = X_train[k,:]@(np.identity(p)-P@R.T)@X_train[k,:].T
                        
                    #Calculate test statistics
                    T_val = X_val@R
                    tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                    QVal = np.zeros((np.shape(X_val)[0],1))
                    for k in range(np.shape(X_val)[0]):
                        tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                        e = X_val[k,:]@(np.identity(p) - R@P.T)
                        QVal[k] = e@e.T
                        # tSquaredVal[k] = X_val[k,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_val[k,:].T
                        # QVal[k] = X_val[k,:]@(np.identity(p)-P@R.T)@X_val[k,:].T
                        
                    #Calculate T^2 threshold following Vicky's definition
                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                    Violations_result[i,counter-1] += np.abs(Violations-alpha)
                    # PLS_para = PLS.coef_.reshape(-1,1)
                    # yhat_val = np.dot(X_val, PLS_para)
                    # MSE_result[i] += np.sum((yhat_val-y_val)**2)/y.shape[0]
                    
            Violations_mean = np.sum(Violations_result,axis = 1)/counter
                 
            Violations_std = np.std(Violations_result, axis = 1)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0]]
            Violations_bar = Violations_min + Violations_std[idx[0]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            # print('PLS: A_min = {0:2d}, A_oster = {1:2d}'.format(Components[idx[0]], Components[idx_final[0]])) # REMOVEME
        with open(hyper_params_file_name, 'wb') as w:
            pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
    
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(y)
        y = scalery.transform(y)
        
        #fit the final model using opt hyper_params
        PLS_final = PLSRegression(scale = False, n_components=int(hyper_params['num_components'])).fit(X,y)
        # T1 = PLS_model.x_scores_
        # R = PLS_model.x_loadings_
        # P = (np.linalg.pinv(R)).transpose()
        #Change definitions to the following for all PLS algorithms
        P = PLS_final.x_loadings_
        R = PLS_final.x_rotations_
        T = X@R
        #print(P.transpose()@R)
        # T_test = X_test@R
        
        #Calculate train statistics
        Scm = np.linalg.pinv(T.T@T/(N-1))
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = T[i,:]@Scm@T[i,:].T
            e = X[i,:]@(np.identity(p) - R@P.T)
            Q[i] = e@e.T
            # tSquared[i] = X[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X[i,:].T
            # Q[i] = X[i,:]@(np.identity(p)-P@R.T)@X[i,:].T
        
        if X_test is not None:
            #Calculate test statistics
            X_test = scaler.transform(X_test)
            T_test = X_test@R
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = T_test[i,:]@Scm@T_test[i,:].T
                e = X_test[i,:]@(np.identity(p) - R@P.T)
                QTest[i] = e@e.T
                # tSquaredTest[i] = X_test[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_test[i,:].T
                # QTest[i] = X_test[i,:]@(np.identity(p)-P@R.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
        
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'])
    
    #%%% CVA
    elif method == 'CVA':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        #Define hyperparameters
        lags = np.array(range(1,4))
        Components = np.array(range(1,min(N,p,10)+1))
        # lags = np.array(range(1,2))
        # Components = np.array(range(1,min(N,p,1)+1))
        Violations_result = np.zeros((len(Components),len(lags),K_fold))
        counter = 0
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            #Crossval procedure
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                scaler.fit(y_train)
                y_train = scaler.transform(y_train)
                y_val = scaler.transform(y_val)
                N_train = np.shape(X_train)[0]
                N_val = np.shape(X_val)[0]
                p_train = np.shape(X_train)[1]
                yp_train = np.shape(y_train)[1]
                for i in range(len(Components)):
                    for j in range(len(lags)):
                        #Create past and future vectors
                        P = np.zeros((lags[j]*(p_train+yp_train),N_train-lags[j]))
                        F = np.zeros((lags[j]*yp_train,N_train-lags[j]))
                        P_val = np.zeros((lags[j]*(p_train+yp_train),N_val-lags[j]))
                        F_val = np.zeros((lags[j]*yp_train,N_val-lags[j]))
                        for m in range(N_train-lags[j]):
                            for n in range(lags[j]):
                                P[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_train[m-1-n,:].T
                                P[lags[j]+n*p_train:lags[j]+(n+1)*p_train,m-lags[j]] = X_train[m-1-n,:].T
                            for n in range(lags[j]):
                                F[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_train[m+n,:].T
                        for m in range(N_val-lags[j]):
                            for n in range(lags[j]):
                                P_val[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_val[m-1-n,:].T
                                P_val[lags[j]+n*p_train:lags[j]+(n+1)*p_train,m-lags[j]] = X_val[m-1-n,:].T
                            for n in range(lags[j]):
                                F_val[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_val[m+n,:].T
                        P = P.T
                        F = F.T
                        P_val = P_val.T
                        F_val = F_val.T
                        #Scale vectors
                        scaler_P = StandardScaler(with_mean=True, with_std=True)
                        scaler_P.fit(P)
                        P_scale = scaler_P.transform(P)
                        P_scale_val = scaler_P.transform(P_val)
                        scaler_F = StandardScaler(with_mean=True, with_std=True)
                        scaler_F.fit(F)
                        F_scale = scaler_F.transform(F)
                        # F_scale_val = scaler_F.transform(F_val)
                        #Apply cva to P and F vectors
                        Sxx = 1/(np.shape(P)[0]-1)*P_scale.T@P_scale
                        Syy = 1/(np.shape(F)[0]-1)*F_scale.T@F_scale
                        Sxy = 1/(np.shape(F)[0]-1)*P_scale.T@F_scale
                        
                        
                        U, S, V = np.linalg.svd(fractional_matrix_power(Sxx, -0.5)@Sxy@fractional_matrix_power(Syy,-0.5), full_matrices=True)
                           
                        J = U.T@fractional_matrix_power(Sxx,-0.5)
                        Jd = J[:i+1,:].T 
                        Jr = J[i+1:-10,:].T 
                        
                        #Calculate train statistics
                        tSquared = np.zeros((np.shape(P)[0],1))
                        trSquared = np.zeros((np.shape(P)[0],1))
                        Q = np.zeros((np.shape(P)[0],1))
                        for k in range(np.shape(P)[0]):
                            tSquared[k] = P_scale[k,:]@Jd@Jd.T@P_scale[k,:].T
                            trSquared[k] = P_scale[k,:]@Jr@Jr.T@P_scale[k,:].T
                            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale[k,:].T
                            Q[k] = rt.T@rt
                            
                        #Calculate val statistics
                        tSquaredVal = np.zeros((np.shape(P_val)[0],1))
                        trSquaredVal = np.zeros((np.shape(P_val)[0],1))
                        QVal = np.zeros((np.shape(P_val)[0],1))
                        for k in range(np.shape(P_val)[0]):
                            tSquaredVal[k] = P_val[k,:]@Jd@Jd.T@P_val[k,:].T
                            trSquaredVal[k] = P_val[k,:]@Jr@Jr.T@P_val[k,:].T
                            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale_val[k,:].T
                            QVal[k] =rt.T@rt
                        lagp = np.shape(P)[1]  
                        thresholdTsquared, thresholdTrsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = None, S = None, TrSquared = trSquared, lagp = lagp)                
                        TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                        QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                        TrViolations = np.where(np.any(trSquaredVal > thresholdTrsquared, axis = 1))[0]
                        # Violations = len(np.unique(np.concatenate((TViolations,QViolations,TrViolations))))/np.shape(QVal)[0]
                        Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                        Violations_result[i,j,counter-1] += np.abs(Violations-alpha)
            
            Violations_mean = np.sum(Violations_result,axis = 2)/counter
                 
            Violations_std = np.std(Violations_result, axis = 2)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}     
            
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            # print('CVA: (A, h)_min = ({0:2d}, {1:2d}), (A, h)_oster = ({2:2d}, {3:2d})'.format(Components[idx[0]], lags[idx[0]], Components[idx_final[0]], lags[idx_final[0]])) # REMOVEME
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(y)
        y = scalery.transform(y)
        
        #Full dataset run
        N_train = np.shape(X)[0]
        
        p_train = np.shape(X)[1]
        yp_train = np.shape(y)[1]
        
        P = np.zeros((hyper_params['lag']*(p_train+yp_train),N_train-hyper_params['lag']))
        F = np.zeros((hyper_params['lag']*yp_train,N_train-hyper_params['lag']))
        for m in range(N_train-hyper_params['lag']):
            for n in range(hyper_params['lag']):
                # P[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m-n,:].T
                # P[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X[m-n,:].T
                P[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m-1-n,:].T
                P[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X[m-1-n,:].T
            for n in range(hyper_params['lag']):
                # F[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m+n,:].T
                F[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m+n,:].T
        P = P.T
        F = F.T
        scaler_P = StandardScaler(with_mean=True, with_std=True)
        scaler_P.fit(P)
        P_scale = scaler_P.transform(P)
        scaler_F = StandardScaler(with_mean=True, with_std=True)
        scaler_F.fit(F)
        F_scale = scaler_F.transform(F)
        # F_scale_test = scaler_F.transform(F_test)
        # Apply cva to P and F vectors
        
        Sxx = 1/(np.shape(P)[0]-1)*P_scale.T@P_scale
        Syy = 1/(np.shape(F)[0]-1)*F_scale.T@F_scale
        Sxy = 1/(np.shape(F)[0]-1)*P_scale.T@F_scale
        
        U, S, V = np.linalg.svd(fractional_matrix_power(Sxx, -0.5)@Sxy@fractional_matrix_power(Syy,-0.5), full_matrices=True)
           
        J = U.T@fractional_matrix_power(Sxx,-0.5)
        Jd = J[:hyper_params['num_components']+1,:].T 
        Jr = J[hyper_params['num_components']+1:,:].T 
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(P)[0],1))
        trSquared = np.zeros((np.shape(P)[0],1))
        Q = np.zeros((np.shape(P)[0],1))
        for i in range(np.shape(P)[0]):
            tSquared[i] = P_scale[i,:]@Jd@Jd.T@P_scale[i,:].T
            trSquared[i] = P_scale[i,:]@Jr@Jr.T@P_scale[i,:].T
            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale[i,:].T
            Q[i] = rt.T@rt
            
        if X_test is not None:
            #Calculate test statistics
            X_test = scaler.transform(X_test)
            y_test = scalery.transform(y_test)
            N_test = np.shape(X_test)[0]
            P_test = np.zeros((hyper_params['lag']*(p_train+yp_train),N_test-hyper_params['lag']))
            F_test = np.zeros((hyper_params['lag']*yp_train,N_test-hyper_params['lag']))
            for m in range(N_test-hyper_params['lag']):
                for n in range(hyper_params['lag']):
                    # P_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m-n,:].T
                    # P_test[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X_test[m-n,:].T
                    P_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m-1-n,:].T
                    P_test[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X_test[m-1-n,:].T
                for n in range(hyper_params['lag']):
                    # F_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m+n,:].T  
                    F_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m+n,:].T 
            P_test = P_test.T
            F_test = F_test.T
            P_scale_test = scaler_P.transform(P_test)
            tSquaredTest = np.zeros((np.shape(P_test)[0],1))
            trSquaredTest = np.zeros((np.shape(P_test)[0],1))
            QTest = np.zeros((np.shape(P_test)[0],1))
            for i in range(np.shape(P_test)[0]):
                tSquaredTest[i] = P_scale_test[i,:]@Jd@Jd.T@P_scale_test[i,:].T
                trSquaredTest[i] = P_scale_test[i,:]@Jr@Jr.T@P_scale_test[i,:].T
                rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale_test[i,:].T
                QTest[i] =rt.T@rt
        else:
            tSquaredTest = None
            trSquaredTest = None
            QTest = None
        
        lagp = np.shape(P)[1]  
        thresholdTsquared, thresholdTrsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'], latent = None, S = None, TrSquared = trSquared, lagp = lagp)      
    
    #%%% DPCA
    elif method == 'DPCA':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        Components = np.array(range(1,min(N,p,10)+1))
        lags = np.array(range(1,4))
        Violations_result = np.zeros((len(Components),len(lags),K_fold))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, X_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                # scaler = StandardScaler(with_mean=True, with_std=True)
                # scaler.fit(X_train)
                # X_train = scaler.transform(X_train)
                # X_val = scaler.transform(X_val)
                for j in range(len(lags)):
                    #Get dimensions
                    N_train = np.shape(X_train)[0]
                    N_val = np.shape(X_val)[0]               
                    p_train = np.shape(X)[1]
                    Xdyn_train = np.zeros((N_train - lags[j],(lags[j]+1)*p_train))
                    Xdyn_val = np.zeros((N_val - lags[j],(lags[j]+1)*p_train))
                    for m in range(N_train - lags[j]):
                        for n in range(p_train):
                                Xdyn_train[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_train[m:m+lags[j]+1,n]
                    
                    for m in range(N_val - lags[j]):
                        for n in range(p_train):
                                Xdyn_val[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_val[m:m+lags[j]+1,n]
                    # Scale dynamics matrices
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    scaler.fit(Xdyn_train)
                    Xdyn_train = scaler.transform(Xdyn_train)
                    Xdyn_val = scaler.transform(Xdyn_val)
                    #Use new dynamic matrices as inputs
                    # X_train = Xdyn_train
                    # X_val = Xdyn_val
                    p_dyn = np.shape(Xdyn_train)[1]
                    
                    pca = PCA(n_components = Components[-1]).fit(Xdyn_train)
                    i_idx = []
                    for i in range(len(Components)):
                        i_idx.append(i)
                        latent = pca.explained_variance_
                        P = pca.components_[i_idx,:].T #obtain PCA loadings
                    
                        latent = pca.explained_variance_
                        P = pca.components_.T #obtain PCA loadings
                        S = np.cov(Xdyn_train,rowvar = False)
                        
                        #Calculate train statistics
                        tSquared = np.zeros((np.shape(Xdyn_train)[0],1))
                        Q = np.zeros((np.shape(Xdyn_train)[0],1))
                        for k in range(np.shape(Xdyn_train)[0]):
                            tSquared[k] = Xdyn_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xdyn_train[k,:].T
                            Q[k] = Xdyn_train[k,:]@(np.identity(p_dyn)-P@P.T)@Xdyn_train[k,:].T
                            
                        #Calculate val statistics
                        tSquaredVal = np.zeros((np.shape(Xdyn_val)[0],1))
                        QVal = np.zeros((np.shape(Xdyn_val)[0],1))
                        for k in range(np.shape(Xdyn_val)[0]):
                            tSquaredVal[k] = Xdyn_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xdyn_val[k,:].T
                            QVal[k] = Xdyn_val[k,:]@(np.identity(p_dyn)-P@P.T)@Xdyn_val[k,:].T
                            
                        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)                
                        TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                        QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                        Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                        Violations_result[i,j,counter-1] += np.abs(Violations-alpha)
                        
            Violations_mean = np.sum(Violations_result,axis = 2)/counter
                 
            Violations_std = np.std(Violations_result, axis = 2)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            # hyper_params['lag'] = 2
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        #fit the final model using opt hyper_params
        
        # pre-process data
        # scaler = StandardScaler(with_mean=True, with_std=True)
        # scaler.fit(X)
        # X = scaler.transform(X)
        # X_test = scaler.transform(X_test)
        
        #Get dimensions
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        Xdyn = np.zeros((N - hyper_params['lag'],(hyper_params['lag']+1)*p))
        for m in range(N - hyper_params['lag']):
            for n in range(p):
                    Xdyn[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X[m:m+hyper_params['lag']+1,n]
        
        # Scale dynamics matrices
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(Xdyn)
        Xdyn = scaler.transform(Xdyn)
        
        #Use new dynamic matrices as inputs
        X = Xdyn.copy()
        p_dyn = np.shape(X)[1]
        pcaFinal = PCA(n_components = hyper_params['num_components']).fit(X)
        latent = pcaFinal.explained_variance_
        P = pcaFinal.components_.T #obtain PCA loadings
        S = np.cov(X,rowvar = False)
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = X[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X[i,:].T
            Q[i] = X[i,:]@(np.identity(p_dyn)-P@P.T)@X[i,:].T
        
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0]               
            Xdyn_test = np.zeros((N_test - hyper_params['lag'],(hyper_params['lag']+1)*p))
            for m in range(N_test - hyper_params['lag']):
                for n in range(p):
                        Xdyn_test[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X_test[m:m+hyper_params['lag']+1,n]
            Xdyn_test = scaler.transform(Xdyn_test)
            X_test = Xdyn_test.copy()
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = X_test[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_test[i,:].T
                QTest[i] = X_test[i,:]@(np.identity(p_dyn)-P@P.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'], latent = latent, S = S)
    
    #%%% DPLS
    elif method == 'DPLS':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        Components = np.array(range(1,min(N,p,10)+1))
        lags = np.array(range(1,6))
        Violations_result = np.zeros((len(Components),len(lags),K_fold))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                # X_train = scaler.transform(X_train)
                # X_val = scaler.transform(X_val)
                # scaler.fit(y_train)
                # y_train = scaler.transform(y_train)
                # y_val = scaler.transform(y_val)
                for j in range(len(lags)):
                    #Get dimensions
                    N_train = np.shape(X_train)[0]
                    N_val = np.shape(X_val)[0]               
                    p_train = np.shape(X)[1]
                    yp_train = np.shape(y)[1]
                    Xdyn_train = np.zeros((N_train - lags[j],(lags[j]+1)*p_train))
                    Xdyn_val = np.zeros((N_val - lags[j],(lags[j]+1)*p_train))
                    ydyn_train = np.zeros((N_train - lags[j],yp_train))
                    ydyn_val = np.zeros((N_val - lags[j],yp_train))
                    for m in range(N_train - lags[j]):
                        ydyn_train[m,:] = y_train[m+lags[j],:]
                        for n in range(p_train):
                            Xdyn_train[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_train[m:m+lags[j]+1,n]
                    
                    for m in range(N_val - lags[j]):
                        ydyn_val[m,:] = y_val[m+lags[j],:]
                        for n in range(p_train):
                            Xdyn_val[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_val[m:m+lags[j]+1,n]
                    # Scale dynamics matrices
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    scaler.fit(Xdyn_train)
                    Xdyn_train = scaler.transform(Xdyn_train)
                    Xdyn_val = scaler.transform(Xdyn_val)
                    scaler.fit(ydyn_train)
                    ydyn_train = scaler.transform(ydyn_train)
                    ydyn_val = scaler.transform(ydyn_val)
                    #Use new dynamic matrices as inputs
                    X_train = Xdyn_train.copy()
                    X_val = Xdyn_val.copy()
                    y_train = ydyn_train.copy()
                    y_val = ydyn_val.copy()
                    p_dyn = np.shape(X_train)[1]
                    # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(X_train,y_train)
                    i_idx = []
                    for i in range(len(Components)):
                        # i_idx.append(i)
                        PLS_model = PLSRegression(scale = False, n_components = Components[i]).fit(X_train,y_train)
                        P = PLS_model.x_loadings_
                        R = PLS_model.x_rotations_
                        # R = PLS.x_loadings_[:,i_idx]
                        # P = (np.linalg.pinv(R)).transpose()
                        T = X_train@R
                        # T_val = X_val@R
                        
                        #Calculate train statistics
                        Scm = np.linalg.pinv(T.T@T/(np.shape(X_train)[0] - 1))
                        tSquared = np.zeros((np.shape(X_train)[0],1))
                        Q = np.zeros((np.shape(X_train)[0],1))
                        for k in range(np.shape(X_train)[0]):
                            tSquared[k] = T[k,:]@Scm@T[k,:].T
                            e = X_train[k,:]@(np.identity(p_dyn) - R@P.T)
                            Q[k] = e@e.T
                            # tSquared[k] = X_train[k,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_train[k,:].T
                            # Q[k] = X_train[k,:]@(np.identity(p_dyn)-P@R.T)@X_train[k,:].T
                            
                        #Calculate test statistics
                        T_val = X_val@R
                        tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                        QVal = np.zeros((np.shape(X_val)[0],1))
                        for k in range(np.shape(X_val)[0]):
                            tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                            e = X_val[k,:]@(np.identity(p_dyn) - R@P.T)
                            QVal[k] = e@e.T
                            # tSquaredVal[k] = X_val[k,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_val[k,:].T
                            # QVal[k] = X_val[k,:]@(np.identity(p_dyn)-P@R.T)@X_val[k,:].T
                            
                        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                        TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                        QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                        Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                        Violations_result[i,j,counter-1] += np.abs(Violations-alpha)
            Violations_mean = np.sum(Violations_result,axis = 2)/counter
                 
            Violations_std = np.std(Violations_result, axis = 2)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            # print('DPLS: (A, h)_min = ({0:2d}, {1:2d}), (A, h)_oster = ({2:2d}, {3:2d})'.format(Components[idx[0]], lags[idx[0]], Components[idx_final[0]], lags[idx_final[0]])) # REMOVEME
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        #fit the final model using opt hyper_params
        
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(y)
        y = scalery.transform(y)
        #Get dimensions
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        yp = np.shape(y)[1]
        Xdyn = np.zeros((N - hyper_params['lag'],(hyper_params['lag']+1)*p))
        ydyn = np.zeros((N - hyper_params['lag'],yp))
        for m in range(N - hyper_params['lag']):
            ydyn[m,:] = y[m+hyper_params['lag'],:]
            for n in range(p):
                Xdyn[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X[m:m+hyper_params['lag']+1,n]
        
        # Scale dynamics matrices
        scalerd = StandardScaler(with_mean=True, with_std=True)
        scalerd.fit(Xdyn)
        Xdyn = scalerd.transform(Xdyn)
        scalerdy = StandardScaler(with_mean=True, with_std=True)
        scalerdy.fit(ydyn)
        ydyn = scalerdy.transform(ydyn)
        #Use new dynamic matrices as inputs
        X = Xdyn.copy()
        y = ydyn.copy()
        p_dyn = np.shape(X)[1]
        
        PLS_final = PLSRegression(scale = False, n_components=int(hyper_params['num_components'])).fit(X,y)
        P = PLS_final.x_loadings_
        R = PLS_final.x_rotations_
        # R = PLS_final.x_loadings_
        # P = (np.linalg.pinv(R)).transpose()
        T = X@R
        # T_test = X_test@R
        
        #Calculate train statistics
        Scm = np.linalg.pinv(T.T@T/(N-1))
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = T[i,:]@Scm@T[i,:].T
            e = X[i,:]@(np.identity(p_dyn) - R@P.T)
            Q[i] = e@e.T
            # tSquared[i] = X[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X[i,:].T
            # Q[i] = X[i,:]@(np.identity(p_dyn)-P@R.T)@X[i,:].T
            
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0]   
            X_test = scaler.transform(X_test)
            y_test = scalery.transform(y_test)
            Xdyn_test = np.zeros((N_test - hyper_params['lag'],(hyper_params['lag']+1)*p))
            ydyn_test = np.zeros((N_test - hyper_params['lag'],yp))
            for m in range(N_test - hyper_params['lag']):
                ydyn_test[m,:] = y_test[m+hyper_params['lag'],:]
                for n in range(p):
                    Xdyn_test[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X_test[m:m+hyper_params['lag']+1,n]
            Xdyn_test = scalerd.transform(Xdyn_test)
            ydyn_test = scalerdy.transform(ydyn_test)
            X_test = Xdyn_test.copy()
            y_test = ydyn_test.copy()
            T_test = X_test@R
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = T_test[i,:]@Scm@T_test[i,:].T
                e = X_test[i,:]@(np.identity(p_dyn) - R@P.T)
                QTest[i] = e@e.T
                # tSquaredTest[i] = X_test[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X_test[i,:].T
                # QTest[i] = X_test[i,:]@(np.identity(p_dyn)-P@R.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'])
    
    #%%% KPCA
    elif method == 'KPCA':
        num_iter = K_fold*Nr
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        
        Components = np.array(range(1,min(N,p,10)+1))
        Kernels = ['rbf','poly']
        gamma = [1/5**2,1/10**2,1/20**2,1/50**2,1/100**2,1/150**2,1/200**2]
        degree = [2,3]
        Violations_result = np.zeros((len(Components),len(Kernels),len(gamma),len(degree),K_fold*Nr))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, X_val in CVpartition(X, y, Type = 'Re_KFold', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                X_train_backup = deepcopy(X_train)
                X_val_backup = deepcopy(X_val)
                N_train = np.shape(X_train)[0]
                counter += 1
                for j in range(len(Kernels)):
                    for m in range(len(gamma)):
                        if Kernels[j] == 'rbf':
                            x_train = pairwise_kernels(X_train_backup, metric = Kernels[j], gamma = gamma[m])
                            x_val = pairwise_kernels(X_val_backup, X_train_backup, metric = Kernels[j], gamma = gamma[m])
                            #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                            unitX = np.ones((np.shape(x_train)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                            unitX_val = np.ones((np.shape(x_val)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                            X_train = x_train-unitX@x_train-x_train@unitX+unitX@x_train@unitX
                            X_val = x_val-unitX_val@x_train-x_val@unitX+unitX_val@x_train@unitX
                            
                            p_kern = np.shape(X_train)[1]
                            pca = PCA(n_components = Components[-1]).fit(X_train)
                            i_idx = []
                            for i in range(len(Components)):
                                i_idx.append(i)
                                latent = pca.explained_variance_
                                P = pca.components_[i_idx,:].T #obtain PCA loadings
                                S = np.cov(X_train,rowvar = False)
                                
                                #Calculate train statistics
                                tSquared = np.zeros((np.shape(X_train)[0],1))
                                Q = np.zeros((np.shape(X_train)[0],1))
                                for k in range(np.shape(X_train)[0]):
                                    tSquared[k] = X_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_train[k,:].T
                                    Q[k] = X_train[k,:]@(np.identity(p_kern)-P@P.T)@X_train[k,:].T
                                    
                                #Calculate val statistics
                                tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                                QVal = np.zeros((np.shape(X_val)[0],1))
                                for k in range(np.shape(X_val)[0]):
                                    tSquaredVal[k] = X_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_val[k,:].T
                                    QVal[k] = X_val[k,:]@(np.identity(p_kern)-P@P.T)@X_val[k,:].T
                                    
                                thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)
                                TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                Violations_result[i,j,m,:,counter-1] += np.abs(Violations-alpha)
                                
                        if Kernels[j] == 'poly':
                            for n in range(len(degree)):                   
                                x_train = pairwise_kernels(X_train_backup, metric = Kernels[j], gamma = gamma[m], degree = degree[n])
                                x_val = pairwise_kernels(X_val_backup, X_train_backup, metric = Kernels[j], gamma = gamma[m], degree = degree[n])
                            
                                #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                unitX = np.ones((np.shape(x_train)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                                unitX_val = np.ones((np.shape(x_val)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                                X_train = x_train-unitX@x_train-x_train@unitX+unitX@x_train@unitX
                                X_val = x_val-unitX_val@x_train-x_val@unitX+unitX_val@x_train@unitX
                                
                                p_kern = np.shape(X_train)[1]
                                pca = PCA(n_components = Components[-1]).fit(X_train)
                                i_idx = []
                                for i in range(len(Components)):
                                    i_idx.append(i)
                                    latent = pca.explained_variance_
                                    P = pca.components_[i_idx,:].T #obtain PCA loadings
                                    S = np.cov(X_train,rowvar = False)
                                    
                                    #Calculate train statistics
                                    tSquared = np.zeros((np.shape(X_train)[0],1))
                                    Q = np.zeros((np.shape(X_train)[0],1))
                                    for k in range(np.shape(X_train)[0]):
                                        tSquared[k] = X_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_train[k,:].T
                                        Q[k] = X_train[k,:]@(np.identity(p_kern)-P@P.T)@X_train[k,:].T
                                        
                                    #Calculate val statistics
                                    tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                                    QVal = np.zeros((np.shape(X_val)[0],1))
                                    for k in range(np.shape(X_val)[0]):
                                        tSquaredVal[k] = X_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_val[k,:].T
                                        QVal[k] = X_val[k,:]@(np.identity(p_kern)-P@P.T)@X_val[k,:].T
                                        
                                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)
                                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                    Violations_result[i,j,m,n,counter-1] += np.abs(Violations-alpha)
                                    
            Violations_mean = np.sum(Violations_result,axis = 4)/counter
                 
            Violations_std = np.std(Violations_result, axis = 4)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3]]/np.sqrt(num_iter)
    
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
            
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['kernel'] = Kernels[idx_final[1]]
            hyper_params['gamma'] = gamma[idx_final[2]]
            if hyper_params['kernel'] == 'poly':
                hyper_params['degree'] = degree[idx_final[3]]
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)

        # Scaling
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X_s = scaler.transform(X)
        
        #relvant kernel options: rbf (gamma), poly (gamma, order)
        if hyper_params['kernel'] == 'rbf':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
        elif hyper_params['kernel'] == 'poly':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
        
        #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
        unitX = np.ones((np.shape(x)[0],np.shape(x)[0]))/np.shape(x)[0]
        X = x-unitX@x-x@unitX+unitX@x@unitX
        
        p_kern = np.shape(X)[1]
        #fit the final model using opt hyper_params
        pcaFinal = PCA(n_components = hyper_params['num_components']).fit(X)
        latent = pcaFinal.explained_variance_
        P = pcaFinal.components_.T #obtain PCA loadings
        S = np.cov(X,rowvar = False)
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = X[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X[i,:].T
            Q[i] = X[i,:]@(np.identity(p_kern)-P@P.T)@X[i,:].T
            
        if X_test is not None:
            #Calculate test statistics
            X_test = scaler.transform(X_test)
            #relvant kernel options: rbf (gamma), poly (gamma, order)
            if hyper_params['kernel'] == 'rbf':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
            elif hyper_params['kernel'] == 'poly':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
            unitX_test = np.ones((np.shape(x_test)[0],np.shape(x)[0]))/np.shape(x)[0]
            X_test = x_test-unitX_test@x-x_test@unitX+unitX_test@x@unitX
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = X_test[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_test[i,:].T
                QTest[i] = X_test[i,:]@(np.identity(p_kern)-P@P.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'], latent = latent, S = S)
    
    #%%% KPLS
    elif method == 'KPLS':
        num_iter = K_fold*Nr
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        
        Components = np.array(range(1,min(N,p,10)+1))
        Kernels = ['rbf','poly']
        gamma = [1/5**2,1/10**2,1/20**2,1/50**2,1/100**2,1/150**2,1/200**2]
        degree = [2,3]
        Violations_result = np.zeros((len(Components),len(Kernels),len(gamma),len(degree),K_fold*Nr))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = 'Re_KFold', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                scaler.fit(y_train)
                y_train = scaler.transform(y_train)
                y_val = scaler.transform(y_val)
                X_train_backup = deepcopy(X_train)
                X_val_backup = deepcopy(X_val)
                N_train = np.shape(X_train)[0]
                counter += 1
                for j in range(len(Kernels)):
                    for m in range(len(gamma)):
                        if Kernels[j] == 'rbf':
                            x_train = pairwise_kernels(X_train_backup, metric = Kernels[j], gamma = gamma[m])
                            x_val = pairwise_kernels(X_val_backup, X_train_backup, metric = Kernels[j], gamma = gamma[m])
                            #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                            unitX = np.ones((np.shape(x_train)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                            unitX_val = np.ones((np.shape(x_val)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                            X_train = x_train-unitX@x_train-x_train@unitX+unitX@x_train@unitX
                            X_val = x_val-unitX_val@x_train-x_val@unitX+unitX_val@x_train@unitX
                            N_train = np.shape(X_train)[0]
                            N_val = np.shape(X_val)[0]
                            
                            p_kern = np.shape(X_train)[1]
                            # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(X_train,y_train)
                            i_idx = []
                            for i in range(len(Components)):
                                PLS_model = PLSRegression(scale = False, n_components = Components[i]).fit(X_train,y_train)
                                P = PLS_model.x_loadings_
                                R = PLS_model.x_rotations_
                                # i_idx.append(i)
                                # R = PLS.x_loadings_[:,i_idx]
                                # P = (np.linalg.pinv(R)).transpose()
                                T = X_train@R
                                # T_test = X_test@R
                                
                                #Calculate train statistics
                                Scm = np.linalg.pinv(T.T@T/(np.shape(X_train)[0] - 1))
                                tSquared = np.zeros((np.shape(X_train)[0],1))
                                Q = np.zeros((np.shape(X_train)[0],1))
                                for k in range(np.shape(X_train)[0]):
                                    tSquared[k] = T[k,:]@Scm@T[k,:].T
                                    e = X_train[k,:]@(np.identity(p_kern) - R@P.T)
                                    Q[k] = e@e.T
                                    # tSquared[k] = X_train[k,:]@R@np.linalg.pinv(T.T@T/(N_train-1))@R.T@X_train[k,:].T
                                    # Q[k] = X_train[k,:]@(np.identity(p_kern)-P@R.T)@X_train[k,:].T
                                    
                                #Calculate val statistics
                                T_val = X_val@R
                                tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                                QVal = np.zeros((np.shape(X_val)[0],1))
                                for k in range(np.shape(X_val)[0]):
                                    tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                                    e = X_val[k,:]@(np.identity(p_kern) - R@P.T)
                                    QVal[k] = e@e.T
                                    # tSquaredVal[k] = X_val[k,:]@R@np.linalg.pinv(T.T@T/(N_val-1))@R.T@X_val[k,:].T
                                    # QVal[k] = X_val[k,:]@(np.identity(p_kern)-P@R.T)@X_val[k,:].T
                                thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                                TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                Violations_result[i,j,m,:,counter-1] += np.abs(Violations-alpha)
                                
                        if Kernels[j] == 'poly':
                            for n in range(len(degree)):                   
                                x_train = pairwise_kernels(X_train_backup, metric = Kernels[j], gamma = gamma[m], degree = degree[n])
                                x_val = pairwise_kernels(X_val_backup, X_train_backup, metric = Kernels[j], gamma = gamma[m], degree = degree[n])
                            
                                #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                unitX = np.ones((np.shape(x_train)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                                unitX_val = np.ones((np.shape(x_val)[0],np.shape(x_train)[0]))/np.shape(x_train)[0]
                                X_train = x_train-unitX@x_train-x_train@unitX+unitX@x_train@unitX
                                X_val = x_val-unitX_val@x_train-x_val@unitX+unitX_val@x_train@unitX
                                N_train = np.shape(X_train)[0]
                                N_val = np.shape(X_val)[0]
                                
                                p_kern = np.shape(X_train)[1]
                                # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(X_train,y_train)
                                i_idx = []
                                for i in range(len(Components)):
                                    PLS_model = PLSRegression(scale = False, n_components = Components[i]).fit(X_train,y_train)
                                    P = PLS_model.x_loadings_
                                    R = PLS_model.x_rotations_
                                    # i_idx.append(i)
                                    # R = PLS.x_loadings_[:,i_idx]
                                    # P = (np.linalg.pinv(R)).transpose()
                                    T = X_train@R
                                    # T_test = X_test@R
                                    
                                    #Calculate train statistics
                                    Scm = np.linalg.pinv(T.T@T/(np.shape(X_train)[0] - 1))
                                    tSquared = np.zeros((np.shape(X_train)[0],1))
                                    Q = np.zeros((np.shape(X_train)[0],1))
                                    for k in range(np.shape(X_train)[0]):
                                        tSquared[k] = T[k,:]@Scm@T[k,:].T
                                        e = X_train[k,:]@(np.identity(p_kern) - R@P.T)
                                        Q[k] = e@e.T
                                        # tSquared[k] = X_train[k,:]@R@np.linalg.pinv(T.T@T/(N_train-1))@R.T@X_train[k,:].T
                                        # Q[k] = X_train[k,:]@(np.identity(p_kern)-P@R.T)@X_train[k,:].T
                                        
                                    #Calculate val statistics
                                    T_val = X_val@R
                                    tSquaredVal = np.zeros((np.shape(X_val)[0],1))
                                    QVal = np.zeros((np.shape(X_val)[0],1))
                                    for k in range(np.shape(X_val)[0]):
                                        tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                                        e = X_val[k,:]@(np.identity(p_kern) - R@P.T)
                                        QVal[k] = e@e.T
                                        # tSquaredVal[k] = X_val[k,:]@R@np.linalg.pinv(T.T@T/(N_val-1))@R.T@X_val[k,:].T
                                        # QVal[k] = X_val[k,:]@(np.identity(p_kern)-P@R.T)@X_val[k,:].T
                                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                    Violations_result[i,j,m,n,counter-1] += np.abs(Violations-alpha)
                                    
                                                    
            Violations_mean = np.sum(Violations_result,axis = 4)/counter
                 
            Violations_std = np.std(Violations_result, axis = 4)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
            
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['kernel'] = Kernels[idx_final[1]]
            hyper_params['gamma'] = gamma[idx_final[2]]
            if hyper_params['kernel'] == 'poly':
                hyper_params['degree'] = degree[idx_final[3]]
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X_s = scaler.transform(X)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(y)
        y = scalery.transform(y)
        
        #relvant kernel options: rbf (gamma), poly (gamma, order)
        if hyper_params['kernel'] == 'rbf':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
        elif hyper_params['kernel'] == 'poly':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
        
        #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
        unitX = np.ones((np.shape(x)[0],np.shape(x)[0]))/np.shape(x)[0]
        X = x-unitX@x-x@unitX+unitX@x@unitX
        
        p_kern = np.shape(X)[1]
        
        PLS_final = PLSRegression(scale = False, n_components=int(hyper_params['num_components'])).fit(X,y)
        P = PLS_final.x_loadings_
        R = PLS_final.x_rotations_
        # R = PLS_final.x_loadings_
        # P = (np.linalg.pinv(R)).transpose()
        T = X@R
        # T_test = X_test@R
        
        #Calculate train statistics
        Scm = np.linalg.pinv(T.T@T/(N-1))
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = T[i,:]@Scm@T[i,:].T
            e = X[i,:]@(np.identity(p_kern) - R@P.T)
            Q[i] = e@e.T
            # tSquared[i] = X[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X[i,:].T
            # Q[i] = X[i,:]@(np.identity(p_kern)-P@R.T)@X[i,:].T
            
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0]
            X_test = scaler.transform(X_test)
            y_test = scalery.transform(y_test)
            #relvant kernel options: rbf (gamma), poly (gamma, order)
            if hyper_params['kernel'] == 'rbf':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
            elif hyper_params['kernel'] == 'poly':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
            unitX_test = np.ones((np.shape(x_test)[0],np.shape(x)[0]))/np.shape(x)[0]
            X_test = x_test-unitX_test@x-x_test@unitX+unitX_test@x@unitX
            T_test = X_test@R
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = T_test[i,:]@Scm@T_test[i,:].T
                e = X_test[i,:]@(np.identity(p_kern) - R@P.T)
                QTest[i] = e@e.T
                # tSquaredTest[i] = X_test[i,:]@R@np.linalg.pinv(T.T@T/(N_test-1))@R.T@X_test[i,:].T
                # QTest[i] = X_test[i,:]@(np.identity(p_kern)-P@R.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'])
    
    #%%% DKPCA
    elif method == 'DKPCA':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        
        Components = np.array(range(1,min(N,p,10)+1))
        lags = np.array(range(1,3))
        Kernels = ['rbf','poly']
        gamma = [1/5**2,1/10**2,1/20**2,1/50**2,1/100**2,1/150**2,1/200**2]
        degree = [2,3]
        
        Violations_result = np.zeros((len(Components),len(lags),len(Kernels),len(gamma),len(degree),K_fold))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, X_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                # scaler = StandardScaler(with_mean=True, with_std=True)
                # scaler.fit(X_train)
                # X_train = scaler.transform(X_train)
                # X_val = scaler.transform(X_val)
                for j in range(len(lags)):
                    #Get dimensions                
                    N_train = np.shape(X_train)[0]
                    N_val = np.shape(X_val)[0]               
                    p_train = np.shape(X)[1]
                    Xdyn_train = np.zeros((N_train - lags[j],(lags[j]+1)*p_train))
                    Xdyn_val = np.zeros((N_val - lags[j],(lags[j]+1)*p_train))
                    for m in range(N_train - lags[j]):
                        for n in range(p_train):
                                Xdyn_train[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_train[m:m+lags[j]+1,n]
                    
                    for m in range(N_val - lags[j]):
                        for n in range(p_train):
                                Xdyn_val[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_val[m:m+lags[j]+1,n]
                    # Scale dynamics matrices
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    scaler.fit(Xdyn_train)
                    Xdyn_train = scaler.transform(Xdyn_train)
                    Xdyn_val = scaler.transform(Xdyn_val)
                    #Use new dynamic matrices as inputs
                    # X_train = Xdyn_train
                    # X_val = Xdyn_val
                    # X_train_backup = deepcopy(Xdyn_train)
                    # X_val_backup = deepcopy(Xdyn_val)
                    N_train = np.shape(X_train)[0]
                    for q in range(len(Kernels)):
                        for r in range(len(gamma)):
                            #Create nonlinear matrix using kernel methods
                            if Kernels[q] == 'rbf':
                                xk_train = pairwise_kernels(Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                xk_val = pairwise_kernels(Xdyn_val, Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                unitX = np.ones((np.shape(xk_train)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                unitX_val = np.ones((np.shape(xk_val)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                Xk_train = xk_train-unitX@xk_train-xk_train@unitX+unitX@xk_train@unitX
                                Xk_val = xk_val-unitX_val@xk_train-xk_val@unitX+unitX_val@xk_train@unitX
                                p_kern = np.shape(Xk_train)[1]
                                
                                pca = PCA(n_components = Components[-1]).fit(Xk_train)
                                i_idx = []
                                for i in range(len(Components)):
                                    i_idx.append(i)
                                    latent = pca.explained_variance_
                                    P = pca.components_[i_idx,:].T #obtain PCA loadings
                                    S = np.cov(Xk_train,rowvar = False)
                                    
                                    #Calculate train statistics
                                    tSquared = np.zeros((np.shape(Xk_train)[0],1))
                                    Q = np.zeros((np.shape(Xk_train)[0],1))
                                    for k in range(np.shape(Xk_train)[0]):
                                        tSquared[k] = Xk_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xk_train[k,:].T
                                        Q[k] = Xk_train[k,:]@(np.identity(p_kern)-P@P.T)@Xk_train[k,:].T
                                        
                                    #Calculate val statistics
                                    tSquaredVal = np.zeros((np.shape(Xk_val)[0],1))
                                    QVal = np.zeros((np.shape(Xk_val)[0],1))
                                    for k in range(np.shape(Xk_val)[0]):
                                        tSquaredVal[k] = Xk_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xk_val[k,:].T
                                        QVal[k] = Xk_val[k,:]@(np.identity(p_kern)-P@P.T)@Xk_val[k,:].T
                                        
                                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)
                                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                    Violations_result[i,j,q,r,:,counter-1] += np.abs(Violations-alpha)
                                        
                            elif Kernels[q] == 'poly':
                                for s in range(len(degree)):
                                    xk_train = pairwise_kernels(Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                    xk_val = pairwise_kernels(Xdyn_val, Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                    #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                    unitX = np.ones((np.shape(xk_train)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                    unitX_val = np.ones((np.shape(xk_val)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                    Xk_train = xk_train-unitX@xk_train-xk_train@unitX+unitX@xk_train@unitX
                                    X_val = xk_val-unitX_val@xk_train-xk_val@unitX+unitX_val@xk_train@unitX
                                    p_kern = np.shape(Xk_train)[1]
                                    
                                    pca = PCA(n_components = Components[-1]).fit(Xk_train)
                                    i_idx = []
                                    for i in range(len(Components)):
                                        i_idx.append(i)
                                        latent = pca.explained_variance_
                                        P = pca.components_[i_idx,:].T #obtain PCA loadings
                                        S = np.cov(Xk_train,rowvar = False)
                                        
                                        #Calculate train statistics
                                        tSquared = np.zeros((np.shape(Xk_train)[0],1))
                                        Q = np.zeros((np.shape(Xk_train)[0],1))
                                        for k in range(np.shape(Xk_train)[0]):
                                            tSquared[k] = Xk_train[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xk_train[k,:].T
                                            Q[k] = Xk_train[k,:]@(np.identity(p_kern)-P@P.T)@Xk_train[k,:].T
                                            
                                        #Calculate val statistics
                                        tSquaredVal = np.zeros((np.shape(Xk_val)[0],1))
                                        QVal = np.zeros((np.shape(Xk_val)[0],1))
                                        for k in range(np.shape(Xk_val)[0]):
                                            tSquaredVal[k] = Xk_val[k,:]@P@np.linalg.pinv(P.T@S@P)@P.T@Xk_val[k,:].T
                                            QVal[k] = Xk_val[k,:]@(np.identity(p_kern)-P@P.T)@Xk_val[k,:].T
                                            
                                        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i], latent = latent, S = S)
                                        TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                        QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                        Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                        Violations_result[i,j,q,r,s,counter-1] += np.abs(Violations-alpha)
                                    
                                        
                    
            Violations_mean = np.sum(Violations_result,axis = 5)/counter
                 
            Violations_std = np.std(Violations_result, axis = 5)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3],idx[4]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3],idx[4]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            hyper_params['kernel'] = Kernels[idx_final[2]]
            hyper_params['gamma'] = gamma[idx_final[3]]
            if hyper_params['kernel'] == 'poly':
                hyper_params['degree'] = degree[idx_final[4]]
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        
        #fit the final model using opt hyper_params
        
        # Scale data
        # scaler = StandardScaler(with_mean=True, with_std=True)
        # scaler.fit(X)
        # X = scaler.transform(X)
        # X_test = scaler.transform(X_test)
        
        #Get dimensions
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        Xdyn = np.zeros((N - hyper_params['lag'],(hyper_params['lag']+1)*p))
        for m in range(N - hyper_params['lag']):
            for n in range(p):
                    Xdyn[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X[m:m+hyper_params['lag']+1,n]
        
        # Scale dynamics matrices
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(Xdyn)
        Xdyn = scaler.transform(Xdyn)
        #Use new dynamic matrices as inputs
        X_s = Xdyn.copy()
        #Create nonlinear matrix using kernel methods
        if hyper_params['kernel'] == 'rbf':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
        elif hyper_params['kernel'] == 'poly':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
        
        #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
        unitX = np.ones((np.shape(x)[0],np.shape(x)[0]))/np.shape(x)[0]
        X = x-unitX@x-x@unitX+unitX@x@unitX
                
        N = np.shape(X)[0]
        p_kern = np.shape(X)[1]
        pcaFinal = PCA(n_components = hyper_params['num_components']).fit(X)
        latent = pcaFinal.explained_variance_
        P = pcaFinal.components_.T #obtain PCA loadings
        S = np.cov(X,rowvar = False)
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = X[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X[i,:].T
            Q[i] = X[i,:]@(np.identity(p_kern)-P@P.T)@X[i,:].T
            
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0]               
            Xdyn_test = np.zeros((N_test - hyper_params['lag'],(hyper_params['lag']+1)*p))
            for m in range(N_test - hyper_params['lag']):
                for n in range(p):
                    Xdyn_test[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X_test[m:m+hyper_params['lag']+1,n]
            Xdyn_test = scaler.transform(Xdyn_test)
            X_test = Xdyn_test.copy()
            #Create nonlinear matrix using kernel methods
            if hyper_params['kernel'] == 'rbf':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
            elif hyper_params['kernel'] == 'poly':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
            unitX_test = np.ones((np.shape(x_test)[0],np.shape(x)[0]))/np.shape(x)[0]
            X_test = x_test-unitX_test@x-x_test@unitX+unitX_test@x@unitX
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = X_test[i,:]@P@np.linalg.pinv(P.T@S@P)@P.T@X_test[i,:].T
                QTest[i] = X_test[i,:]@(np.identity(p_kern)-P@P.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'], latent = latent, S = S)
    
    #%%% DKPLS
    elif method == 'DKPLS':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        Components = np.array(range(1,min(N,p,10)+1))
        lags = np.array(range(1,3))
        Kernels = ['rbf','poly']
        gamma = [1/5**2,1/10**2,1/20**2,1/50**2,1/100**2,1/150**2,1/200**2]
        degree = [2,3]
        Violations_result = np.zeros((len(Components),len(lags),len(Kernels),len(gamma),len(degree),K_fold))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                # scaler = StandardScaler(with_mean=True, with_std=True)
                # scaler.fit(X_train)
                # X_train = scaler.transform(X_train)
                # X_val = scaler.transform(X_val)
                # scaler.fit(y_train)
                # y_train = scaler.transform(y_train)
                # y_val = scaler.transform(y_val)
                for j in range(len(lags)):
                    # print([i,j])
                    #Get dimensions                
                    N_train = np.shape(X_train)[0]
                    N_val = np.shape(X_val)[0]               
                    p_train = np.shape(X_train)[1]
                    yp_train = np.shape(y_train)[1]
                    Xdyn_train = np.zeros((N_train - lags[j],(lags[j]+1)*p_train))
                    Xdyn_val = np.zeros((N_val - lags[j],(lags[j]+1)*p_train))
                    ydyn_train = np.zeros((N_train - lags[j],yp_train))
                    ydyn_val = np.zeros((N_val - lags[j],yp_train))
                    for m in range(N_train - lags[j]):
                        ydyn_train[m,:] = y_train[m+lags[j],:]
                        for n in range(p_train):
                                Xdyn_train[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_train[m:m+lags[j]+1,n]
                    
                    for m in range(N_val - lags[j]):
                        ydyn_val[m,:] = y_val[m+lags[j],:]
                        for n in range(p_train):
                                Xdyn_val[m,n*(lags[j]+1):(n+1)*(lags[j]+1)] = X_val[m:m+lags[j]+1,n]
                    # Scale dynamics matrices
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    scaler.fit(Xdyn_train)
                    Xdyn_train = scaler.transform(Xdyn_train)
                    Xdyn_val = scaler.transform(Xdyn_val)
                    scaler.fit(ydyn_train)
                    ydyn_train = scaler.transform(ydyn_train)
                    ydyn_val = scaler.transform(ydyn_val)
                    N_train = np.shape(X_train)[0]
                    for q in range(len(Kernels)):
                        for r in range(len(gamma)):
                            #Create nonlinear matrix using kernel methods
                            if Kernels[q] == 'rbf':
                                xk_train = pairwise_kernels(Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                xk_val = pairwise_kernels(Xdyn_val, Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                unitX = np.ones((np.shape(xk_train)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                unitX_val = np.ones((np.shape(xk_val)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                Xk_train = xk_train-unitX@xk_train-xk_train@unitX+unitX@xk_train@unitX
                                Xk_val = xk_val-unitX_val@xk_train-xk_val@unitX+unitX_val@xk_train@unitX
                                p_kern = np.shape(Xk_train)[1]
                                N_train = np.shape(Xk_train)[0]
                                N_val = np.shape(Xk_val)[0]
                                
                                # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(Xk_train,ydyn_train)
                                # i_idx =[]
                                for i in range(len(Components)):
                                    PLS_model = PLSRegression(scale = False, n_components = Components[i]).fit(Xk_train,ydyn_train)
                                    P = PLS_model.x_loadings_
                                    R = PLS_model.x_rotations_
                                    # i_idx.append(i)
                                    # R = PLS.x_loadings_[:,i_idx]
                                    # P = (np.linalg.pinv(R)).transpose()
                                    T = Xk_train@R
                                    
                                    #Calculate train statistics
                                    Scm = np.linalg.pinv(T.T@T/(np.shape(Xk_train)[0] - 1))
                                    tSquared = np.zeros((np.shape(Xk_train)[0],1))
                                    Q = np.zeros((np.shape(Xk_train)[0],1))
                                    for k in range(np.shape(Xk_train)[0]):
                                        tSquared[k] = T[k,:]@Scm@T[k,:].T
                                        e = Xk_train[k,:]@(np.identity(p_kern) - R@P.T)
                                        Q[k] = e@e.T
                                        # tSquared[k] = Xk_train[k,:]@R@np.linalg.pinv(T.T@T/(N_train-1))@R.T@Xk_train[k,:].T
                                        # Q[k] = Xk_train[k,:]@(np.identity(p_kern)-P@R.T)@Xk_train[k,:].T
                                        
                                    #Calculate val statistics
                                    T_val = Xk_val@R
                                    tSquaredVal = np.zeros((np.shape(Xk_val)[0],1))
                                    QVal = np.zeros((np.shape(Xk_val)[0],1))
                                    for k in range(np.shape(Xk_val)[0]):
                                        tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                                        e = Xk_val[k,:]@(np.identity(p_kern) - R@P.T)
                                        QVal[k] = e@e.T
                                        # tSquaredVal[k] = Xk_val[k,:]@R@np.linalg.pinv(T.T@T/(N_val-1))@R.T@Xk_val[k,:].T
                                        # QVal[k] = Xk_val[k,:]@(np.identity(p_kern)-P@R.T)@Xk_val[k,:].T
                                    thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                                    TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                    QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                    Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                    Violations_result[i,j,q,r,:,counter-1] += np.abs(Violations-alpha)
                                    
                                       
                            elif Kernels[q] == 'poly':
                                for s in range(len(degree)):
                                    xk_train = pairwise_kernels(Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                    xk_val = pairwise_kernels(Xdyn_val, Xdyn_train, metric = Kernels[q], gamma = gamma[r])
                                    #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
                                    unitX = np.ones((np.shape(xk_train)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                    unitX_val = np.ones((np.shape(xk_val)[0],np.shape(xk_train)[0]))/np.shape(xk_train)[0]
                                    Xk_train = xk_train-unitX@xk_train-xk_train@unitX+unitX@xk_train@unitX
                                    Xk_val = xk_val-unitX_val@xk_train-xk_val@unitX+unitX_val@xk_train@unitX
                                    p_kern = np.shape(Xk_train)[1]
                                    
                                    # PLS = PLSRegression(scale = False, n_components = Components[-1]).fit(Xk_train,ydyn_train)
                                    i_idx =[]
                                    for i in range(len(Components)):
                                        PLS_model = PLSRegression(scale = False, n_components = Components[i]).fit(Xk_train,ydyn_train)
                                        P = PLS_model.x_loadings_
                                        R = PLS_model.x_rotations_
                                        # i_idx.append(i)
                                        # R = PLS.x_loadings_[:,i_idx]
                                        # P = (np.linalg.pinv(R)).transpose()
                                        T = Xk_train@R
                                        
                                        #Calculate train statistics
                                        Scm = np.linalg.pinv(T.T@T/(np.shape(Xk_train)[0] - 1))
                                        tSquared = np.zeros((np.shape(Xk_train)[0],1))
                                        Q = np.zeros((np.shape(Xk_train)[0],1))
                                        for k in range(np.shape(Xk_train)[0]):
                                            tSquared[k] = T[k,:]@Scm@T[k,:].T
                                            e = Xk_train[k,:]@(np.identity(p_kern) - R@P.T)
                                            Q[k] = e@e.T
                                            # tSquared[k] = Xk_train[k,:]@R@np.linalg.pinv(T.T@T/(N_train-1))@R.T@Xk_train[k,:].T
                                            # Q[k] = Xk_train[k,:]@(np.identity(p_kern)-P@R.T)@Xk_train[k,:].T
                                            
                                        #Calculate val statistics
                                        T_val = Xk_val@R
                                        tSquaredVal = np.zeros((np.shape(Xk_val)[0],1))
                                        QVal = np.zeros((np.shape(Xk_val)[0],1))
                                        for k in range(np.shape(Xk_val)[0]):
                                            tSquaredVal[k] = T_val[k,:]@Scm@T_val[k,:].T
                                            e = Xk_val[k,:]@(np.identity(p_kern) - R@P.T)
                                            QVal[k] = e@e.T
                                            # tSquaredVal[k] = Xk_val[k,:]@R@np.linalg.pinv(T.T@T/(N_val-1))@R.T@Xk_val[k,:].T
                                            # QVal[k] = Xk_val[k,:]@(np.identity(p_kern)-P@R.T)@Xk_val[k,:].T
                                        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = Components[i])
                                        TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                        QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                        Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                        Violations_result[i,j,q,r,s,counter-1] += np.abs(Violations-alpha)
                                    
                    
            Violations_mean = np.sum(Violations_result,axis = 5)/counter
                 
            Violations_std = np.std(Violations_result, axis = 5)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3],idx[4]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3],idx[4]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            hyper_params['kernel'] = Kernels[idx_final[2]]
            hyper_params['gamma'] = gamma[idx_final[3]]
            if hyper_params['kernel'] == 'poly':
                hyper_params['degree'] = degree[idx_final[4]]
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        #fit the final model using opt hyper_params
        
        # pre-process data
        # scaler = StandardScaler(with_mean=True, with_std=True)
        # scaler.fit(X)
        # X = scaler.transform(X)
        # X_test = scaler.transform(X_test)
        # scaler.fit(y)
        # y = scaler.transform(y)
        # y_test = scaler.transform(y_test)
        
        #Get dimensions
        N = np.shape(X)[0]              
        p = np.shape(X)[1]
        yp = np.shape(y)[1]
        Xdyn = np.zeros((N - hyper_params['lag'],(hyper_params['lag']+1)*p))
        ydyn = np.zeros((N - hyper_params['lag'],yp))
        for m in range(N - hyper_params['lag']):
            ydyn[m,:] = y[m+hyper_params['lag'],:]
            for n in range(p):
                    Xdyn[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X[m:m+hyper_params['lag']+1,n]
        
        # Scale dynamics matrices
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(Xdyn)
        Xdyn = scaler.transform(Xdyn)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(ydyn)
        ydyn = scalery.transform(ydyn)
        # Use dynamic matrices as inputs
        X_s = Xdyn.copy()
        y = ydyn.copy()
        #Create nonlinear matrix using kernel methods
        if hyper_params['kernel'] == 'rbf':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
        elif hyper_params['kernel'] == 'poly':
            x = pairwise_kernels(X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
        
        #Center kernel matrices according to the following instructions: https://stats.stackexchange.com/questions/131140/what-exactly-is-the-procedure-to-compute-principal-components-in-kernel-pca
        unitX = np.ones((np.shape(x)[0],np.shape(x)[0]))/np.shape(x)[0]
        X = x-unitX@x-x@unitX+unitX@x@unitX
                
        N = np.shape(X)[0]
        p_kern = np.shape(X)[1]
        PLS_final = PLSRegression(scale = False, n_components=int(hyper_params['num_components'])).fit(X,y)
        P = PLS_final.x_loadings_
        R = PLS_final.x_rotations_
        # R = PLS_final.x_loadings_
        # P = (np.linalg.pinv(R)).transpose()
        T = X@R
        
        #Calculate train statistics
        Scm = np.linalg.pinv(T.T@T/(N-1))
        tSquared = np.zeros((np.shape(X)[0],1))
        Q = np.zeros((np.shape(X)[0],1))
        for i in range(np.shape(X)[0]):
            tSquared[i] = T[i,:]@Scm@T[i,:].T
            e = X[i,:]@(np.identity(p_kern) - R@P.T)
            Q[i] = e@e.T
            # tSquared[i] = X[i,:]@R@np.linalg.pinv(T.T@T/(N-1))@R.T@X[i,:].T
            # Q[i] = X[i,:]@(np.identity(p_kern)-P@R.T)@X[i,:].T
            
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0] 
            Xdyn_test = np.zeros((N_test - hyper_params['lag'],(hyper_params['lag']+1)*p))
            ydyn_test = np.zeros((N_test - hyper_params['lag'],yp))
            for m in range(N_test - hyper_params['lag']):
                ydyn_test[m,:] = y_test[m+hyper_params['lag'],:]
                for n in range(p):
                    Xdyn_test[m,n*(hyper_params['lag']+1):(n+1)*(hyper_params['lag']+1)] = X_test[m:m+hyper_params['lag']+1,n]
            Xdyn_test = scaler.transform(Xdyn_test)
            ydyn_test = scalery.transform(ydyn_test)
            X_test = Xdyn_test.copy()
            y_test = ydyn_test.copy()
            #Create nonlinear matrix using kernel methods
            if hyper_params['kernel'] == 'rbf':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'])
            elif hyper_params['kernel'] == 'poly':
                x_test = pairwise_kernels(X_test, X_s, metric = hyper_params['kernel'], gamma = hyper_params['gamma'], degree = hyper_params['degree'])
            unitX_test = np.ones((np.shape(x_test)[0],np.shape(x)[0]))/np.shape(x)[0]
            X_test = x_test-unitX_test@x-x_test@unitX+unitX_test@x@unitX
            T_test = X_test@R
            tSquaredTest = np.zeros((np.shape(X_test)[0],1))
            QTest = np.zeros((np.shape(X_test)[0],1))
            for i in range(np.shape(X_test)[0]):
                tSquaredTest[i] = T_test[i,:]@Scm@T_test[i,:].T
                e = X_test[i,:]@(np.identity(p_kern) - R@P.T)
                QTest[i] = e@e.T
                # tSquaredTest[i] = X_test[i,:]@R@np.linalg.pinv(T.T@T/(N_test-1))@R.T@X_test[i,:].T
                # QTest[i] = X_test[i,:]@(np.identity(p_kern)-P@R.T)@X_test[i,:].T
        else:
            tSquaredTest = None
            QTest = None
            
        thresholdTsquared, thresholdQ = getControlLimits(tSquared,Q,alpha, Tmethod, Qmethod, method, Components = hyper_params['num_components'])
    
    #%%% KDE-CVA
    elif method == 'KDE-CVA':
        num_iter = K_fold
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        #Define hyperparameters
        lags = np.array(range(1,4))
        Components = np.array(range(1,min(N,p,10)+1))
        hTSquaredFactors = np.array([1,2,1/2,3,1/3,5,1/5])
        hQFactors = np.array([1,2,1/2,3,1/3,5,1/5])
        # lags = np.array([2])#[1,2,3,4,5]
        # Components = np.array([2])
        # hTSquaredFactors = np.array([1])
        # hQFactors = np.array([1])
        Violations_result = np.zeros((len(Components),len(lags),len(hTSquaredFactors),len(hQFactors),K_fold))
        
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            #Crossval procedure
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = 'Timeseries', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                scaler.fit(y_train)
                y_train = scaler.transform(y_train)
                y_val = scaler.transform(y_val)
                N_train = np.shape(X_train)[0]
                N_val = np.shape(X_val)[0]
                p_train = np.shape(X_train)[1]
                yp_train = np.shape(y_train)[1]
                for i in range(len(Components)):
                    for j in range(len(lags)):                           
                        #Create past and future vectors
                        P = np.zeros((lags[j]*(p_train+yp_train),N_train-lags[j]))
                        F = np.zeros((lags[j]*yp_train,N_train-lags[j]))
                        P_val = np.zeros((lags[j]*(p_train+yp_train),N_val-lags[j]))
                        F_val = np.zeros((lags[j]*yp_train,N_val-lags[j]))
                        for m in range(N_train-lags[j]):
                            for n in range(lags[j]):
                                P[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_train[m-1-n,:].T
                                P[lags[j]+n*p_train:lags[j]+(n+1)*p_train,m-lags[j]] = X_train[m-1-n,:].T
                            for n in range(lags[j]):
                                F[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_train[m+n,:].T
                        for m in range(N_val-lags[j]):
                            for n in range(lags[j]):
                                P_val[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_val[m-1-n,:].T
                                P_val[lags[j]+n*p_train:lags[j]+(n+1)*p_train,m-lags[j]] = X_val[m-1-n,:].T
                            for n in range(lags[j]):
                                F_val[n*yp_train:(n+1)*yp_train,m-lags[j]] = y_val[m+n,:].T
                        P = P.T
                        F = F.T
                        P_val = P_val.T
                        F_val = F_val.T
                        #Scale vectors
                        scaler_P = StandardScaler(with_mean=True, with_std=True)
                        scaler_P.fit(P)
                        P_scale = scaler_P.transform(P)
                        P_scale_val = scaler_P.transform(P_val)
                        scaler_F = StandardScaler(with_mean=True, with_std=True)
                        scaler_F.fit(F)
                        F_scale = scaler_F.transform(F)
                        # F_scale_val = scaler_F.transform(F_val)
                        #Apply cva to P and F vectors
                        Sxx = 1/(np.shape(P)[0]-1)*P_scale.T@P_scale
                        Syy = 1/(np.shape(F)[0]-1)*F_scale.T@F_scale
                        Sxy = 1/(np.shape(F)[0]-1)*P_scale.T@F_scale
                        
                        U, S, V = np.linalg.svd(fractional_matrix_power(Sxx, -0.5)@Sxy@fractional_matrix_power(Syy,-0.5), full_matrices=True)
                           
                        J = U.T@fractional_matrix_power(Sxx,-0.5)
                        Jd = J[:i+1,:].T 
                        Jr = J[i+1:,:].T 
                        
                        #Calculate train statistics
                        tSquared = np.zeros((np.shape(P)[0],1))
                        trSquared = np.zeros((np.shape(P)[0],1))
                        Q = np.zeros((np.shape(P)[0],1))
                        for k in range(np.shape(P)[0]):
                            tSquared[k] = P_scale[k,:]@Jd@Jd.T@P_scale[k,:].T
                            trSquared[k] = P_scale[k,:]@Jr@Jr.T@P_scale[k,:].T
                            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale[k,:].T
                            Q[k] = rt.T@rt
                            
                        #Calculate val statistics
                        tSquaredVal = np.zeros((np.shape(P_val)[0],1))
                        trSquaredVal = np.zeros((np.shape(P_val)[0],1))
                        QVal = np.zeros((np.shape(P_val)[0],1))
                        for k in range(np.shape(P_val)[0]):
                            tSquaredVal[k] = P_val[k,:]@Jd@Jd.T@P_val[k,:].T
                            trSquaredVal[k] = P_val[k,:]@Jr@Jr.T@P_val[k,:].T
                            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale_val[k,:].T
                            QVal[k] =rt.T@rt
                        for q in range(len(hTSquaredFactors)):
                            for r in range(len(hQFactors)):
                                N = np.shape(tSquared)[0]
                                hTSquaredopt = 1.06 * np.std(tSquared) * N**(-0.2)
                                hTrSquaredopt = 1.06 * np.std(trSquared) * N**(-0.2)
                                hQopt = 1.06 * np.std(Q) * N**(-0.2)
                                kdeTSquared = sm.nonparametric.KDEUnivariate(tSquared)
                                kdeTSquared.fit(kernel='gau', bw=hTSquaredopt*hTSquaredFactors[q])
                                idxTSquared = np.argmin(abs(kdeTSquared.cdf - (1-alpha)))
                                thresholdTsquared = kdeTSquared.support[idxTSquared]
                                kdeTrSquared = sm.nonparametric.KDEUnivariate(trSquared)
                                kdeTrSquared.fit(kernel='gau', bw=hTrSquaredopt*hTSquaredFactors[q])
                                idxTrSquared = np.argmin(abs(kdeTrSquared.cdf - (1-alpha)))
                                thresholdTrsquared = kdeTrSquared.support[idxTrSquared]
                                kdeQ = sm.nonparametric.KDEUnivariate(Q)
                                kdeQ.fit(kernel='gau', bw=hQopt*hQFactors[r])
                                idxQ = np.argmin(abs(kdeQ.cdf - (1-alpha)))
                                thresholdQ = kdeQ.support[idxQ]
                                TViolations = np.where(np.any(tSquaredVal > thresholdTsquared, axis = 1))[0]
                                TrViolations = np.where(np.any(trSquaredVal > thresholdTrsquared, axis = 1))[0]
                                QViolations = np.where(np.any(QVal > thresholdQ, axis = 1))[0]
                                # Violations = len(np.unique(np.concatenate((TViolations,QViolations,TrViolations))))/np.shape(QVal)[0]
                                Violations = len(np.unique(np.concatenate((TViolations,QViolations))))/np.shape(QVal)[0]
                                Violations_result[i,j,q,r,counter-1] += np.abs(Violations-alpha)
                
            
            Violations_mean = np.sum(Violations_result,axis = 4)/counter
                 
            Violations_std = np.std(Violations_result, axis = 4)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
            
            hyper_params = {}     
            
            hyper_params['num_components'] = Components[idx_final[0]]
            hyper_params['lag'] = lags[idx_final[1]]
            hyper_params['hTSquaredFactor'] = hTSquaredFactors[idx_final[2]]
            hyper_params['hQFactor'] = hQFactors[idx_final[3]]
            # hyper_params['num_components'] = 2
            # hyper_params['lag'] = 2
            # hyper_params['hTSquaredFactor'] = 1
            # hyper_params['hQFactor'] = 1
            # hyper_params['num_components'] = 2
            # hyper_params['lag'] = 2
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
        
        # pre-process data
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        scalery = StandardScaler(with_mean=True, with_std=True)
        scalery.fit(y)
        y = scalery.transform(y)
        
        #Full dataset run
        N_train = np.shape(X)[0]
        
        p_train = np.shape(X)[1]
        yp_train = np.shape(y)[1]
        
        P = np.zeros((hyper_params['lag']*(p_train+yp_train),N_train-hyper_params['lag']))
        F = np.zeros((hyper_params['lag']*yp_train,N_train-hyper_params['lag']))
        for m in range(N_train-hyper_params['lag']):
            for n in range(hyper_params['lag']):
                # P[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m-n,:].T
                # P[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X[m-n,:].T
                P[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m-1-n,:].T
                P[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X[m-1-n,:].T
            for n in range(hyper_params['lag']):
                # F[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m+n,:].T
                F[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y[m+n,:].T
        P = P.T
        F = F.T
        scaler_P = StandardScaler(with_mean=True, with_std=True)
        scaler_P.fit(P)
        P_scale = scaler_P.transform(P)
        scaler_F = StandardScaler(with_mean=True, with_std=True)
        scaler_F.fit(F)
        F_scale = scaler_F.transform(F)
        # F_scale_test = scaler_F.transform(F_test)
        # Apply cva to P and F vectors
        
        Sxx = 1/(np.shape(P)[0]-1)*P_scale.T@P_scale
        Syy = 1/(np.shape(F)[0]-1)*F_scale.T@F_scale
        Sxy = 1/(np.shape(F)[0]-1)*P_scale.T@F_scale
        
        U, S, V = np.linalg.svd(fractional_matrix_power(Sxx, -0.5)@Sxy@fractional_matrix_power(Syy,-0.5), full_matrices=True)
           
        J = U.T@fractional_matrix_power(Sxx,-0.5)
        Jd = J[:hyper_params['num_components']+1,:].T 
        Jr = J[hyper_params['num_components']+1:,:].T 
        
        #Calculate train statistics
        tSquared = np.zeros((np.shape(P)[0],1))
        trSquared = np.zeros((np.shape(P)[0],1))
        Q = np.zeros((np.shape(P)[0],1))
        for i in range(np.shape(P)[0]):
            tSquared[i] = P_scale[i,:]@Jd@Jd.T@P_scale[i,:].T
            trSquared[i] = P_scale[i,:]@Jr@Jr.T@P_scale[i,:].T
            rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale[i,:].T
            Q[i] = rt.T@rt
            
        if X_test is not None:
            #Calculate test statistics
            N_test = np.shape(X_test)[0]
            X_test = scaler.transform(X_test)
            y_test = scalery.transform(y_test)
            P_test = np.zeros((hyper_params['lag']*(p_train+yp_train),N_test-hyper_params['lag']))
            F_test = np.zeros((hyper_params['lag']*yp_train,N_test-hyper_params['lag']))
            for m in range(N_test-hyper_params['lag']):
                for n in range(hyper_params['lag']):
                    # P_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m-n,:].T
                    # P_test[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X_test[m-n,:].T
                    P_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m-1-n,:].T
                    P_test[hyper_params['lag']+n*p_train:hyper_params['lag']+(n+1)*p_train,m-hyper_params['lag']] = X_test[m-1-n,:].T
                for n in range(hyper_params['lag']):
                    # F_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m+n,:].T  
                    F_test[n*yp_train:(n+1)*yp_train,m-hyper_params['lag']] = y_test[m+n,:].T 
            P_test = P_test.T
            F_test = F_test.T
            P_scale_test = scaler_P.transform(P_test)
            tSquaredTest = np.zeros((np.shape(P_test)[0],1))
            trSquaredTest = np.zeros((np.shape(P_test)[0],1))
            QTest = np.zeros((np.shape(P_test)[0],1))
            for i in range(np.shape(P_test)[0]):
                tSquaredTest[i] = P_scale_test[i,:]@Jd@Jd.T@P_scale_test[i,:].T
                trSquaredTest[i] = P_scale_test[i,:]@Jr@Jr.T@P_scale_test[i,:].T
                rt = (np.identity(np.shape(Jd)[0])-Jd@Jd.T)@P_scale_test[i,:].T
                QTest[i] =rt.T@rt
        else:
            tSquaredTest = None
            trSquaredTest = None
            QTest = None
        
        #KDE estimation of control limits (cdf shows cumulative density function for support)
        N = np.shape(tSquared)[0]
        hTSquaredopt = 1.06 * np.std(tSquared) * N**(-0.2)
        hTrSquaredopt = 1.06 * np.std(trSquared) * N**(-0.2)
        hQopt = 1.06 * np.std(Q) * N**(-0.2)
        kdeTSquared = sm.nonparametric.KDEUnivariate(tSquared)
        kdeTSquared.fit(kernel='gau', bw=hTSquaredopt*hyper_params['hTSquaredFactor'])
        idxTSquared = np.argmin(abs(kdeTSquared.cdf - (1-alpha)))
        thresholdTsquared = kdeTSquared.support[idxTSquared]
        kdeTrSquared = sm.nonparametric.KDEUnivariate(trSquared)
        kdeTrSquared.fit(kernel='gau', bw=hTrSquaredopt*hyper_params['hTSquaredFactor'])
        idxTrSquared = np.argmin(abs(kdeTrSquared.cdf - (1-alpha)))
        thresholdTrsquared = kdeTrSquared.support[idxTrSquared]
        kdeQ = sm.nonparametric.KDEUnivariate(Q)
        kdeQ.fit(kernel='gau', bw=hQopt*hyper_params['hQFactor'])
        idxQ = np.argmin(abs(kdeQ.cdf - (1-alpha)))
        thresholdQ = kdeQ.support[idxQ]
    
    #%%% SVDD
    elif method == 'SVDD':
        num_iter = K_fold*Nr
        N = np.shape(X)[0]
        p = np.shape(X)[1]
        #Cross-validation
        Cs = [1,1/2,2,1/3,3,1/5,5]
        gammas = [1/5**2,1/10**2,1/20**2,1/50**2,1/100**2,1/150**2,1/200**2]
        kernels = ['rbf','poly']
        degrees = [2,3]
        Violations_result = np.zeros((len(Cs),len(gammas),len(kernels),len(degrees),K_fold*Nr))
         
        if if_have_hyper_params:
            with open(hyper_params_file_name, 'rb') as handle:
                hyper_params, Violations_result, Violations_mean, Violations_std, Violations_min, Violations_bar, idx_final = pickle.load(handle)
        else:
            counter = 0
            for X_train, X_val in CVpartition(X, y, Type = 'Re_KFold', K = K_fold, Nr = Nr, if_have_output = if_have_output):
                counter += 1
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                for i in range(len(Cs)):
                    for j in range(len(gammas)):
                        for k in range(len(kernels)):
                            if kernels[k] == 'rbf':
                                svdd = BaseSVDD(C = Cs[i], gamma = gammas[j], kernel = kernels[k], display = 'off')
                                svdd.fit(X_train)
                                radius = svdd.radius
                                distance = svdd.get_distance(X_train)
                                distance_val = svdd.get_distance(X_val)
                                Violations = np.sum(distance_val > radius)/np.shape(distance_val)[0]
                                Violations_result[i,j,k,:,counter-1] += np.abs(Violations-alpha)
                            elif kernels[k] == 'poly':
                                for m in range(len(degrees)):
                                    svdd = BaseSVDD(C = Cs[i], gamma = gammas[j], kernel = kernels[k], degree = degrees[m], display = 'off')
                                    svdd.fit(X_train)
                                    radius = svdd.radius
                                    distance = svdd.get_distance(X_train)
                                    distance_val = svdd.get_distance(X_val)
                                    Violations = np.sum(distance_val > radius)/np.shape(distance_val)[0]
                                    Violations_result[i,j,k,m,counter-1] += np.abs(Violations-alpha)
            Violations_mean = np.sum(Violations_result,axis = 4)/counter
                 
            Violations_std = np.std(Violations_result, axis = 4)
            
            idx = np.unravel_index(np.argmin(Violations_mean, axis=None), Violations_mean.shape)
                    
            Violations_min = Violations_mean[idx[0],idx[1],idx[2],idx[3]]
            Violations_bar = Violations_min + Violations_std[idx[0],idx[1],idx[2],idx[3]]/np.sqrt(num_iter)
            
            ind_lower = Violations_mean <= Violations_bar
            final_ind = np.where(ind_lower)
            minimum = 10000
            for i in range(len(final_ind[0])):
                sumIdx = 0
                idx_final_trial = []
                for j in range(len(final_ind)):
                    sumIdx += final_ind[j][i]
                    idx_final_trial.append(final_ind[j][i])
                if sumIdx < minimum:
                    minimum = sumIdx
                    idx_final = idx_final_trial
                    
            hyper_params = {}             
            hyper_params['C'] = Cs[idx_final[0]]
            hyper_params['gamma'] = gammas[idx_final[1]]
            hyper_params['kernel'] = kernels[idx_final[2]]
            if hyper_params['kernel'] == 'poly':
                hyper_params['degree'] = degrees[idx_final[3]]
            with open(hyper_params_file_name, 'wb') as w:
                pickle.dump([hyper_params, Violations_result, Violations_mean, Violations_std,  Violations_min, Violations_bar, idx_final], w)
                        
        # Scaling
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        # Final model
        svdd = BaseSVDD(C = hyper_params['C'], gamma = hyper_params['gamma'], kernel = hyper_params['kernel'], display='off')
        svdd.fit(X)
        radius = svdd.radius
        distance = svdd.get_distance(X)
        
        if X_test is not None:
            N_test = np.shape(X_test)[0]  
            X_test = scaler.transform(X_test)
            distance_test = svdd.get_distance(X_test)
        else:
            distance_test = None
    
    #%% Print outcomes
    if method == 'SVDD':
        if plot_training:
            PlotStatisticsSVDD(distance,radius,method,'traning')
        if X_test is not None and plot_testing:
            PlotStatisticsSVDD(distance_test,radius,method,'testing')
        return(distance,distance_test,hyper_params,Violations_result,radius,Violations_mean,Violations_min,idx_final)
    elif method in ['CVA', 'KDE-CVA']:
        if plot_training:
            PlotStatisticsCVA(tSquared,thresholdTsquared,trSquared,thresholdTrsquared,Q,thresholdQ,method,'traning')
        if X_test is not None and plot_testing:
            PlotStatisticsCVA(tSquaredTest,thresholdTsquared,trSquaredTest,thresholdTrsquared,QTest,thresholdQ,method,'testing')
        return(tSquared,trSquared,tSquaredTest,trSquared,trSquaredTest,Q,QTest,hyper_params,Violations_result,thresholdTsquared,thresholdTrsquared,thresholdQ,Violations_mean,Violations_min,idx_final)
    else:
        # PlotStatistics(tSquared,tSquaredTest,thresholdTsquared,Q,QTest,thresholdQ)
        if plot_training:
            PlotStatistics(tSquared,thresholdTsquared,Q,thresholdQ,method,'traning')
        if X_test is not None and plot_testing:
            PlotStatistics(tSquaredTest,thresholdTsquared,QTest,thresholdQ,method,'testing')
        return(tSquared,tSquaredTest,Q,QTest,hyper_params,Violations_result,thresholdTsquared,thresholdQ,Violations_mean,Violations_min,idx_final,Violations_std,Violations_bar)
    
    