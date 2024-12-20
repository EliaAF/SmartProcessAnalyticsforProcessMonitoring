# -*- coding: utf-8 -*-
"""
mvn_R.py
Version: 1.0.0
Date: 2022/08/03
Author: Elia Arnese-Feffin elia249@mit.edu/elia.arnesefeffin@phd.unipd.it

This code is based on ace_R.py by Weike Sun, provided as part of the
Smart Process Analytics (SPA) code, available at:
	https://github.com/vickysun5/SmartProcessAnalytics

# GNU General Public License version 3 (GPL-3.0) ------------------------------

Smart Process Analytics for Process Monitoring
Copyright (C) 2022 Elia Arnese-Feffin

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

#%% Load packages

# Numerical Python
import numpy as np
# R-to-Python interface
import rpy2
# Set R_HOME environment key if not preset by the system
import os
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
# R-to-Python interface
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
# Input-outpt tools
from io import StringIO
# System package
import sys

# Load utils package in R
utils = importr("utils")
# Set loading preferences
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}

# Load MVN package in R
try:
    # MVN found automatically given the R_HOME registry key in Windows
    mvn = importr('MVN', robject_translations = d)
except:
    # MVN not found automatically, need to insert installation path
    mvn = importr('MVN', robject_translations = d, lib_loc = '/Library/Frameworks/R.framework/Resources/library')

# Function for multivariate normality test
def multnormtest (X, test = 'hz', alpha = 0.01, ddof = 0, scale = False):
    '''
    Multivariate normality test
 	
 	The function assess the multivariate normality of a dataset by means of
    Royston's test, Henze-Zirkler test, or Mardia's combined tests. Royston's
    test should be used by deafult, but such a test can be performend only up
    to 2000 observations. Therefore, Henze-Zirkler test should be used on
    larger datasets; however, this test yields meaningless results on datasets
    containing more than 50 variables (the variance of the lognormal
    distribution used to determine the p-value tends to 0, causing the
    propagation of numerical errors). Mardia's combined test (on both
    multivariate skewness and multivariate kurtosis) should be therefore used
    for datasets with more than 50 variables and more than 2000 observations.
    The R library  MVN is used, therefore a link to the R language by means of
    the rpy2 package is required.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)\n
        test : {'hz', 'royston', 'mardia_comb', 'mardia_skew', 'mardia_comb'}
            Multivariate normality test to be performed
            
            * 'hz' : Henze-Zirkler test (default)
            * 'royston' : Royston's test
            * 'mardia_comb' : Mardia's combined test (data are deemed as normal
            only if they are by both Mardia's skwness and kurtosis test)
            * 'mardia_skew' : Mardia's skewness test
            * 'mardia_kurt' : Mardia's kurtosis test
            
        alpha : float
            Significance level for normality test
        ddof : int
            Degrees of freedom of the variance for Royston's test and
            Mardia's tests, can be either 0 for normalizing the
                    variance by N (default), or 1 for normalizing by N - 1
        scale: bool
            Logical flag to scale columns of the dataset to unit variance
    
    Returns
    _______
        norm : bool
            True is dataset is multivariate normal
        p_value : float
            p-value of the chosen test, if test = 'mardia_comb' a tuple with
            p-values of Mardia's skwness and kurtosis is returned
        stat : float
            Value of the statistic of the chosen test, if test = 'mardia_comb'
            a tuple with statistics of Mardia's skwness and kurtosis is
            returned
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations and number of variables
    N, V = np.shape(X)
    
    # Check is the requested test is valid
    if test not in ['hz', 'royston', 'mardia_comb', 'mardia_skew', 'mardia_kurt']:
        raise Exception('Requested test is not supported')
    
    # R objects production ----------------------------------------------------
    
    # Activare R object
    rpy2.robjects.numpy2ri.activate()

    # Convert NP array to R matrix
    Xr = ro.r.matrix(X, nrow = N, ncol = V)
    # Assign to R object
    ro.r.assign('X', Xr)

    # Multivariate normality test ---------------------------------------------
    
    # Check if Mardia's test
    if 'mardia' in test:
        # Perform Mardia's test
        mvn_res = mvn.mvn(Xr, alpha = alpha, mvnTest = 'mardia', covariance = bool(ddof), scale = scale, desc = False)
        # Save sys.stdout
        save_stdout = sys.stdout
        # Initialize string imput-output object
        printed = StringIO()
        # Redefine ys.stdout
        sys.stdout = printed
        # Print results of Mardia's test
        print(mvn_res.rx2('multivariateNormality'))
        # Restore ys.stdout
        sys.stdout = save_stdout
        # Get the printed results and split them on spaces
        printed = printed.getvalue().split()
        # Handle correctly each test
        if test == 'mardia_comb':
            # Get test statistic
            stat = (np.float64(printed[8]), np.float64(printed[14]))
            # Get p-value
            p_value = (np.float64(printed[9]), np.float64(printed[15]))
            # Evaluate significance
            norm = np.logical_not(any(np.array(p_value) < alpha))
        elif test == 'mardia_skew':
            # Get test statistic
            stat = np.float64(printed[8])
            # Get p-value
            p_value = np.float64(printed[9])
            # Evaluate significance
            norm = np.logical_not(p_value < alpha)
        elif test == 'mardia_kurt':
            # Get test statistic
            stat = np.float64(printed[14])
            # Get p-value
            p_value = np.float64(printed[15])
            # Evaluate significance
            norm = np.logical_not(p_value < alpha)
    else:
        # Perform Henze-Zirkler or Royston's test
        mvn_res = mvn.mvn(Xr, alpha = alpha, mvnTest = test, covariance = bool(ddof), scale = scale, desc = False)
        # Get test statistic
        stat = mvn_res.rx2('multivariateNormality')[1][0]
        # Get p-value
        p_value = mvn_res.rx2('multivariateNormality')[2][0]
        # Evaluate significance
        norm = np.logical_not(p_value < alpha)

    return (norm, p_value, stat)