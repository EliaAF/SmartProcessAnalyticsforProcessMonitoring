# -*- coding: utf-8 -*-
"""
ace_R.py
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

# Load utils package in R
utils = importr("utils")
# Set loading preferences
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}

# Load acepack package in R
try:
    # acepack found automatically given the R_HOME registry key in Windows
    ace = importr('acepack', robject_translations = d)
except:
    # acepack not found automatically, need to insert installation path
    ace = importr('acepack', robject_translations = d, lib_loc = '')

# Function for maximal correlation coefficient
def maxcorrcoef (x, y, cat = None):
    '''
    Maximal correlation coefficient by alternating conditional expectation
	
	The function estimates the maximal correlation coefficient between two
    variables by means of the alternating conditional expectation (ACE)
    algorithm. The R library acepack is used for ACE, therefore a link to the
    R language by means of the rpy2 package is required.
    
    Parameters
    __________
        x : array_like, 1d
            Predictor variable, must have shape (N, )
        y : array_like, 1d
            Response variable, must have shape (N, )
        cat : {None, list}
            List of integers specifying if x and/or y are categorical variables
            
            * cat = np.array([1], dtype = int) if x is categorial
            * cat = np.array([0], dtype = int) if y is categorical
            * cat = np.array([0, 1], dtype = int) if both x and y are
            categorical
    
    Returns
    _______
        maxcorr : float
            Maximal correlation coefficient between x and y
    '''
    # Dataset info ------------------------------------------------------------

    # Give depth to arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Number of observations and number of predictor variables
    N, V = np.shape(x)

    # R objects production ----------------------------------------------------

    # Activare R object
    rpy2.robjects.numpy2ri.activate()

    # Convert predictor NP array to R matrix
    xr = ro.r.matrix(x, nrow = N, ncol = V)
    # Assign to R object
    ro.r.assign('x', xr)
    # Convert response NP array to R matrix
    yr = ro.r.matrix(y, nrow = N, ncol = 1)
    # Assign to R object
    ro.r.assign('y', yr)

    # Check for categorical variables
    if cat is not None:
        # Convert array of pointers to categorical variables to R matrix
        catr = ro.r.matrix(cat, nrow = np.shape(cat)[0], ncol = 1)
    else:
        # Assign cat as R NULL
        catr = ro.NULL
    # Assign to R object
    ro.r.assign('cat', catr)

    # Maximal correlation coefficient -----------------------------------------

    # Transform variables by ACE algoithm
    ace_res = ace.ace(xr, yr, cat = catr, delrsq = 0.001)

    # Get transformed variables
    tx = ace_res.rx2('tx')
    ty = ace_res.rx2('ty')

    # Compute maximal correlation coefficient
    maxcorr = np.array(ro.r.cor(tx, ty))[0][0]
    
    return maxcorr