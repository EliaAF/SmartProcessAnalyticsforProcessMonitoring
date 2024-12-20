# -*- coding: utf-8 -*-
"""
dataset_property_assessment.py
Version: 1.0.0
Date: 2022/08/03
Author: Elia Arnese-Feffin elia249@mit.edu/elia.arnesefeffin@phd.unipd.it

This code is based on dataset_property_new.py by Weike Sun, provided as part
of the Smart Process Analytics (SPA) code, available at:
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
# Multivariate normality test (R language, need rpy2)
import mvn_R
# Univariate normality tests
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import normal_ad as anderson
# Linear regression
from sklearn.linear_model import LinearRegression
# F distribution
from scipy.stats import f
# ACE algorithm (R language, need rpy2)
import ace_R
# Normal distribution
from scipy.stats import norm
# Empirical percentile from samples
from scipy.stats import percentileofscore
# Statmodels time series analysis tools
import statsmodels.tsa.stattools as smtsa

#%% Normality

def non_normality_test (X, alpha = 0.01):
    '''
    Test for multivariate normality of a dataset
	
	The function test the distribution of the dataset. The Royston's test is
    used by default if the dataset contains up to 2000 observations; the
    Henze-Zirkler test is used for datasets with more than 2000 observations
    and up to 50 variables; combined Mardia's test (check on both skewness and
    kurtosis) is used for all other cases. Univariate normality of all
    variables in the dataset is also test, even though independently on the
    multivariate normality test. Shapiro-Wilk test is used if  the dataset
    contains up to 5000 observations; the Anderson-Darling test is used
    otherwise. A data matrix with N rows and V columns is expected as input,
    where N is the number of observations and V is the number of variables. 
    The R library MVN is used for all the multivariate normality tests,
    therefore a link to the R language by means of the rpy2 package is
    required.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
        alpha : float
            Significance level for normality test
    
    Returns
    _______
        sig_nn : bool
            True if the dataset is not multivariate normal
        p_value : float
            p-value of the multivariate normality test
        map_nn : array
            Logical array of shape shape (V, ), where component map_nn[v] is
            True if X[:, v] is not univariate normal
    '''
    # Dataset info ------------------------------------------------------------
    
    # Number of observations and number of variables
    N, V = np.shape(X)
    
    # Univariate normality test -----------------------------------------------
    
    # Pre-allocation of p-value array
    p_value_uni = np.zeros(V)

    # Check numebr of observations
    if N <= 5000:
        # Loop over variables
        for v in range(V):
            # Perform Shapiro-Wilk test
            p_value_uni[v] = shapiro(X[:, v])[1]
    else:
        # Loop over variables
        for v in range(V):
            # Perform Anderson-Darling test
            p_value_uni[v] = anderson(X[:, v])[1]
    
    # Check for significance of non-normality in each variable
    map_nn = np.bool_(p_value_uni < alpha)
    
    # Multivariate normality test ---------------------------------------------
    
    # Check numebr of observations
    if N <= 2000:
        # Perform Royston's test
        norm, p_value = mvn_R.multnormtest(X, test = 'royston', alpha = alpha, ddof = 0, scale = False)[0:2]
    else:
        # Check numebr of variables
        if V <= 50:
            # Perform Henze-Zirkler test
            norm, p_value = mvn_R.multnormtest(X, test = 'hz', alpha = alpha, ddof = 0, scale = False)[0:2]
        else:
            # Perform Mardia's test
            norm, p_value = mvn_R.multnormtest(X, test = 'mardia_comb', alpha = alpha, ddof = 0, scale = False)[0:2]
            # Return only the lowest p-value
            p_value = np.min(p_value)
    
    # Significance of non-normality
    sig_nn = np.logical_not(norm)
    
    return (sig_nn, p_value, map_nn)

#%% Nonlinearity

def quadtest (x, y):
    '''
    Quadratic test for nonlinear correlation
	
	The function performs the quadratic test for nonlinear correlation
    detection between two variables. It is suggested to centre variables to
    zero mean and to scale them to unit variance.
    
    Parameters
    __________
        x : array_like, 1d
            Predictor variable, must have shape (N, )
        y : array_like, 1d
            Response variable, must have shape (N, )
    
    Returns
    _______
        p_value : float
            p-value of the quadratic test on x and y
    '''
    
    # Dataset info ------------------------------------------------------------
    
    # Give depth to arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # Number of observations and number of variables
    N, V = np.shape(x)
    
    # Linear regression model -------------------------------------------------
    
    # Initialize model
    reg_lin = LinearRegression(fit_intercept = False)
    # Fit the model
    reg_lin = reg_lin.fit(x, y)
    # Apply the model
    pred_lin = reg_lin.predict(x)
    # Compute sum of squared errors
    sse_lin = np.sum((y - pred_lin)**2)
    
    # Quadratic regression model ----------------------------------------------
    
    # Initialize model
    reg_quad = LinearRegression(fit_intercept = False)
    # Fit the model
    reg_quad = reg_quad.fit(np.concatenate((x**2, x), axis = 1), y)
    # Apply the model
    pred_quad = reg_quad.predict(np.concatenate((x**2, x), axis = 1))
    # Compute sum of squared errors
    sse_quad = np.sum((y - pred_quad)**2)
    
    # Quadratic test significance ---------------------------------------------
    
    # Copute F-value
    F_value = (sse_lin - sse_quad)/(sse_quad/(N - 2))
    # Compute p-value
    p_value = 1 - f.cdf(F_value, 1, N - 2)
    # Null p_value if below maine precision
    p_value = 0 if p_value < 10*np.finfo(np.float64).eps else p_value
    
    return p_value

def corrcoef_matrix (X):
    '''
    Matrix of linear correlation coefficients among the variables in X
	
	The function evaluates the linear correlation coefficient between each
    couple of vairables in the dataset. It is suggested to centre variables to
    zero mean and to scale them to unit variance.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
    
    Returns
    _______
        LC : array
            Matrix of linear coerrelation coefficients, component [r, c]
            is coefficient between X[:, r] and X[:, c]
    '''
    
    # Compute matrix of linear correlation coefficients
    LC = np.corrcoef(X, rowvar = False)
    
    return LC

"""
def maxcorrcoef_matrix_no_cat (X, force_sym = True):
    '''
    Matrix of maximal correlation coefficients among the variables in X
	
	The function evaluates the maximal correlation coefficient between each
    couple of variables in the dataset. It is suggested to centre variables to
    zero mean and to scale them to unit variance. In principle, maximal
    correlation coefficient between variables x and y could not be the same as
    of the one between variables y and x. The two coefficients are usually the
    same if the nonlinear relationship between the variables is invertible.
    However, the two coefficients can differ due to random chance if the
    variables are nearly uncorrelated. The R library acepack is used for the
    maximal correlation analysis, therefore a link to the R language by means
    of the rpy2 package is required.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
        force_sym : bool
            If True, the maximal correlation matrix is forced to be symmetric
            by evaluating only the maximal correlaton coefficiet between
            X[:, r] and X[:, c] and assuming it is the same as the one between
            X[:, c] and X[:, r] (which is not evaluated)
            
    Returns
    _______
        MC : array
            Matrix of maximal coerrelation coefficients, component [r, c]
            is coefficient between X[:, r] and X[:, c]
    '''
    # Dataset info ------------------------------------------------------------

    # Number of variables
    V = np.shape(X)[1]
    
    # Matrix of maximal correlation coefficients ------------------------------
    
    # Pre-allocation
    MC = np.identity(V)
    
    # Check if matrix requested symmetrical
    if force_sym:
        # Loop over columns
        for c in range(0, V - 1):
            # Loop over rows
            for r in range(c + 1, V):
                # Compute maximal correlation coefficient between X[:, r] and X[:, c]
                MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c])
                # Copy other coefficient
                MC[c, r] = MC[r, c]
    else:
        # Loop over columns
        for c in range(0, V - 1):
            # Loop over rows
            for r in range(c + 1, V):
                # Compute maximal correlation coefficient between X[:, r] and X[:, c]
                MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c])
                # Compute other coefficient
                MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r])
    
    return MC
"""

def maxcorrcoef_matrix (X, cat = None, force_sym = True):
    '''
    Matrix of maximal correlation coefficients among the variables in X
	
	The function evaluates the maximal correlation coefficient between each
    couple of variables in the dataset. It is suggested to centre variables to
    zero mean and to scale them to unit variance. In principle, maximal
    correlation coefficient between variables x and y could not be the same as
    of the one between variables y and x. The two coefficients are usually the
    same if the nonlinear relationship between the variables is invertible.
    However, the two coefficients can differ ue to random chance if the
    variables are nearly uncorrelated. Symmetry is assumed by default, but it
    can be deactivated at will of the user. The R library acepack is used for
    the maximal correlation analysis, therefore a link to the R language by
    means of the rpy2 package is required.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
        cat : {None, list}
            List of integer indices of categorical variables in X, where i
            in cat if X[:, i] is categorical
        force_sym : bool
            If True, the maximal correlation matrix is forced to be symmetric
            by evaluating only the maximal correlation coefficiet between
            X[:, r] and X[:, c] and assuming it is the same as the one between
            X[:, c] and X[:, r] (which is not evaluated)
            
    Returns
    _______
        MC : array
            Matrix of maximal coerrelation coefficients, component [r, c]
            is coefficient between X[:, r] and X[:, c]
    '''
    # Dataset info ------------------------------------------------------------

    # Number of variables
    V = np.shape(X)[1]
    
    # Check for categorical variables
    if cat is None:
        # Propagate the None
        cat_log = None
    else:
        # Convert list of inidice of categoricals to logical vector
        cat_log = np.zeros(V, dtype = np.bool_)
        cat_log[cat] = True
    
    # Matrix of maximal correlation coefficients ------------------------------
    
    # Pre-allocation
    MC = np.identity(V)
    
    # Check for categorical variables
    if cat_log is None:
        # Check if matrix requested symmetrical
        if force_sym:
            # Loop over columns
            for c in range(0, V - 1):
                # Loop over rows
                for r in range(c + 1, V):
                    # Compute maximal correlation coefficient between X[:, r] and X[:, c]
                    MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c])
                    # Copy other coefficient
                    MC[c, r] = MC[r, c]
        else:
            # Loop over columns
            for c in range(0, V - 1):
                # Loop over rows
                for r in range(c + 1, V):
                    # Compute maximal correlation coefficient between X[:, r] and X[:, c]
                    MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c])
                    # Compute other coefficient
                    MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r])
    else:
        # Loop over columns
        for c in range(0, V - 1):
            # Loop over rows
            for r in range(c + 1, V):
                # Check for catgorical vairables
                if not cat_log[r] and not cat_log[c]:
                    # No categorical variables in data
                    MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c])
                    # Check if matrix requested symmetrical
                    if force_sym:
                        # Symmetrical, just copy
                        MC[c, r] = MC[r, c]
                    else:
                        # Non-symmetrical, compute
                        MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r])
                else:
                    if cat_log[r] and not cat_log[c]:
                        # Predictor variable X[:, r] is categorical
                        MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c], cat = np.array([1], dtype = np.int64))
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC[c, r] = MC[r, c]
                        else:
                            # Non-symmetrical, flipped reponse variable X[:, r] is categorical
                            MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r], cat = np.array([0], dtype = np.int64))
                    elif not cat_log[r] and cat_log[c]:
                        # Response variable X[:, c] is categorical
                        MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c], cat = np.array([0], dtype = np.int64))
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC[c, r] = MC[r, c]
                        else:
                            # Non-symmetrical, flipped predictor variable X[:, c] is categorical
                            MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r], cat = np.array([1], dtype = np.int64))
                    else:
                        # Predictor and response variables X[:, r] and X[:, c] are both categorical
                        MC[r, c] = ace_R.maxcorrcoef(X[:, r], X[:, c], cat = np.array([0, 1], dtype = np.int64))
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC[c, r] = MC[r, c]
                        else:
                            # Non-symmetrical, compute
                            MC[c, r] = ace_R.maxcorrcoef(X[:, c], X[:, r], cat = np.array([0, 1], dtype = np.int64))
    
    return MC

def quadtest_matrix (X, cat = None):
    '''
    Matrix of p-values from quadratic tests among the variables in X
	
	The function performs the quadratic test between each couple of variables
    in the dataset. It is suggested to centre variables to zero mean and to
    scale them to unit variance. The p-value of the quadratic test between
    variables x and y is different as of the one between variables y and x.
    Also note that categorical variables cannot be assessed by the quandartic
    test, therefore they are excluded and a p-value of 1 is inserted in the
    matrix.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
        cat : {None, list}
            List of integer indices of categorical variables in X, where i
            in cat if X[:, i] is categorical
            
    Returns
    _______
        QT : array
            Matrix of p-values from quadratic tests, component [r, c]
            from test between X[:, r] and X[:, c]
    '''
    # Dataset info ------------------------------------------------------------

    # Number of variables
    V = np.shape(X)[1]
    
    # Check for categorical variables
    if cat is None:
        # Propagate the None
        cat_log = None
    else:
        # Convert list of inidice of categoricals to logical vector
        cat_log = np.zeros(V, dtype = np.bool_)
        cat_log[cat] = True
    
    # Matrix of p-values from quadratic tests ---------------------------------
    
    # Pre-allocation
    QT = np.identity(V)
    
    # Check for categorical variables
    if cat_log is None:
        # Loop over columns
        for c in range(0, V - 1):
            # Loop over rows
            for r in range(c + 1, V):
                # Perform quadratic testbetween X[:, r] and X[:, c]
                QT[r, c] = quadtest(X[:, r], X[:, c])
                # Perform other test
                QT[c, r] = quadtest(X[:, c], X[:, r])
    else:
        # Loop over columns
        for c in range(0, V - 1):
            # Check if response variables X[:, c] is categorical
            if cat_log[c]:
                # Response variables X[:, c] is categorical, skip column
                QT[:, c] = 1
                QT[c, :] = 1
            else:
                # Loop over rows
                for r in range(c + 1, V):
                    # Check if predictor variables X[:, r] is categorical
                    if cat_log[r]:
                        # Predictor variables X[:, r] is categorical
                        QT[r, c] = 1
                        QT[c, r] = 1
                    else:
                        # Perform quadratic testbetween X[:, r] and X[:, c]
                        QT[r, c] = quadtest(X[:, r], X[:, c])
                        # Perform other test
                        QT[c, r] = quadtest(X[:, c], X[:, r])

    return QT

def maxcorrcoef_conflim (x, y, cat = None, alpha = 0.01, N_boot = 1000, CL_method = 'bias_corr_acc', random_state = None):
    '''
    Confidence limits of maximal correlation coefficient

 	The function estimates the confidence limits of the maximal correlation
    coefficient. The distribution of the coefficient is approximated by means
    of the bootrap approach. Confidence limits of can be estimated using the
    bias-corrected and accelerated method or the ``common'' quantile-based
    method. The former method is stronly recommended to account for the bias in
    the estimate of the coefficient due to sampling with replacement used to
    generate the bootstrap samples (which could inflate the values of the
    coefficient). Using the quantile-based method is discouraged. The R library
    acepack is used for the maximal correlation analysis, therefore a link to
    the R language by means of the rpy2 package is required.
    
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
            * cat = np.array([0, 1], dtype = int) if both x and y are\
            categorical
        
        alpha : float
            Significance level for estimating limits
        N_boot : int
            Number of boostrap samples to be generated
        CL_method : {'bias_corr_acc', 'quantile'}
            Method for estimating confidence limits of the maximal correlation
            coefficient
            
            * 'bias_corr_acc' : bias-corrected and accelerated confidence
            limits (default)
            * 'quantile' : quantile-based confidence limits
        
        random_state : {None, int, numpy rng instance}
            Numpy random number generator seed or instance, if None a random
            generator is initialized (default)
    
    Returns
    _______
        MC_lcl : float
            Lower confidence limit of maximal correlation coefficient
        MC_ucl : float
            Upper confidence limit of maxmimal correlation coefficient
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations
    N = np.shape(x)[0]

    # Make random number generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(seed = random_state)
    else:
        rng = random_state
        
    # Boostrap distribution ---------------------------------------------------
    
    # Matrix of indices for resampling
    idx_boot = rng.integers(low = 0, high = N, size = (N_boot, N))
    # Make array of samples
    x_boot = x[idx_boot, ...]
    y_boot = y[idx_boot, ...]
    
    # Pre-allocations
    MC_boot = np.empty(N_boot)
    
    # Loop over boostrap samples
    for i in range(N_boot):
        # Compute maximal correlation coefficient
        MC_boot[i] = ace_R.maxcorrcoef(x_boot[i], y_boot[i], cat = cat)
    
    # Bootstrap confindence limits --------------------------------------------
    
    # Compute point estimates of maximal correlation coefficient
    MC_point = ace_R.maxcorrcoef(x, y, cat = cat)
    
    # Maximum value of safety counter
    sc_max = 10
    # Initialize safety counter
    sc = 0
    
    # Significance levels for two-tail confidence limits
    alphas = (alpha/2, 1 - alpha/2)
    
    # Select method
    if CL_method == 'quantile':
        # Ensure that point estimates are within boostrap samples
        while (MC_point < np.min(MC_boot) or MC_point > np.max(MC_boot)) and (sc < sc_max):
            # Increment safety counter
            sc += 1
            # Regenerate bootstrap samples
            idx_boot = rng.integers(low = 0, high = N, size = (N_boot, N))
            x_boot = x[idx_boot, ...]
            y_boot = y[idx_boot, ...]
            MC_boot = np.empty(N_boot)
            for i in range(N_boot):
                MC_boot[i] = ace_R.maxcorrcoef(x_boot[i], y_boot[i], cat = cat)
        
        # Quantile-based confidence limits
        MC_cl = np.quantile(MC_boot, alphas)
        
    elif CL_method == 'bias_corr_acc':
        # Bias-corrected and accelerated confidence limits
        
        # Compute bias-correction factor --------------------------------------
        
        # Ensure that point estimates are within boostrap samples
        while (MC_point < np.min(MC_boot) or MC_point > np.max(MC_boot)) and (sc < sc_max):
            # Increment safety counter
            sc += 1
            # Regenerate bootstrap samples
            idx_boot = rng.integers(low = 0, high = N, size = (N_boot, N))
            x_boot = x[idx_boot, ...]
            y_boot = y[idx_boot, ...]
            MC_boot = np.empty(N_boot)
            for i in range(N_boot):
                MC_boot[i] = ace_R.maxcorrcoef(x_boot[i], y_boot[i], cat = cat)
            
        # Quantile of the statistics in the bootstrap distributions
        q = percentileofscore(MC_boot, MC_point)/100
        # Bias-correction factor
        BC_fac = norm.ppf(q)
    
        # Compute acceleration factor -----------------------------------------
        
        # Index vector for jackknife
        idx = np.ones(N, dtype = np.bool_)
        # Pre-allocation
        MC_jack = np.empty(N)
        # Loop over observations
        for n in range(N):
           # Remove observation n
           idx[n] = False
           # Compute statistics
           MC_jack[n] = ace_R.maxcorrcoef(x[idx], y[idx], cat = cat)
           # Put back observation n
           idx[n]= True
        # Mean statistics over jackknife
        MC_jack_mean = np.mean(MC_jack)
        # Compute acceleration factor
        num = np.sum((MC_jack_mean - MC_jack)**3)
        den = 6*np.sum((MC_jack_mean - MC_jack)**2)**(3/2)
        A_fac = num/den
    
        # Correct significance levels -----------------------------------------
        
        # Normal deviates
        zetas = norm.ppf(alphas)
    
        # Sigificance level of lower confidence limit
        num_lcl = BC_fac + zetas[0]
        alpha_lcl = norm.cdf(BC_fac + num_lcl/(1 - A_fac*num_lcl))
    
        # Significance level of upper confidence limit
        num_ucl = BC_fac + zetas[1]
        alpha_ucl = norm.cdf(BC_fac + num_ucl/(1 - A_fac*num_ucl))
        
        # Significance levels for two-tail confidence limits
        alphas_BAc = (alpha_lcl, alpha_ucl)
    
        # Bias-corrected and accelerated confidence limits --------------------
    
        # Compute bias-corrected and accelerated confidence limits
        MC_cl = np.quantile(MC_boot, alphas_BAc)
    else:
        raise Exception('Confidence limit method not supported')
    
    return (MC_cl[0], MC_cl[1])

def maxcorrcoef_conflim_matrix (X, cat = None, force_sym = True, alpha = 0.01, N_boot = 1000, CL_method = 'bias_corr_acc', random_state = None):
    '''
    Matrix of bootstrap confidence limits of the maximal correlation
    coefficients among the variables in X
	
	The function estimates the confidence limits of maximal correlation
    coefficient between each couple of variables in the dataset using the
    bootstrap approach. Details on how this is done are resported in the
    docstring of the function maxcorrcoef_conflim. It is suggested to centre
    variables to zero mean and to scale them to unit variance. The R library
    acepack is used for the maximal correlation analysis, therefore a link to
    the R language by means of the rpy2 package is required.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)
        cat : {None, list}
            List of integer indices of categorical variables in X, where i
            in cat if X[:, i] is categorical
        force_sym : bool
            If True, the matrices of confidence limits areforced to be
            symmetric by evaluating only the limits of the coefficiet between
            X[:, r] and X[:, c] and assuming it is the same as the one between
            X[:, c] and X[:, r] (which is not evaluated)
        
        alpha : float
            Significance level for estimating limits
        N_boot : int
            Number of boostrap samples to be generated
        CL_method : {'bias_corr_acc', 'quantile'}
            Method for estimating confidence limits of the maximal correlation
            coefficient
            
            * 'bias_corr_acc' : bias-corrected and accelerated confidence
            limits (default)
            * 'quantile' : quantile-based confidence limits
        
        random_state : {None, int, numpy rng instance}
            Numpy random number generator seed or instance, if None a random
            generator is initialized (default)
    
    Returns
    _______
        MC_lcl : float
            Matrix of lower confidence limits of maximal correlation
            coefficients, component [r, c] is limit of coefficient between
            X[:, r] and X[:, c]
        MC_ucl : float
            Matrix of upper confidence limits of maximal correlation
            coefficients, component [r, c] is limit of coefficient between
            X[:, r] and X[:, c]
    '''
    # Dataset info ------------------------------------------------------------

    # Number of variables
    V = np.shape(X)[1]
    
    # Check for categorical variables
    if cat is None:
        # Propagate the None
        cat_log = None
    else:
        # Convert list of inidice of categoricals to logical vector
        cat_log = np.zeros(V, dtype = np.bool_)
        cat_log[cat] = True
    
    # Matrix of maximal correlation coefficients ------------------------------
    
    # Pre-allocation
    MC_lcl = np.zeros((V, V))
    MC_ucl = np.zeros((V, V))
    
    # Check for categorical variables
    if cat_log is None:
        # Check if matrix requested symmetrical
        if force_sym:
            # Loop over columns
            for c in range(0, V - 1):
                # Loop over rows
                for r in range(c + 1, V):
                    # Confidence limits of maximal correlation coefficient between X[:, r] and X[:, c]
                    MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                        X[:, r], X[:, c],
                        alpha = alpha,
                        N_boot = N_boot,
                        CL_method = CL_method,
                        random_state = random_state
                    )
                    # Copy other limits
                    MC_lcl[c, r] = MC_lcl[r, c]
                    MC_ucl[c, r] = MC_ucl[r, c]
        else:
            # Loop over columns
            for c in range(0, V - 1):
                # Loop over rows
                for r in range(c + 1, V):
                    # Confidence limits of maximal correlation coefficient between X[:, r] and X[:, c]
                    MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                        X[:, r], X[:, c],
                        alpha = alpha,
                        N_boot = N_boot,
                        CL_method = CL_method,
                        random_state = random_state
                    )
                    # Compute other limits
                    MC_lcl[c, r], MC_ucl[c, r] = maxcorrcoef_conflim(
                        X[:, c], X[:, r],
                        alpha = alpha,
                        N_boot = N_boot,
                        CL_method = CL_method,
                        random_state = random_state
                    )
    else:
        # Loop over columns
        for c in range(0, V - 1):
            # Loop over rows
            for r in range(c + 1, V):
                # Check for catgorical vairables
                if not cat_log[r] and not cat_log[c]:
                    # No categorical variables in data
                    MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                        X[:, r], X[:, c],
                        alpha = alpha,
                        N_boot = N_boot,
                        CL_method = CL_method,
                        random_state = random_state
                    )
                    # Check if matrix requested symmetrical
                    if force_sym:
                        # Symmetrical, just copy
                        MC_lcl[c, r] = MC_lcl[r, c]
                        MC_ucl[c, r] = MC_ucl[r, c]
                    else:
                        # Non-symmetrical, compute
                        MC_lcl[c, r], MC_ucl[c, r] = maxcorrcoef_conflim(
                            X[:, c], X[:, r],
                            alpha = alpha,
                            N_boot = N_boot,
                            CL_method = CL_method,
                            random_state = random_state
                        )
                else:
                    if cat_log[r] and not cat_log[c]:
                        # Predictor variable X[:, r] is categorical
                        MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                            X[:, r], X[:, c],
                            cat = np.array([1], dtype = np.int64),
                            alpha = alpha,
                            N_boot = N_boot,
                            CL_method = CL_method,
                            random_state = random_state
                        )
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC_lcl[c, r] = MC_lcl[r, c]
                            MC_ucl[c, r] = MC_ucl[r, c]
                        else:
                            # Non-symmetrical, flipped reponse variable X[:, r] is categorical
                            MC_lcl[c, r], MC_ucl[c, r] = maxcorrcoef_conflim(
                                X[:, c], X[:, r],
                                cat = np.array([0], dtype = np.int64),
                                alpha = alpha,
                                N_boot = N_boot,
                                CL_method = CL_method,
                                random_state = random_state
                            )
                    elif not cat_log[r] and cat_log[c]:
                        # Response variable X[:, c] is categorical
                        MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                            X[:, r], X[:, c],
                            cat = np.array([0], dtype = np.int64),
                            alpha = alpha,
                            N_boot = N_boot,
                            CL_method = CL_method,
                            random_state = random_state
                        )
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC_lcl[c, r] = MC_lcl[r, c]
                            MC_ucl[c, r] = MC_ucl[r, c]
                        else:
                            # Non-symmetrical, flipped predictor variable X[:, c] is categorical
                            MC_lcl[c, r], MC_ucl[c, r] = maxcorrcoef_conflim(
                                X[:, c], X[:, r],
                                cat = np.array([1], dtype = np.int64),
                                alpha = alpha,
                                N_boot = N_boot,
                                CL_method = CL_method,
                                random_state = random_state
                            )
                    else:
                        # Predictor and response variables X[:, r] and X[:, c] are both categorical
                        MC_lcl[r, c], MC_ucl[r, c] = maxcorrcoef_conflim(
                            X[:, r], X[:, c],
                            cat = np.array([0, 1], dtype = np.int64),
                            alpha = alpha,
                            N_boot = N_boot,
                            CL_method = CL_method,
                            random_state = random_state
                        )
                        # Check if matrix requested symmetrical
                        if force_sym:
                            # Symmetrical, just copy
                            MC_lcl[c, r] = MC_lcl[r, c]
                            MC_ucl[c, r] = MC_ucl[r, c]
                        else:
                            # Non-symmetrical, compute
                            MC_lcl[c, r], MC_ucl[c, r] = maxcorrcoef_conflim(
                                X[:, c], X[:, r],
                                cat = np.array([0, 1], dtype = np.int64),
                                alpha = alpha,
                                N_boot = N_boot,
                                CL_method = CL_method,
                                random_state = random_state
                            )
    
    return (MC_lcl, MC_ucl)

def nonlinearity_test (X, frac_nl = 0.1, cat = None, force_sym = True, tol_dif = 0.4, thr_hmc = 0.92, tol_dif_hmc = 0.1, deflate_MC = 'median', alpha_boot = 0.01, N_boot = 1000, CL_method_boot = 'bias_corr_acc', random_state = None, alpha_qt = 0.01, correction = True):
    '''
    Detection of significant nonlinear correlation among the variables in X
	
	The function evaluates the presence of nonlinear correlation by performing
	simultaneously linear correlation analysis, maximal correlation analysis,
    and quadratic test. Concerning the maximal correlation coefficient, the
    well known inflation problem for nearly uncorrelated variables is
    counteracted ``deflating'' the value of the coefficient. The method for
    setting the deflation factor can be chosen by the user. On the other hand,
    the false-positive-rate of the quadratic test is counteracted applying the
    Bonferroni correction to the nominal significance level of the test. The
    dataset is deemed as featuring significant nonlinearity if the fraction of
    variables involved in significant nonlinear relationships is greater than
    a preset threshold, by default 0.1. A data matrix with N rows and V columns
    is expected as input, where N is the number of observations and V is the
    number of variables. It is suggested to centre variables to zero mean and
    to scale them to unit variance. The R library acepack is used for the
    maximal correlation analysis, therefore a link to the R language by means
    of the rpy2 package is required.
    
    Inputs:
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)\n
        frac_nl : float
            Minimum fraction of variables involved in nonlinear relationships
            to deem the dataset as significantly nonlinear
        cat : {None, list}
            List of integer indices of categorical variables in X, where i
            in cat if X[:, i] is categorical
        force_sym : bool
            If True, the maximal correlation matrix is forced to be symmetric
            by evaluating only the maximal correlation coefficiet between
            X[:, r] and X[:, c] and assuming it is the same as the one between
            X[:, c] and X[:, r] (which is not evaluated)
        tol_dif : float
            Tolerance for difference between maximal correlation and absolute
            linear correlation
        thr_hmc : float
            Threshold of maximal correlation for difference tolerance
            refinement
        tol_dif_hmc : float
            Tolerance for difference between maximal correlation and absolute
            linear correlation when maximal correlation is above thr_hmc
        deflate_MC : {'none', 'median', 'upper', 'bootstrap'}
            Method for deflating the maximal correlation coefficient
            
            * 'none' : no deflation is applied
            * 'median' : deflation factor is set to the median value of the
            maximal correlation coefficient of two uncorrelated standard normal
            variables with the same number of observations of the ones being
            assessed (default)
            * 'upper' : deflation factor is set to the 99% upper confidence
            limit of the maximal correlation coefficient of two uncorrelated
            standard normal variables  with the same number of observations
            of the ones being assessed
            * 'bootstrap' : deflation factor is set to the lower confidence
            interval of the maximal correlation coefficient (point estimate
            minus (1 - alpha_boot/2)*100% confidence limit) estimated with the
            bootstrap approach
            
        alpha_boot : float
            Significance level for estimating the confidence limits of the
            maximal correlation coefficient by boostrap approach (has effect
            only if deflate_MC = 'bootstrap')
        N_boot : int
            Number of boostrap samples to be generated for estimating
            confidence limits of the maximal correlation coefficient by
            boostrap approach
        CL_method_boot : {'bias_corr_acc', 'quantile'}
            Method for estimating confidence limits of the maximal correlation
            coefficient by bootstrap approach
            
                * 'bias_corr_acc' : bias-corrected and accelerated confidence
                limits (default)
                * 'quantile' : quantile-based confidence limits
                
        random_state : {None, int, numpy rng instance}
            Numpy random number generator seed or instance, if None a random
            generator is initialized (default)
        alpha_qt : float
            Significance level for quaratic test
        correction : bool
            Logical flag to enable the Bonferroni correction to adjust the
            nominal significance level of the quadratic test (provided as
            alpha_qt) dividing it by the  number of tested being performend,
            which is V_nc*(V_nc - 1), where V_nc is the number of
            non-categorical variables in the dataset
    
    Returns
    _______
        sig_nl : bool
            True if nonlinearity in the dataset is significant
        frac_nl_vars : float
            Fraction of variables involved in significant nonlinear
            relationships
        map_nl : array
            Logical array of shape shape (V, ), where component map_nl[v] is
            True if X[:, v] is involved in significant nonlinear relationships
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations and number of variables
    N, V = np.shape(X)
    
    # Correction of significance level ----------------------------------------
    
    # Check if categorical variables
    if cat is None:
        # Number of tests to be performed
        N_tests = int(V*(V - 1))
    else:
        # Number of non-categorical variables
        V_nc = V - len(cat)
        # Number of tests to be performed
        N_tests = int(V_nc*(V_nc - 1))
    
    # Check if correction is requested
    if correction:
        # Correct significance level of p-value (Bonferroni correction, very conservative)
        alpha_qt_adj = alpha_qt/N_tests
    else:
        # No correction required
        alpha_qt_adj = alpha_qt

    # Compute coefficients ----------------------------------------------------

    # Matrix of linear correlation coefficients
    LC = corrcoef_matrix(X)
    
    # Matrix of quadratic test p-values
    QT = quadtest_matrix(X, cat = cat)
    
    # Matrix of maximal correlation coefficients
    MC = maxcorrcoef_matrix(X, cat = cat, force_sym = force_sym)
    
    # Deflation factor for maximal correlation coefficient --------------------
    
    # Selector of deflation method
    if deflate_MC == 'none':
        # No deflation
        df = 0
    elif deflate_MC == 'median':
        # Defaltion factor is median
        df = 3.1075604666208005*(N**(- 0.5268877345291182))
    elif deflate_MC == 'upper':
        # Deflation factor is upper control limit
        df = 4.568213258443067*(N**(- 0.5010835837678528))
    elif deflate_MC == 'bootstrap':
        # Boostrap confidence limits
        MC_lcl, MC_ucl = maxcorrcoef_conflim_matrix(X, cat = cat, force_sym = force_sym, alpha = alpha_boot, N_boot = N_boot, CL_method = CL_method_boot, random_state = random_state)
        # Deflation factor is lower confidence interval
        df = MC - MC_lcl 
    else:
        raise Exception('Deflation method not supported')

    # Significance of nonlinearity -------------------------------------------
    
    # Significance of difference between maximal correlation and absolute linear correlation
    sig_diff = ((MC - df) - np.abs(LC)) > tol_dif
    # Significance of difference between maximal correlation and absolute linear correlation for high maximal
    sig_diff_hmc = np.logical_and(MC > thr_hmc, (MC - np.abs(LC)) > tol_dif_hmc)
    # Overall significance of maximal correlation
    sig_mc = np.logical_or(sig_diff, sig_diff_hmc)
    
    # Significance of quadratic test
    sig_qt = QT < alpha_qt_adj
    
    # Significance of nonlinearity
    sig_def = np.logical_or(sig_mc, sig_qt)

    # Check for significant nonlinearity in each variable
    map_nl = np.logical_or(np.any(sig_def, axis = 0), np.any(sig_def, axis = 1))
    # Collapse in fraction
    frac_nl_vars = np.sum(map_nl)/V
    # Nonlinearity detected in the dataset in "given percentage of variables mode"
    sig_nl = np.bool_(frac_nl_vars > frac_nl)
            
    return (sig_nl, frac_nl_vars, map_nl)

#%% Dynamics

def significant_lags_acf (x, nlags = None, alpha = 0.01, approach = 'ljung_box', variance = 'bartlett', correction = True, major = False):
    '''
    Detection of significant dynamics in the time series x
    
    The function evaluates the dynamics by testing significant ceofficients in
    the autocorrelation function of x. By default, lags of significant
    coefficients are determined by means of the Ljung-Box Q statistic, and the
    significance level is adjusted by the Bonferroni correction, therefore
    dividing the nominal significance level by the number of lags being tested.
    
    Parameters
    __________
        x : array_like, 1d
            Time series to be tested, must have shape (N, )
        nlags : {None, int}
            Number of lags to be assessed, default is minimum between 20 and
            N//2 - 1 when None is given
        alpha : float
            Significance level for assessment of coefficients
        approach : {'ljung_box', 'conf_lim', 'p_value'}
            Approach to significance assessment of coefficients
            
            * 'ljung_box' : a coefficient is significant if the p-value of its
            Ljung-Box Q statistics is less than the significance level
            (default)
            * 'conf_lim' : a coefficient is signficant if beyond confidence
            limits
            * 'p_value' : a coefficient is significant if its p-value is less
            than the significance level
                
        variance : {'bartlett', 'large_sample', 'lag_corr'}
            Estimator of the variance of the distribution of coefficients
            
            * 'barlett' : variance from Bartlett's formula (default)
            * 'large_sample' : variance from large-sample approximation
            * 'lag_corr' : variance from lag-corrected approach
            
        correction : bool
            Logical flag to enable the Bonferroni correction to adjust the
            nominal significance level (provided as alpha) dividing it by the
            number of lags being tested
        major : bool
            Logical flag to enable the major lag constraint, according to which
            the only coefficients deemed as significant are the ones with lower
            order, ordered and starting from 1 (in other words, significant
            coefficients can be only subsets of contiguous natural numbers
            starting at 1)
    
    Returns
    _______
        lags_acf : array
            Indexes of lags of the significant coefficients in autocorrelation
            function
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations
    N = np.shape(x)[0]
    
    # Numebr of lags to be considerd
    if nlags is None:
        nlags = np.minimum(20, N//2 - 1)
    
    # Correction of significance level ----------------------------------------
    
    # Check if correction is requested
    if correction:
        # Correct significance level of p-value (Bonferroni correction, very conservative)
        alpha_adj = alpha/nlags
    else:
        # No correction required
        alpha_adj = alpha
    
    # Autocorrelation function ------------------------------------------------
    
    # Compute autocorrelation function and compute Ljung-Box Q statistic
    ACF = smtsa.acf(x, nlags = nlags, alpha = alpha_adj, qstat = True, fft = False)
    # Get autocorrelation function
    acf = ACF[0][1:]
    # Get p-value of Ljung-Box statistics
    p_value_Q = ACF[3]
    
    # Variance of the distribution of coefficients ----------------------------
    
    # Selector for kind of variance
    if variance == 'bartlett':
        # Variance according to Barlett's formula: initialize
        var = np.empty(nlags)
        # Variance of first coefficient as large sample approximation
        var[0] = 1/N
        # Variance of remaining coefficients as Barlett's formula
        var[1:] = (1 + 2 * np.cumsum(acf[0:-1]**2))/N
    elif variance == 'large_sample':
        # Variance according to large sample approximation
        var = 1/N
    elif variance == 'lag_corr':
        # Lag-corrected variance
        var = 1/np.linspace(N - 1, N - nlags, nlags)
    else:
        raise Exception('Requested variance estimator is not supported')
    
    # Significance of coefficients --------------------------------------------
    
    # Selector for approach to significance assessment
    if approach == 'ljung_box':
        # Flag significant coefficients
        acf_lag_detection = p_value_Q < alpha_adj
    elif approach == 'conf_lim':
        # Compute normal deviate limit
        acf_lim = norm.ppf(1 - alpha_adj/2)*np.sqrt(var)
        # Flag significant coefficients
        acf_lag_detection = np.abs(acf) > acf_lim
    elif approach == 'p_value':
        # Compute p-values
        p_value_acf = norm.cdf(- np.abs(acf)/np.sqrt(var))
        # Flag significant coefficients
        acf_lag_detection = p_value_acf < alpha_adj
    
    else:
        raise Exception('Requested approach is not supported')
    
    # Lags corresponding to significant coefficients
    acf_lag_detection = p_value_Q < alpha_adj
    lags_acf = np.array([i + 1 for i, u in enumerate(acf_lag_detection) if u == True])
    
    # Lag constraint ----------------------------------------------------------
    
    # Check if only major lags are requested
    if major:
        # Check if any lag was deemed as significant
        if lags_acf.shape[0] != 0:
            # Initialise list of major lags
            lags_acf_maj = []
            # Loop over lags and keep only major ones
            for l in range(0, lags_acf.shape[0]):
                # Check if current lag is major
                if lags_acf[l] == l + 1:
                    # Save lag
                    lags_acf_maj.append(l + 1)
                else:
                    # Break cycle
                    break
            # Convert list to NumPy array and save to significant lags
            lags_acf = np.array(lags_acf_maj)
    
    return lags_acf

def significant_lags_pacf (x, nlags = None, alpha = 0.01, approach = 'conf_lim', variance = 'large_sample', correction = True, major = False):
    '''
    Detection of significant dynamics in the time series x
    
    The function evaluates the dynamics by testing significant coefficients in
    the partial autocorrelation function of x. By default, lags of significant
    coefficients are determined using the normal deviate test, and the
    significance level is adjusted by the Bonferroni correction, therefore
    dividing the nominal significance level by the number of lags being tested.
    
    Parameters
    __________
        x : array_like, 1d
            Time series to be tested, must have shape (N, )
        nlags : {None, int}
            Number of lags to be assessed, default is minimum between 20 and
            N//2 - 1 when None is given
        alpha : float
            Significance level for assessment of coefficients
        approach : {'conf_lim', 'p_value'}
            Approach to significance assessment of coefficients
            
            * 'conf_lim' : a coefficient is signficant if beyond confidence
            limits (default)
            * 'p_value' : a coefficient is significant if its p-value is less
            than the significance level
                
        variance : {'large_sample', 'lag_corr'}
            Estimator of the variance of the distribution of coefficients
            
            * 'large_sample' : variance from large-sample approximation
            (default)
            * 'lag_corr' : variance from lag-corrected approach
            
        correction : bool
            Logical flag to enable the Bonferroni correction to adjust the
            nominal significance level (provided as alpha) dividing it by the
            number of lags being tested
        major : bool
            Logical flag to enable the major lag constraint, according to which
            the only coefficients deemed as significant are the ones with lower
            order, ordered and starting from 1 (in other words, significant
            coefficients can be only subsets of contiguous natural numbers
            starting at 1)
    
    Returns
    _______
        lags_pacf : array
            Indexes of lags of the significant coefficients in partial
            autocorrelation function
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations
    N = np.shape(x)[0]
    
    # Numebr of lags to be considerd
    if nlags is None:
        nlags = np.minimum(20, N//2 - 1)
    
    # Correction of significance level ----------------------------------------
    
    # Check if correction is requested
    if correction:
        # Correct significance level of p-value (Bonferroni correction, very conservative)
        alpha_adj = alpha/nlags
    else:
        # No correction required
        alpha_adj = alpha

    # Partial autocorrelation function ----------------------------------------
    
    # Compute autocorrelation function
    PACF = smtsa.pacf(x, nlags = nlags, alpha = alpha_adj, method = 'ywm')
    # Get partial autocorrelation function
    pacf = PACF[0][1:]
    
    # Variance of the distribution of coefficients ----------------------------
    
    # Selector for kind of variance
    if variance == 'large_sample':
        # Variance according to large sample approximation
        var = 1/N
    elif variance == 'lag_corr':
        # Lag-corrected variance
        var = 1/np.linspace(N - 1, N - nlags, nlags)
    else:
        raise Exception('Requested variance method is not supported')
    
    # Significance of coefficients --------------------------------------------
    
    # Selector for approach to significance assessment
    if approach == 'conf_lim':
        # Compute normal deviate limit
        pacf_lim = norm.ppf(1 - alpha_adj/2)*np.sqrt(var)
        # Flag significant coefficients
        pacf_lag_detection = np.abs(pacf) > pacf_lim
    elif approach == 'p_value':
        # Compute p-values
        p_value_pacf = norm.cdf(- np.abs(pacf)/np.sqrt(var))
        # Flag significant coefficients
        pacf_lag_detection = p_value_pacf < alpha_adj
    else:
        raise Exception('Requested approach is not supported')
    
    # Lags corresponding to significant coefficients
    lags_pacf = np.array([i + 1 for i, u in enumerate(pacf_lag_detection) if u == True])
    
    # Lag constraint ----------------------------------------------------------
    
    # Check if only major lags are requested
    if major:
        # Check if any lag was deemed as significant
        if lags_pacf.shape[0] != 0:
            # Initialise list of major lags
            lags_pacf_maj = []
            # Loop over lags and keep only major ones
            for l in range(0, lags_pacf.shape[0]):
                # Check if current lag is major
                if lags_pacf[l] == l + 1:
                    # Save lag
                    lags_pacf_maj.append(l + 1)
                else:
                    # Break cycle
                    break
            # Convert list to NumPy array and save to significant lags
            lags_pacf = np.array(lags_pacf_maj)
    
    return lags_pacf

def significant_lags_ccf (x, y, nlags = None, alpha = 0.01, approach = 'conf_lim', variance = 'large_sample', correction = True, major = False):
    '''
    Detection of significant relationships in dynamics of the time series x
    and y
	
	The function evaluates the interaction of dynamics by testing significant
    coefficients in the cross-correlation function of x and y. By default, lags
    of significant coefficients are determined using the normal deviate test,
    and the significance level is adjusted by the Bonferroni correction,
    therefore dividing the nominal significance level by the number of lags
    being tested.
    
    Parameters
    __________
        x : array_like, 1d
            First time series to be tested, must have shape (N, )
        y : array_like, 1d
            Second time series to be tested, must have shape (N, )
        nlags : {None, int}
            Number of lags to be assessed, default is minimum between 20 and
            N//2 - 1 when None is given
        alpha : float
            Significance level for assessment of coefficients
        approach : {'conf_lim', 'p_value'}
            Approach to significance assessment of coefficients
            
            * 'conf_lim' : a coefficient is signficant if beyond confidence
            limits (default)
            * 'p_value' : a coefficient is significant if its p-value is less
            than the significance level
                
        variance : {'large_sample', 'lag_corr'}
            Estimator of the variance of the distribution of coefficients
            
            * 'large_sample' : variance from large-sample approximation
            (default)
            * 'lag_corr' : variance from lag-corrected approach
            
        correction : bool
            Logical flag to enable the Bonferroni correction to adjust the
            nominal significance level (provided as alpha) dividing it by the
            number of lags being tested
        major : bool
            Logical flag to enable the major lag constraint, according to which
            the only coefficients deemed as significant are the ones with lower
            order, ordered and starting from 1 (in other words, significant
            coefficients can be only subsets of contiguous natural numbers
            starting at 1)
    
    Returns
    _______
        lags_ccf : array
            Indexes of lags of the significant coefficients in
            cross-correlation function
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations
    N = np.shape(x)[0]
    
    # Numebr of lags to be considerd
    if nlags is None:
        nlags = np.min(20, N//2 - 1)
    
    # Correction of significance level ----------------------------------------
    
    # Check if correction is requested
    if correction:
        # Correct significance level of p-value (Bonferroni correction, very conservative)
        alpha_adj = alpha/nlags
    else:
        # No correction required
        alpha_adj = alpha
    
    # Cross-correlation function ------------------------------------------------
    
    # Compute cross-correlation function
    ccf = smtsa.ccf(x, y, fft = False)
    # Cut number of lags
    ccf = ccf[1:nlags + 1]
    
    # Variance of the distribution of coefficients ----------------------------
    
    # Selector for kind of variance
    if variance == 'large_sample':
        # Variance according to large sample approximation
        var = 1/N
    elif variance == 'lag_corr':
        # Lag-corrected variance
        var = 1/np.linspace(N - 1, N - nlags, nlags)
    else:
        raise Exception('Requested variance method is not supported')
    
    # Significance of coefficients --------------------------------------------
    
    # Selector for approach to significance assessment
    if approach == 'conf_lim':
        # Compute normal deviate limit
        ccf_lim = norm.ppf(1 - alpha_adj/2)*np.sqrt(var)
        # Flag significant coefficients
        ccf_lag_detection = np.abs(ccf) > ccf_lim
    elif approach == 'p_value':
        # Compute p-values
        p_value_ccf = norm.cdf(- np.abs(ccf)/np.sqrt(var))
        # Flag significant coefficients
        ccf_lag_detection = p_value_ccf < alpha_adj
    else:
        raise Exception('Requested approach is not supported')
    
    # Lags corresponding to significant coefficients
    lags_ccf = np.array([i + 1 for i, u in enumerate(ccf_lag_detection) if u == True])
    
    # Major lags --------------------------------------------------------------
    
    # Check if only major lags are requested
    if major:
        # Check if any lag was deemed as significant
        if lags_ccf.shape[0] != 0:
            # Initialise list of major lags
            lags_ccf_maj = []
            # Loop over lags and keep only major ones
            for l in range(0, lags_ccf.shape[0]):
                # Check if current lag is major
                if lags_ccf[l] == l + 1:
                    # Save lag
                    lags_ccf_maj.append(l + 1)
                else:
                    # Break cycle
                    break
            # Convert list to NumPy array and save to significant lags
            lags_ccf = np.array(lags_ccf_maj)
    
    return lags_ccf

def dynamics_test (X, frac_dyn = 0.1, alpha = 0.01, cat = None, nlags = None, approach = 'ljung_box', variance = 'bartlett', correction = True, major = False):
    '''
    Detection of significant dynamics in the variables in X
 	
 	The function evaluates the presence of dynamics by detecting the numbers of
    significant coefficients in the autocorrelation functions of each
    non-categorical varibale in the dataset to be tested. The dataset is deemed
    as featuring significant dynamics if the fraction of variables showing
    dynamic behaviour (number of significant lags in the autocorrelation
    function greater than 0) is greater than a preset threshold, by default
    0.1. Note that categorical variables are neither considered in the dynamics
    assessment nor counted when computig the fraction of dynamic variables. A
    data matrix with N rows and V columns is expected as input, where N is the
    number of observations and V is the number of variables. It is suggested to
    centre variables to zero mean and to scale them to unit variance.
    
    Parameters
    __________
        X : array_like, 2d
            Dataset to be assessed, must have shape (N, V)\n
        frac_dyn : float
            Minimum fraction of dynamic variables to deem the dataset as
            significantly dynamic
        alpha : float
            Significance level for assessment of coefficients of
            autocorrelation function
        cat : {None, list}
            List of integer indices of categorical variables in X, where i
            in cat if X[:, i] is categorical
        nlags : {None, int}
            Number of lags to be assessed, default is minimum between 20 and
            N//2 - 1 when None is given
        approach : {'ljung_box', 'conf_lim', 'p_value'}
            Approach to significance assessment of coefficients of
            autocorrelation function
            
            * 'ljung_box' : a coefficient is significant if the p-value of its
            Ljung-Box Q statistics is less than the significance level
            (default)
            * 'conf_lim' : a coefficient is signficant if beyond confidence
            limits
            * 'p_value' : a coefficient is significant if its p-value is less
            than the significance level
                
        variance : {'bartlett', 'large_sample', 'lag_corr'}
            Estimator of the variance of the distribution of coefficients of
            autocorrelation function
            
            * 'barlett' : variance from Bartlett's formula (default)
            * 'large_sample' : variance from large-sample approximation
            * 'lag_corr' : variance from lag-corrected approach
            
        correction : bool
            Logical flag to enable the Bonferroni correction to adjust the
            nominal significance level (provided as alpha) dividing it by the
            number of lags being tested
        major : bool
            Logical flag to enable the major lag constraint, according to which
            the only coefficients deemed as significant are the ones with lower
            order, ordered and starting from 1 (in other words, significant
            coefficients can be only subsets of contiguous natural numbers
            starting at 1)
    
    Returns
    _______
        sig_dyn : bool
            True if dynamics in the dataset is significant
        frac_dyn_vars : float
            Fraction of non-categorical variables featuring significant
            dynamics
        map_dyn : array
            Logical array of shape shape (V, ), where component map_dyn[v] is
            True if X[:, v] features significant dynamics
    '''
    # Dataset info ------------------------------------------------------------

    # Number of observations and number of variables
    N, V = np.shape(X)
    
    # Numebr of lags to be considerd
    if nlags is None:
        nlags = np.minimum(20, N//2 - 1)
    
    # Check for categorical variables
    if cat is None:
        # All variables to be assessed
        idx_tba = np.arange(V, dtype = int)
    else:
        # Remove categorical variables
        idx_tba = np.delete(np.arange(V, dtype = int), cat)
    # Number of variables to be assessed
    V_tba = np.shape(idx_tba)[0]

    # Significant lags --------------------------------------------------------

    # Pre-allocation of lag array
    acf_lags = np.zeros(V)
    
    # Loop over variables
    for i in idx_tba:
        # Number of significant lags in autocorrelaton function
        lags_acf = significant_lags_acf(X[:, i], nlags = nlags, alpha = alpha, approach = approach, variance = variance, correction = correction, major = major)
        # Count significant lags
        acf_lags[i] = np.shape(lags_acf)[0]

    # Significance of dynamics ------------------------------------------------

    # Check for significant dynamics in each variable
    map_dyn = np.bool_(acf_lags != 0)
    # Collapse in fraction
    frac_dyn_vars = np.sum(map_dyn)/V_tba
    # Dynamics detected in the dataset in "given percentage of variables mode"
    sig_dyn = np.bool_(frac_dyn_vars > frac_dyn)
            
    return (sig_dyn, frac_dyn_vars, map_dyn)