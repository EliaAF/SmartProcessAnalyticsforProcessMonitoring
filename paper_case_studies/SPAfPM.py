#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPAfPM.py
Version: 0.8.0
Date: 2024/02/26
Author: Elia Arnese-Feffin elia249@mit.edu/elia.arnesefeffin@phd.unipd.it

This code is based on Smart_Process_Analytics.py by Weike Sun,
provided as part of the Smart Process Analytics (SPA) code, available at:
	https://github.com/vickysun5/SmartProcessAnalytics

# GNU General Public License version 3 (GPL-3.0) ------------------------------

Smart Process Analytics for Process Monitoring - main file
Copyright (C) 2022– Elia Arnese-Feffin

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

'''NOTE -----------------------------------------------------------------------
This file can be used to reproduce the case study in the SPAfPM paper. Review
the code below and adjust the settings to run the four case studies. Make sure
to download and install the SVDD package (see link below) before running the
cases.
    SVDD: https://github.com/iqiukp/SVDD-Python
'''

# System interface
import os
# Nice format for strings
import textwrap

# Numerical Python
import numpy as np
# Tabular data structures
import pandas as pd
# Data-pre-processing
from sklearn.preprocessing import StandardScaler

# Functions for dataset assessment
import dataset_property_assessment as dpa
# Functions for data analytics method running
import data_analytics_methods as dam

# General plots
import matplotlib as mpl
import matplotlib.pyplot as plt
# Statistical plots
import seaborn as sns

# Copy variables
from copy import deepcopy

#%% Preliminary operations

# Key to print information on progress
printinfo = True

# Library of models with characteristics
model_library = pd.DataFrame(
    data = {
        'model'                 : [
            'PCA',
            'PLS',
            'CVA',
            'SVDD',
            'DPCA',
            'KPCA',
            'DKPCA',
            'DPLS',
            'KPLS',
            'DKPLS',
            'KDE-CVA'
        ],
        'is_nonlinear'         : [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        'is_dynamic'            : [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        'is_quality_relevant'   : [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    }
)

#%% Acquire settings

# Print info
if printinfo:
    print(
        '#' + 59*'-' + '#',
        '| Smart Process Analytics for Process Monitoring (SPAfPM)   |',
        '| Version 0.8.0, 2024/02/26                                 |',
        '| (c) Fabian Mohr & Elia Arnese Feffin                      |',
        '| Covered by GNU General Public License version 3 (GPL-3.0) |',
        '#' + 59*'-' + '#',
        sep = os.linesep,
        end = 2*'\n'
    )

'''NOTE -----------------------------------------------------------------------
Review the code and make sure the following settings are properly set in order
to run SPAfPM
'''

# Print info
if printinfo:
    print(
        '#' + 60*'-',
        'Acquiring general settings (listing mode)',
        sep = os.linesep,
        end = 2*'\n'
    )

'''NOTE -----------------------------------------------------------------------
File types allowed are:
    .csv:   separator is comma
    .txt:   separator is tab
    .xlsx:  excel file
Bear in mind the following points:
    * Data are expected to be arranged with variables as columns and
      observations as rows
    * Names of variables and observations are allowed in data files, but must
      be in the first row and in the first columns, respectively
    * Data should not contain missing values
'''

# What is the name of file of training data (with file extension)? [string]
# train_data_name = ''
train_data_name = 'datasets/NumericalLinear_NOC.xlsx'
# train_data_name = 'datasets/TEP_NOC.xlsx'
# train_data_name = 'datasets/ContCarSim_NOC.xlsx'
# train_data_name = 'datasets/MetalEtch_NOC.xlsx'

# Are there variable names in data files? [boolean]
header_found = False
if header_found:
    header = 0
else:
    header = None

# Are there observation names in data files? [boolean]
index_found = False
if index_found:
    index = 0
else:
    index = None

'''NOTE -----------------------------------------------------------------------
A selector is provided to decided wether to use all variables for model
building, or just a subset. Note that this subset refers to variables to be
used as input data (the X matrix). In the case quality-relevant monitoring is
requested, quality variables (the Y matrix) should not be included also in
input data.
'''

# Do you want to use only a subset of variables in data? [boolean]
use_subset = True
if use_subset:
    # What are the indices of input variables? [list of integers]
    # process_variable_index = []
    process_variable_index = [*range(15)]   # Linear
    # process_variable_index = [*range(52)]  # TEP no Y
    # process_variable_index = [*range(22)] + [*range(41, 52)]  # TEP yes Y
    # process_variable_index = [*range(8)]   # ContCarSim
    # process_variable_index = [*range(19)]   # MetalEtch no Y
    # process_variable_index = [*range(9)] + [*range(10, 19)]  # MetalEtch yes Y
else:
    process_variable_index = None

# Do you wish to monitor only faults affecting quality? [boolean]
is_quality_relevant = False
if is_quality_relevant:
    # What is the index of the quality variable? [list of integers]
    # quality_variable_index = []
    quality_variable_index = [15]   # Linear
    # quality_variable_index = [34]   # TEP
    # quality_variable_index = [8]    # ContCarSim
    # quality_variable_index = [9]    # MetalEtch
else:
    # General monitoring
    quality_variable_index = None

# Are there discrete/non-numerical variables in data (e. g., binary)? [boolean]
categorical_found = False
if categorical_found:
    # Raise a warning
    print(
        Warning('Warning: Categorical variables found in the data\n'
            + textwrap.fill(textwrap.dedent('''\
                The use of discrete/non-numerical variables is discouraged for
                process monitoring model development: please, evaluate the need
                of such variables. Automatic model selection will be performed,
                if requested. However, SVDD is recommended as default model for
                dataset containing categorical variables.\
            '''))
        ),
        sep = os.linesep
    )
    # What are the indices of the discrete/non-numerical variables? [list of integers]
    categorical_index = []
    # categorical_index = [3, 16]  # MetalEtch
else:
    categorical_index = None

# Do you want plot designation of variables? [boolean]
plot_variable_roles = True

'''NOTE -----------------------------------------------------------------------
While training data should represent normal operating conditions only, testing
data can include faulty observations. In order to estimate the Type I and
Type II error-rates, testing data should come with an additional column
containing a binary variable named 'is_falut'. The variable has value 0 if the
observation is normal operating conditions, while has value 1 if observation is
faulty. The variable 'is_fault' should be addedd as the last column of the
dataset.
'''

# Do you have a testing dataset for performance assessment? [boolean]
have_testing_data = True
if have_testing_data:
    # What is the name of file of testing data (with file extension)? [string]
    # test_data_name = ''
    test_data_name = 'datasets/NumericalLinear_F01.xlsx'
    # test_data_name = 'datasets/NumericalLinear_F02.xlsx'
    # test_data_name = 'datasets/NumericalLinear_F03.xlsx'
    # test_data_name = 'datasets/TEP_F01.xlsx'
    # test_data_name = 'datasets/TEP_F02.xlsx'
    # test_data_name = 'datasets/TEP_F04.xlsx'
    # test_data_name = 'datasets/TEP_F05.xlsx'
    # test_data_name = 'datasets/TEP_F06.xlsx'
    # test_data_name = 'datasets/TEP_F07.xlsx'
    # test_data_name = 'datasets/TEP_F08.xlsx'
    # test_data_name = 'datasets/TEP_F10.xlsx'
    # test_data_name = 'datasets/TEP_F11.xlsx'
    # test_data_name = 'datasets/TEP_F12.xlsx'
    # test_data_name = 'datasets/TEP_F13.xlsx'
    # test_data_name = 'datasets/TEP_F14.xlsx'
    # test_data_name = 'datasets/TEP_F16.xlsx'
    # test_data_name = 'datasets/TEP_F17.xlsx'
    # test_data_name = 'datasets/TEP_F18.xlsx'
    # test_data_name = 'datasets/TEP_F19.xlsx'
    # test_data_name = 'datasets/TEP_F20.xlsx'
    # test_data_name = 'datasets/TEP_F21.xlsx'
    # test_data_name = 'datasets/ContCarSim_F01.xlsx'
    # test_data_name = 'datasets/MetalEtch_F01.xlsx'
    # test_data_name = 'datasets/MetalEtch_F10.xlsx'
    # test_data_name = 'datasets/MetalEtch_F16.xlsx'
else:
    # No testing data available
    test_data_name = None

# Do you wish to automatically select models based on preliminary data interrogation? [boolean]
dataset_assessment = True
if dataset_assessment:
    # Print info
    if printinfo:
        print(
            'Model to be selected automatically based on data interrogation',
            sep = os.linesep
        )
    
    '''NOTE -------------------------------------------------------------------
    Criteria for dataset assessment can be tuned by the user. This must be done
    by directly acting on the code in the section 'Preliminary data
    interrogation'. However, note that parameters of criteria are grouped into
    two families. Some parameters are degrees of freedom to the user (for
    instance, the significance level of the multivariate normality test, or the
    fraction of dynamic variables to deem the dataset as dynamic); other
    parameters have highly technical meanings and may have a relevant effect on
    the sensitivity/selectivity of critria. Parameters in this latter family
    have been optimized during the development of SPAfPM. Manipulating them is
    still allowed, although suggested only to expert users.
    '''
    
    # Do you believe training data are non-normally distributed? [boolean, None if unknown]
    is_non_normal = None
    # Do you believe variables are involved in nonlinear relationships? [boolean, None if unknown]
    is_nonlinear = None
    # Do you believe variables feature dynamic behaviour? [boolean, None if unknown]
    is_dynamic = None
    
    # Do you want to plot results of preliminary dataset assessment? [boolean]
    plot_dataset_assessment = True
    
else:
    # Print info
    if printinfo:
        print(
            'Model to be selected automatically among the listed ones',
            sep = os.linesep
        )
    
    '''NOTE ---------------------------------------------------------------
    Available keys for general monitoring:
        ['PCA', 'DPCA', 'KPCA', 'DKPCA', 'SVDD']
    Available keys for quality-relevant monitoring:
        ['PLS', 'DPLS', 'KPLS', 'DKPLS', 'CVA', 'KDE-CVA']
    See 'model_library' variable for info on characteristics
    '''
    
    # What models do you wish to test? [List of strings]
    model_list = []
    
    # Look for selected models in the library
    selected = model_library['model'].isin(model_list)
    
    # Check if quality-relevant monitoring has been requested
    if is_quality_relevant:
        # Raise an error if no model for quality-relavnt monitoring has been listed
        if not np.any(model_library.loc[selected, 'is_quality_relevant']):
            raise Exception('No quality-relevant model selected\n'
                + textwrap.fill(textwrap.dedent('''\
                    Quality-relevant monitoring requested, but no adequate
                    model listed.\
                '''))
            )
    
    # Check compatibility of listed models
    is_static = np.any(~model_library.loc[selected, 'is_dynamic'])
    is_dynamic = np.any(model_library.loc[selected, 'is_dynamic'])
    is_general = np.any(~model_library.loc[selected, 'is_quality_relevant'])
    is_quality_relevant = np.any(model_library.loc[selected, 'is_quality_relevant'])
    # Raise a warning if both static and dynamic models
    if is_static and is_dynamic:
        print(
            '',
            Warning('Warning: Incosistent internal validation scheme\n'
                + textwrap.fill(textwrap.dedent('''\
                    Both static and dynamic models have been requested:
                    comparison might be misleading and/or unreliable
                    due to the different internal validation schemes.\
                '''))
            ),
            sep = os.linesep
        )
    # Raise a warning if both general and quality-relevant monitoring
    if is_general and is_quality_relevant:
        print(
            '',
            Warning('Warning: Incosistent monitoring objective\n'
                + textwrap.fill(textwrap.dedent('''\
                    Models for both general and quality-relevant monitoring
                    have been requested: comparing models with different aims
                    might be misleading, and validation could not be reliable.\
                ''')),
            ),
            sep = os.linesep
        )
    
    # No plot needed
    plot_dataset_assessment = False

# What is the nominal significance level of control limits? [float]
alpha_nominal = 0.01

'''NOTE -------------------------------------------------------------------
Estimateds of control limits for T^2 and Q are required by almost all models.
Available estimators are the following:
    T^2:
        ['chi2_distribution', 'F_distribution', 'KDE_distribution']
    Q:
        ['chi2_distribution', 'Jackson_Mudholkar', 'KDE_distribution']
Note some relevant exceptions
    * SVDD does not rely on the T^2 and Q statistics, and it does not require
      any estimator for control limits, which are built directly in model
      calibration
    * CVA and KDE-CVA use an additional statistic, T_r^2, which control limit
      can be estimated with the following estimators:
        ['chi2_distribution', 'F_distribution', 'KDE_distribution']
Note that estimators are required for all three statistics potentially used.
However, only estimators required by models selected in preliminary data
interrogation are effectively used.
'''

# KDE limits are currently inactive. They are built in directly in KDE-CVA, but no other method can use them as for now.

# Which estimator should be used for for control limits of T^2 statistic? [string]
estimator_Tsq_limit = 'chi2_distribution'

# Which estimator should be used for for control limits of Q statistic? [string]
estimator_Q_limit = 'chi2_distribution'

# Which estimator should be used for for control limits of T_r^2 statistic? [string]
estimator_Trsq_limit = 'chi2_distribution'

'''NOTE -------------------------------------------------------------------
Bear in in mind the following naming for internal validation:
    * cross-validation refers to the internal validation procedure used for
      static models, and is performed by r-repeated k-fold, where data are
      randomly split in k groups at each repetiton, the k-fold validation is
      repeated r times
    * forward-validation refers to the internal validation procedure used for
      dynamic models, and is performed by k-growing-window, where data are
      split in k blocks of contiguous observations
Note that values of number of folds and repetitions are required for both
cross- and forward-validation. However, only values required by the approach
relevant to models selected in preliminary data interrogation are effectively
used.
'''

# How many folds should be used for cross-validation (static models)? [int]
CV_folds = 5
# How many folds should be performed for cross-validation (static models)? [int]
CV_repetitions = 10
# How many folds should be used for forward-validation (dynamic models)? [int]
FV_folds = 5

'''NOTE -------------------------------------------------------------------
Hyperparameters can be determined by interna validation (based on the training
dataset) or provided as an external file; in the latter case, internal
validation is skipped, resulting in a significant reduction in the
computational workload for model calibration. Hyperparameters are saved using
the pickle packages in files named:
    hyper_params_summary_<casename>_<model>.pkl
where <casename> is extracted from train_data_name as:
    <path>/<casename>.ext
where path may or may not be specified, while <model> is the name of the model
to which hyperparameters are referred. Note that the file name are generated
automatically and, if no hyperparameter file matching the above format is
found, internal validation is used to produce such a file.
'''

# Do you already have hyperparameters saved? [boolean]
have_hyperparameters = True

# Do you want to plot monitoring statistics in model training? [boolean]
plot_training_results = True

# Check if testing data are provided
if have_testing_data:
    # Do you want to plot monitoring statistics in model testing? [boolean]
    plot_testing_results = True
else:
    # No plot of validation results
    plot_testing_results = False

# Print info
if printinfo:
    print(
        '',
        'Done',
        '#' + 60*'-',
        sep = os.linesep,
        end = 2*'\n'
    )

# Summary of settings ---------------------------------------------------------

# Print info
if printinfo:
    # Header
    print(
        '#' + 60*'-',
        'Summary of settings',
        sep = os.linesep,
        end = 2*'\n'
    )
    # Training data
    print(
        'Training data file:',
        f'\t{train_data_name:s}',
        sep = os.linesep,
        end = '\n'
    )
    # Testing data
    if have_testing_data:
        print(
            'Testing data file:',
            f'\t{test_data_name:s}',
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'Testing data not provided',
            sep = os.linesep,
            end = '\n'
        )
        
    # Headers in data files
    if header_found:
        print(
            'Labels of variables extracted by training data file',
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'Labels of variables generated automatically',
            sep = os.linesep,
            end = '\n'
        )
    # Index in data files
    if index_found:
        print(
            'Labels of obervations extracted by training data file',
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'Labels of observations generated automatically',
            sep = os.linesep,
            end = 2*'\n'
        )
    
    # Quality relevant monitoring
    if is_quality_relevant:
        print(
            'Indices of quality variables:',
            '\t' + str(quality_variable_index),
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'No variables designated as quality',
            sep = os.linesep,
            end = '\n'
        )
    # Use only a subset of variables
    if use_subset:
        print(
            'Indices of process variables:',
            '\t' + str(process_variable_index),
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'All other variables designated as process',
            sep = os.linesep,
            end = '\n'
        )
    # Categorical variables
    if categorical_found:
        print(
            'Indices of discrete/non-numerical variables:',
            '\t' + str(categorical_index),
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'All variables are continuous and numerical',
            sep = os.linesep,
            end = 2*'\n'
        )
    
    # Preliminary data interrogation
    if dataset_assessment:
        print(
            'Candidate models selected according to data characteristics',
            sep = os.linesep,
            end = '\n'
        )
        # Non-normality
        if is_non_normal is None:
            print(
                'Non-normality to be tested on data',
                sep = os.linesep,
                end = '\n'
            )
        else:
            if is_non_normal:
                print(
                    'Distribution of data is assumed to be non-normal',
                    sep = os.linesep,
                    end = '\n'
                )
            else:
                print(
                    'Distribution of data is assumed to be normal',
                    sep = os.linesep,
                    end = '\n'
                )
        # Nonlinearity
        if is_nonlinear is None:
            print(
                'Nonlinearity to be tested on data',
                sep = os.linesep,
                end = '\n'
            )
        else:
            if is_nonlinear:
                print(
                    'Data are assumed to be nonlinear',
                    sep = os.linesep,
                    end = '\n'
                )
            else:
                print(
                    'Data are assumed to be linear',
                    sep = os.linesep,
                    end = '\n'
                )
        # Dynamics
        if is_dynamic is None:
            print(
                'Dynamics to be tested on data',
                sep = os.linesep,
                end = 2*'\n'
            )
        else:
            if is_dynamic:
                print(
                    'Data are assumed to be dynamic',
                    sep = os.linesep,
                    end = 2*'\n'
                )
            else:
                print(
                    'Data are assumed to be static',
                    sep = os.linesep,
                    end = 2*'\n'
                )
    else:
        print(
            'Candidate models listed by user:',
            '\t' + ', '.join(model_list),
            sep = os.linesep,
            end = 2*'\n'
        )
    
    # Significance level of control limits
    print(
        'Significance level for control limits:',
        f'\t{alpha_nominal:1.4f}',
        sep = os.linesep,
        end = '\n'
    )
    # Estimator for T^2 limit
    print(
        'Estimator for T^2 control limit:',
        f'\t{estimator_Tsq_limit:s}',
        sep = os.linesep,
        end = '\n'
    )
    # Estimator for Q limit
    print(
        'Estimator for Q control limit:',
        f'\t{estimator_Q_limit:s}',
        sep = os.linesep,
        end = '\n'
    )
    # Estimator for T_r^2 limit
    print(
        'Estimator for Tr^2 control limit:',
        f'\t{estimator_Trsq_limit:s}',
        sep = os.linesep,
        end = 2*'\n'
    )
    
    # Hyperparameters file provided
    if have_hyperparameters:
        print(
            'Hypeparameter files provided',
            sep = os.linesep,
            end = 2*'\n'
        )
    else:
        # Internal validation
        print(
            'Internal validation is used (on training data)',
            sep = os.linesep,
            end = '\n'
        )
        # Number of folds for cross-validation
        print(
            'Number of folds for cross-validation:',
            f'\t{CV_folds:d}',
            sep = os.linesep,
            end = '\n'
        )
        # Number of repetitions for cross-validation
        print(
            'Number of repetitions for cross-validation:',
            f'\t{CV_repetitions:d}',
            sep = os.linesep,
            end = '\n'
        )
        # Number of folds for forward-validation
        print(
            'Number of folds for forward-validation:',
            f'\t{FV_folds:d}',
            sep = os.linesep,
            end = 2*'\n'
    )
    
    # Plots
    if not any([plot_variable_roles, plot_dataset_assessment, plot_training_results, plot_testing_results]):
        print(
            'No plot required',
            sep = os.linesep,
            end = '\n'
        )
    else:
        print(
            'Plots will be produced for:',
            sep = os.linesep,
            end = '\n'
        )
        if plot_variable_roles:
            print(
                '\tDesignation of variable roles',
                sep = os.linesep,
                end = '\n'
            )
        if plot_dataset_assessment:
            print(
                '\tResults of preliminary data interrogation',
                sep = os.linesep,
                end = '\n'
            )
        if plot_training_results:
            print(
                '\tMonitoring statistics on training data',
                sep = os.linesep,
                end = '\n'
            )
        if plot_testing_results:
            print(
                '\tMonitoring statistics on testing data',
                sep = os.linesep,
                end = '\n'
            )
    
    # Footer
    print(
        '#' + 60*'-',
        sep = os.linesep,
        end = 2*'\n'
    )

#%% Data import and pre-processing

# Print info
if printinfo:
    print(
        '#' + 60*'-',
        'Importing data',
        sep = os.linesep,
        end = 2*'\n'
    )

# Import data -----------------------------------------------------------------

# Selector for training data
if train_data_name[-4:] == '.txt':
    # Import training data from text file
    data_train = pd.read_table(train_data_name, header = header, index_col = index)
if train_data_name[-4:] == '.csv':
    # Import training data from csv file
    data_train = pd.read_csv(train_data_name, header = header, index_col = index)
elif train_data_name[-5:] == '.xlsx':
    # Import training data from Excel file
    data_train = pd.read_excel(train_data_name, header = header, index_col = index, engine = 'openpyxl')
else:
    # File type not supported, raise exception
    raise Exception('Unrecognized file format\n'
        + textwrap.fill(textwrap.dedent('''\
            Format of data file not supported: please convert data file to
            '.txt', '.csv', or '.xlsx'.\
        '''))
    )
# Print info
if printinfo:
    print(
        'Training data imported from ' + train_data_name,
        sep = os.linesep
    )

# Check if testing data are provided
if have_testing_data:
    # Selector for testing data
    if test_data_name[-4:] == '.txt':
        # Import testing data from text file
        data_test = pd.read_table(test_data_name, header = header, index_col = index)
    if test_data_name[-4:] == '.csv':
        # Import testing data from csv file
        data_test = pd.read_csv(test_data_name, header = header, index_col = index)
    elif test_data_name[-5:] == '.xlsx':
        # Import testing data from Excel file
        data_test = pd.read_excel(test_data_name, header = header, index_col = index, engine = 'openpyxl')
    else:
        # File type not supported, raise exception
        raise Exception('Unrecognized file format\n'
            + textwrap.fill(textwrap.dedent('''\
                Format of data file not supported: please convert data file to
                '.txt', '.csv', or '.xlsx'.\
            '''))
        )
    # Print info
    if printinfo:
        print(
            'Testing data imported from ' + test_data_name,
            sep = os.linesep
        )
else:
    # Empty testing data
    data_test = None
    # Print info
    if printinfo:
        print(
            'No testing data provided',
            sep = os.linesep
        )

# Assign roles to variables ---------------------------------------------------

# Print info
if printinfo:
    print(
        'Verifying roles of variables',
        sep = os.linesep
    )

# Number of variables in raw dataset
num_vars = data_train.shape[1]
# Check if there is header in data
if header_found:
    # Import names of variables
    var_names = data_train.columns.tolist()
else:
    # Generate standard names of variables
    var_names = ['V' + str(v).zfill(len(str(num_vars))) for v in range(num_vars)]

# Check if quality-relevant monitoring is required
if is_quality_relevant:
    # Check if only a subset of variables it ot be used as inputs
    if use_subset:
        # Look for overlaps between process and quality variables
        overlaps = [v in process_variable_index for v in quality_variable_index]
        # Remove overlaps
        if any(overlaps):
            # Raise warning
            print(
                '',
                Warning('Warning: process/quality variables overlap\n'
                    + textwrap.fill(textwrap.dedent('''\
                        Some of the quality variables found also in process
                        data: overlapping variables will be removed from
                        process data.\
                    ''')),
                ),
                sep = os.linesep
            )
            # Remove overlaps
            process_variable_index = [v for v in process_variable_index if v not in quality_variable_index]
    else:
        # All variables as inputs, besides quality ones
        process_variable_index = [v for v in range(num_vars) if v not in quality_variable_index]
else:
    # Check if only a subset of variables it ot be used as inputs
    if not use_subset:
        # All variables as inputs
        process_variable_index = [*range(num_vars)]

# Make table for variable roles
var_roles = pd.DataFrame(
    data = np.zeros((num_vars, 4), dtype = bool),
    columns = ['is_input', 'is_output', 'is_cat', 'is_unused']
)
# Assign roles of variables
var_roles.loc[process_variable_index, 'is_input'] = True
if is_quality_relevant:
    var_roles.loc[quality_variable_index, 'is_output'] = True
if categorical_found:
    var_roles.loc[categorical_index, 'is_cat'] = True
var_roles.loc[np.logical_not(np.logical_or(var_roles['is_input'], var_roles['is_output'])), 'is_unused'] = True

# Check if plot of variable roles is requested
if plot_variable_roles:
    
    # Settings for plot
    
    # Figure size
    figsize = (9, 3)
    # Resolution of figure
    figdpi = 330
    # Font size
    fontsize = 10
    # Colour for positive result
    color_pos = (0.65, 0.0, 0.15)
    # Colour for negative result
    color_neg = (0.2, 0.2, 0.6)
    
    # Discrete colormap
    cmap_d = mpl.colors.ListedColormap([color_neg, color_pos])
    
    # Figure
    fig, ax = plt.subplot_mosaic(
        [
          ['VR', 'cb'],
        ],
        gridspec_kw = dict(
            width_ratios = [1, 0.03]
        ),
        figsize = figsize,
        dpi = figdpi)
    
    # Variable properties plot
    g = sns.heatmap(
        ax = ax['VR'],
        data = var_roles.T,
        xticklabels = 'auto',
        yticklabels = 'auto',
        square = False,
        cmap = cmap_d,
        vmin = 0, vmax = 1,
        cbar_ax = ax['cb'],
        cbar_kws = {
            'label': None,
            'ticks': [0, 1],
        },
        linewidths = 0.2,
        linecolor = 'gray',
        annot = False,
        annot_kws = {'size': fontsize}
    )
    # Fix colorbar ticks and label
    cbar = g.collections[0].colorbar
    cbar.set_ticks([0.25, 0.75], labels = ['No', 'Yes'])
    cbar.ax.tick_params(labelsize = fontsize)
    # Fix ticks and labels
    ticklabels = [var_names[int(i.get_text())] for i in g.get_xticklabels()]
    g.set_xticklabels(ticklabels, rotation = 90, size = fontsize)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, size = fontsize)
    
    # Global title
    fig.suptitle('Designation of variables', fontsize = fontsize*1.3, fontweight = 'bold')
    # Fix layout
    fig.set_tight_layout(True)
    

# Adjust indices of categorical variables
cat_vars = np.where(var_roles.loc[var_roles['is_input'], 'is_cat'])[0].tolist()
# Adjust number of variables
num_vars = len(process_variable_index)
# Select names of variables
var_names = [vn for v, vn in enumerate(var_names) if var_roles.loc[v, 'is_input']]
# Adjust indices of categorical variables
cat_vars = np.where(var_roles.loc[var_roles['is_input'], 'is_cat'])[0].tolist()

# Make data matrices ----------------------------------------------------------

# Print info
if printinfo:
    print(
        'Arranging data matrices',
        sep = os.linesep
    )

# Get process variables in training data
X_train = data_train.loc[:, var_roles['is_input'].to_list()].to_numpy(copy = True)
if have_testing_data:
    # Get process variables in testing data
    X_test = data_test.loc[:, var_roles['is_input'].to_list() + [False]].to_numpy(copy = True)
    # Get flag for faulty observations is testing data
    is_fault_test = data_test.loc[:, data_test.columns[[- 1]]].to_numpy(copy = True)
else:
    # Empty testing data
    X_test = None
    is_fault_test = None

# Check if quality relevant monitoring
if is_quality_relevant:
    # Get quality variables in training data
    Y_train = data_train.loc[:, var_roles['is_output'].to_list()].to_numpy(copy = True)
    # Check if testing data are provided
    if have_testing_data:
        # Get quality variables in testing data
        Y_test = data_test.loc[:, var_roles['is_output'].to_list() + [False]].to_numpy(copy = True)
    else:
        # Empty testing data
        Y_test = None
else:
    # No quality data
    Y_train = None
    Y_test = None

# Data pre-processing ---------------------------------------------------------

# Print info
if printinfo:
    print(
        'Pre-processing data',
        sep = os.linesep
    )

# Fit autoscaler to training data
preprocessor_X = StandardScaler(with_mean = True, with_std = True).fit(X_train)
# Pre-process training data
X_train_s = preprocessor_X.transform(X_train)
# Check if testing data are provided
if have_testing_data:
    # Pre-process testing data
    X_test_s = preprocessor_X.transform(X_test)

# Check if quality relevant monitoring
if is_quality_relevant:
    # Fit autoscaler to training data
    preprocessor_Y = StandardScaler(with_mean = True, with_std = True).fit(Y_train)
    # Pre-process training data
    Y_train_s = preprocessor_Y.transform(Y_train)
    # Check if testing data are provided
    if have_testing_data:
        # Pre-process testing data
        Y_test_s = preprocessor_Y.transform(Y_test)

# Print info
if printinfo:
    print(
        '',
        'Done',
        '#' + 60*'-',
        sep = os.linesep,
        end = 2*'\n'
    )

#%% Preliminary data interrogation

# Degrees of freedom of criteria ----------------------------------------------

'''NOTE -----------------------------------------------------------------------
The following parameters can be freely tuned. They represent degrees of freedom
of criteria used in the preliminary data interrogation.
'''

# Significance level for multivariate normality test
alpha_nn_test = 0.01

# Fraction of variables involved in significant nonlinear relationsips to deem data as nonlinear
frac_nl_test = 0.1

# Fraction of variables featuring significant dynamics to deem data as dynamic
frac_dyn_test = 0.1
# Nominal significance level for autocorrelation coefficients
alpha_acf_nominal = 0.01

# Expert settings of criteria -------------------------------------------------

'''NOTE -----------------------------------------------------------------------
The following parameters have been optimized during the development of SPAfPM.
They still can be manipulated, but their meaning is highly techical. Modifying
these parameters is suggested only to expert users. See docstrings of functions
'non_normality_test', 'nonlinearity_test' and 'dynamics_test' in the module
'dataset_property_assessment.py' for details on parameters and allowed values.
'''

# Force symmetry of maximal correlation coefficient matrix
force_sym_mc = True
# Tolerance on difference MC - abs(LC) if MC <= thr_hmc
tol_dif_mc_lc = 0.4
# Threshold of maximal correlation coefficient for tolerance refinement
thr_hmc = 0.92
# Tolerance on difference MC - abs(LC) if MC > thr_hmc
tol_dif_mc_lc_hmc = 0.1
# Approach to deflation of maximal correlation coefficient if MC <= thr_hmc
deflate_mc = 'median' # ['none', 'median', 'upper', 'bootstrap']
# Significance level for deflation of MC based on bootstrap confidence limits
alpha_boot_cl_mc = 0.01
# Number of bootstrap repetitions for confidence limits of MC
N_boot_cl_mc = 1000
# Estimator of bootstrap confidence limits of MC
method_boot_cl_mc = 'bias_corr_acc' # ['bias_corr_acc', 'quantile']
# Random number generator intance or seed for bootstrap resampling
random_state_boot = None
# Nominal significance level for quadratic test
alpha_qt_nominal = 0.01
# Bonferroni coerrection for significance level of quadratic test
correction_qt = True

# Number of lags to be assessed in autocorrelation fucntion (automatically selected if None)
nlags_acf = None
# Approach to assess significance of autocorrelation coefficients
approach_acf = 'ljung_box' # ['ljung_box', 'conf_lim', 'p_value']
# Variance estimator for distribution of autocorrelation coefficients
variance_acf = 'bartlett' # ['bartlett', 'large_sample', 'lag_corr']
# Bonferroni coerrection for significance level of autocorrelation test
correction_acf = True
# Major lag constraint for significance of autocorrelation coefficients
major_acf = False

# Property assessment ---------------------------------------------------------

# Check if data interrogation is to be performed
if dataset_assessment:
    # Print info
    if printinfo:
        print(
            '#' + 60*'-',
            'Performing preliminary data interrogation',
            sep = os.linesep,
            end = 2*'\n'
        )
    
    # Check if non-normality assessment is required
    if is_non_normal is None:
        # Perform non-normality assessment
        is_non_normal, p_value, map_nn = dpa.non_normality_test(X_train_s,
            alpha = alpha_nn_test
        )
        # Print info
        if printinfo:
            if is_non_normal:
                print(
                    'Distribution of data is non-normal',
                    f'\tp-value = {p_value:1.4f}',
                    f'\tthreshold = {alpha_nn_test:1.4f}',
                    sep = os.linesep
                )
            else:
                print(
                    'Distribution of data is normal',
                    f'\tp-value = {p_value:1.4f}',
                    f'\tthreshold = {alpha_nn_test:1.4f}',
                    sep = os.linesep
                )
    else:
        # Empty results
        p_value = None
        map_nn = np.full(num_vars, np.nan)
        # Print info
        if printinfo:
            if is_non_normal:
                print(
                    'Distribution of data is assumed to be non-normal',
                    sep = os.linesep
                )
            else:
                print(
                    'Distribution of data is assumed to be normal',
                    sep = os.linesep
                )
    
    # Check if nonlinearity assessment is required
    if is_nonlinear is None:
        # Perform nonlinearity assessment
        is_nonlinear, frac_nl_vars, map_nl = dpa.nonlinearity_test(X_train_s,
            frac_nl = frac_nl_test,
            cat = cat_vars,
            force_sym = force_sym_mc,
            tol_dif = tol_dif_mc_lc,
            thr_hmc = thr_hmc,
            tol_dif_hmc = tol_dif_mc_lc_hmc,
            deflate_MC = deflate_mc,
            alpha_boot = alpha_boot_cl_mc,
            N_boot = N_boot_cl_mc,
            CL_method_boot = method_boot_cl_mc,
            random_state = random_state_boot,
            alpha_qt = alpha_qt_nominal,
            correction = correction_qt
        )
        # Print info
        if printinfo:
            if is_nonlinear:
                print(
                    'Data feature significant nonlinearity',
                    f'\tfrac_nl_vars = {frac_nl_vars:1.4f}',
                    f'\tthreshold = {frac_nl_test:1.4f}',
                    sep = os.linesep
                )
            else:
                print(
                    'Data do not feature significant nonlinearity',
                    f'\tfrac_nl_vars = {frac_nl_vars:1.4f}',
                    f'\tthreshold = {frac_nl_test:1.4f}',
                    sep = os.linesep
                )
    else:
        # Empty results
        frac_nl_vars = None
        map_nl = np.full(num_vars, np.nan)
        # Print info
        if printinfo:
            if is_nonlinear:
                print(
                    'Data are assumed to be nonlinear',
                    sep = os.linesep
                )
            else:
                print(
                    'Data are assumed to be linear',
                    sep = os.linesep
                )
    
    # Check if dynamics assessment is required
    if is_dynamic is None:
        # Perform dynamics assessment
        is_dynamic, frac_dyn_vars, map_dyn = dpa.dynamics_test(X_train_s,
            frac_dyn = frac_dyn_test,
            alpha = alpha_acf_nominal,
            cat = cat_vars,
            nlags = nlags_acf,
            approach = approach_acf,
            variance = variance_acf,
            correction = correction_acf,
            major = major_acf
        )
        # Print info
        if printinfo:
            if is_dynamic:
                print(
                    'Data feature significant dynamics',
                    f'\tfrac_dyn_vars = {frac_dyn_vars:1.4f}',
                    f'\tthreshold = {frac_dyn_test:1.4f}',
                    sep = os.linesep
                )
            else:
                print(
                    'Data do not feature significant dynamics',
                    f'\tfrac_dyn_vars = {frac_dyn_vars:1.4f}',
                    f'\tthreshold = {frac_dyn_test:1.4f}',
                    sep = os.linesep
                )
    else:
        # Empty results
        frac_dyn_vars = None
        map_dyn = np.full(num_vars, np.nan)
        # Print info
        if printinfo:
            if is_dynamic:
                print(
                    'Data are assumed to be dynamic',
                    sep = os.linesep
                )
            else:
                print(
                    'Data are assumed to be static',
                    sep = os.linesep
                )

    # Plots of outcomes of preliminary data interrogation ---------------------

    # Check if plot is requested
    if plot_dataset_assessment:
        # Print info
        if printinfo:
            print(
                '',
                'Plotting outcomes of preliminary data interrogation',
                sep = os.linesep
            )
        
        # Settings of the plot
        
        # Figure size
        figsize = (9, 6)
        # Resolution of figure
        figdpi = 300
        # Font size
        fontsize = 10
        # Colour for positive result
        color_pos = (0.65, 0.0, 0.15)
        # Colour for negative result
        color_neg = (0.2, 0.2, 0.6)
        # Lines to mark discrete/non-numerical variables variables
        catlines = True
        
        # Discrete colormap
        cmap_d = mpl.colors.ListedColormap([color_neg, color_pos])
        # Array of indices of discrete/non-numerical variables
        cat_array = np.array(cat_vars)
        
        # Mask for plot
        mask = np.zeros((3, num_vars), dtype = bool)
        if p_value is None:
            mask[0, :] = True
        if frac_nl_vars is None:
            mask[1, :] = True
        if frac_dyn_vars is None:
            mask[2, :] = True
        else:
            if cat_vars is not None:
                mask[2, cat_vars] = True

        # Figure
        fig, ax = plt.subplot_mosaic(
            [
              ['NN', 'NL', 'DYN'],
              ['P', 'P', 'P'],
              ['Pcb', 'Pcb', 'Pcb']
            ],
            gridspec_kw = dict(
                height_ratios = [1, 0.5, 0.1]
            ),
            figsize = figsize,
            dpi = figdpi)

        # Non-normality plot
        # Check if performed non-normality test
        if p_value is not None:
            # Exponent of lower bound of y axis
            minexp = int(np.floor(np.log10(np.minimum(p_value, alpha_nn_test))))
            if 10**minexp == alpha_nn_test:
                minexp -= 1
            # Is dataset non-normal?
            if is_non_normal:
                # Color
                nn_bar_color = color_pos
                # Text
                nn_text = 'Data deemed as non-normal'
            else:
                # Color
                nn_bar_color = color_neg
                # Text
                nn_text = 'Data deemed as normal'
            # Plot
            ax['NN'].fill([-0.4, -0.4, 0.4, 0.4], [1, p_value, p_value, 1], color = nn_bar_color)
            ax['NN'].hlines(alpha_nn_test, -0.5, 0.5, color = 'k', linestyle = '--')
            # ax['NN'].text(0, alpha_norm*0.2, nn_text, size = fontsize, ha = 'center', va = 'top')
            ax['NN'].set_yscale('log')
            ax['NN'].set_xlim([-0.5, 0.5])
            ax['NN'].set_ylim([10**minexp, 1e0])
        else:
            # Just set bounds of axes
            ax['NN'].set_xlim([-0.5, 0.5])
            ax['NN'].set_ylim([0, 1])
            # Is dataset non-normal?
            if is_non_normal:
                # Text
                nn_text = 'Data assumed non-normal'
            else:
                # Text
                nn_text = 'Data assumed normal'
        # Fix ticks and labels
        ax['NN'].set_xticks([0], labels = [nn_text], rotation = 0, size = fontsize)
        ax['NN'].set_ylabel('p_value')
        ax['NN'].set_title('Non-normality p-value', size = fontsize)

        # Nonlinearity plot
        # Check if performed nonlinearity test
        if frac_nl_vars is not None:
            # Is dataset nonlinear?
            if is_nonlinear:
                # Color
                nl_bar_color = color_pos
                # Text
                nl_text = 'Data deemed as nonlinear'
            else:
                # Color
                nl_bar_color = color_neg
                # Text
                nl_text = 'Data deemed as linear'
            # Plot
            ax['NL'].bar(0, frac_nl_vars, width = 0.8, color = nl_bar_color)
            ax['NL'].hlines(frac_nl_test, -0.5, 0.5, color = 'k', linestyle = '--')
            # ax['NL'].text(0, 1.05, nl_text, size = fontsize, ha = 'center', va = 'center')
        else:
            # Is dataset nonlinear?
            if is_nonlinear:
                # Text
                nl_text = 'Data assumed nonlinear'
            else:
                # Text
                nl_text = 'Data assumed linear'
        # Set bounds of axes
        ax['NL'].set_xlim([-0.5, 0.5])
        ax['NL'].set_ylim([0, 1.1])
        # Fix ticks and labels
        ax['NL'].set_xticks([0], labels = [nl_text], rotation = 0, size = fontsize)
        ax['NL'].set_ylabel('nl_frac')
        ax['NL'].set_title('Nonlinearity fraction', size = fontsize)

        # Dynamics plot
        # Check if performed dynamics test
        if frac_dyn_vars is not None:
            # Is dataset dynamics?
            if is_dynamic:
                # Color
                dyn_bar_color = color_pos
                # Text
                dyn_text = 'Data deemed as dynamic'
            else:
                # Color
                dyn_bar_color = color_neg
                # Text
                dyn_text = 'Data deemed as static'
            # Plot
            ax['DYN'].bar(0, frac_dyn_vars, width = 0.8, color = dyn_bar_color)
            ax['DYN'].hlines(frac_dyn_test, -0.5, 0.5, color = 'k', linestyle = '--')
            # ax['DYN'].text(0, 1.05, dyn_text, size = fontsize, ha = 'center', va = 'center')
        else:
            # Is dataset dynamics?
            if is_dynamic:
                # Text
                dyn_text = 'Data assumed dynamic'
            else:
                # Text
                dyn_text = 'Data assumed static'
        # Set bounds of axes
        ax['DYN'].set_xlim([-0.5, 0.5])
        ax['DYN'].set_ylim([0, 1.1])
        # Fix ticks and labels
        ax['DYN'].set_xticks([0], labels = [dyn_text], rotation = 0, size = fontsize)
        ax['DYN'].set_ylabel('dyn_frac')
        ax['DYN'].set_title('Dynamics fraction', size = fontsize)

        # Variable properties plot
        g = sns.heatmap(
            ax = ax['P'],
            data = np.concatenate((map_nn.reshape(-1, 1), map_nl.reshape(-1, 1), map_dyn.reshape(-1, 1)), axis = 1).T,
            mask = mask,
            xticklabels = 'auto',
            yticklabels = 'auto',
            square = False,
            cmap = cmap_d,
            vmin = 0, vmax = 1,
            cbar_ax = ax['Pcb'],
            cbar_kws = {
                'orientation': 'horizontal',
                'label': 'Is non-normal/nonlinear/dynamic?',
                'ticks': [0, 1],
            },
            linewidths = 0.2,
            linecolor = 'gray',
            annot = False,
            annot_kws = {'size': fontsize}
        )
        # Lines over categorical variables
        if cat_vars is not None and catlines:
            ax['P'].vlines(cat_array + 0.5, 0, 3, color = 'gray')
        # Fix colorbar ticks and label
        cbar = g.collections[0].colorbar
        cbar.set_ticks([0.25, 0.75], labels = ['No', 'Yes'])
        cbar.ax.tick_params(labelsize = fontsize)
        cbar.ax.set_xlabel(cbar.ax.get_xlabel(), fontsize = fontsize)
        # Fix ticks and labels
        ticklabels = [var_names[int(i.get_text())] for i in g.get_xticklabels()]
        g.set_xticklabels(ticklabels, rotation = 90, size = fontsize)
        g.set_yticklabels(['NN', 'NL', 'DYN'], rotation = 0, size = fontsize)
        g.set_title('Properties of variables', size = fontsize)
        
        # Global title
        fig.suptitle('Characteristics of data', fontsize = fontsize*1.3, fontweight = 'bold')
        # Fix layout
        fig.set_tight_layout(True)
    
    # Model pre-selection -----------------------------------------------------
    
    # Index for nonlinear models
    idx_nl = (model_library['is_nonlinear'] == is_nonlinear).to_numpy()
    # Index for dynamic models
    idx_dyn = (model_library['is_dynamic'] == is_dynamic).to_numpy()
    # Index for quality-relevant models
    idx_qr = (model_library['is_quality_relevant'] == is_quality_relevant).to_numpy()
    
    # Index of pre-selected models
    idx_comb = np.logical_and.reduce((idx_nl, idx_dyn, idx_qr))
    
    # Select models
    model_list = model_library.loc[idx_comb, 'model'].to_list()
    
    # Print info
    if printinfo:
        print(
            '',
            'Candidate models selected from preliminary data interrogation',
            '\t' + ', '.join(model_list),
            sep = os.linesep
        )
        if len(model_list) > 1:
            print(
                'Final model will be selected based on validation peformance',
                sep = os.linesep
            )
    
    # Print info
    if printinfo:
        print(
            '',
            'Done',
            '#' + 60*'-',
            sep = os.linesep,
            end = 2*'\n'
        )
else:
    # Print info
    if printinfo:
        print(
            '#' + 60*'-',
            'Preliminary data interrogation not requested',
            sep = os.linesep,
            end = 2*'\n'
        )
        print(
            'Candidate models listed by user',
            '\t' + ', '.join(model_list),
            sep = os.linesep
        )
        if len(model_list) > 1:
            print(
                'Final model will be selected based on validation peformance',
                sep = os.linesep
            )
        print(
            '',
            'Done',
            '#' + 60*'-',
            sep = os.linesep,
            end = 2*'\n'
        )

'''**********************************************************************++*'''
'''**********************************************************************++*'''
'''**********************************************************************++*'''
'''**********************************************************************++*'''
'''**********************************************************************++*'''

#%% Model running

# Print info
if printinfo:
    print(
        '#' + 60*'-',
        'Running models',
        sep = os.linesep,
        end = 2*'\n'
    )

Violations_chosen_Summary = np.zeros((len(model_list)))
Type1Fault_tr = np.zeros((len(model_list))) #Detected fault when no fault occured in training
IVI = np.zeros((len(model_list)))
Type1Fault = np.zeros((len(model_list))) #Detected fault when no fault occured in testing
Type2Fault = np.zeros((len(model_list))) #Not detected fault even though it occured in testing

hyper_params_summary = {}
ValidationFolds = {}
idxNoFaultTest = np.where(is_fault_test == 0)[0]
idxFaultTest = np.where(is_fault_test == 1)[0]

idx = [train_data_name.rfind('/'), train_data_name.rfind('.')]
if idx[0] == -1:
    idx[0] = 0;
case_name = train_data_name[idx[0] + 1:idx[1]]

index = 0
for model_index in model_list:
    
    # Print info
    if printinfo:
        print(
            f'Running {model_index:s}',
            sep = os.linesep
        )
    
    # hyper_params_file_name = 'hyper_params_summary_' + case_name + '_' + model_index + '.pkl'
    hyper_params_file_name = 'hyperparameters/hyper_params_summary_' + case_name + '_' + model_index + '.pkl'
    
    if model_index == 'CVA' or model_index == 'KDE-CVA' or model_index == 'DPCA' or model_index == 'DKPCA' or model_index == 'DPLS' or model_index == 'DKPLS':
        TotalFolds = FV_folds
        K_fold = FV_folds
    else:
        TotalFolds = CV_folds*CV_repetitions
        K_fold = CV_folds
    # Run resulting algorithms
    if is_quality_relevant:
        if model_index == 'CVA' or model_index == 'KDE-CVA':
            [
                tSquared,
                trSquared,
                tSquaredTest,
                trSquared,
                trSquaredTest,
                Q,
                QTest,
                hyper_params,
                Violations_result,
                thresholdTsquared,
                thresholdTrsquared,
                thresholdQ,
                Violations_mean,
                Violations_min,
                idx_final
            ] = dam.ApplyAlgorithm(X_train,
                                   X_test,
                                   model_index,
                                   alpha = alpha_nominal,
                                   y = Y_train,
                                   y_test = Y_test,
                                   K_fold = K_fold,
                                   Nr = CV_repetitions,
                                   if_have_output = is_quality_relevant,
                                   Tmethod = estimator_Tsq_limit,
                                   Qmethod = estimator_Q_limit,
                                   if_have_hyper_params = have_hyperparameters,
                                   hyper_params_file_name = hyper_params_file_name,
                                   plot_training = plot_training_results,
                                   plot_testing = plot_testing_results
            )
            if 'lag' in hyper_params:
                idxFaultTest = idxFaultTest[:-hyper_params['lag']]
            Violations_chosen_Summary[index] = Violations_mean.item(tuple(idx_final))
            TViolations1_tr = np.where(np.any(tSquared > thresholdTsquared, axis = 1))[0]
            TrViolations1_tr = np.where(np.any(trSquared > thresholdTrsquared, axis = 1))[0]
            QViolations1_tr = np.where(np.any(Q > thresholdQ, axis = 1))[0]
            Type1Fault_tr[index] = len(np.unique(np.concatenate((TViolations1_tr,TrViolations1_tr,QViolations1_tr))))/np.shape(Q)[0]
            IVI[index] = np.abs(Type1Fault_tr[index] - alpha_nominal)
            # Print info
            if printinfo:
                print(
                    f'\tInternal validation performance measure = {Violations_chosen_Summary[index]:1.4f}',
                    f'\tTraining: type I error rate =  {Type1Fault_tr[index]:1.4f}',
                    sep = os.linesep
                )
            if have_testing_data:
                TViolations1 = np.where(np.any(tSquaredTest[idxNoFaultTest] > thresholdTsquared, axis = 1))[0]
                TrViolations1 = np.where(np.any(trSquaredTest[idxNoFaultTest] > thresholdTrsquared, axis = 1))[0]
                QViolations1 = np.where(np.any(QTest[idxNoFaultTest] > thresholdQ, axis = 1))[0]
                Type1Fault[index] = len(np.unique(np.concatenate((TViolations1,TrViolations1,QViolations1))))/np.shape(QTest[idxNoFaultTest])[0]
                TViolations2 = np.where(np.any(tSquaredTest[idxFaultTest] > thresholdTsquared, axis = 1))[0]
                TrViolations2 = np.where(np.any(trSquaredTest[idxFaultTest] > thresholdTrsquared, axis = 1))[0]
                QViolations2 = np.where(np.any(QTest[idxFaultTest] > thresholdQ, axis = 1))[0]
                Type2Fault[index] = 1 - len(np.unique(np.concatenate((TViolations2,TrViolations2,QViolations2))))/np.shape(QTest[idxFaultTest])[0]
                if printinfo:
                    print(
                        f'\tTesting:  type I error rate =  {Type1Fault[index]:1.4f}',
                        f'\tTesting:  type II error rate = {Type2Fault[index]:1.4f}',
                        sep = os.linesep
                    )
            index += 1
        else:
            [
                tSquared,
                tSquaredTest,
                Q,
                QTest,
                hyper_params,
                Violations_result,
                thresholdTsquared,
                thresholdQ,
                Violations_mean,
                Violations_min,
                idx_final,
                T,
                T1
            ] = dam.ApplyAlgorithm(X_train,
                                   X_test,
                                   model_index,
                                   alpha = alpha_nominal,
                                   y = Y_train,
                                   y_test = Y_test,
                                   K_fold = K_fold,
                                   Nr = CV_repetitions,
                                   if_have_output = is_quality_relevant,
                                   Tmethod = estimator_Tsq_limit,
                                   Qmethod = estimator_Q_limit,
                                   if_have_hyper_params = have_hyperparameters,
                                   hyper_params_file_name = hyper_params_file_name,
                                   plot_training = plot_training_results,
                                   plot_testing = plot_testing_results
            )
            if 'lag' in hyper_params:
                idxFaultTest = idxFaultTest[:-hyper_params['lag']]
            Violations_chosen_Summary[index] = Violations_mean.item(tuple(idx_final))
            TViolations1_tr = np.where(np.any(tSquared > thresholdTsquared, axis = 1))[0]
            QViolations1_tr = np.where(np.any(Q > thresholdQ, axis = 1))[0]
            Type1Fault_tr[index] = len(np.unique(np.concatenate((TViolations1_tr,QViolations1_tr))))/np.shape(Q)[0]
            IVI[index] = np.abs(Type1Fault_tr[index] - alpha_nominal)
            # Print info
            if printinfo:
                print(
                    f'\tInternal validation performance measure = {Violations_chosen_Summary[index]:1.4f}',
                    f'\tTraining: type I error rate =  {Type1Fault_tr[index]:1.4f}',
                    sep = os.linesep
                )
            if have_testing_data:
                TViolations1 = np.where(np.any(tSquaredTest[idxNoFaultTest] > thresholdTsquared, axis = 1))[0]
                QViolations1 = np.where(np.any(QTest[idxNoFaultTest] > thresholdQ, axis = 1))[0]
                Type1Fault[index] = len(np.unique(np.concatenate((TViolations1,QViolations1))))/np.shape(QTest[idxNoFaultTest])[0]
                TViolations2 = np.where(np.any(tSquaredTest[idxFaultTest] > thresholdTsquared, axis = 1))[0]
                QViolations2 = np.where(np.any(QTest[idxFaultTest] > thresholdQ, axis = 1))[0]
                Type2Fault[index] = 1 - len(np.unique(np.concatenate((TViolations2,QViolations2))))/np.shape(QTest[idxFaultTest])[0]
                if printinfo:
                    print(
                        f'\tTesting:  type I error rate =  {Type1Fault[index]:1.4f}',
                        f'\tTesting:  type II error rate = {Type2Fault[index]:1.4f}',
                        sep = os.linesep
                    )
            index += 1
    else:
        if model_index == 'SVDD':
            [
                distance,
                distance_test,
                hyper_params,
                Violations_result,
                radius,
                Violations_mean,
                Violations_min,
                idx_final
            ] = dam.ApplyAlgorithm(X_train,
                                   X_test,
                                   model_index,
                                   alpha = alpha_nominal,
                                   K_fold = K_fold,
                                   Nr = CV_repetitions,
                                   if_have_output = is_quality_relevant,
                                   if_have_hyper_params = have_hyperparameters,
                                   hyper_params_file_name = hyper_params_file_name,
                                   plot_training = plot_training_results,
                                   plot_testing = plot_testing_results
            )
            if 'lag' in hyper_params:
                idxFaultTest = idxFaultTest[:-hyper_params['lag']]
            Violations_chosen_Summary[index] = Violations_mean.item(tuple(idx_final))
            Type1Fault_tr[index] = len(np.where(np.any(distance > radius, axis = 1))[0])/np.shape(distance)[0]
            IVI[index] = np.abs(Type1Fault_tr[index] - alpha_nominal)
            # Print info
            if printinfo:
                print(
                    f'\tInternal validation performance measure = {Violations_chosen_Summary[index]:1.4f}',
                    f'\tTraining: type I error rate =  {Type1Fault_tr[index]:1.4f}',
                    sep = os.linesep
                )
            if have_testing_data:
                Type1Fault[index] = len(np.where(np.any(distance_test[idxNoFaultTest] > radius, axis = 1))[0])/np.shape(distance_test[idxNoFaultTest])[0]
                Type2Fault[index] = 1 - len(np.where(np.any(distance_test[idxFaultTest] > radius, axis = 1))[0])/np.shape(distance_test[idxFaultTest])[0]
                if printinfo:
                    print(
                        f'\tTesting:  type I error rate =  {Type1Fault[index]:1.4f}',
                        f'\tTesting:  type II error rate = {Type2Fault[index]:1.4f}',
                        sep = os.linesep
                    )
            index += 1
        else:
            [
                tSquared,
                tSquaredTest,
                Q,
                QTest,
                hyper_params,
                Violations_result,
                thresholdTsquared,
                thresholdQ,
                Violations_mean,
                Violations_min,
                idx_final,
                Violations_std,
                Violations_bar
            ] = dam.ApplyAlgorithm(X_train,
                                   X_test,
                                   model_index,
                                   alpha = alpha_nominal,
                                   K_fold = K_fold,
                                   Nr = CV_repetitions,
                                   if_have_output = is_quality_relevant,
                                   Tmethod = estimator_Tsq_limit,
                                   Qmethod = estimator_Q_limit,
                                   if_have_hyper_params = have_hyperparameters,
                                   hyper_params_file_name = hyper_params_file_name,
                                   plot_training = plot_training_results,
                                   plot_testing = plot_testing_results
            )
            if 'lag' in hyper_params:
                idxFaultTest = idxFaultTest[:-hyper_params['lag']]
            Violations_chosen_Summary[index] = Violations_mean.item(tuple(idx_final))
            TViolations1_tr = np.where(np.any(tSquared > thresholdTsquared, axis = 1))[0]
            QViolations1_tr = np.where(np.any(Q > thresholdQ, axis = 1))[0]
            Type1Fault_tr[index] = len(np.unique(np.concatenate((TViolations1_tr,QViolations1_tr))))/np.shape(Q)[0]
            IVI[index] = np.abs(Type1Fault_tr[index] - alpha_nominal)
            # Print info
            if printinfo:
                print(
                    f'\tInternal validation performance measure = {Violations_chosen_Summary[index]:1.4f}',
                    f'\tTraining: type I error rate =  {Type1Fault_tr[index]:1.4f}',
                    sep = os.linesep
                )
            if have_testing_data:
                TViolations1 = np.where(np.any(tSquaredTest[idxNoFaultTest] > thresholdTsquared, axis = 1))[0]
                QViolations1 = np.where(np.any(QTest[idxNoFaultTest] > thresholdQ, axis = 1))[0]
                Type1Fault[index] = len(np.unique(np.concatenate((TViolations1,QViolations1))))/np.shape(QTest[idxNoFaultTest])[0]
                TViolations2 = np.where(np.any(tSquaredTest[idxFaultTest] > thresholdTsquared, axis = 1))[0]
                QViolations2 = np.where(np.any(QTest[idxFaultTest] > thresholdQ, axis = 1))[0]
                Type2Fault[index] = 1 - len(np.unique(np.concatenate((TViolations2,QViolations2))))/np.shape(QTest[idxFaultTest])[0]
                if printinfo:
                    print(
                        f'\tTesting:  type I error rate =  {Type1Fault[index]:1.4f}',
                        f'\tTesting:  type II error rate = {Type2Fault[index]:1.4f}',
                        sep = os.linesep
                    )
            index += 1  
    ValidationSummaryModel = np.zeros((TotalFolds))  
    for i in range(TotalFolds):
        idx_use = deepcopy(idx_final)
        idx_use.append(i)
        ValidationSummaryModel[i] = Violations_result.item(tuple(idx_use))
    ValidationFolds[model_index] = ValidationSummaryModel
    hyper_params_summary[model_index] = hyper_params

if len(model_list) == 1:
    best_index = 0
    multiple_index = False
else:
    best_index = np.argmin(Violations_chosen_Summary)
    multi_best = Violations_chosen_Summary == Violations_chosen_Summary[best_index]
    multiple_index = np.sum(multi_best) > 1

if printinfo:
    if multiple_index:
        print(
            '',
            'The following models show equivalent validation performances',
            '\t' + ', '.join(model_list[i] for i in np.where(multi_best)[0]),
            'Final selection based on type II error rate on an independent',
            'dataset is recommended',
            sep = os.linesep
        )
    else:
        if dataset_assessment:
            print(
                '',
                f'{model_list[best_index]:s} recommended based on dataset properties and validation performance',
                sep = os.linesep
            )
        else:
            print(
                '',
                f'{model_list[best_index]:s} recommended based on validation peformance',
                sep = os.linesep
            )

# Print info
if printinfo:
    print(
        '',
        'Done',
        '#' + 60*'-',
        sep = os.linesep,
        end = 2*'\n'
    )