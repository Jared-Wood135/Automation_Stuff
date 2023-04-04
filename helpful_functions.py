# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. Dataframe Specific
    a. split
    b. sample_dataframe
4. Regression Modeling
    a. regression_prediction
    b. get_eval_stats
    c. select_kbest
    d. rfe
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to provide a litany of hopefully helpful functions to use for data science
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

# Basic imports (Vectorization, Dataframe, Visualizations)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats

# Sklearn specific
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# =======================================================================================================
# Imports END
# Imports TO split
# split START
# =======================================================================================================

def split(df):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets

    INPUT:
    df = Dataframe to be split

    OUTPUT:
    train = Dataframe split for training
    validate = Dataframe split for validation
    test = Dataframe split for testing
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349)
    print(f"train.shape:{train.shape}\nvalidate.shape:{validate.shape}\ntest.shape:{test.shape}")
    return train, validate, test

# =======================================================================================================
# split END
# split TO sample_dataframe
# sample_dataframe START
# =======================================================================================================

def sample_dataframe(train, validate, test):
    '''
    Takes train, validate, test dataframes and reduces the shape to no more than 1000 rows by taking
    the percentage of 1000/len(train) then applying that to train, validate, test dataframes.

    INPUT:
    train = Split dataframe for training
    validate = Split dataframe for validation
    test = Split dataframe for testing

    OUTPUT:
    train_sample = Reduced size of original split dataframe of no more than 1000 rows
    validate_sample = Reduced size of original split dataframe of no more than 1000 rows
    test_sample = Reduced size of original split dataframe of no more than 1000 rows
    '''
    ratio = 1000/len(train)
    train_samples = int(ratio * len(train))
    validate_samples = int(ratio * len(validate))
    test_samples = int(ratio * len(test))
    train_sample = train.sample(train_samples)
    validate_sample = validate.sample(validate_samples)
    test_sample = test.sample(test_samples)
    return train_sample, validate_sample, test_sample

# =======================================================================================================
# sample_dataframe END
# sample_dataframe TO regression_prediction
# regression_prediction START
# =======================================================================================================

def regression_prediction(train, x_cols, y_col):
    '''
    Takes a training dataframe and the x_columns to train the machine off of in order to best predict
    the y_column and adds those prediction values to the train dataframe.

    INPUT: 
    train = Dataframe used for training purposes
    x_cols = LIST of column names to train the machine on
    y_col = LIST of column names for the machine to predict values of

    OUTPUT:
    train_updated = Dataframe updated with the prediction columns added to it
    '''
    train_updated = train
    lm = LinearRegression()
    lm.fit(train[x_cols], train[y_col])
    train_updated['yhat'] = np.round(lm.predict(train[x_cols]))
    return train_updated

# =======================================================================================================
# regression_prediction END
# regression_prediction TO get_eval_stats
# get_eval_stats START
# =======================================================================================================

def get_eval_stats(df, actual_col, baseline_col, prediction_col):
    '''
    Takes in 4 inputs and returns summary evaluation statistics for the baseline, the prediction,
    and the summary evaluation of the difference between the two...

    INPUT VALUES:
    df = Pandas dataframe name
    actual_col = Column name containing actual values
    baseline_col = Column name containing baseline prediction of actual_col
    prediction_col = Column name containing predicted values of actual_col

    OUTPUT VALUES:
    3 Tables with summary statistics for the baseline, prediction, and difference
    (Stats like SSE, ESS, TSS, MSE, RMSE, R2, etc.)
    '''
    baseline = df[actual_col].mean()
    base_residual = df[actual_col] - baseline
    pred_residual = df[actual_col] - df[prediction_col]
    SSE_base = (base_residual ** 2).sum()
    SSE_pred = (pred_residual ** 2).sum()
    SSE_diff = int(SSE_pred - SSE_base)
    ESS_base = sum((df[baseline_col] - df[actual_col]) ** 2)
    ESS_pred = sum((df[prediction_col] - df[actual_col]) ** 2)
    ESS_diff = int(ESS_pred - ESS_base)
    TSS_base = SSE_base + ESS_base
    TSS_pred = SSE_pred + ESS_pred
    TSS_diff = int(TSS_pred - TSS_base)
    MSE_base = SSE_base / len(df)
    MSE_pred = SSE_pred / len(df)
    MSE_diff = int(MSE_pred - MSE_base)
    RMSE_base = MSE_base ** .5
    RMSE_pred = MSE_pred ** .5
    RMSE_diff = int(RMSE_pred - RMSE_base)
    R2_base = 1 - (SSE_base/TSS_base)
    R2_pred = 1 - (SSE_pred/SSE_base)
    print(f'\033[35m===== {baseline_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_base:.2f}\n\033[32mESS:\033[0m {ESS_base:.2f}\n\033[32mTSS:\033[0m {TSS_base:.2f}\n\033[32mMSE:\033[0m {MSE_base:.2f}\n\033[32mRMSE:\033[0m {RMSE_base:.2f}\n')
    print(f'\033[35m===== {prediction_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_pred:.2f}\n\033[32mESS:\033[0m {ESS_pred:.2f}\n\033[32mTSS:\033[0m {TSS_pred:.2f}\n\033[32mMSE:\033[0m {MSE_pred:.2f}\n\033[32mRMSE:\033[0m {RMSE_pred:.2f}\n\033[32mR2:\033[0m {R2_pred:.2f}\n')
    print(f'\033[35m===== {prediction_col} - {baseline_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_diff:.2f}\n\033[32mESS:\033[0m {ESS_diff:.2f}\n\033[32mTSS:\033[0m {TSS_diff:.2f}\n\033[32mMSE:\033[0m {MSE_diff:.2f}\n\033[32mRMSE:\033[0m {RMSE_diff:.2f}\n')

# =======================================================================================================
# get_eval_stats END
# get_eval_stats TO select_kbest
# select_kbest START
# =======================================================================================================

def select_kbest(predictors, target, k_features):
    '''
    Takes in a predictors and target dataframes as well as how many features (column names) you want
    to be selected.

    INPUT:
    predictors = Dataframe with ONLY the predictor columns and their respective values. MUST BE INT, FLOAT.
    target = Dataframe with ONLY the target column and their respective values.
    k_features = The number of top performing features you want to be returned.

    OUTPUT:
    top = A list of column names that are the top performing features.
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    _ = kbest.fit(predictors, target)
    top = predictors.columns[kbest.get_support()].to_list()
    return top

# =======================================================================================================
# select_kbest END
# select_kbest TO rfe
# rfe START
# =======================================================================================================

def rfe(predictors, target, k_features):
    '''
    Takes in a predictors and target dataframes as well as how many features (column names) you want
    to be selected.

    INPUT:
    predictors = Dataframe with ONLY the predictor columns and their respective values. MUST BE INT, FLOAT.
    target = Dataframe with ONLY the target column and their respective values.
    k_features = The number of top performing features you want to be returned.

    OUTPUT:
    top = A dataframe ordered from best to worst performing features.
    '''
    LR = LinearRegression()
    rfe = RFE(LR, n_features_to_select=k_features)
    rfe.fit(predictors, target)
    top = pd.DataFrame({'rfe_ranking' : rfe.ranking_},
                 index=predictors.columns)
    top = pd.DataFrame(top.rfe_ranking.sort_values())
    return top

# =======================================================================================================
# rfe END
# rfe TO 
#  START
# ======================================================================================================