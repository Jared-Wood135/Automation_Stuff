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
# regression_prediction TO 
#  START
# =======================================================================================================
