# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
D 1. Orientation
D 2. Imports
D 3. Correlation
D    1. PearsonR
D    2. Spearman's
4. T-Test
    1. 1 Sample
    2. 2 Sample
    3. Values
5. Significance
    1. 3+ Sample
6. Chi
    1. Contingency
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to assist with and expedite the process of exploring data and 
their statistical significance, correlations, and other statistically useful information as
well as automating visualizations along with outputs to enrich understanding.
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from scipy import stats

# =======================================================================================================
# Imports END
# Imports TO Correlation(PearsonR)
# Correlation(PearsonR) START
# =======================================================================================================

def correlation_pearson(data, x_str, y_str):
    corr, pval = stats.pearsonr(data[x_str], data[y_str])
    print(f'\033[32mCorrelation=\033[0m {corr}\n\033[32mP-value=\033[0m {pval}')
    sns.scatterplot(data[x_str], data[y_str])
    plt.title(f'Relationship between {x_str} and {y_str}')
    plt.show()

# =======================================================================================================
# Correlation(PearsonR) END
# Correlation(PearsonR) TO Correlation(Spearman's)
# Correlation(Spearman's) START
# =======================================================================================================

def correlation_spearman(data, x_str, y_str):
    corr, pval = stats.spearmanr(data[x_str], data[y_str])
    print(f'\033[32mCorrelation=\033[0m {corr}\n\033[32mP-value=\033[0m {pval}')
    sns.scatterplot(data[x_str], data[y_str])
    plt.title(f'Relationship between {x_str} and {y_str}')
    plt.show()

# =======================================================================================================
# Correlation(Spearman's) END
# Correlation(Spearman's) TO T-Test(1-Sample)
# T-Test(1-Sample) START
# =======================================================================================================

# stats.ttest_1samp()

# =======================================================================================================
# T-Test(1-Sample) END
# T-Test(1-Sample) TO T-Test(2-Sample)
# T-Test(2-Sample) START
# =======================================================================================================

# stats.ttest_ind()

# =======================================================================================================
# T-Test(2-Sample) END
# T-Test(2-Sample) TO T-Test(Values)
# T-Test(Values) START
# =======================================================================================================

# stats.ttest_ind_from_values()

# =======================================================================================================
# T-Test(Values) Sample END
# T-Test(Values) TO Significance 3+ Sample
# Significance 3+ Sample START
# =======================================================================================================

# stats.f_oneway()

# =======================================================================================================
# Significance 3+ Sample END
# Significance 3+ TO Chi(Contingency)
# Chi(Contingency) START
# =======================================================================================================

# stats.chi2_contingency()

# =======================================================================================================
# Chi(Contingency) END
# =======================================================================================================