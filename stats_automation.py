# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
D 1. Orientation
D 2. Imports
D 3. Correlation
D    1. PearsonR
D    2. Spearman's
D4. T-Test
D    1. 1 Sample
D    2. 2 Sample
D    3. Values
#5. Significance
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
    '''
    Returns a correlation and p-value as well as a visualization using the pearson methodology
    (pearsonr used for straight line correlation)
    stats.pearsonr(mpg.year, mpg.cty)
    Tests the relationship between car year and city mileage
    '''
    corr, pval = stats.pearsonr(data[x_str], data[y_str])
    print(f'\033[32mCorrelation =\033[0m {corr}\n\033[32mP-value =\033[0m {pval}')
    sns.scatterplot(data[x_str], data[y_str])
    plt.title(f'Relationship between {x_str} and {y_str}')
    plt.show()

# =======================================================================================================
# Correlation(PearsonR) END
# Correlation(PearsonR) TO Correlation(Spearman's)
# Correlation(Spearman's) START
# =======================================================================================================

def correlation_spearman(data, x_str, y_str):
    '''
    Returns a correlation and p-value as well as a visualization using the spearman methodology
    (spearmanr used for non-straight line correlation like a parabola or exponential growth)
    stats.spearmanr(mpg.displ, mpg.cyl)
    Tests the correlation between displ and cyl
    '''
    corr, pval = stats.spearmanr(data[x_str], data[y_str])
    print(f'\033[32mCorrelation =\033[0m {corr}\n\033[32mP-value =\033[0m {pval}')
    sns.scatterplot(data[x_str], data[y_str])
    plt.title(f'Relationship between {x_str} and {y_str}')
    plt.show()

# =======================================================================================================
# Correlation(Spearman's) END
# Correlation(Spearman's) TO T-Test(1-Sample)
# T-Test(1-Sample) START
# =======================================================================================================

def ttest1samp(data, subset, setColStr):
    '''
    Returns a statistic and p-value as well as a visualization using the T-Test, 1-Sample methodology
    (ttest_1samp is used for comparing a subset against the entire population)
    stats.ttest_1samp(mpg[mpg.year == 1999].cty, mpg.cty.mean())
    Tests the relationship of city mileage of 1999 cars to the city mileage of all cars
    '''
    stat, pval = stats.ttest_1samp(subset, data[setColStr].mean())
    print(f'\033[32mStatistic =\033[0m {stat}\n\033[32mP-value =\033[0m {pval}')
    sns.distplot(subset, label='Subset')
    sns.distplot(data[setColStr], label='Overall')
    plt.title(f'Overall {setColStr} Distribution')
    plt.legend()
    plt.show()

# =======================================================================================================
# T-Test(1-Sample) END
# T-Test(1-Sample) TO T-Test(2-Sample)
# T-Test(2-Sample) START
# =======================================================================================================

def ttestind(subset_var1, subset_var2):
    '''
    Returns a statistic and p-value as well as a visualization using the T-Test, independent methodology
    (ttest_ind is used for comparing two independent subsets and a shared value)
    stats.ttest_ind(mpg[mpg.year == 1999].cty, mpg[mpg.year == 2008].cty)
    Tests the relationship of city mileage between 1999 and 2008 vehicles
    '''
    stat, pval = stats.ttest_ind(subset_var1, subset_var2)
    print(f'\033[32mStatistic =\033[0m {stat}\n\033[32mP-value =\033[0m {pval}')
    sns.distplot(subset_var1, label='Subset 1')
    sns.distplot(subset_var2, label='Subset 2')
    plt.title(f'Relationship between {subset_var1} and {subset_var2}')
    plt.legend()
    plt.show()

# =======================================================================================================
# T-Test(2-Sample) END
# T-Test(2-Sample) TO T-Test(Values)
# T-Test(Values) START
# =======================================================================================================

def ttestval(mean1, std1, trials1, mean2, std2, trials2):
    '''
    Returns a statistic and p-value as well as a visualization using the T-Test, value methodology
    (ttest_ind_from_stats is used for comparing two independent subsets using known values)
    stats.ttest_ind_from_stats(90, 15, 50, 150, 10, 50)
    Tests the relationship of the first independent value of {mean:90, std:15, trial:50}
    and the second independent value of {mean:150, std:10, trial:50}
    '''
    stat, pval = stats.ttest_ind_from_stats(mean1, std1, trials1, mean2, std2, trials2)
    print(f'\033[32mStatistic =\033[0m {stat}\n\033[32mP-value =\033[0m {pval}')
    value1 = np.random.normal(mean1, std1, trials1)
    value2 = np.random.normal(mean2, std2, trials2)
    sns.distplot(value1, label='Value 1')
    sns.distplot(value2, label='Value 2')
    plt.title(f'Relationship between Value 1 and Value 2')
    plt.legend()
    plt.show()

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