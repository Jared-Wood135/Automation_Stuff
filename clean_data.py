# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. IDontFuckingKnow
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientaion START
# =======================================================================================================

'''
Purpose: 
    - Automate dataframe cleaning to expedite data science pipeline

Methodology:
    1. Input desired dataframe
    2. Bin all columns - Dictionary
        a. Numerical-Ready
            - Int/Float column without any issues
            - List of pertinent column names
        b. Numerical-Issue
            - Int/Float column with 'Null', 'NaN', 'inf', etc.
            - List of pertinent column names
        c. Datetime-Ready
            - Date column without any issues
            - List of pertinent column names
        d. Datetime-Issue
            - Date column with 'Null', 'NaN', 'inf', etc.
            - List of pertinent column names
        e. Object/Unique-Ready
            - Object column with all unique values and no issues
            - List of pertinent column names
        f. Object/Unique-Issue
            - Object column with 'Null', 'NaN', 'inf', etc.
            - List of pertinent column names
        g. Object/Input Req.-Ready
            - Object column with no issues
            - List of pertinent column names
        h. Object/Input Req.-Issue
            - Object column with 'Null', 'NaN', 'inf', etc.
            - List of pertinent column names
        i. Object/Categorical-Ready
            - Object column identified as categorical with no issues
            - Empty list for input on 'Object/Input Req.'
        j. Object/Location-Ready
            - Object column identified as location with no issues
            - Empty list for input on 'Object/Input Req.'
        k. Object/Comment-Ready
            - Object column identified as location with no issues
            - Empty list for input on 'Object/Input Req.'
    3. Correct issue keys from dictionary above
        a. Int/Float
            - 0
            - Mean
            - Mode
        b. Datetime
            - Mode
        c. Object(No spec. char)
            - Empty/0/' '
            - Mode
        d. Object(spec. char)
            - Change dtype?
            - 0
            - Mean
            - Mode
            - Empty/' '
    4. Return dictionary of alterations made
    5. Return cleaned dictionary

Status:
    - Incomplete
    - In Progress
    - Bin data (Testing)
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

# NOT ALL INCLUSIVE
import numpy as np
import pandas as pd
import itertools
from pydataset import data

# =======================================================================================================
# Imports END
# Imports TO IDontFuckingKnow?
# IDontFuckingKnow START
# =======================================================================================================

# Testing

# =======================================================================================================
# IDontFuckingKnow END
# =======================================================================================================