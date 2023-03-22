# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
D = DONE
# = IN-PROGRESS
    1. Orientation
    2. Imports
    3. Main Menu
D    4. Numpy
D        1. Numpy Random Menu
D            1. randint 
D            2. random
D            3. choice
#    5. Pandas
D        1. Creating Dataframes
D            1. Dictionary
D            2. SQL
D            3. Read Files
##        2. Navigate Dataframes
###            1. info
            2. describe
            3. dtype
        3. Manipulate Dataframes
            1. Reset index
            2. Columns
                1. delete
                2. drop
                3. rename
            3. Groupby
            4. Joins
            5. Aggregate
    6. Matplotlib.pyplot
    7. Seaborn
    8. Scipy.Stats
    9. Clear Terminal Function
    10. Start Program
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this document is to remind myself and anyone else who finds this useful
of key lines of code, their syntax, and their purpose for the numpy, pandas, 
matplotlib.pyplot, seaborn, and scipy.stats libraries...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

# Basic vectorization/dataframe purposes:
# - numpy (np)
# - pandas (pd)
import numpy as np
'''
'numpy' is used for vectorization, arrays, and probabilities...
Common alias is 'np'...
'''

import pandas as pd
'''
'pandas' is used to read various files and ultimately create dataframes that can be used
similarly to things like 'MySQL'...
Common alias is 'pd'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# Visualizations and graphing purposes:
# - matplotlib.pyplot (plt)
# - seaborn (sns)
import matplotlib.pyplot as plt
'''
'matplotlib.pyplot' is a sub-library of 'matplotlib' and is used for creating visualizations
of various data...
Common alias is 'plt'...
'''

import seaborn as sns
'''
'seaborn' is similar to 'matplotlib', but generally makes visualizations faster to create...
Common alias is 'sns'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# Statistical testing purposes:
# - scipy.stats as stats
import scipy.stats as stats
'''
'scipy.stats' is a sub-library of 'scipy' and is used to obtain statistical information...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# Useful tools to use with modeling:
'''
- Scores/evaluations
    - from sklearn.metrics import classification_report
        - Used to get accuracy, precision, recall, f-1, and support scores of model
    - from sklearn.metrics import confusion_matrix
        - Used to get a confusion matrix of TP, FP, TN, FN
'''


# Classification modeling purposes:
'''
- Decision Tree Classifier
    - sklearn.tree import DecisionTreeClassifier
        - Used for Decision Tree model machine learning type
    - sklearn.tree import export_text
        - Used to export text of what the model is doing
    - sklearn.tree import plot_tree
        - Used to visualize what the model is doing
- Random Forest Classifier
    - sklearn.ensemble import RandomForestClassifier
        - Used for Random Forest model machine learning type
- K-Nearest Neighbors
    - sklearn.neighbors import KNeighborsClassifier
        - Used for K-Nearest Neighbors model machine learning type
- Logistic Regression
    - sklearn.linear_model import LogisticRegression
        - Used for Logistic Regression model machine learning
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# To have functionality similar to command line interface (CLI)
# - os
import os
'''
This is used specifically to clear the terminal as you navigate this script
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# For example/play datasets:
# - pydataset
from pydataset import data
'''
This is used specifically to demonstrate databases and give better visuals of some explanations
'''


# =======================================================================================================
# Imports END
# Imports TO Main Menu
# Main Menu START
# =======================================================================================================

def menu():
    '''
    Opens a menu for easier navigation when looking for specific information...
    NEEDS WORK:
        - pandas
        - matplotlib.pyplot
        - seaborn
        - scipy.stats
    '''
    while True:
        print('\033[32m==========> MAIN MENU <==========\033[0m')
        print('\033[36m(1)\033[0m numpy')
        print('\033[36m(2)\033[0m pandas')
        print('\033[36m(3)\033[0m matplotlib.pyplot')
        print('\033[36m(4)\033[0m seaborn')
        print('\033[36m(5)\033[0m scipy.stats')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            numpyMenu()
            break
        elif menuin == '2':
            clear()
            pandasMenu()
            break
        elif menuin == '3':
            clear()
            print('matplotlib.pyplot')
            break
        elif menuin == '4':
            clear()
            print('seaborn')
            break
        elif menuin == '5':
            clear()
            print('scipy.stats')
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')


# =======================================================================================================
# Main Menu END
# Main Menu TO Numpy
# Numpy START
# =======================================================================================================

def numpyMenu():
    '''
    Menu of numpy specific help...
    '''
    while True:
        print('\033[32m==========> NUMPY MENU <==========\033[0m')
        print("\033[32mMenu of numpy specific help...\033[0m")
        print('\033[36m(1)\033[0m random')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            numpyRandomMenu()
            break
        elif menuin.lower() == 'm':
            clear()
            print('\033[33mReturning to Main Menu\033[0m')
            menu()
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')


# =======================================================================================================
# Numpy END
# Numpy TO Numpy Random
# Numpy Random START
# =======================================================================================================

def numpyRandomMenu():
    '''
    Help menu for numpy's randomizer functions...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM MENU <==========\033[0m')
        print("\033[32mHelp menu for numpy's randomizer functions...\033[0m")
        print('\033[36m(1)\033[0m randint')
        print('\033[36m(2)\033[0m random')
        print('\033[36m(3)\033[0m choice')
        print('\033[36m(N)\033[0m Return to Numpy Menu')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            numpyRandomRandintHelp()
            break
        if menuin == '2':
            clear()
            numpyRandomRandomHelp()
            break
        if menuin == '3':
            clear()
            numpyRandomChoiceHelp()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Menu\033[0m')
            numpyMenu()
            break
        elif menuin.lower() == 'm':
            clear()
            print('\033[33mReturning to Main Menu\033[0m')
            menu()
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Numpy Random END
# Numpy Random TO Numpy Random Randint
# Numpy Random Randint START
# =======================================================================================================

def numpyRandomRandintHelp():
    '''
    Gives help for numpy's random randint function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-RANDINT HELP <==========\033[0m')
        print("\033[32mGives help for numpy's random randint function...\033[0m")
        print('\033[35mRaw Syntax:\033[0m np.random.randint(#, #, [#, #])')
        print('\033[35mINPUT:\033[0m np.random.randint(1, 6, [2, 5])')
        print('\033[35mOUTPUT:\033[0m\n', np.random.randint(1, 6, [2, 5]))
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print('\033[35mRaw Syntax:\033[0m np.random.randint(\033[36m#\033[0m, \033[36m#\033[0m, [\033[36m#\033[0m, \033[36m#\033[0m])')
            print('\033[35mAliased Syntax:\033[0m np.random.randint(\033[36m<Min Range>\033[0m, \033[36m<Max Range>\033[0m, [\033[36m<Rows>\033[0m, \033[36m<Columns>\033[0m])')
            print('\033[35mnp.random.randint:\033[0m Function that randomly chooses an integer from the numpy.random function')
            print('\033[35m(<Min Range>, <Max Range>,:\033[0m Defines the min then max range values. (MAX VALUE IS EXCLUSIVE)')
            print('\033[35m[<Rows>, <Columns>]):\033[0m Defines the rows and columns to generate an array by')
            print('\033[33mReturning to Numpy Random Menu\033[0m\n')
            numpyRandomMenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyRandomMenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Numpy Random Randint END
# Numpy Random Randint TO Numpy Random Random
# Numpy Random Random START
# =======================================================================================================

def numpyRandomRandomHelp():
    '''
    Gives help for numpy's random random function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-RANDOM HELP <==========\033[0m')
        print("\033[32mGives help for numpy's random random function...\033[0m")
        print('\033[35mRaw Syntax:\033[0m np.random.random([#, #])')
        print('\033[35mINPUT:\033[0m np.random.random([2, 5])')
        print('\033[35mOUTPUT:\033[0m\n', np.random.random([2, 5]))
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print('\033[35mRaw Syntax:\033[0m np.random.random([\033[36m#\033[0m, \033[36m#\033[0m])')
            print('\033[35mAliased Syntax:\033[0m np.random.random([\033[36m<Rows>\033[0m, \033[36m<Columns>\033[0m])')
            print('\033[35mnp.random.random:\033[0m Function that randomly generates a percentage from 0 to 1 from the numpy.random function')
            print('\033[35m[<Rows>, <Columns>]):\033[0m Defines the rows and columns to generate an array by')
            print('\033[33mReturning to Numpy Random Menu\033[0m\n')
            numpyRandomMenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyRandomMenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Numpy Random Random END
# Numpy Random Random TO Numpy Random Choice
# Numpy Random Choice START
# =======================================================================================================

def numpyRandomChoiceHelp():
    '''
    Gives help for numpy's random choice function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-CHOICE HELP <==========\033[0m')
        print("\033[32mGives help for numpy's random choice function...\033[0m")
        print('\033[35mRaw Syntax:\033[0m np.random.choice([# or str], [#, #], p=[#])')
        print('\033[35mINPUT:\033[0m np.random.choice([1, \'yeet\', 3], [2, 5], p=[0.2, 0.8, 0]])')
        print('\033[35mOUTPUT:\033[0m\n', np.random.choice([1, 'yeet', 3], [2, 5], p=[.2, .8, 0]))
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print('\033[35mRaw Syntax:\033[0m np.random.choice([\033[36m# or str\033[0m], [\033[36m#, #\033[0m], p=[\033[36m#\033[0m])')
            print('\033[35mAliased Syntax:\033[0m np.random.random([\033[36m<List of values>\033[0m], [\033[36m<Rows>\033[0m, \033[36m<Columns>\033[0m], p=[\033[36m<Percent chance in order of list of values>\033[0m])')
            print('\033[35mnp.random.choice:\033[0m Function that selects from the list of values by their inputted percent chance from the numpy.random function')
            print('\033[35m[<List of values>]):\033[0m Values that will be randomly selected (Can be int, str, float)')
            print('\033[35m[<Rows>, <Columns>]):\033[0m Defines the rows and columns to generate an array by')
            print('\033[35mp=[#]):\033[0m Defines the percent chance of a value in order from left to right in list of values')
            print('\033[33mReturning to Numpy Random Menu\033[0m\n')
            numpyRandomMenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyRandomMenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Numpy Random Choice END
# Numpy Random Choice TO Pandas
# Pandas START
# =======================================================================================================

def pandasMenu():
    '''
    Menu of pandas specific help...
    NEEDS WORK:
        - Navigate Dataframe
        - Manipulate Dataframe
    '''
    while True:
        print('\033[32m==========> PANDAS MENU <==========\033[0m')
        print("\033[32mMenu of pandas specific help...\033[0m")
        print('\033[36m(1)\033[0m Creating Dataframes')
        print('\033[36m(2)\033[0m Navigate Dataframes')
        print('\033[36m(3)\033[0m Manipulate Dataframes')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            pandasCreateDataFrame()
            break
        if menuin == '2':
            clear()
            pandasNavigateDataFrame()
            break
        if menuin == '3':
            clear()
            print('Manipulate Dataframes')
            break
        elif menuin.lower() == 'm':
            clear()
            print('\033[33mReturning to Main Menu\033[0m')
            menu()
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas END
# Pandas TO Pandas Creating Dataframe
# Pandas Creating Dataframe START
# =======================================================================================================

def pandasCreateDataFrame():
    '''
    Help menu for pandas's dataframe creation...
    '''
    while True:
        print('\033[32m==========> PANDAS-CREATE DATAFRAME MENU <==========\033[0m')
        print("\033[32mHelp menu for pandas's dataframe creation...\033[0m")
        print('\033[36m(1)\033[0m dictionary')
        print('\033[36m(2)\033[0m sql')
        print('\033[36m(3)\033[0m read file')
        print('\033[36m(N)\033[0m Return to Pandas Menu')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            pandasCreateDataFrameDict()
            break
        if menuin == '2':
            clear()
            pandasCreateDataFrameSQL()
            break
        if menuin == '3':
            clear()
            pandasCreateDataFrameReadFile()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Menu\033[0m')
            pandasMenu()
            break
        elif menuin.lower() == 'm':
            clear()
            print('\033[33mReturning to Main Menu\033[0m')
            menu()
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas Creating Dataframe END
# Pandas Creating Dataframe TO Pandas Creating Dataframe Dictionary
# Pandas Creating Dataframe Dictionary START
# =======================================================================================================

def pandasCreateDataFrameDict():
    '''
    Gives help for pandas's dataframe creation with dictionaries...
    '''
    while True:
        testdict = {
            'fruits' : ['apple', 'banana', 'orange'],
            'quantity' : [5, 10, 15],
            'cost' : [0.98, 1.14, 0.52]
        }
        print('\033[32m==========> PANDAS-CREATE DATAFRAME-DICTIONARY HELP <==========\033[0m')
        print("\033[32mGives help for panda's dataframe creation with dictionaries...\033[0m")
        print('\033[35mRaw Syntax:\033[0m pd.DataFrame(dict)')
        print("\033[35mtestdict:\033[0m\ntestdict = {\n'fruits' : ['apple', 'banana', 'orange'],\n'quantity' : [5, 10, 15],\n'cost' : [0.98, 1.14, 0.52]\n}")
        print('\033[35mINPUT:\033[0m pd.DataFrame(testdict)')
        print('\033[35mOUTPUT:\033[0m\n', pd.DataFrame(testdict))
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print('\033[35mRaw Syntax:\033[0m pd.DataFrame(dict)')
            print('\033[35mAliased Syntax:\033[0m pd.DataFrame(\033[36m<Dictionary Name>\033[0m)')
            print('\033[35mpd.DataFrame:\033[0m Function that creates a pandas dataframe')
            print('\033[35m(<Dictionary Name>):\033[0m Defines the dictionary to create a dataframe from')
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m\n')
            pandasCreateDataFrame()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m')
            pandasCreateDataFrame()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas Creating Dataframe Dictionary END
# Pandas Creating Dataframe Dictionary TO Pandas Creating Dataframe SQL
# Pandas Creating Dataframe SQL START
# =======================================================================================================

def pandasCreateDataFrameSQL():
    '''
    Gives help for pandas's dataframe creation with SQL databases...
    '''
    while True:
        query = (
            '''
            SELECT *
            FROM employees
            '''
        )
        print('\033[32m==========> PANDAS-CREATE DATAFRAME-SQL HELP <==========\033[0m')
        print("\033[32mGives help for panda's dataframe creation with SQL databases...\033[0m")
        print('\033[35mRaw Syntax:\033[0m pd.read_sql(query, url)')
        print("\033[35mquery:\033[0m\nquery = (\n'''\nSELECT *\nFROM employees\n'''\n)")
        print("\033[35murl:\033[0m url = <Your unique URL connection to the SQL database>")
        print('\033[35mINPUT:\033[0m pd.read_sql(query, url)')
        print('\033[35mOUTPUT:\033[0m Dataframe with SQL Query output')
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print('\033[35mRaw Syntax:\033[0m pd.read_sql(query, url)')
            print('\033[35mAliased Syntax:\033[0m pd.read_sql(\033[36m<SQL Query>\033[0m, \033[36m<URL Connection To SQL>\033[0m)')
            print('\033[35mpd.read_sql:\033[0m Function that creates a pandas dataframe by reading a sql query')
            print('\033[35m(<SQL Query>,:\033[0m Uses the SQL Query output to create the pandas dataframe')
            print('\033[35m<URL Connection To SQL>):\033[0m Unique URL in order to connect to desired SQL database')
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m\n')
            pandasCreateDataFrame()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m')
            pandasCreateDataFrame()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas Creating Dataframe SQL END
# Pandas Creating Dataframe SQL TO Pandas Creating Dataframe Read Files
# Pandas Creating Dataframe Read Files START
# =======================================================================================================

def pandasCreateDataFrameReadFile():
    '''
    Gives help for pandas's dataframe creation with reading external files...
    '''
    while True:
        print('\033[32m==========> PANDAS-CREATE DATAFRAME-READ FILE HELP <==========\033[0m')
        print("\033[32mGives help for pandas's dataframe creation with reading external files...\033[0m")
        print("\033[35mRaw Syntax:\033[0m pd.read_<file-type>(<url>, sep='<seperator>')")
        print("\033[35murl:\033[0m url = <URL connection to file>")
        print("\033[35mINPUT:\033[0m pd.read_csv(<url>, sep=',')")
        print('\033[35mOUTPUT:\033[0m Dataframe with file values seperated by a comma')
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print("\033[35mRaw Syntax:\033[0m pd.read_<file-type>(<url>, sep='<seperator>')")
            print("\033[35mAliased Syntax:\033[0m pd.read_<file-type>(\033[36m<url>\033[0m, \033[36msep='<seperator>'\033[0m)")
            print('\033[35mpd.read_<file-type>:\033[0m Function that creates a pandas dataframe by reading a specific file type')
            print('\033[35m(<url>,:\033[0m Uses the url to locate the file you want')
            print("\033[35msep='<seperator>'):\033[0m Seperates the values in the file by a specific seperator")
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m\n')
            pandasCreateDataFrame()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m')
            pandasCreateDataFrame()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas Creating Dataframe Read Files END
# Pandas Creating Dataframe Read Files TO Pandas Navigate Dataframe
# Pandas Navigate Dataframe START
# =======================================================================================================

def pandasNavigateDataFrame():
    '''
    Help menu for pandas's dataframe navigation...
    NEEDS WORK:
        1. showDoc
        2. info
        3. describe
        4. dtype
    '''
    while True:
        print('\033[32m==========> PANDAS-NAVIGATE DATAFRAME MENU <==========\033[0m')
        print("\033[32mHelp menu for pandas's dataframe navigation...\033[0m")
        print('\033[36m(1)\033[0m showDoc')
        print('\033[36m(2)\033[0m info')
        print('\033[36m(3)\033[0m describe')
        print('\033[36m(4)\033[0m dtype')
        print('\033[36m(N)\033[0m Return to Pandas Menu')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            print('showDoc')
            break
        if menuin == '2':
            clear()
            print('info')
            break
        if menuin == '3':
            clear()
            print('describe')
            break
        if menuin == '4':
            clear()
            print('dtype')
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Menu\033[0m')
            pandasMenu()
            break
        elif menuin.lower() == 'm':
            clear()
            print('\033[33mReturning to Main Menu\033[0m')
            menu()
            break
        elif menuin.lower() == 'q':
            clear()
            print('\033[35mHope it helped!\033[0m')
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas Navigate Dataframe END
# Pandas Navigate Dataframe TO Pandas Navigate Dataframe Info
# Pandas Navigate Dataframe Info START
# =======================================================================================================

### CHECKPOINT ###
def pandasNavigateDataframeInfo():
    '''
    Gives help for pandas's dataframe navigation with info...
    '''
    while True:
        print('\033[32m==========> PANDAS-NAVIGATE DATAFRAME-INFO HELP <==========\033[0m')
        print("\033[32mGives help for pandas's dataframe navigation with showDoc...\033[0m")
        print("\033[35mRaw Syntax:\033[0m pd.read_<file-type>(<url>, sep='<seperator>')")
        print("\033[35murl:\033[0m url = <URL connection to file>")
        print("\033[35mINPUT:\033[0m pd.read_csv(<url>, sep=',')")
        print('\033[35mOUTPUT:\033[0m Dataframe with file values seperated by a comma')
        menuin = input('\033[33mWould you like an explanation? (Y/N)\033[0m\n')
        if menuin.lower() == 'y':
            clear()
            print("\033[35mRaw Syntax:\033[0m pd.read_<file-type>(<url>, sep='<seperator>')")
            print("\033[35mAliased Syntax:\033[0m pd.read_<file-type>(\033[36m<url>\033[0m, \033[36msep='<seperator>'\033[0m)")
            print('\033[35mpd.read_<file-type>:\033[0m Function that creates a pandas dataframe by reading a specific file type')
            print('\033[35m(<url>,:\033[0m Uses the url to locate the file you want')
            print("\033[35msep='<seperator>'):\033[0m Seperates the values in the file by a specific seperator")
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m\n')
            pandasCreateDataFrame()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Pandas Create Dataframe Menu\033[0m')
            pandasCreateDataFrame()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Pandas END
# Pandas TO Matplotlib.pyplot
# Matplotlib.pyplot START
# =======================================================================================================



# =======================================================================================================
# Matplotlib.pyplot END
# Matplotlib.pyplot TO Seaborn
# Seaborn START
# =======================================================================================================



# =======================================================================================================
# Seaborn END
# Seaborn TO Scipy.stats
# Scipy.stats START
# =======================================================================================================



# =======================================================================================================
# Scipy.stats END
# Scipy.stats TO Clear Terminal Function
# Clear Terminal Function START
# =======================================================================================================

def clear():
    '''
    Clears terminal to reduce clutter...
    '''
    os.system('clear')

# =======================================================================================================
# Clear Terminal END
# Clear Terminal TO Start Program
# Start Program START
# =======================================================================================================

menu()

# =======================================================================================================
# Start Program END
# =======================================================================================================