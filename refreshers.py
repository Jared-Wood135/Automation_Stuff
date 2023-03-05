# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
D = DONE
# = IN-PROGRESS
    1. Orientation
    2. Imports
    3. Main Menu
#    4. Numpy
D        1. Numpy Random Menu
D            1. randint 
D            2. random
D            3. choice
    5. Pandas
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

import numpy as np
'''
'numpy' is used for vectorization, arrays, and probabilities...
Common alias is 'np'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd
'''
'pandas' is used to read various files and ultimately create dataframes that can be used
similarly to things like 'MySQL'...
Common alias is 'pd'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
'''
'matplotlib.pyplot' is a sub-library of 'matplotlib' and is used for creating visualizations
of various data...
Common alias is 'plt'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

import seaborn as sns
'''
'seaborn' is similar to 'matplotlib', but generally makes visualizations faster to create...
Common alias is 'sns'...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

import scipy.stats as stats
'''
'scipy.stats' is a sub-library of 'scipy' and is used to obtain statistical information...
'''

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

import os
'''
This is used specifically to clear the terminal as you navigate this script
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
        - numpy
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
            numpymenu()
            break
        elif menuin == '2':
            clear()
            print('Works')
            break
        elif menuin == '3':
            clear()
            print('Works')
            break
        elif menuin == '4':
            clear()
            print('Works')
            break
        elif menuin == '5':
            clear()
            print('Works')
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

def numpymenu():
    '''
    Menu of numpy specific help...
    '''
    while True:
        print('\033[32m==========> NUMPY MENU <==========\033[0m')
        print('\033[36m(1)\033[0m random')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            numpyrandommenu()
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


# ///////////////////////////////////////////////////////////////////////////////////////////////////////

def numpyrandommenu():
    '''
    Help menu for numpy's randomizer functions...
    NEEDS WORK:
        - random
        - choice
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM MENU <==========\033[0m')
        print('\033[36m(1)\033[0m randint')
        print('\033[36m(2)\033[0m random')
        print('\033[36m(3)\033[0m choice')
        print('\033[36m(N)\033[0m Return to Numpy Menu')
        print('\033[36m(M)\033[0m Return to Main Menu')
        print('\033[36m(Q)\033[0m Exit Program')
        menuin = input('\n\033[33mWhat do you want help with?\033[0m\n')
        if menuin == '1':
            clear()
            numpyrandomrandinthelp()
            break
        if menuin == '2':
            clear()
            numpyrandomrandomhelp()
            break
        if menuin == '3':
            clear()
            numpyrandomchoicehelp()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Menu\033[0m')
            numpymenu()
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

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

def numpyrandomrandinthelp():
    '''
    Gives help for numpy's random randint function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-RANDINT HELP <==========\033[0m')
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
            numpyrandommenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyrandommenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

def numpyrandomrandomhelp():
    '''
    Gives help for numpy's random random function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-RANDOM HELP <==========\033[0m')
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
            numpyrandommenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyrandommenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# ///////////////////////////////////////////////////////////////////////////////////////////////////////

def numpyrandomchoicehelp():
    '''
    Gives help for numpy's random choice function...
    '''
    while True:
        print('\033[32m==========> NUMPY-RANDOM-CHOICE HELP <==========\033[0m')
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
            numpyrandommenu()
            break
        elif menuin.lower() == 'n':
            clear()
            print('\033[33mReturning to Numpy Random Menu\033[0m')
            numpyrandommenu()
            break
        else:
            clear()
            print('\033[31mInvalid Input\033[0m')

# =======================================================================================================
# Numpy END
# Numpy TO Pandas
# Pandas START
# =======================================================================================================



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