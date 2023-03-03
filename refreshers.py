# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. Main Menu
4. Numpy
5. Pandas
6. Matplotlib.pyplot
7. Seaborn
8. Scipy.Stats
9. Clear Terminal Function
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
# ////////////////////////////////
import pandas as pd
'''
'pandas' is used to read various files and ultimately create dataframes that can be used
similarly to things like 'MySQL'...
Common alias is 'pd'...
'''
# ////////////////////////////////
import matplotlib.pyplot as plt
'''
'matplotlib.pyplot' is a sub-library of 'matplotlib' and is used for creating visualizations
of various data...
Common alias is 'plt'...
'''
# ////////////////////////////////
import seaborn as sns
'''
'seaborn' is similar to 'matplotlib', but generally makes visualizations faster to create...
Common alias is 'sns'...
'''
# ////////////////////////////////
import scipy.stats as stats
'''
'scipy.stats' is a sub-library of 'scipy' and is used to obtain statistical information...
'''
# ////////////////////////////////
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
            print('Works')
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
menu()

# =======================================================================================================
# Main Menu END
# Main Menu TO Numpy
# Numpy START
# =======================================================================================================

def numpymenu():
    '''
    Menu of numpy specific help...
    '''

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
# Clear Terminal Function END
# =======================================================================================================