import numpy as np


def randint():
    '''
    Returns an array of random integers with inputted ranges and shape values
    '''
    minrange = int(input("MIN Range Value(INCLUSIVE):\n"))
    maxrange = int(input("MAX Range Value(INCLUSIVE):\n")) + 1
    rows = int(input("Amount of rows:\n"))
    columns = int(input("Amount of columns:\n"))
    return np.random.randint(minrange, maxrange, size=[rows, columns]) 
test = randint()

def randpct():
    '''
    Returns an array of random percent values with inputted ranges and shape values
    '''
    rows = int(input("Amount of rows:\n"))
    columns = int(input("Amount of columns:\n"))
    return np.random.random([rows, columns])
randpct()