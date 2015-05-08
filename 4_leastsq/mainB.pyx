# cython: language_level=3

import numpy as np
import auxB as leastsq
import math, random
import sys

#
# Fitting functions
#
def f0(x):
    return math.pow(x, 0)

def f1(x):
    return math.pow(x, 1)

def f2(x):
    return math.pow(x, 2)

def f3(x):
    return math.pow(x, 3)

def f4(x):
    return math.pow(x, 4)


#
# Main function
#
def mainB():
    """
    Test of the least squares fitting
    """
    # Make test-data and write for plot
    n = 12
    a = -1.4
    b = 1.4
    x, y, dy = makedata(n, a, b)

    # Do the fitting
    fitfunc = [f0, f1, f2]
    c, dc, S = leastsq.singular_fit(fitfunc, x, y, dy)

    # Write the fit to plot
    print('\n')
    nx = 120
    for i in range(nx):
        xx = a + (b-a)*i / nx
        print(xx, leastsq.evalfit(fitfunc, c, xx), sep='\t')

    # Errors on the fit to plot
    print('\n')
    for i in range(nx):
        xx = a + (b-a)*i / nx
        print(xx, leastsq.evalfit(fitfunc, c+dc, xx), sep='\t')

    print('\n')
    for i in range(nx):
        xx = a + (b-a)*i / nx
        print(xx, leastsq.evalfit(fitfunc, c-dc, xx), sep='\t')



#
# Function to produce the data to fit and write for plot
#
def makedata(n, a, b):
    """
    Generates test-data and writes for plotting with Gnuplot
    """
    # Function to generate data
    def data(x):
        return 1 + 0.6*x - 1.3*math.pow(x, 2)

    # Initialize
    x = np.zeros(n, dtype='float64')
    y = np.zeros(n, dtype='float64')
    dy = np.zeros(n, dtype='float64')

    # Fill the test-data and print for plot
    for i in range(n):
        x[i] = a + (b-a)*i / (n-1)
        y[i] = data(x[i]) + (0.7*random.random() - 0.2)
        dy[i] = 0.2 + 0.8*random.random()
        print(x[i], y[i], dy[i], sep='\t')

    # Return the data
    return x, y, dy
