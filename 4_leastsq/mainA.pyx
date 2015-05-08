# cython: language_level=3

import numpy as np
import auxA as leastsq
import math, random

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
def mainA():
    """
    Test of the least squares fitting
    """
    # Make test-data and write for plot
    n = 10
    a = -0.9
    b = 0.9
    x, y, dy = makedata(n, a, b)

    # Do the fitting
    fitfunc = [f0, f1, f2]
    c, dc, S = leastsq.qrfit(fitfunc, x, y, dy)

    # Write the fit to plot
    print('\n')
    nx = 100
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
        return 1 + 2*x + 3*math.pow(x, 2)

    # Initialize
    x = np.zeros(10, dtype='float64')
    y = np.zeros(10, dtype='float64')
    dy = np.zeros(10, dtype='float64')

    # Fill the test-data and print for plot
    for i in range(10):
        x[i] = a + (b-a)*i / (n-1)
        y[i] = data(x[i]) + (random.random()-0.5)
        dy[i] = 0.1 + random.random()
        print(x[i], y[i], dy[i], sep='\t')

    # Return the data
    return x, y, dy
