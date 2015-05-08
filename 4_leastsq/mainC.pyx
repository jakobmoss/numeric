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
def mainC():
    """
    Analysis of the straight line fit to two points
    """
    # Make the test points and write them for plot
    n = 2
    a = -1.1
    b = 1.1
    x, y, dy = makedata(n, a, b)

    # Make a linear fit
    fitfunc = [f0, f1]
    c, dc, S = leastsq.qrfit(fitfunc, x, y, dy)

    # Write the fit to plot
    print('\n')
    nx = 20
    grid = np.linspace(a-0.3, b+0.3, nx)
    for i in range(nx):
        xx = grid[i]
        print(xx, leastsq.evalfit(fitfunc, c, xx), sep='\t')


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
        dy[i] = 0.1 + 0.6*random.random()
        print(x[i], y[i], dy[i], sep='\t')

    # Return the data
    return x, y, dy
