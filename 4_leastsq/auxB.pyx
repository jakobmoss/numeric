# cython: language_level=3

# General modules
import numpy as np
import math




#
# Function to evaluate a fit
#
def evalfit(flist, c, x):
    """
    Evaluates a fit of the form \sum{c_i * f_i(x)} at point x.
    """
    return sum([c[i] * flist[i](x) for i in range(len(flist))])
    
