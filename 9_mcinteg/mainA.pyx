# cython: language_level=3

# General modules
import numpy as np
import math
import auxA as mcint
import sys

#
# Test-functions to integrate
#
def f1(q):
    """
    y * cos(x)
    """
    return q[1] * math.cos(q[0])   


#
# Main function
#
def mainA():
    """
    Test of the Monte Carlo routines
    """
    ##############
    # Function 1 #
    ##############
    # Initializations
    a = np.array([-math.pi/2.0, 0])
    b = np.array([math.pi/2.0, 1])
    exact1 = 1
