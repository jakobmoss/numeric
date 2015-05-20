# cython: language_level=3

import numpy as np
import auxA as minimize
import math, sys

# Global variables
import globvar

# Different functions to root-find
from datafun import *


#
# Main function
#
def mainA():
    """
    Test of the minimization methods
    """
    ## PART ZERO  --> Pretty print!
    print('\n********************************************')
    print('* Test of Newton\'s method for minimization *')
    print('********************************************\n')
