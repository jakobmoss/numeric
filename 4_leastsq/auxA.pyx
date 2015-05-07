# cython: language_level=3

# General modules
import numpy as np
import math

# QR-decomposition routines using Given's rotation
import givens as qr


#
# Function to fit data using QR-decomposition
#
def qrfit(flist, x, y, dy):
    """
    
    Arguments:
    - `flist`:
    - `x`:
    - `y`:
    - `dy`:
    """
    # Initializations
    n = len(x)
    m = len(flist)
    A = np.zeros((n, m), dtype='float64')
    b = np.zeros(n, dtype='float64')

    # Fill A and b  -->  Loop over all data points
    for i in range(n):
        b[i] = y[i] / dy[i]  # Weighting by error

        # Loop over fitting functions
        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]
