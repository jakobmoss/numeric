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
    Calculates the fit of the form \sum{c_i * f_i(x)} to the data 
    (x, y) with error dy on y, using least mean squares and 
    QR-decomposition with Given's rotation.

    Returns the fitting coefficients as a vector and the covariance
    matrix.

    Arguments:
    - `flist`: List of functions to fit to the data
    - `x`: X-data [vector]
    - `y`: Y-data [vector]
    - `dy`: Error on y-data [vector]
    """
    # Initializations
    n = len(x)
    m = len(flist)
    A = np.zeros((n, m), dtype='float64')
    c = np.zeros(n, dtype='float64')
    dc = np.zeros(n, dtype='float64')

    # Fill A and c  -->  Loop over all data points
    for i in range(n):
        c[i] = y[i] / dy[i]  # Weighting by error

        # Loop over fitting functions
        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]

    # Decompose using Given's rotation and solve by in-place backsub
    qr.decomp(A)
    qr.solve(A, c)

    # Explictly build the R-matrix and find the inverse
    Rinv = np.zeros((n, n), dtype='float64')
    qr.inverse(qr.build_r(A), Rinv)

    # Calculate the covariance matrix S
    S = np.dot(Rinv, np.transpose(Rinv))

    # Convert S into uncertainties on the coefficients
    for i in range(m):
        dc[i] = math.sqrt(S[i,i])

    # Return the fitting coefficients and the covariance matrix
    return c, dc, S

#
# Function to evaluate a fit
#
def evalfit(flist, c, x):
    """
    Evaluates a fit of the form \sum{c_i * f_i(x)} at point x.

    Arguments: See qrfit
    """
    return sum([c[i] * flist[i](x) for i in range(len(flist))])
    
