# cython: language_level=3

# General modules
import numpy as np
import math
import sys

# Module with Jacobi diagonalization
import jacobi

#
# Function to fit using singular value decomposition
#
def singular_fit(flist, x, y, dy):
    """
    
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
    b = np.zeros(n, dtype='float64')
    dc = np.zeros(m, dtype='float64')

    # Fill A and b  -->  Loop over all data points
    for i in range(n):
        b[i] = y[i] / dy[i]  # Weighting by error

        # Loop over fitting functions
        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]

    # Decompose and solve using singular value decomposition  -->  c
    U, s, V = decomp(A)
    c = solve(U, s, V, b)
    
    # Build the covariance matrix S
    VDinv = np.zeros((m, m), dtype='float64')
    for j in range(m):
        dinv_j = 1 / math.pow(s[j] ,2)
        for i in range(m):
            VDinv[i, j] = V[i, j] * dinv_j
    S = np.dot(VDinv, V.T)

    # Convert S into uncertainties on the coefficients
    for i in range(m):
        dc[i] = math.sqrt(S[i,i])

    # Return the fitting coefficients and the covariance matrix
    return c, dc, S


#
# Function to do singular value decomposition
#
def decomp(A):
    """
    Computes the singular value decomposition using Jacobi's algorithm
    for diagonalization. The matrix A is decomposed into A = U*S*V^T,
    where U = A*V*D^{-1/2} and S = D^{1/2}. D is the diagonal matrix 
    containing the eigenvalues of A and V contains the corresponding
    eigenvectors.

    Returns U, s, V ; where s is a vector of the eivenvalues.

    Arguments:
    - `A`: Input matix.
    """
    # Initialization
    n = A.shape[0]
    m = A.shape[1]
    U = np.zeros(A.shape, dtype='float64')
    s = np.zeros(m, dtype='float64')
    V = np.zeros((m, m), dtype='float64')

    # Diagonalization of the matrix (A^T * A)
    jacobi.diag(np.dot(A.T, A), s, V)

    # Calculate the matrix U
    U = np.dot(A, V)
    for i in range(m):
        s[i] = math.sqrt(s[i])
        U[:, i] /= s[i]

    # Return the decomposition
    return U, s, V


#
# Function to solve a singular value decomposed system
#
def solve(U, s, V, b):
    """
    Finds the least mean square solution to the system (U*S*V^T)*x = b
    from singular value decomposition, where the diagonal elemenst of
    S is kept in the vector s. 

    Returns the solution as a vector (not in-place).

    Specific arguments:
    - `b`: Right-hand side of the system.
    """
    # Calculate the right-hand side of the projected equation
    #   -->  S*V^T*x = U^T*b
    y = np.dot(U.T, b)

    # Solve the diagonal system in place for y  -->  S*y = U^T*b
    for i in range(len(y)):
        y[i] /= s[i]

    # Obtain solution  -->  x = V*y
    x = np.dot(V, y)

    # Return the solution
    return x


#
# Function to evaluate a fit
#
def evalfit(flist, c, x):
    """
    Evaluates a fit of the form \sum{c_i * f_i(x)} at point x.
    """
    return sum([c[i] * flist[i](x) for i in range(len(flist))])
    
