# cython: language_level=3

# General modules
import numpy as np
import math

# Module with Jacobi diagonalization
import jacobi

#
# Function to fit using singular value decomposition
#



#
# Function to do singular value decomposition
#
def singular(A):
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
    U = np.zeros(A.shape, dtype='float64')
    s = np.zeros(A.shape[0], dtype='float64')
    V = np.zeros(A.shape, dtype='float64')

    # Diagonalization of the matrix (A^T * A)
    jacobi.diag(np.dot(A.T, A), s, V)

    # Calculate the matrix U
    U = np.dot(A, V)
    for i in range(A.shape[1]):
        s[i] = math.sqrt(s[i])
        U[i] /= s[i]

    # Return the decomposition
    return U, s, V


#
# Function to solve a singular value decomposed system
#
def singular_solve(U, s, V, b):
    """
    Finds the least mean square solution to the system (U*S*V^T)*x = b
    from singular value decomposition, where the diagonal elemenst of
    S is kept in the vector s. 

    Returns

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
    x = np.dot[V, y]

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
    
