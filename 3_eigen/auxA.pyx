# cython: language_level=3

import numpy as np


#
# Function to calculate the QR-decomposition
#
def decomp(Q, R):
    """
    Calculates the QR-decomposition in-place (i.e. the original matrix is 
    replaced) using stabilized Gram-Schmidt.

    Arguments:
    - `Q`: Matrix of dimension (n x m) to be transformed <type: double>
    - `R`: Empty matrix of dimension (m x m) <type: double>
    """
    # Loop over number of columns
    for i in range(Q.shape[1]):
        qi = Q[:, i]  # Pointer to the i'th column
        R[i, i] = np.sqrt(np.dot(qi, qi))  # Fill R_ii
        qi /= R[i, i]  # Scale by 1/R_ii

        # Normalize to all columns to the right
        for j in range(i+1, Q.shape[1]):
            qj = Q[:, j]  # Pointer to the j'th column
            R[i, j] = np.dot(qi, qj)  # Fill R_ij
            qj -= qi*R[i, j]  # Do the normalization


#
# General function to do back-substitution (used by qr-solve)
#
def backsub(U, b):
    """
    Solves the upper triangular system Ux = b by in-place back-substitution
    Arguments:
    - `U`: Upper triangular matrix
    - `b`: Vector containing the right-hand side -- the solution is stored here
    """
    # Loop backwards through elements
    for i in reversed(range(U.shape[1])):
        s = b[i]  # Temporary storage of i'th entry

        # Subtract each higher element
        for j in range(i+1, U.shape[1]):
            s -= U[i, j] * b[j]

        # Store in original vector
        b[i] = s / U[i, i]
    

#
# Funtion to solve a QR-decomp. system using back-sub
#
def solve(Q, R, b):
    """
    Solves the system (QR)x = b by in-place back-substitution
    Arguments:
    - `Q`: From QR-decomposition
    - `R`: From QR-decomposition
    - `b`: Vector containing the right-hand side -- the solution is stored here
    """
    # Calculate the desired right-hand side
    QTb = np.dot(np.transpose(Q), b)

    # Solve Rx = Q^{T}b by backsubstitution
    backsub(R, QTb)

    # Store into original vector
    for i in range(len(b)):
        b[i] = QTb[i]


#
# Function to calculate the absolute value of a QR-determinant
#
def absdet(R):
    """
    Calculates the absolute value of the determinant of a QR-decomposed matrix 
    using just R (since det(Q)^2 = 1).

    Arguments:
    - `R`: From QR-decomposition (upper-triangular)
    """
    # To store determinant
    value = 1

    # Loop over dimension (quadratic)
    for i in range(len(R)):
        value *= R[i, i]

    # Return value
    return value


#
# Function to calculate the inverse
#
def inverse(Q, R, Ainv):
    """
    Calculates the inverse of the matrix A using the decomposition A = QR

    Arguments:
    - `Q`:
    - `R`:
    - `Ainv`:
    """
    # Initialize A^{-1} as the identity matrix
    for i in range(len(Ainv)):
        Ainv[i, i] = 1

    # Solve for each column using back-substitution
    for i in range(Q.shape[1]):
        solve(Q, R, Ainv[:, i])
