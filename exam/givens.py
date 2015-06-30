# Modules
import numpy as np
import math


#
# Function to calculate the QR-decomposition
#
def decomp(A):
    """
    Calculates the QR-decomposition in-place (i.e. the original matrix is
    replaced) using Given's Rotation.

    R is stored in the upper-triangular part of the matrix, while the Given's
    rotations are stored in the lower part. Q is not built explicitly.

    Arguments:
    - `A`: Matrix of dimension (n x m) to be transformed <type: double>
    """
    # Dimensions of matrix
    rows = A.shape[0]
    cols = A.shape[1]

    # Loop over columns
    for q in range(cols):

        # Loop over rows under the diagonal
        for p in range(q+1, rows):
            theta = math.atan2(A[p, q], A[q, q])  # To eliminate A_pq

            # Loop over columns from q and to the right
            for k in range(q, cols):
                xq = A[q, k]
                xp = A[p, k]
                A[q, k] = xq*math.cos(theta) + xp*math.sin(theta)
                A[p, k] = -xq*math.sin(theta) + xp*math.cos(theta)

            # Store theta in zeroed element
            A[p, q] = theta


#
# Function to rotate vector (used to solve by backsub)
#
def rotate_vec(QR, b):
    """
    Multiply the vector b by the Given's matrix G = Q^T to transform Ax = b
    into Rx = Gb, which can be solved.

    Arguments:
    - `QR`: Resulting matrix from Given's decomposition
    - `b`: Vector. The result is stored here.
    """
    # Dimensions of matrix
    rows = QR.shape[0]
    cols = QR.shape[1]

    # Loop over all rows below the diagonal (i.e. recover the rotations)
    for q in range(cols):
        for p in range(q+1, rows):
            theta = QR[p, q]

            # Get corresponding entries in b
            bq = b[q]
            bp = b[p]

            # Apply on b
            b[q] = bq*math.cos(theta) + bp*math.sin(theta)
            b[p] = -bq*math.sin(theta) + bp*math.cos(theta)


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
def solve(QR, b):
    """
    Solves the system (QR)x = b by in-place back-substitution
    Arguments:
    - `QR`: From Given's QR-decomposition
    - `b`: Vector containing the right-hand side -- the solution is stored here
    """
    # Apply Given's rotations to b
    rotate_vec(QR, b)

    # Solve Rx = Gb by backsubstitution
    backsub(QR, b)


#
# Function to calculate the of a QR-determinant
#
def det(QR):
    """
    Calculates the value of the determinant of a Given's QR-decomposed matrix

    Arguments:
    - `QR`: From Given's-decomposition
    """
    # To store determinant
    value = 1

    # Loop over diagonal dimension (i.e. columns)
    for i in range(QR.shape[1]):
        value *= QR[i, i]

    # Return value
    return value


#
# Function to calculate the inverse
#
def inverse(QR, Ainv):
    """
    Calculates the inverse of the matrix A using the decomposition A = QR

    Arguments:
    - `QR`: From Given's decomposition
    - `Ainv`: Empty matrix (where the inverse is stored)
    """
    # Initialize A^{-1} as the identity matrix
    for i in range(len(Ainv)):
        Ainv[i, i] = 1

    # Solve for each column using back-substitution
    for i in range(QR.shape[1]):
        solve(QR, Ainv[:, i])


#
# Function to explictly build the matrix R
#
def build_r(QR):
    """
    Builds and returns the matrix R from the Given's decomposition

    Arguments:
    - `QR`: From Given's decomposition
    """
    # Initialization
    n = QR.shape[1]
    R = np.zeros((n, n), dtype='float64')

    # Build!
    for i in range(n):
        for j in range(i+1):
            R[j, i] = QR[j, i]

    # Return
    return R


#
# Function to explictly build the matrix Q
#
def build_q(QR):
    """
    Builds and returns the matrix Q from the Given's decomposition

    Arguments:
    - `QR`: From Given's decomposition
    """
    # Initialization
    Q = np.zeros(QR.shape, dtype='float64')

    # Build!
    for i in range(QR.shape[0]):

        # Unit vector is created and the Given's rotation applied
        ei = np.zeros(QR.shape[0], dtype='float64')
        ei[i] = 1
        rotate_vec(QR, ei)

        for j in range(QR.shape[1]):
            Q[i, j] = ei[j]

    # Return
    return Q
