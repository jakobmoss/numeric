###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
#
# Time-stamp: <2015-06-30 16:13:22 moss>
#
# Implementation of the routines
############################################

#
# Modules
#
import numpy as np
import numpy.linalg as la
import givens as qr


#
# Inverse (power) iteration method
#
def inviter(A0, N=10, shift=0):
    """
    Inverse iteration algorithm to determine an eigenvalue and with
    corresponding eigenvector.

    Returns the approximation of the eigenvalue and -vector.

    Arguments:
    - `A0`: Matrix
    - `N`: Number of iterations (default 10)
    - `shift`: Initial guess on eigenvalue (if not set, the routine will
               converge towards the one of lowest magnitude)
    """
    # Work on a copy of the matrix
    A = np.copy(A0)

    # Initialize normalized arbitary vector --> v_0
    v = np.random.random(A.shape[0])
    v /= la.norm(v)

    # Perform a shift?
    if shift:
        I = np.eye(A.shape[0], dtype='float')
        A -= shift * I

    # Make QR-decomposition with Givens's rotation
    qr.decomp(A)

    # BEGIN the iteration
    for k in range(20):
        qr.solve(A, v)    # Solves A v_{k} = v_{k-1} in-place
        v /= la.norm(v)   # Normalize v_k

    # Estimate eigenvalue using the Rayleigh quotient
    lamb = np.dot(np.dot(v, A0), v)

    return lamb, v
