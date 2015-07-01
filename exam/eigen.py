###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
#
# Time-stamp: <2015-07-01 11:39:03 moss>
#
# Implementation of the routines
############################################

# Import general modules
import numpy as np
import numpy.linalg as la

# Import home-made module containing QR-routines using Given's rotation
import givens as qr


#
# Inverse (power) iteration method
#
def inviter(A0, N=10, shift=0):
    """
    Inverse iteration algorithm to determine an eigenvalue and with
    corresponding eigenvector. Uses Given's rotation for QR-decomposition and
    backsubstitution.

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

    # BEGIN  -->  INVERSE ITERATION
    # Make in-place QR-decomposition using Givens's rotation
    qr.decomp(A)

    # Do the iteration. Everything done in-place.
    for k in range(N):
        qr.solve(A, v)    # Solves A v_{k} = v_{k-1} in-place
        v /= la.norm(v)   # Normalize v_k
    # Estimate eigenvalue using the Rayleigh quotient
    lamb = np.dot(np.dot(v, A0), v)
    # END  -->  INVERSE ITERATION

    # Return current estimate of eigenvalue and -vector
    return lamb, v
