###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
#
# Time-stamp: <2015-07-02 13:01:59 moss>
#
# Implementation of the routines
############################################

# Import general modules
import sys
import numpy as np
import numpy.linalg as la

# Import home-made module containing QR-routines using Given's rotation
import givens as qr


#
# Inverse (power) iteration method
#
def inviter(A0, N=10, shift=0, override=False, v0=0):
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
    - `override`: If True, the initial vector will not be generated by this
                  routine, but has to be supplied by the user (for testing)
    - `v0`: Only if override=True. The initial non-zero vector.
    """
    # Work on a copy of the matrix
    A = np.copy(A0)

    # Initialize normalized arbitary vector --> v_0 (or use the user-supplied)
    if not override:
        v = np.random.random(A.shape[0])
        v /= la.norm(v)
    else:
        v = np.copy(v0)

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


#
# Inverse (power) iteration method with updating estimates
#
def inviter_up(A0, N=10, shift=0, Nup=2, override=False, v0=0):
    """
    Inverse iteration algorithm to determine an eigenvalue and with
    corresponding eigenvector. Uses Given's rotation for QR-decomposition and
    backsubstitution. Updates current eigenvalue estimate for (hopefully)
    faster convergence.

    Returns the approximation of the eigenvalue and -vector.

    Arguments:
    - `A0`: Matrix
    - `N`: Number of iterations (default 10)
    - `shift`: Initial guess on eigenvalue (if not set, the routine will
               converge towards the one of lowest magnitude)
    - `Nup`: The eigenvalue estimate will be updated every Nup iterations
    - `override`: If True, the initial vector will not be generated by this
                  routine, but has to be supplied by the user (for testing)
    - `v0`: Only if override=True. The initial non-zero vector.
    """
    # Work on a copy of the matrix and initialize identity matrix
    A = np.copy(A0)
    I = np.eye(A.shape[0], dtype='float')

    # Initialize normalized arbitary vector --> v_0 (or use the user-supplied)
    if not override:
        v = np.random.random(A.shape[0])
        v /= la.norm(v)
    else:
        v = np.copy(v0)

    # Perform a shift?
    if shift:
        A -= shift * I

    # BEGIN  -->  INVERSE ITERATION
    # In the first iteration we always want to QR-decompose
    changed = True

    # Do the iteration. Everything done in-place.
    for k in range(N):
        # If the matrix has changed: Make in-place QR-decomposition
        if changed:
            qr.decomp(A)
            changed = False

        # Solve using the decomposed matrix
        qr.solve(A, v)    # Solves A v_{k} = v_{k-1} in-place
        v /= la.norm(v)   # Normalize v_k

        # Every `Nup` iterations: Update the estimate of the eigenvalue
        # using the Rayleigh quotient
        if (k % Nup) == (Nup - 1):
            rlamb = np.dot(np.dot(v, A0), v)
            A = A0 - rlamb*I
            changed = True

    # Make final estimate of the eigenvalue using the Rayleigh quotient
    lamb = np.dot(np.dot(v, A0), v)
    # END  -->  INVERSE ITERATION

    # Return current estimate of eigenvalue and -vector
    return lamb, v


#
# Inverse (power) iteration method with updating estimates and acc. goal
#
def inviter_acc(A0, acc=1e-9, shift=0, Nup=2, override=False, v0=0):
    """
    Inverse iteration algorithm to determine an eigenvalue and with
    corresponding eigenvector. Uses Given's rotation for QR-decomposition and
    backsubstitution. Updates current eigenvalue estimate for (hopefully)
    faster convergence. Stops when desired accuracy is reached.

    Returns the approximation of the eigenvalue and -vector, the estimated
    error from the convergence criterion, and the number of iterations used.

    Arguments:
    - `A0`: Matrix
    - `acc`: Desired accuracy (default 1e-12)
    - `shift`: Initial guess on eigenvalue (if not set, the routine will
               converge towards the one of lowest magnitude)
    - `Nup`: The eigenvalue estimate will be updated every Nup iterations
    - `override`: If True, the initial vector will not be generated by this
                  routine, but has to be supplied by the user (for testing)
    - `v0`: Only if override=True. The initial non-zero vector.
    """
    # Maximum number of iterations allowed
    NMAX = 25

    # Work on a copy of the matrix and initialize identity matrix
    A = np.copy(A0)
    I = np.eye(A.shape[0], dtype='float')

    # Initialize normalized arbitary vector --> v_0 (or use the user-supplied)
    if not override:
        v = np.random.random(A.shape[0])
        v /= la.norm(v)
    else:
        v = np.copy(v0)

    # Perform a shift?
    if shift:
        A -= shift * I

    # BEGIN  -->  INVERSE ITERATION
    # In the first iteration we always want to QR-decompose
    changed = True

    # Do the iteration. Everything done in-place.
    for k in range(NMAX):
        # For convergence criterion -->  w = v_{k-1}
        w = np.copy(v)

        # If the matrix has changed: Make in-place QR-decomposition
        if changed:
            qr.decomp(A)
            changed = False

        # Solve using the decomposed matrix
        qr.solve(A, v)    # Solves A v_{k} = v_{k-1} in-place
        v /= la.norm(v)   # Normalize v_k

        # Every `Nup` iterations: Update the estimate of the eigenvalue
        # using the Rayleigh quotient
        if (k % Nup) == (Nup - 1):
            rlamb = np.dot(np.dot(v, A0), v)
            A = A0 - rlamb*I
            changed = True

        # Convergence criterion: How much has the Rayleigh estimate changed
        #  + Minimum one update of the estimate should be performed
        dv = abs(abs(np.dot(np.dot(v, A0), v)) - abs(np.dot(np.dot(w, A0), w)))
        if (dv < acc) and (k > Nup):
            converged = True
            iters = k+1
            break

    # Make final estimate of the eigenvalue using the Rayleigh quotient
    lamb = np.dot(np.dot(v, A0), v)
    # END  -->  INVERSE ITERATION

    # Flag a warning if the criterion is not met!
    if not converged:
        print('INVITER: Did not converge in', NMAX, 'iterations!',
              file=sys.stderr)
        iters = -1

    # Return current estimate of eigenvalue and -vector
    return lamb, v, dv, iters
