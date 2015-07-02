###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
#
# Time-stamp: <2015-07-02 20:55:23 moss>
#
# Jacobi eigenvalue routines
############################################

# General modules
import math
import numpy as np
import sys


#
# Function to diagonalize eigenvalue by eigenvalue eliminating the
# greatest element first.
#
def diag_eig2(A0, d, V, first='small', halt=False):
    """
    Performs matrix diagonalization (on a real and symmetric matrix)
    using the Jacobi eigenvalue method; solving for one eigenvalue at
    a time. This algorithm zeros out one row at a time, always
    eliminating the greatest off-diagonal element.

    Returns the number of rotations used.

    Arguments:
    - `A`: Real, symmetric input matrix to diagonalize. Upper triangle
           is destroyed.
    - `d`: Empty target vector. The eigenvalues are stored here.
    - `V`: Empty target matrix. The corresponding eigenvectors are
           stored here.
    """
    # Initializations
    A = np.copy(A0)  # Work on a copy
    n = A.shape[0]
    rotations = 0

    # Store all diagonal elements in d and initialize V as the identity matrix
    for i in range(n):
        d[i] = A[i, i]
        V[i, i] = 1

    # BEGIN SWEEPING  -->  Row-by-row
    for p in range(n-1):
        # Testing: Stop after only the first eigenvalue have been found
        if halt and (p > 0):
            print('Jacobi: On purpose, only the first eigenvalue',
                  'have been determined!', file=sys.stderr)
            break

        changed = True

        # Iterate until convergence
        while changed:
            changed = False

            # Locate largest element in row  --> Start with first element
            q = p + 1
            apq = abs(A[p, q])

            # Compare with element to the right and store if greater
            for i in range(p+2, n):
                apix = abs(A[p, i])  # api is a reserved keyword
                if apix > apq:
                    q = i
                    apq = apix

            # Perform the rotation with greatest element as target
            changed, rotations = rotation(A, d, V, n, p, q, changed,
                                          rotations, first, True)

    # END SWEEPING

    # Return the number of rotations used
    return rotations


#
# Function used by the different Jacobi algorithms to perform the rotation
#
def rotation(A, d, V, n, p, q, changed, rotations, first, prev_rows):
    """
    Perform a Jacobi rotation - to be used in the different algorithms

    Specific arguments:
    - `prev_rows`: Whether previous rows are assumed to have been
                   eliminated or not. This can reduce the number of
                   operations required for the non-cyclic methods.
    """
    # Get different entries
    app = d[p]
    aqq = d[q]
    apq = A[p, q]

    # Whether to find smallest or largest eigenvalue first
    if first.lower() in ['small', 'smallest']:
        phi = 0.5 * math.atan2(2*apq, aqq-app)
    elif first.lower() in ['large', 'largest']:
        phi = 0.5 * (math.pi + math.atan2(2*apq, aqq-app))
    else:
        print('Jacobi: Missing flag \'first\'! Aborting!',
              file=sys.stderr)
        exit()

    # Calculate different coefficients to zero out A_pq
    c = math.cos(phi)
    s = math.sin(phi)

    # Calculate the new diagonal elements and compare with the old
    app_new = c*c*app - 2*s*c*apq + s*s*aqq
    aqq_new = s*s*app + 2*s*c*apq + c*c*aqq

    if (app_new != app) or (aqq_new != aqq):
        changed = True
        rotations += 1

        # Update the diagonal-element vector and A_pq
        d[p] = app_new
        d[q] = aqq_new
        A[p, q] = 0

        # Loop over all elements with i != p,q.
        # First part doesn't affect the matrix if eliminating eigenvalue
        # by eigenvalue (i.e. no reason to do it!)
        if not prev_rows:
            for i in range(p):
                aip = A[i, p]
                aiq = A[i, q]
                A[i, p] = c*aip - s*aiq
                A[i, q] = c*aiq + s*aip
        for i in range(p+1, q):
            apix = A[p, i]  # api is a reserved keyword
            aiq = A[i, q]
            A[p, i] = c*apix - s*aiq
            A[i, q] = c*aiq + s*apix
        for i in range(q+1, n):
            apix = A[p, i]
            aqi = A[q, i]
            A[p, i] = c*apix - s*aqi
            A[q, i] = c*aqi + s*apix
        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c*vip - s*viq
            V[i, q] = c*viq + s*vip

    # Return the status
    return changed, rotations
