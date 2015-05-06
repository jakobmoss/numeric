# cython: language_level=3

import numpy as np
import math


#
# Function to diagonalize with cyclic sweeps
#
def cyclic(A, d, V):
    """
    Performs matrix diagonalization (on a real and symmetric matrix) using the 
    Jacobi eigenvalue method with cyclic sweeps. Returns the number of rotations
    used.

    Arguments:
    - `A`: Real, symmetric input matrix to diagonalize. Upper triangle is destroyed.
    - `d`: Empty target vector. The eigenvalues are stored here.
    - `V`: Empty target matrix. The corresponding eigenvectors are stored here.
    """
    # Initializations
    n = A.shape[0]
    rotations = 0
    changed = True

    # Store all diagonal elements in d and initialize V as the identity matrix
    for i in range(n):
        d[i] = A[i, i]
        V[i, i] = 1

    # Iterate until convergence
    while changed:
        changed = False

        # BEGIN CYCLIC SWEEP  -->  Loop over columns to the right of diagonal
        for p in range(n):
            for q in range(p+1, n):

                # Perform the rotation
                changed, rotations = rotation(A, d, V, n, p, q, changed, rotations)
                
        # END CYCLIC SWEEP

    # Return the number of rotations used
    return rotations


def rotation(A, d, V, n, p, q, changed, rotations):
    """
    Perform a Jacobi rotation - to be used in the different algorithms
    """
    # Get different entries
    app = d[p]
    aqq = d[q]
    apq = A[p, q]

    # Calculate different coefficients to zero out A_pq
    phi = 0.5 * math.atan2(2*apq, aqq-app)
    c = math.cos(phi)
    s = math.sin(phi)

    # Calculate the new diagonal elements and compare with the old
    app_new = c*c*app - 2*s*c*apq + s*s*aqq
    aqq_new = s*s*app + 2*s*c*apq + c*c*aqq
                
    if (app_new != app) or (aqq_new != aqq) :
        changed = True
        rotations += 1

        # Update the diagonal-element vector and A_pq
        d[p] = app_new
        d[q] = aqq_new
        A[p, q] = 0

        # Loop over all elements with i != p,q
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
