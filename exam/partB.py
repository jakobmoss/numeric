###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 20:34:16 moss>
#
# Part B
###########################################

#
# Modules
#

# General
import sys
import numpy as np
import numpy.linalg as la

# Homemade routines for iterative determination of eigenvalues
import eigen


#
# Internal auxiliary functions for organizing the testing
#
def __initsys(verbose=True):
    """
    Initialize testing matrix and NumPy eigenvalues- and vectors.
    Prints to stdout if verbose flag is not unset
    """
    # Test-matrix
    A = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float')

    # Results from NumPy routine
    npval, npvec = la.eig(A)

    # Verbose output?
    if verbose:
            print('Test-matrix: A =\n', A)
            print('\nEigenvalues by NumPy :\n', npval)
            print('Eigenvectors NumPy   :\n', npvec)

    # Return
    return A, npval, npvec


def __basictest(A, iters, guesses):
    """
    Basic test of the inverse iteration algorithm with and without a shift
    Prints to stdout.
    """
    # Pretty print!
    print('\n\n*************************************************')
    print('** Finding eigenvalues using inverse iteration **')
    print('*************************************************')

    # Eigenvalue of leat magnitude
    iters = 10
    print('\nFinding eigenvalue of least magnitude using', iters, 'iterations')
    valmin, vecmin = eigen.inviter_up(A)
    print('Estimated eigenvalue  :', valmin)
    print('Estimated eigenvector :', vecmin.T)

    # Test if different eigenvalues can be found using a shift
    print('\n\n-- Searching for different eigenvalues by introducing a shift')
    for guess in guesses:
        val, vec = eigen.inviter_up(A, N=iters, shift=guess)
        print('\nFinding eigenvalue near', guess, 'using', iters,
              'iterations...')
        print('Eigenvalue by inverse iteration  :', val)
        print('Eigenvector by inverse iteration :', vec.T)


#
# Main of Part B
#
def main(**options):
    # Basic test of the functionality
    if options['basic']:
        A, npval, npvec = __initsys()
        Niter = 10
        eigenguess = [-13, -7, -1, 12]
        __basictest(A, Niter, eigenguess)
