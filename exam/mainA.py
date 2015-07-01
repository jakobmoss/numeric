###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 10:53:00 moss>
#
# Part A
###########################################

#
# Modules
#

# General
import sys
import numpy as np
import numpy.linalg as la

# For passing options
import argparse

# Homemade routines for iterative determination of eigenvalues
import eigen


#
# Auxiliary functions for organizing the testing
#
def initsys():
    """
    Initialize testing matrix and NumPy eigenvalues- and vectors.
    Prints to stdout.
    """
    # Test-matrix
    A = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float')
    print('Test-matrix: A =\n', A)

    # Results from NumPy routine
    npval, npvec = la.eig(A)
    print('\nEigenvalues by NumPy :\n', npval)
    print('Eigenvectors NumPy   :\n', npvec)

    # Return
    return A, npval, npvec


def basictest(A, iters, guesses):
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
    valmin, vecmin = eigen.inviter(A)
    print('Estimated eigenvalue  :', valmin)
    print('Estimated eigenvector :', vecmin.T)

    # Test if different eigenvalues can be found using a shift
    print('\n\n-- Searching for different eigenvalues by introducing a shift')
    for guess in guesses:
        val, vec = eigen.inviter(A, iters, guess)
        print('\nFinding eigenvalue near', guess, 'using', iters,
              'iterations...')
        print('Eigenvalue by inverse iteration  :', val)
        print('Eigenvector by inverse iteration :', vec.T)


#
# Main of Part A
#
def mainA(**options):
    # Get testing-matrix and eigens determined by NumPy
    A, npval, npvec = initsys()

    # Basic test of the algorithm
    if options['basic']:
        Niter = 10
        eigenguess = [-13, -7, -1, 12]
        basictest(A, Niter, eigenguess)

    # Test of convergence as a function of iterations
    elif options['convergence']:
        print('Option works', file=sys.stderr)

    # No options given
    else:
        print('No option given!', file=sys.stderr)


#
# Parsing of arguments to the script
#
def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Inverse iteration method')

    # Add and gather keywords
    parser.add_argument('--basic', action='store_true',
                        help='Basic test of the algorithm')
    parser.add_argument('--convergence', action='store_true',
                        help='Test of the convergence rate')
    args = parser.parse_args()

    # Run the real main-function with the parsed options
    mainA(**vars(args))

#
# If the file is called directly: Run the main!
#
if __name__ == '__main__':
    main()
