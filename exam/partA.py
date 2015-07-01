###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 13:47:01 moss>
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


def __convtest(A, exacteigen, Nmax):
    """
    Test of the convergence as a function of the number of iterations
    """
    # Header
    print('# PartA: Data for convergence test')

    # Test the different number of iterations
    for N in range(Nmax):
        val, vec = eigen.inviter(A, N)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(N, diff[0]))

    # Test the convergence along the way by keeping the initial vector
    vinit = np.random.random(A.shape[0])
    vinit /= la.norm(vinit)
    print('\n')  # New index in Gnuplot
    for N in range(Nmax):
        val, vec = eigen.inviter(A, N, override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(N, diff[0]))


#
# Main of Part A
#
def main(**options):
    # Basic test of the algorithm
    if options['basic']:
        A, npval, npvec = __initsys()  # Testing-matrix and eigens from NumPy
        Niter = 10
        eigenguess = [-13, -7, -1, 12]
        __basictest(A, Niter, eigenguess)

    # Test of convergence as a function of iterations
    elif options['convergence']:
        A, npval, npvec = __initsys(verbose=False)
        minev = npval[abs(npval) <= min(abs(npval))]  # Eigval of min magnitude
        maxiter = 25
        __convtest(A, minev, maxiter)

    # No options given
    else:
        print('No option given!', file=sys.stderr)


#
# If the file is called directly: Run the main!
#
if __name__ == '__main__':
    main()
