###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-02 12:59:13 moss>
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


def __convtest(A, exacteigen, Nmax):
    """
    Test of the convergence as a function of the number of iterations
    """
    # Header
    print('# PartB: Data for convergence test')

    # Test the convergence along the way by keeping the initial vector
    vinit = np.random.random(A.shape[0])
    vinit /= la.norm(vinit)

    # Without updating the estimate (making sure the update is never triggered)
    for n in range(Nmax):
        val, vec = eigen.inviter_up(A, N=n, Nup=Nmax+1,
                                    override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(n, diff[0]))

    # New index in Gnuplot
    print('\n')

    # Updating every 5th time
    for n in range(Nmax):
        val, vec = eigen.inviter_up(A, N=n, Nup=5,
                                    override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(n, diff[0]))

    # Updating every 3rd time
    print('\n')
    for n in range(Nmax):
        val, vec = eigen.inviter_up(A, N=n, Nup=3,
                                    override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(n, diff[0]))

    # Updating every 2nd time
    print('\n')
    for n in range(Nmax):
        val, vec = eigen.inviter_up(A, N=n, Nup=2,
                                    override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(n, diff[0]))

    # Updating every time
    print('\n')
    for n in range(Nmax):
        val, vec = eigen.inviter_up(A, N=n, Nup=1,
                                    override=True, v0=vinit)
        diff = abs(exacteigen - val)
        print('{0:3d} {1:17.9e}'.format(n, diff[0]))


def __acctest(A, exacteigen):
    """
    Accurary test of convergence criterion
    """
    # Pretty print!
    print('\n\n********************************************************')
    print('** Testing convergence criterion of inverse iteration **')
    print('********************************************************')

    # Test the convergence criterion
    Ntest = 5
    eps = 1e-9
    print('Running', Ntest, 'test runs with acc =', eps)

    for N in range(Ntest):
        val, vec, dv, iters = eigen.inviter_acc(A, Nup=3, acc=eps)
        diff = abs(exacteigen - val)
        print('\nTest #', N+1)
        print('Found eigenvalue of minimum magnitude in', iters, 'iterations')
        print('Eigenvalue              =', val)
        print('Estimted error          =', dv)
        print('Error compared to NumPy =', diff[0])


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

    # Test of convergence as a function of iterations and Rayleigh updates
    elif options['convergence']:
        A, npval, npvec = __initsys(verbose=False)
        minev = npval[abs(npval) <= min(abs(npval))]
        maxiter = 25
        __convtest(A, minev, maxiter)

    # Test of the convergence criterion
    elif options['criterion']:
        A, npval, npvec = __initsys()
        minev = npval[abs(npval) <= min(abs(npval))]
        __acctest(A, minev)

    # No options given
    else:
        print('No option given!', file=sys.stderr)
