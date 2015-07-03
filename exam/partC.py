###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-03 09:50:45 moss>
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
import timeit

# Homemade routines for iterative determination of eigenvalues
import eigen

# Homemade routines for finding eigenvalues with Jacobi rotations
import jacobi


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


def __basictest(A):
    """
    Basic test of the inverse iteration algorithm with and without a shift
    Prints to stdout.
    """
    # Pretty print!
    print('\n\n*****************************************************')
    print('** Comparing inverse iteration and Jacobian method **')
    print('*****************************************************')

    # Settings
    eps = 1e-9
    nup = 3

    # Eigenvalue of least magnitude using inverse iteration
    print('\n--Estimating eigenvalue of least magnitude using inverse',
          'iteration with convergence goal of acc = ', eps,
          'updating the estimate every', nup, 'iterations')
    t = timeit.default_timer()
    val, vec, dv, iters = eigen.inviter_acc(A, Nup=nup, acc=eps)
    dt = timeit.default_timer() - t
    print('Used', iters, 'iterations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalue  :', val)
    print('Estimated eigenvector :', vec.T)

    # Full Jacobian method
    print('\n--Estimating all eigenvalues using Jacobian',
          'eigenvalue-by-eigenvalue method')
    d = np.zeros(A.shape[0], dtype='float')
    V = np.zeros(A.shape, dtype='float')
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V, first='large')
    dt = timeit.default_timer() - t
    print('-Finding the largest eigenvalue first used', rot, 'rotations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalues  :', d)
    print('Estimated eigenvectors :\n', V)

    d = np.zeros(A.shape[0], dtype='float')
    V = np.zeros(A.shape, dtype='float')
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V, first='small')
    dt = timeit.default_timer() - t
    print('\n-Finding the smallest eigenvalue first used', rot, 'rotations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalues  :', d)
    print('Estimated eigenvectors :\n', V)

    # Only the largest eigenvalue with Jacobian
    print('\n\n--Estimating the largest eigenvalue using Jacobian',
          'eigenvalue-by-eigenvalue method')
    d = np.zeros(A.shape[0], dtype='float')
    V = np.zeros(A.shape, dtype='float')
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V, first='large', halt=True)
    dt = timeit.default_timer() - t
    print('Finding only the largest eigenvalue used', rot, 'rotations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalue  :', d[0])
    print('Estimated eigenvector :\n', V[:, 0])

    # Eigenvalue of leat magnitude using inverse iteration
    guess = 1e9
    print('\n-Finding the same eigenvalue with inverse iteration',
          '(same settings as above) and a crazy shift = ', guess)
    t = timeit.default_timer()
    val, vec, dv, iters = eigen.inviter_acc(A, shift=guess, Nup=nup,
                                            acc=eps)
    dt = timeit.default_timer() - t
    print('Used', iters, 'iterations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalue  :', val)
    print('Estimated eigenvector :', vec.T)

    # Better guess....
    guess = 25
    print('\n-Inverse iter with a more reasonable guess = ', guess)
    t = timeit.default_timer()
    val, vec, dv, iters = eigen.inviter_acc(A, shift=guess, Nup=nup,
                                            acc=eps)
    dt = timeit.default_timer() - t
    print('Used', iters, 'iterations')
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('Estimated eigenvalue  :', val)
    print('Estimated eigenvector :', vec.T)


def __speedtest(N=10):
    """
    Test the running time of the different methods
    Prints to stdout.
    """
    # Pretty print!
    print('\n\n** Comparing execution time **')

    # Random (real and symmeric) matrix
    print('Using a random, symmetric', N, 'x', N, 'matrix')
    a = np.random.rand(N, N) * 10
    A = (a + a.T)/2

    # NumPy
    t = timeit.default_timer()
    val1, vec1 = la.eig(A)
    dt = timeit.default_timer() - t
    print('\n-NumPy (all eigenvalues) running time  : {0:.4f} ms'.format(100*dt))

    # Jacobian (full)
    d = np.zeros(A.shape[0], dtype='float')
    V = np.zeros(A.shape, dtype='float')
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V, first='large')
    dt = timeit.default_timer() - t
    print('-Jacobi (all eigenvalues) running time : {0:.4f} ms'.format(100*dt))

    # Jacobian (first)
    d = np.zeros(A.shape[0], dtype='float')
    V = np.zeros(A.shape, dtype='float')
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V, first='large', halt=True)
    dt = timeit.default_timer() - t
    print('-Jacobi (only largest ev) running time : {0:.4f} ms'.format(100*dt))

    # Inverse iter (least magnitude)
    t = timeit.default_timer()
    val, vec, dv, iters = eigen.inviter_acc(A, Nup=2, acc=1e-9)
    dt = timeit.default_timer() - t
    print('-Inverse iteration with convergence criterion and update every 2nd')
    print(' iter (only min mag) running time      : {0:.4f} ms'.format(100*dt))

    # Inverse iter (least magnitude) without convergence criterion
    t = timeit.default_timer()
    val, vec = eigen.inviter_up(A, Nup=2, N=5)
    dt = timeit.default_timer() - t
    print('-Inverse iteration with 5 iterations and update every 2nd iter')
    print(' (only min mag) running time           : {0:.4f} ms'.format(100*dt))

    # Inverse iteration with guess
    minev = val1[abs(val1) <= min(abs(val1))]
    t = timeit.default_timer()
    val, vec, dv, iters = eigen.inviter_acc(A, Nup=3, acc=1e-9, shift=minev-.1)
    dt = timeit.default_timer() - t
    print('-Inverse iteration with convergence criterion and update every 3rd')
    print(' iter (with guess) running time        : {0:.4f} ms'.format(100*dt))


#
# Main of Part C
#
def main(**options):
    # Basic test of the functionality
    if options['basic']:
        A, npval, npvec = __initsys()
        __basictest(A)
        if options['speed']:
            __speedtest(15)

    # No options given
    else:
        print('No option given!', file=sys.stderr)
