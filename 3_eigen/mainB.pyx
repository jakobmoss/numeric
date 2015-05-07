# cython: language_level=3

import numpy as np
import auxB as jacobi
import timeit


#
# Testing routines
#
def test_cyclic(A, d, V):
    """
    Testing cyclic Jacobi routine
    """
    Aorg = np.copy(A)
    
    # Run the test
    print('\n\n -- Testing cyclic Jacobi diagonalization -- ')
    print('A = \n', A)
    t = timeit.default_timer()
    rot = jacobi.diag_cyclic(A, d, V)
    dt = timeit.default_timer() - t

    print('\nNumber of rotations used = ', rot)
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('\nEigenvalues of A :\nd = \n', d)
    print('\nEigenvectors of A:\nV = \n', V)

    print('\nOrthogonality of eigenvectors:')
    for i in range(len(A)):
        for j in range(len(A)):
            forprint = 'V_' + str(i) + ' * V_' + str(j) + ' ='
            print(forprint, np.dot(V[:,i], V[:,j]))

    print('\nCheck of eigenvalues and -vectors:')
    for i in range(len(A)):
        forprint1 = 'A * V_' + str(i) + ' =\n'
        forprint2 = 'd_' + str(i) + ' * V_' + str(i) + '=\n'
        print(forprint1, np.dot(Aorg, V[:, i]))
        print(forprint2, np.dot(d[i], V[:, i]))

    print('\nCheck of diagonalization:\nV^T A V =')
    print(np.dot(np.dot(np.transpose(V), Aorg), V))

    
def test_row(A, d, V):
    """
    Testing eigenvalue-by-eigenvalue Jacobi routine
    """
    Aorg = np.copy(A)
    
    # Run the test
    print('\n\n -- Testing row-by-row Jacobi diagonalization -- ')
    print('A = \n', A)
    t = timeit.default_timer()
    rot = jacobi.diag_eig(A, d, V)
    dt = timeit.default_timer() - t

    print('\nNumber of rotations used = ', rot)
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('\nEigenvalues of A :\nd = \n', d)
    print('\nEigenvectors of A:\nV = \n', V)

    print('\nOrthogonality of eigenvectors:')
    for i in range(len(A)):
        for j in range(len(A)):
            forprint = 'V_' + str(i) + ' * V_' + str(j) + ' ='
            print(forprint, np.dot(V[:,i], V[:,j]))

    print('\nCheck of eigenvalues and -vectors:')
    for i in range(len(A)):
        forprint1 = 'A * V_' + str(i) + ' =\n'
        forprint2 = 'd_' + str(i) + ' * V_' + str(i) + '=\n'
        print(forprint1, np.dot(Aorg, V[:, i]))
        print(forprint2, np.dot(d[i], V[:, i]))

    print('\nCheck of diagonalization:\nV^T A V =')
    print(np.dot(np.dot(np.transpose(V), Aorg), V))


def test_great(A, d, V):
    """
    Testing 'row-by-row with greatest element first' Jacobi routines
    """
    Aorg = np.copy(A)
    
    # Run the test
    print('\n\n -- Testing greatst element Jacobi diagonalization -- ')
    print('A = \n', A)
    t = timeit.default_timer()
    rot = jacobi.diag_eig2(A, d, V)
    dt = timeit.default_timer() - t

    print('\nNumber of rotations used = ', rot)
    print('Running time: {0:.4f} ms'.format(100*dt))
    print('\nEigenvalues of A :\nd = \n', d)
    print('\nEigenvectors of A:\nV = \n', V)

    print('\nOrthogonality of eigenvectors:')
    for i in range(len(A)):
        for j in range(len(A)):
            forprint = 'V_' + str(i) + ' * V_' + str(j) + ' ='
            print(forprint, np.dot(V[:,i], V[:,j]))

    print('\nCheck of eigenvalues and -vectors:')
    for i in range(len(A)):
        forprint1 = 'A * V_' + str(i) + ' =\n'
        forprint2 = 'd_' + str(i) + ' * V_' + str(i) + '=\n'
        print(forprint1, np.dot(Aorg, V[:, i]))
        print(forprint2, np.dot(d[i], V[:, i]))

    print('\nCheck of diagonalization:\nV^T A V =')
    print(np.dot(np.dot(np.transpose(V), Aorg), V))


def speedtest():
    """
    Testing the speed and number of rotations used by the different
    algorithms
    """
    print(' -- Speed test of the Jacobi routines --')

    # Generate symmetric 6 x 6 random matrix for testing
    a = np.random.rand(6, 6) * 10
    A = (a + a.T)/2
    print('\nTesting on sym. rand. 6 x 6 matrix:\nA =\n', A)

    # Cyclic
    print('\n * Cyclic method:')
    A1 = np.copy(A)
    d1 = np.zeros(A.shape[0], dtype='float64')
    V1 = np.zeros(A.shape, dtype='float64')
    t1 = timeit.default_timer()
    rot1 = jacobi.diag_cyclic(A1, d1, V1)
    dt1 = timeit.default_timer() - t1
    print('Number of rotations used = ', rot1)
    print('Running time: {0:.4f} ms'.format(100*dt1))

    # Row-by-row
    print('\n * Row-by-row method:')
    A2 = np.copy(A)
    d2 = np.zeros(A.shape[0], dtype='float64')
    V2 = np.zeros(A.shape, dtype='float64')
    t2 = timeit.default_timer()
    rot2 = jacobi.diag_eig(A2, d2, V2)
    dt2 = timeit.default_timer() - t2
    print('Number of rotations used = ', rot2)
    print('Running time: {0:.4f} ms'.format(100*dt2))

    # Row-by-row with greatest element first
    print('\n * Greatest-element method:')
    A3 = np.copy(A)
    d3 = np.zeros(A.shape[0], dtype='float64')
    V3 = np.zeros(A.shape, dtype='float64')
    t3 = timeit.default_timer()
    rot3 = jacobi.diag_eig2(A3, d3, V3)
    dt3 = timeit.default_timer() - t3
    print('Number of rotations used = ', rot3)
    print('Running time: {0:.4f} ms'.format(100*dt3))


    # We are done!
    print('\n----------------------------------------')


#
# Main function
#
def mainB():
    """
    Test of the routines for diagonalization
    """

    # General test
    speedtest()
    
    # Test the cyclic algorithm
    A1 = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float64')
    d1 = np.zeros(A1.shape[0], dtype='float64')
    V1 = np.zeros(A1.shape, dtype='float64')
    t1 = timeit.default_timer()
    test_cyclic(A1, d1, V1)

    # Test the row-by-row algorithm
    A2 = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float64')
    d2 = np.zeros(A2.shape[0], dtype='float64')
    V2 = np.zeros(A2.shape, dtype='float64')
    test_row(A2, d2, V2)

    # Test the greatest element algorithm
    A3 = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float64')
    d3 = np.zeros(A3.shape[0], dtype='float64')
    V3 = np.zeros(A3.shape, dtype='float64')
    test_great(A3, d3, V3)
