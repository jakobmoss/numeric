# cython: language_level=3

import numpy as np
import auxB as jacobi


#
# Testing routines
#
def test_cyclic(A, d, V):
    """
    Testing cyclic Jacobi routine
    """
    Aorg = np.copy(A)
    
    # Run the test
    print(' -- Testing cyclic Jacobi diagonalization -- ')
    print('A = \n', A)
    rot = jacobi.diag_cyclic(A, d, V)

    print('\nNumber of rotations used = ', rot)
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
    rot = jacobi.diag_eig(A, d, V)

    print('\nNumber of rotations used = ', rot)
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
    rot = jacobi.diag_eig2(A, d, V)

    print('\nNumber of rotations used = ', rot)
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

    
#
# Main function
#
def mainB():
    """
    Test of the routines for diagonalization
    """

    # Test the cyclic algorithm
    A1 = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float64')
    d1 = np.zeros(A1.shape[0], dtype='float64')
    V1 = np.zeros(A1.shape, dtype='float64')
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
