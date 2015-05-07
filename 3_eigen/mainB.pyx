# cython: language_level=3

import numpy as np
import auxB as jacobi


#
# Main function
#
def mainB():
    """
    Test of the routines for diagonalization
    """
        # Initialization of test matrices
#    A = np.array([[3, 1, -1], [1, 3, -1], [-1, -1, 5]], dtype='float64')
    A = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2],
                  [10, 9, 2, 3]], dtype='float64')
    Aorg = np.copy(A)
    d = np.zeros(A.shape[0], dtype='float64')
    V = np.zeros(A.shape, dtype='float64')

    # Run the test
    print(' -- Testing Jacobi diagonalization -- ')
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
