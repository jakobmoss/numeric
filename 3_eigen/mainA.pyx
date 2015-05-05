# cython: language_level=3

import numpy as np
import auxA as jacobi


#
# Main function
#
def mainA():
    """
    Test of the routines for diagonalization
    """
    # Initialization of test matrices
    A = np.array([[3, 1, -1], [1, 3, -1], [-1, -1, 5]], dtype='float64')
    d = np.zeros(A.shape[0], dtype='float64')
    V = np.zeros(A.shape, dtype='float64')

    # Run the test
    print(' -- Testing Jacobi diagonalization -- ')
    print('A = \n', A)
    rot = jacobi.diag(A, d, V)

    print('Number of rotations = \n', rot)
