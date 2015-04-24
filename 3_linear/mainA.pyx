# cython: language_level=3

import numpy as np
import auxA as qr


#
# Main function
#
def mainA():
    """
    Test of the routines for solving linear equations
    """

    # Double precision test-matrix
#    A = np.array([[-4, -4, 2], [0, 2, 4], [-3, -2, 3]], dtype='float64')
    A = np.array([[4, -6, 3], [3, -5, 8], [5, 4, -7]], dtype='float64')
    R = np.zeros((A.shape[1], A.shape[1]), dtype='float64')

    b = np.array([9, 22, 25], dtype='float64')
    
    print(' -- Before GS --')
    print('A =')
    print(A)
    print('R =')
    print(R)    
    print('b =')
    print(b)

    # Do the decomposition
    qr.decomp(A, R)
    
    print('\n -- After GS --')
    print('Q =')
    print(A)
    print('R =')
    print(R)
    print('QR =')
    print(np.dot(A, R))

    # Solve by back-sub
    qr.solve(A, R, b)

    print('x =')
    print(b)
    print('A x =')
    print(np.dot(np.dot(A, R), b))
