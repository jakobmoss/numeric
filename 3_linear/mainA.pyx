# cython: language_level=3

import numpy as np
import auxA as qr

#
# Function to test the routines on a square matrix
#
def test(A, b):
    # Storage arrays
    R = np.zeros((A.shape[1], A.shape[1]), dtype='float64')
    Ainv = np.zeros(A.shape, dtype='float64')

    print(' -- Testing on --')
    print('A =')
    print(A)
    print('b =')
    print(b)

    # Do the decomposition
    qr.decomp(A, R)
    
    print('\n -- After Gram-Schmidt --')
    print('Q =')
    print(A)
    print('R =')
    print(R)
    print('QR =')
    print(np.dot(A, R))

    # Solve by back-sub
    qr.solve(A, R, b)

    print('\n -- After back-substitution --')
    print('x =')
    print(b)
    print('A x =')
    print(np.dot(np.dot(A, R), b))

    # Calculate absolute value of A's determinant
    det = qr.absdet(R)
    print('\n -- Determinant and inverse --')
    print('|det(A)| =', det, '\n')

    # Calculate the inverse of A
    qr.inverse(A, R, Ainv)
    print('A^{-1} =')
    print(Ainv)
    print('A A^{-1} =')
    print(np.dot(np.dot(A, R), Ainv))


#
# Function to test the Gram-Schmidt on a long matrix
#
def test_long(A):
    # Storage arrays
    R = np.zeros((A.shape[1], A.shape[1]), dtype='float64')

    print(' -- Testing on --')
    print('A =')
    print(A)

    # Do the decomposition
    qr.decomp(A, R)
    
    print('\n -- After Gram-Schmidt --')
    print('Q =')
    print(A)
    print('R =')
    print(R)
    print('QR =')
    print(np.dot(A, R))

#
# Main function
#
def mainA():
    """
    Test of the routines for solving linear equations
    """
    # Test 1
    A = np.array([[-4, -4, 2], [0, 2, 4], [-3, -2, 3]], dtype='float64')
    b = np.array([9, 22, 25], dtype='float64')
    test(A, b)

    # Test 2
    print('\n------------\n')
    AA = np.array([[3, 4, 5], [-8, 10, 7], [-5, -6, 0], [-4, 0, 4]], dtype='float64')
    test_long(AA)
