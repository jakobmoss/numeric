# cython: language_level=3

import numpy as np
import auxB as givens

#
# Function to test the Givens's rotation routines on a square matrix
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

    # Make a copy of the original matrix
    orgA = np.copy(A)
    
    # Do the decomposition
    givens.decomp(A)
    
    print('\n -- After Givens rotation --')
    print('QR =')
    print(A)

    # Solve by back-sub
    givens.solve(A, b)

    print('\n -- After back-substitution --')
    print('x =')
    print(b)
    print('A x =')
    print(np.dot(orgA, b))

    # Calculate value of A's determinant
    det = givens.det(A)
    print('\n -- Determinant and inverse --')
    print('|det(A)| =', det, '\n')

    # Calculate the inverse of A
    givens.inverse(A, Ainv)
    print('A^{-1} =')
    print(Ainv)
    print('A A^{-1} =')
    print(np.dot(orgA, Ainv))


#
# Function to test Given's rotation on a long matrix
#
def test_long(A):
    # Storage arrays
    R = np.zeros((A.shape[1], A.shape[1]), dtype='float64')

    print(' -- Testing on long matrix --')
    print('A =')
    print(A)

    # Do the decomposition
    givens.decomp(A)
    
    # print('\n -- After Given's rotation --')
    print('QR =')
    print(A)


#
# Main function
#
def mainB():
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
