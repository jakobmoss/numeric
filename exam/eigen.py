# Modules
import numpy as np
import numpy.linalg as la
import givens as qr


#
# Functions
#
def inviter(A0):
    """
    Inverse iteration algorithm to determine eigenvalues and -vectors

    Arguments:
    - `A0`: Matrix
    """
    # Work on a copy of the matrix
    A = np.copy(A0)

    # Initialize normalized arbitary vector
    v = np.random.random(A.shape[0])
    v /= la.norm(v)

    # Decompose and solve
    for k in range(2):
        qr.decomp(A)
        qr.solve(A, v)
    v /= la.norm(v)

    return v

#
# Run
#

# Test-matrix
A = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2], [10, 9, 2, 3]],
             dtype='float64')
print('Test-matrix: A =\n', A)

# Results from NumPy routine
npval, npvec = la.eig(A)
print('\nEigenvalues by NumPy :\n', npval)
print('Eigenvectors NumPy   :\n', npvec)

# Run own alg
vec = inviter(A)
print('\nEigenvector by inverse iteration:', vec.T)
print('\nHas it changed?: A =\n', A)
