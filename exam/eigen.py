# Modules
import numpy as np
import numpy.linalg as la
import givens as qr


#
# Functions
#
def inviter(A0, shift=0):
    """
    Inverse iteration algorithm to determine eigenvalues and -vectors

    Arguments:
    - `A0`: Matrix
    """
    # Work on a copy of the matrix and initialize identity matrix
    A = np.copy(A0)

    # Initialize normalized arbitary vector
    w = np.random.random(A.shape[0])
    w /= la.norm(w)

    # Perform a shift?
    if shift:
        I = np.eye(A.shape[0], dtype='float')
        A -= shift * I

    # Make QR-decomposition with Givens's rotation
    qr.decomp(A)

    # Run the loop
    for k in range(20):
        v = w             # Update current value
        w = np.copy(v)    # v contains (k-1)'th
        qr.solve(A, w)    # Solve Aw = v  <-->  A x_{k} = x_{k-1}
        w /= la.norm(w)   # w contains k'th

    # Estimate eigenvalue using the Rayleigh quotient
    lamb = np.dot(np.dot(w, A0), w)

    return lamb, v

#
# Run
#

# Test-matrix
A = np.array([[-5, 9, -4, 10], [9, -1, 3, 9], [-4, 3, -6, 2], [10, 9, 2, 3]],
             dtype='float')
print('Test-matrix: A =\n', A)

# Results from NumPy routine
npval, npvec = la.eig(A)
print('\nEigenvalues by NumPy :\n', npval)
print('Eigenvectors NumPy   :\n', npvec)

# Run own alg
val, vec = inviter(A, -7)
print('\nEigenvalue by inverse iteration:', val)
print('\nEigenvector by inverse iteration:', vec.T)
