###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 09:40:18 moss>
#
# Part A
###########################################

#
# Modules
#

# General
import numpy as np
import numpy.linalg as la

# Homemade routines for iterative determination of eigenvalues
import eigen


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

# Run own algorithm. Pretty print!
print('\n\n*************************************************')
print('** Finding eigenvalues using inverse iteration **')
print('*************************************************')

# Eigenvalue of leat magnitude
iters = 10
print('\nFinding eigenvalue of least magnitude using', iters, 'iterations')
valmin, vecmin = eigen.inviter(A)
print('Estimated eigenvalue  :', valmin)
print('Estimated eigenvector :', vecmin.T)

# Test if different eigenvalues can be found using a shift
print('\n\n-- Searching for different eigenvalues by introducing a shift')
for guess in [-13, -7, -1, 12]:
    val, vec = eigen.inviter(A, iters, guess)
    print('\nFinding eigenvalue near', guess, 'using', iters, 'iterations...')
    print('Eigenvalue by inverse iteration  :', val)
    print('Eigenvector by inverse iteration :', vec.T)
