# cython: language_level=3

import numpy as np


#
# Function to calculate the QR-decomposition
#
def decomp(Q, R):
    """
    Calculates the QR-decomposition in-place (i.e. the original matrix is 
    replaced) using stabilized Gram-Schmidt.

    Arguments:
    - `Q`: Matrix of dimension (n x m) to be transformed <type: double>
    - `R`: Empty matrix of dimension (m x m) <type: double>
    """
    # Loop over number of columns
    for i in range(Q.shape[1]):
        qi = Q[:, i]  # Pointer to the i'th column
        R[i, i] = np.sqrt(np.dot(qi, qi))  # Fill R_ii
        qi /= R[i, i]  # Scale by 1/R_ii

        # Normalize to all columns to the right
        for j in range(i+1, Q.shape[1]):
            qj = Q[:, j]  # Pointer to the j'th column
            R[i, j] = np.dot(qi, qj)  # Fill R_ij
            qj -= qi*R[i, j]  # Do the normalization
