# cython: language_level=3

import auxB as intpol
from auxB cimport testdata


#
# Main function
#
def mainB():
    """
    Test of the interpolation routines
    """
    
    # Generate test data
    x, y = testdata(10)

    # Print y=0 line for test-data
    print(x[0], end='\t')
    print('0')
    print(x[-1], end='\t')
    print('0')
    print('\n')
    
    # Print test data for plotting
    for i in range(len(x)):
        print(x[i], end='\t')
        print(y[i])
    print('\n')

    # Initialize useful variables
    n = 500
    step = (x[-1] - x[0]) / n

    # Build quadratic spline
    qspline, deriv_qspline, integ_qspline = intpol.qspline(x, y)

    # Evaluate spline
    z = x[0]
    while z < x[-1]:
        print(z, end='\t')
        print(qspline(z))
        z += step
    print('\n')

    # Evaluate derivative
    zz = x[0]
    while zz < x[-1]:
        print(zz, end='\t')
        print(deriv_qspline(zz))
        zz += step
    print('\n')

    # Evaluate integral
    zzz = x[0]
    while zzz < x[-1]:
        print(zzz, end='\t')
        print(integ_qspline(zzz))
        zzz += step
    print('\n')
