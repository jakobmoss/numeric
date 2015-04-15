from __future__ import print_function
from math import cos, sin


#
# Linear interpolation
#
def lin_intpol(x, y, z):
    """
    Calculates the linear interpolation in (x, y) at the point z

    Arguments:
    - `x`: X-data as list
    - `y`: Y-data as list
    - `z`: Point of evaluation (along x) as double
    """
    # Initialize counter variables
    i = 0
    j = len(x) - 1

    # Do a binary search
    while j-i > 1:
        m = int((i+j)/2)
        if z > x[m]:
            i = m
        else:
            j = m

    # Return the interpulated point
    s = y[i] + (y[i+1] - y[i])/(x[i+1] - x[i]) * (z - x[i])
    return s


def testdata(N):
    """
    Generates N points of (x, y)-data to test the different routines
    Arguments:
    - `N`:
    """
    x = [i + 0.5 * sin(i) for i in range(N)]
    y = [i + cos(i*i) for i in range(N)]
    return x, y

#
# Main
#

# Generate test data
x, y = testdata(10)

# Print test data for plotting
for i in range(len(x)):
    print(x[i], end='\t')
    print(y[i])
print('\n')

# Initialize useful variables
n = 50
step = (x[-1] - x[0]) / n

# Linear interpolation
z_lin = x[0]
while z_lin < x[-1]:
    print(z_lin, end='\t')
    print(lin_intpol(x, y, z_lin))
    z_lin += step
