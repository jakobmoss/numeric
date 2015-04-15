from __future__ import print_function
from math import cos, sin


#
# Linear interpolation
#
def lspline(x, y, z):
    """
    Calculates the linear interpolation, i.e. spline with linear polynomials,
    in (x, y) at the point z. Returns S(z).

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

    # Return the interpulated point (1.5) + (1.6)
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


def qspline(x, y):
    """
    Calculates the quadratic spline of (x, y). Returns function to evaluate the
    spline at a given point.

    Arguments:
    - `x`: X-data as list
    - `y`: Y-data as list
    """
    # Shorthand notation
    n = len(x)

    # Calculate auxiliary coefficients (1.6)
    dx = [x[i+1] - x[i] for i in range(n-1)]  # \Delta X
    p = [(y[i+1] - y[i])/dx[i] for i in range(n-1)]  # p_i

    # Calculate spline coefficients by forward recursion (1.11)
    c = [0 for i in range(n-1)]
    for i in range(n-2):
        c[i+1] = (p[i+1] - p[i] - c[i]*dx[i]) / dx[i+1]

    # Backward recursion to give the averaged coefficients (1.12)
    c[n-2] /= 2
    for i in reversed(range(n-2)):
        c[i] = (p[i+1] - p[i] - c[i+1]*dx[i+1]) / dx[i]

    # Function to evaluate the spline at point z using the calculated coeff.
    def S(z):
        # Do a binary search (as in the linear interpol)
        i = 0
        j = n - 1
        while j-i > 1:
            m = int((i+j)/2)
            if z > x[m]:
                i = m
            else:
                j = m

        # Return the spline (1.7)
        return y[i] + p[i]*(z - x[i]) + c[i]*(z-x[i])*(z-x[i+1])

    # Return the spline function
    return S


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
    print(lspline(x, y, z_lin))
    z_lin += step
print('\n')

# Quadratic spline
z_quad = x[0]
quad_spline = qspline(x, y)
while z_quad < x[-1]:
    print(z_quad, end='\t')
    print(quad_spline(z_quad))
    z_quad += step
