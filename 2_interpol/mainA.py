from __future__ import print_function
import auxA as intpol


#
# Function to generate test data
#
def testdata(N):
    """
    Generates N points of (x, y)-data to test the different routines
    Arguments:
    - `N`:
    """
    from math import cos, sin
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
    print(intpol.lspline(x, y, z_lin))
    z_lin += step
print('\n')

# Quadratic spline
z_quad = x[0]
quad_spline = intpol.qspline(x, y)
while z_quad < x[-1]:
    print(z_quad, end='\t')
    print(quad_spline(z_quad))
    z_quad += step
