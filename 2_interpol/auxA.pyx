# cython: language_level=3


#
# Function to generate test data
#
def testdata(unsigned int N):
    """
    Generates N points of (x, y)-data to test the different routines
    """
    # Import math functions
    from math import cos, sin

    # Fill the lists with data
    cdef unsigned int i
    cdef list x = [i + 0.5 * sin(i) for i in range(N)]
    cdef list y = [i + cos(i*i) for i in range(N)]

    # Return the data
    return x, y


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


#
# Quadratic spline interpolation
#
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
