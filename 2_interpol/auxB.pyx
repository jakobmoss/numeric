# cython: language_level=3


#
# Function to generate test data
#
cdef tuple testdata(unsigned int N):
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
# Quadratic spline interpolation
#
def qspline(x, y):
    """
    Calculates the quadratic spline of (x, y). Returns functions to evaluate the
    spline, to evaluate the derivative and to evaluate the integral.

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

    # Calculate the extra coefficients for the ordinary spline (1.13)
    b = [p[i] - c[i]*dx[i] for i in range(n-1)]

    
    # Function to evaluate the optimized spline at point z (1.7)
    def spline(z):
        i = 0
        j = n - 1
        while j-i > 1:
            m = int((i+j)/2)
            if z > x[m]:
                i = m
            else:
                j = m

        # Return evaluation
        return y[i] + p[i]*(z - x[i]) + c[i]*(z-x[i])*(z-x[i+1])

    
    # Function to estimate the derivative of the ordinary spline at point z (1.13)
    def deriv_spline(z):
        i = 0
        j = n - 1
        while j-i > 1:
            m = int((i+j)/2)
            if z > x[m]:
                i = m
            else:
                j = m

        # Return evaluation
        return b[i] + 2*c[i]*(z - x[i])


    # Function to estimate the integral from x0 to z of the spline
    def integral_spline(z):
        i = 0
        j = n - 1
        while j-i > 1:
            m = int((i+j)/2)
            if z > x[m]:
                i = m
            else:
                j = m

        # Integration up till the interval found by binary search
        sum_int = 0
        for k in range(i):
            dz = x[k+1] - x[k]  # Shorthand notation
            sum_int += dz * (y[k] + dz * (b[k]/2.0 + dz*c[k]/3.0)) # Clever way to express it

        # Calculate final step and return evaluation
        dz = z - x[i]
        sum_int += dz * (y[i] + dz * (b[i]/2.0 + dz*c[i]/3.0))
        return sum_int


    # Return the spline functions
    return spline, deriv_spline, integral_spline
