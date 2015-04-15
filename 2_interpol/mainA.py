from __future__ import print_function
from math import cos, sin


# Test data
x = [i + 0.5 * sin(i) for i in range(10)]
y = [i + cos(i*i) for i in range(10)]

# Print test data for plotting
for i in range(len(x)):
    print(x[i], end='\t')
    print(y[i])
