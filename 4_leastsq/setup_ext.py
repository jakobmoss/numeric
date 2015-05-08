# Needed to build Cython
from distutils.core import setup
from Cython.Build import cythonize

# Change to newest gcc
import os
os.environ['CC'] = 'gcc-4.9'

# Do the build
setup(ext_modules=cythonize(['givens.pyx', 'jacobi.pyx']))
