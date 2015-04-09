# Needed to build Cython
from distutils.core import setup
from Cython.Build import cythonize

# Change to newest gcc
import os
os.environ['CC'] = 'gcc-4.9'

# Do the builds
setup(ext_modules=cythonize('helloB.pyx'))
setup(ext_modules=cythonize('userB.pyx'))
