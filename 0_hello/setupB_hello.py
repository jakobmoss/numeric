# Needed to build Cython
from distutils.core import setup
from Cython.Build import cythonize

# Change to newest gcc on Darwin
import os
from sys import platform
if platform.lower() == 'darwin':
    os.environ['CC'] = 'gcc-5'

# Do the builds
setup(ext_modules=cythonize('helloB.pyx'))
