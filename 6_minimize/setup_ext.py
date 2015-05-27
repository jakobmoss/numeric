# Needed to build Cython
from distutils.core import setup
from Cython.Build import cythonize

# Change to newest gcc
import os
os.environ['CC'] = 'gcc-5'

# Do the build
setup(ext_modules=cythonize('givens.pyx'))
setup(ext_modules=cythonize('datafun.pyx'))
setup(ext_modules=cythonize('globvar.pyx'))
