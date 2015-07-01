###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 10:51:40 moss>
#
# Makefile
###########################################

# Default target
default: A
clean: cleanA clean-stuff

# Part A
A: A.out

A.out: mainA.py eigen.py givens.py
	python3 $< --basic > A.out

cleanA:
	$(RM) A.out




# General clean-up
clean-stuff:
	$(RM) *.pyc
	rm -rf __pycache__
