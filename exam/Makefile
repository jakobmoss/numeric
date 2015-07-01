###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 11:08:57 moss>
#
# Makefile
###########################################

# Default target
default: A
clean: cleanA clean-stuff

# Part A
A: A.out

A.out: main.py partA.py eigen.py givens.py
	python3 $< A --basic > $@

cleanA:
	$(RM) A.out




# General clean-up
clean-stuff:
	$(RM) *.pyc
	rm -rf __pycache__
