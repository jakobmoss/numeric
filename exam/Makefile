###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-06-30 16:17:48 moss>
#
# Makefile
###########################################

# Default target
default: A

# Part A
A: A.out

A.out: mainA.py eigen.py givens.py
	python3 $< > A.out

cleanA:
	$(RM) A.out




# General clean-up
clean:
	$(RM) *.pyc
	rm -rf __pycache__
