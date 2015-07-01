###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 13:55:33 moss>
#
# Makefile
###########################################

# Default target
default: A
clean: cleanA clean-stuff

#
# Part A
#
A: output.A.txt plot.A.pdf

# Text-output from basic tests
output.A.txt: main.py partA.py eigen.py givens.py
	python3 $< A --basic > $@

# Visual output from convergence tests
plot.A.pdf: A.plt A.dat
	gnuplot $<
	rm A.plt

A.plt: Makefile
	@echo set term pdfcairo > $@
	@echo set out \'plot.A.pdf\' >> $@
	@echo set title \'Convergence test\' >> $@
	@echo set logscale y >> $@
	@echo set format y \'10^{%L}\' >> $@
	@echo set xlabel \'Number of iterations\' >> $@
	@echo set ylabel \'Error on estimated eigenvalue\' >> $@
	@echo plot \'A.dat\' index 0 title \'Clean start\' \\>> $@
	@echo , \'A.dat\' index 1 title \'Same v0\' >> $@

A.dat: main.py partA.py eigen.py givens.py
	python3 $< A --convergence > $@	

# Housekeeping...
cleanA:
	$(RM) output.A.txt plot.A.pdf A.plt A.dat


#
# General clean-up of stuff created by Python
#
clean-stuff:
	$(RM) *.pyc
	rm -rf __pycache__
