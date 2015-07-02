###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-02 13:05:49 moss>
#
# Makefile
###########################################

# Default target
default: B
all: A B
clean: cleanA cleanB clean-stuff

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
# Part B
#
B: output.B.txt plot.B.pdf

# Text-output
output.B.txt: main.py partB.py eigen.py givens.py
	python3 $< B --criterion > $@

# Visual output from convergence tests
plot.B.pdf: B.plt B.dat
	gnuplot $<
	rm B.plt

B.plt: Makefile
	@echo set term pdfcairo > $@
	@echo set out \'plot.B.pdf\' >> $@
	@echo set title \'Effect of updated estimates on convergence\' >> $@
	@echo set logscale y >> $@
	@echo set format y \'10^{%L}\' >> $@
	@echo set mxtics 5 >> $@
	@echo set xlabel \'Number of iterations\' >> $@
	@echo set ylabel \'Error on estimated eigenvalue\' >> $@
	@echo plot \'B.dat\' index 0  w lp title \'No update\' \\ >> $@ 
	@echo , \'B.dat\' index 1 w lp title \'Every 5th iter\' \\ >> $@ 
	@echo , \'B.dat\' index 2 w lp lt 4 title \'Every 3rd iter\' \\ >> $@ 
	@echo , \'B.dat\' index 3 w lp lt 6 title \'Every 2nd iter\' \\ >> $@ 
	@echo , \'B.dat\' index 4 w lp lt 8 title \'Every iter\' \\ >> $@ 

B.dat: main.py partB.py eigen.py givens.py
	python3 $< B --convergence > $@

# Housekeeping...
cleanB:
	$(RM) output.B.txt


#
# General clean-up of stuff created by Python
#
clean-stuff:
	$(RM) *.pyc
	rm -rf __pycache__
