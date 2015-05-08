# Make all targets
all: A
clean: cleanA
clean-deep: cleanA clean-givens

# Target A
A: plot.A.pdf

plot.A.pdf: A.plt out.A.txt
	gnuplot $<

A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.A.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.A.txt\" index 0 with errorbars title \"data\" \\>> $@
	@echo , \"out.A.txt\" index 1 with lines title \"fit\" \\ >> $@
	@echo , \"out.A.txt\" index 2 with lines title \"fit + err\" \\ >> $@
	@echo , \"out.A.txt\" index 3 with lines title \"fit - err\" >> $@

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx givens.so
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt A.plt build auxA.c auxA.so mainA.so mainA.c



# Build module containing the routines for Given's rotation
# No need to build every time; it is assumes static!
givens.so: setup_givens.py givens.pyx
	python3 $< build_ext --inplace

clean-givens:
	rm -rf givens.c givens.so
