# Make all targets
all: A B C
clean: cleanA cleanB cleanC clean-ext

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
	rm -rf out.A.txt A.plt build auxA.c auxA.so mainA.so mainA.c plot.A.pdf


# Target B
B: plot.B.pdf

plot.B.pdf: B.plt out.B.txt
	gnuplot $<

B.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.B.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.B.txt\" index 0 with errorbars title \"data\" \\ >> $@
	@echo , \"out.B.txt\" index 1 with lines title \"fit\" \\ >> $@
	@echo , \"out.B.txt\" index 2 with lines title \"fit + err\" \\ >> $@
	@echo , \"out.B.txt\" index 3 with lines title \"fit - err\" >> $@

out.B.txt: runB.py mainB.so
	python3 $< > $@

mainB.so: setupB.py mainB.pyx auxB.pyx jacobi.so
	python3 $< build_ext --inplace

cleanB:
	rm -rf out.B.txt B.plt build auxB.c auxB.so mainB.so mainB.c plot.B.pdf


# Target B
C: plot.C.pdf

plot.C.pdf: C.plt out.C.txt
	gnuplot $<

C.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.C.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.C.txt\" index 0 with errorbars title \"data\" \\ >> $@
	@echo , \"out.C.txt\" index 1 using 1:2 w l title \"Linear: y\(x\)\" \\ >> $@
	@echo , \"out.C.txt\" index 1 using 1:3 w lines title \"y\(x\) + dy\" \\ >> $@
	@echo , \"out.C.txt\" index 1 using 1:4 w lines title \"y\(x\) - dy\" >> $@

out.C.txt: runC.py mainC.so
	python3 $< > $@

mainC.so: setupC.py mainC.pyx auxA.pyx givens.so
	python3 $< build_ext --inplace

cleanC:
	rm -rf out.C.txt C.plt build mainC.so mainC.c plot.C.pdf


# Build module containing the routines for Givens and Jacobi algorithms
# No need to build every time; it is assumes static!
givens.so: setup_ext.py givens.pyx
	python3 $< build_ext --inplace

jacobi.so: setup_ext.py jacobi.pyx
	python3 $< build_ext --inplace

clean-ext:
	rm -rf givens.c givens.so jacobi.c jacobi.so
