# Make all targets
all: A B
clean: cleanA cleanB

# Target A
A: plot.A.pdf

plot.A.pdf: A.plt out.A.txt
	gnuplot $<

A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.A.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.A.txt\" index 0 title \"data\" \\ >> $@
	@echo , \"out.A.txt\" index 1 title \"lin\" with lines \\ >> $@
	@echo , \"out.A.txt\" index 2 title \"quad\" with lines >> $@

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf out.A.txt plot.A.pdf A.plt build auxA.c auxA.so mainA.so mainA.c


# Target B
B: plot.B.pdf

plot.B.pdf: B.plt out.B.txt
	gnuplot $<

B.plt:
	@echo set term pdfcairo dashed > $@
	@echo set out \"plot.B.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.B.txt\" index 0 notitle with lines lt 0 \\ >> $@
	@echo , \"out.B.txt\" index 1 title \"data\" \\ >> $@
	@echo , \"out.B.txt\" index 2 title \"qspline\" with lines \\ >> $@
	@echo , \"out.B.txt\" index 3 title \"deriv\" with lines \\ >> $@
	@echo , \"out.B.txt\" index 4 title \"integral\" with lines >> $@

out.B.txt: runB.py mainB.so
	python3 $< > $@

mainB.so: setupB.py mainB.pyx auxB.pyx
	python3 setupB.py build_ext --inplace

cleanB:
	rm -rf out.B.txt plot.B.pdf B.plt build auxB.c auxB.so mainB.so mainB.c
