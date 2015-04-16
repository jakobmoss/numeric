# Make all targets
all: A
clean: cleanA

# Target A
A: out.A.pdf

out.A.pdf: A.plt A.txt
	gnuplot $<

A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"out.A.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"A.txt\" index 0 title \"data\" \\ >> $@
	@echo , \"A.txt\" index 1 title \"lin\" with lines \\ >> $@
	@echo , \"A.txt\" index 2 title \"quad\" with lines >> $@

A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf A.txt out.A.pdf A.plt build auxA.c auxA.so mainA.so mainA.c

