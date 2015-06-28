# Make all targets
all: A
clean: cleanA

# Target A
A: plot.A.pdf

plot.A.pdf: A.plt Aerr.dat
	gnuplot $<
	rm fit.log

A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.A.pdf\" >> $@
	@echo set title \"Error estimate\" >> $@
	@echo set logscale xy >> $@
	@echo set xrange [1:2e8] >> $@
	@echo set xlabel \'N\' >> $@
	@echo set ylabel \'Error\' >> $@
	@echo f\(x\) = a / sqrt\(x\) >>$@
	@echo fit f\(x\) \"Aerr.dat\" via a >> $@
	@echo plot \"Aerr.dat\" title \"Data\" \\>> $@
	@echo , f\(x\) title \"Fit: a/sqrt\(x\)\" >> $@

Aerr.dat: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@ 2> Aerr.dat

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt build auxA.c auxA.so mainA.so mainA.c A.plt plot.A.pdf Aerr.dat
