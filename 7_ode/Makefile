# Make all targets
all: A
clean: cleanA

# Target A
A: plot.A.pdf

plot.A.pdf: A.plt out.A.txt
	gnuplot $<

A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.A.pdf\" >> $@
	@echo set yrange [-1.1:1.1] >> $@
	@echo set xzeroaxis lt 1 dt 3 lw 2 lc \"black\" >> $@
	@echo set key bottom left >> $@
	@echo plot \"out.A.txt\" index 0 title \"y0\" \\>> $@
	@echo , \"out.A.txt\" index 1 title \"y1\" \\ >> $@

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt build auxA.c auxA.so mainA.so mainA.c A.plt plot.A.pdf
