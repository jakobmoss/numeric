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
	@echo set yrange [-1.1:1.1] >> $@
	@echo set xzeroaxis lt 1 dt 3 lw 2 lc \"black\" >> $@
	@echo set key bottom left >> $@
	@echo set title \"y\'\' = -y   \;   RK23\" >> $@
	@echo set xlabel \'x\' >> $@
	@echo set ylabel \'y\' >> $@
	@echo plot \"out.A.txt\" index 0 title \"y\(0\) = 0\" \\>> $@
	@echo , \"out.A.txt\" index 1 title \"y\(0\) = 1\" \\ >> $@
	@echo , sin\(x\), cos\(x\) >> $@

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt build auxA.c auxA.so mainA.so mainA.c A.plt plot.A.pdf


# Target B
B: plot.B.pdf

plot.B.pdf: B.plt out.B.txt
	gnuplot $<

B.plt:
	@echo set term pdfcairo > $@
	@echo set out \"plot.B.pdf\" >> $@
	# @echo set yrange [-1.1:1.1] >> $@
	# @echo set xzeroaxis lt 1 dt 3 lw 2 lc \"black\" >> $@
	# @echo set key bottom left >> $@
	# @echo set title \"y\'\' = -y   \;   RK23\" >> $@
	# @echo set xlabel \'x\' >> $@
	# @echo set ylabel \'y\' >> $@
	# @echo plot \"out.B.txt\" index 0 title \"y\(0\) = 0\" \\>> $@
	# @echo , \"out.B.txt\" index 1 title \"y\(0\) = 1\" \\ >> $@
	# @echo , sin\(x\), cos\(x\) >> $@

out.B.txt: runB.py mainB.so
	python3 $< > $@

mainB.so: setupB.py mainB.pyx auxB.pyx
	python3 $< build_ext --inplace

cleanB:
	rm -rf out.B.txt build auxB.c auxB.so mainB.so mainB.c B.plt plot.B.pdf
