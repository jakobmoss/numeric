# Make all targets
all: A
clean: cleanA

# Target A
A: out.A.pdf

out.A.pdf: out.A.plt out.A.txt
	gnuplot $<

out.A.plt:
	@echo set term pdfcairo > $@
	@echo set out \"out.A.pdf\" >> $@
	@echo set key left >> $@
	@echo plot \"out.A.txt\" index 0 title \"data\" \\ >> $@
	@echo , \"out.A.txt\" index 1 title \"lin\" with lines \\ >> $@
	@echo , \"out.A.txt\" index 2 title \"quad\" with lines >> $@

out.A.txt: mainA.py auxA.so
	python3 $< > out.A.txt

auxA.so: setupA.py
	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf out.A.txt out.A.pdf out.A.plt build auxA.c auxA.so

