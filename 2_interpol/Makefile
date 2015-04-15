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
	@echo plot \"out.A.txt\" >> $@

out.A.txt: mainA.py
	python3 $< > out.A.txt

#helloA.so: setupA.py
#	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf out.A.txt out.A.pdf out.A.plt

