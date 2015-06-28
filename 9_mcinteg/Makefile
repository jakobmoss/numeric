# Make all targets
all: A
clean: cleanA

# Target A
A: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt build auxA.c auxA.so mainA.so mainA.c A.plt plot.A.pdf
