# Make all targets
all: A
clean: cleanA clean-ext

# Target A
A: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx globvar.so
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt build auxA.c auxA.so mainA.so mainA.c A.plt plot.A.pdf


# Build modules with different external routines
globvar.so: setup_ext.py globvar.pyx
	python3 $< build_ext --inplace

clean-ext:
	rm -rf  globvar.c globvar.so
