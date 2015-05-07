# Make all targets
all: A
clean: cleanA
clean-deep: cleanA clean-givens

# Target A
A: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx givens.so
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt A.plt build auxA.c auxA.so mainA.so mainA.c



# Build module containing the routines for Given's rotation
# No need to build every time; it is assumes static!
givens.so: setup_givens.py givens.pyx
	python3 $< build_ext --inplace

clean-givens:
	rm -rf givens.c givens.so
