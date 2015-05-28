# Make all targets
all: A B
clean: cleanA cleanB clean-ext

# Target A
A: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx givens.so datafun.so globvar.so
	python3 $< build_ext --inplace

cleanA:
	rm -rf out.A.txt A.plt build auxA.c auxA.so mainA.so mainA.c


# Target B
B: out.B.txt

out.B.txt: runB.py mainB.so
	python3 $< > $@

mainB.so: setupB.py mainB.pyx auxB.pyx givens.so datafun.so globvar.so
	python3 $< build_ext --inplace

cleanB:
	rm -rf out.B.txt B.plt build auxB.c auxB.so mainB.so mainB.c


# Build module containing the routines for Givens rotation
# No need to build every time; it is assumed static!
givens.so: setup_ext.py givens.pyx
	python3 $< build_ext --inplace

datafun.so: setup_ext.py datafun.pyx
	python3 $< build_ext --inplace

globvar.so: setup_ext.py globvar.pyx
	python3 $< build_ext --inplace

clean-ext:
	rm -rf givens.c givens.so datafun.c datafun.so globvar.c globvar.so
