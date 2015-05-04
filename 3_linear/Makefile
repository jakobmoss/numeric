# Make all targets
all: A B
clean: cleanA cleanB

# Target A
A: out.A.txt

out.A.txt: runA.py mainA.so
	python3 $< > $@

mainA.so: setupA.py mainA.pyx auxA.pyx
	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf out.A.txt A.plt build auxA.c auxA.so mainA.so mainA.c


# Target B
B: out.B.txt

out.B.txt: runB.py mainB.so
	python3 $< test > $@
	echo "\n Speed of Gram-Schmidt on 300 x 300 matrix:\n" >> $@
	\time python3 $< gs > /dev/null 2>> $@
	echo "\n\n Speed of Givens rotation on 300 x 300 matrix:\n" >> $@
	\time python3 $< givens > /dev/null 2>> $@

mainB.so: setupB.py mainB.pyx auxB.pyx mainA.so
	python3 setupB.py build_ext --inplace

cleanB:
	rm -rf out.B.txt B.plt build auxB.c auxB.so mainB.so mainB.c
