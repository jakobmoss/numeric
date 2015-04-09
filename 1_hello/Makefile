# Make all targets
all: A B
clean: cleanA cleanB

# Target A - one file
A: out.A.txt

out.A.txt: helloA.so mainA.py
	python3 mainA.py > out.A.txt

helloA.so: setupA.py
	python3 setupA.py build_ext --inplace

cleanA:
	rm -rf build helloA.c helloA.so out.A.txt


# Target B - two files
B: out.B.txt

out.B.txt: mainB.py helloB.so userB.so
	python3 mainB.py > out.B.txt

helloB.so: setupB_hello.py
	python3 setupB_hello.py build_ext --inplace

userB.so: setupB_user.py
	python3 setupB_user.py build_ext --inplace

cleanB:
	rm -rf build helloB.c helloB.so out.B.txt userB.c userB.so

# Checking
check: checkA checkB

checkA: out.A.txt
	@echo "checking target A ..."
	@printf "Hello, moss\n" > correct.txt
	@diff --brief correct.txt out.A.txt
	@echo "target A seems to be ok ..."
	@rm -f correct.txt

checkB: out.B.txt
	@echo "checking target B ..."
	@printf "Hello, moss\n" > correct.txt
	@diff --brief correct.txt out.B.txt
	@echo "target B seems to be ok ..."
	@rm -f correct.txt
