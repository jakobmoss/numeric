Numerical Methods 2015 - Examination Assignment
=================

**Author:** Jakob Rørsted Mosumgaard

**Project:** 9. Inverse iteration for eigenvalues

Exercise
---------
### A
* Implement the inverse iteration method for determining eigenvalues and -vectors.
This algorithm should converge towards the eigenvalue of least magnitude.
* Implement the inverse iteration method with a shift and check that you are
able to find all eigenvalues of a given matrix (compare with a library routine).
* Investigate how the error on the determined eigenvalue (compared to the library
routine) scales with the number of iterations.

### B
* Implement on-the-fly update of the eigenvalue estimate (i.e. the shift) using
the Rayleigh quotient.
* Investigate how the convergence depends on how often the guess is updated.
* Implement a convergence criterion (and hence, error estimate) and test it.

### C
* Modify the algorithm from exercise 3.b (Jacobi eigenvalue-by-eigenvalue) to
make it able to stop after the first eigenvalue (the largest or smalles) have
been found.
* Compare the modified Jacobi with inverse iteration.
* Compare the running time of the algorithms, compared to the library routine.


Implementation
--------------
The project is implemented in Python3 (version 3.4.3) with Gnuplot
(version 5.0.1) for plotting. NumPy is used extensively for handling of arrays.
The NumPy Linear Algebra (`numpy.linalg`) module is used as the library routine
for calculating eigenvalues and -vectors [actually this is just implemented
using LAPACK].

**Overview of the files:**
* _Makefile_ builds the project. It contains comments on the different targets.
The entire project can be build by just running `make` and `make clean` will
remove all output.
* _main.py_ is the file invoked by `make`, and contains an argument parser, which
calls the different parts.
* _partX.py_ with X = {A, B, C} contains the testing for the different parts of
the exercise.
* _eigen.py_ contains the implementation of the different versions of the
inverse iteration algorithm.
* _givens.py_ contains the routines for making QR-decompositions and solve
systems of equations [slightly modified version from exercise 2b].
* _jacobi.py_ contains the routines for determining eigenvalues and -vectors
using the Jacobi method [modified in it's original form from exercise 3b].


Output
______
The output is labelled with the corresponding exercise letter. Some output is
in text-format, while the visual output is in pdf-format.

**Overview of the files:**
* _output.A.txt_ contains a basic test of the algorithm and shows that all
eigenvalues (and -vectors) can be recovered using a shift.
* _plot.A.pdf_ contains a test of convergence (error compared to NumPy) as a
function of iterations. The points marked 'clean start' correspond to different
runs, while 'same v0' is using the same initial vector, hence it is the
convergence during a iterative run.
* _output.B.txt_ contains a test of the convergence criterion.
* _plot.B.pdf_ shows how the convergence (error compared to NumPy) depends on
how often the guess is updated using the Rayleigh criterion.
* _output.C.txt_ contains a comparison of the inverse iteration method and the
Jacobi method. It also includes a speedtest on a random, symmetric matrix.
* _X.dat_ with X = {A, B} contains data for the plots and are not interesting in
themselves.