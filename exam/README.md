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
**Overview of the files:**
* _main.py_: Hep