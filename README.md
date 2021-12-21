Introduction

Numerical analysis creates, analyzes, and implements algorithms to be able to solve
problems of continuous math originating from real world applications of algebra, geometry and
calculus that involve variables that vary continuously [2]. This paper focuses on utilizing
iterative methods on problems that use a system of linear equations.
When one is given a system of linear equations with a single solution described as
$Ax = b$, where $A$ represents a known real matrix with
$n \times n$ size, and represents a known
vector with size $n \times1$
, one must find the optimal solution for , which represents the vector of
unknowns with size $n \times1$
. There are multiple different ways to approach this problem, and this
paper will focus specifically on five different methods to solve the problem. It is later showcased
that one can use either one of these five methods to solve a problem given a system of linear
equations and the output result should be the same amongst each method.
The five methods are Gaussian Elimination with partial pivoting, Jacobi iterative Method,
Gauss-Seidel iterative method, Successive Over-relaxation (SOR) method, and Conjugate
Gradient method. Although Gaussian Elimination with partial pivoting is not an iterative method,
the solutions should give either a fairly similar result, or a more accurate one. Experiments have
been conducted considering the system of equations $Ax = b$, where $A$ is a $N \times N$ matrix and $b$ is a
vector in $\R^n$. The results of the experiments will be compared showcasing which method is most
efficient to use given a scenario.
