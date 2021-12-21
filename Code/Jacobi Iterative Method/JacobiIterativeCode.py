import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import random
from numpy import linalg as la

question_3 = 1
question_10a = 0
question_10b = 0

def Jacobi_Iterative_Method(A, b, x_0):
    
    D = np.diagflat(np.diag(A))    # D matrix with diagonal elements of 0


    lower_diag = np.tril(A)        # lower triangular matrix with diagonal elements of 0
    L = lower_diag - D             # L matrix
    U = A - lower_diag             # U matrix

    x_k = x_0                      # setting up first iteration to be 

    # the D^(-1) * [ b - ((L + U)* x_0) ]
    x_k1 = np.dot(np.linalg.inv(D), b - np.dot(L + U, x_k)) 
    
    
    return x_k1


if question_3 == 1:
    A = np.array([[7,  3,  -1,  2],  # matrix
                  [3,  8,   1, -4],
                  [-1, 1,   4, -1],
                  [2, -4,  -1,  6]])

    b = np.array([-1, 0, -3, 1])     # vector

    x = np.zeros(4)               # vector x with elements of 0 (initial setup)

    print("\nA :\n")
    print(A)

    print("\nb :\n")
    print(b)

    print("\nx :\n")
    print(x)

    # relative error total and rel_sol values to stop while loop at certain error margin
    tolerance = 1.0e-5
    starting_tol = 1.0
    i = 0

    # Finding solution using the conjugate gradient method
    print('''First find the optimal solution for x using the formulas specified in the Jacobi Iterative method: ''')
    t1_start = time.process_time()

    #for h in range(4):
    while (starting_tol > tolerance):
        print('Iteration # %s\t Value of x: %s' % (i, x))
        i+=1
        # print each time the updated solution
        tempA = x - 0
        x = Jacobi_Iterative_Method(A, b, x)
        starting_tol = np.abs(np.max(tempA - x))

    t1_finish = time.process_time()
    print('Time it took to do the Jacobi Iterative method: ', (t1_finish-t1_start), '\n')

    # Finding solution using the numpy package
    print('''Finally, find the exact optimal solution for x using the linlag.solve module from the numpy package: ''')
    t2_start = time.process_time()
    x_using_imported_package = np.linalg.solve(A, b)
    t2_finish = time.process_time()
    print(x_using_imported_package)

    print('Time it took to do the numpy package: ', (t2_finish-t2_start), '\n')

    abs_error = la.norm(x - x_using_imported_package)
    print("Absolute error: " + str(abs_error))
    rel_error = (la.norm(x - x_using_imported_package)) / la.norm(x_using_imported_package)
    print("Percent error:" + "{:10.4f}".format(rel_error))

    print('--------------------------------------------------------------------')

if question_10a==1:
    error_list=[]
    # A is mxm matrix
    for m in range(8,12):
        # Choose Diagonals
        diagonals = [[1 for _ in range(m-2)],
                    [-4 for _ in range(m-1)],
                    [6 for _ in range(m)],
                    [-4 for _ in range(m-1)],
                    [1 for _ in range(m-2)]]

        # Generate A
        A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()
        # Set diferent values at corners
        A[0, 0:3] = [12., -6., (4/3)]
        A[-1, -3:] = [(4/3), -6., 12.]

        condition_number = la.cond(A)
        print("Condition Number: " + "{:10.8f}".format(condition_number))
        
        randomx = random.sample(range(m), int(m))
        b = np.matmul(A,randomx)

        x = np.zeros(m)               # vector x with elements of 0 (initial setup)

        print('The given linear system of equations: ', '\n', 'A= ', '\n', A)
        print('b= ', '\n', b, '\n')
        
        
        # relative error total and rel_sol values to stop while loop at certain error margin
        tolerance = 1.0e-5
        starting_tol = 1.0
        i = 0
        
        print('''First find the optimal solution for x using the formulas specified in the jacobi iterative method: ''')
        t1_start = time.process_time()
        for h in range(10):
        #while (starting_tol > tolerance):
            print('Iteration # %s\t Value of x: %s' % (i, x))
            i+=1
            # print each time the updated solution
            tempA = x - 0
            print(x)
            x = Jacobi_Iterative_Method(A, b, x)
            starting_tol = np.abs(np.max(tempA-x))
            print("error at each iteration",starting_tol)

        print(randomx)
        t1_finish = time.process_time()
        print('Time it took to do the jacobi iterative method: ', (t1_finish-t1_start), '\n')

        abs_error = la.norm(x - randomx)
        print("Absolute error: " + str(abs_error))
        rel_error = (la.norm(x - randomx)) / la.norm(randomx)
        print("Percent error:" + "{:10.4f}".format(rel_error))

        print('--------------------------------------------------------------------')

if question_10b==1:
    error_list=[]
    # A is mxm matrix
    for m in range(8,12):
        # Choose Diagonals
        diagonals = [[1 for _ in range(m-2)],
                    [-4 for _ in range(m-1)],
                    [6 for _ in range(m)],
                    [-4 for _ in range(m-1)],
                    [1 for _ in range(m-2)]]

        # Generate A
        A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()
        # Set diferent values at corners
        A[0, 0:3] = [12., -6., (4/3)]
        A[-1, -3:] = [(12/25), (24/25), (12/25)]
        A[-2, -3:] = [(-93/25), (111/25), (-43/25)]

        condition_number = la.cond(A)
        print("Condition Number: " + "{:10.8f}".format(condition_number))
        
        randomx = random.sample(range(m), int(m))
        b = np.matmul(A,randomx)


        x = np.zeros(m)               # vector x with elements of 0 (initial setup)

        print('The given linear system of equations: ', '\n', 'A= ', '\n', A)
        print('b= ', '\n', b, '\n')
        
        
        # relative error total and rel_sol values to stop while loop at certain error margin
        tolerance = 1.0e-10
        starting_tol = 1.0
        i = 0
        
        # Finding solution using the conjugate gradient method
        print('''First find the optimal solution for x using the formulas specified in the jacobi iterative method: ''')
        t1_start = time.process_time()
        for h in range(10):
        #while (starting_tol > tolerance):
            print('Iteration # %s\t Value of x: %s' % (i, x))
            i+=1
            # print each time the updated solution
            tempA = x - 0
            print(x)
            x = Jacobi_Iterative_Method(A, b, x)
            starting_tol = np.abs(np.max(tempA-x))
            print("error at each iteration",starting_tol)

        print(randomx)
        t1_finish = time.process_time()
        print('Time it took to do the jacobi iterative method: ', (t1_finish-t1_start), '\n')

        abs_error = la.norm(x - randomx)
        print("Absolute error: " + str(abs_error))
        rel_error = (la.norm(x - randomx)) / la.norm(randomx)
        print("Percent error:" + "{:10.4f}".format(rel_error))

        print('--------------------------------------------------------------------')



