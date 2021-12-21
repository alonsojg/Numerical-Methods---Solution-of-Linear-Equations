# Written by: Angelo Scaria

# Import required packages
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import random
from numpy import linalg as la

# Flags for the different questions
question3 = 1
question10a = 0
question10b = 0

''' Set the function to be used to find 
the optimal solution, x for the given 
system of linear equations '''
def conjugate_gradient(A, b, x=None):

    ''' Setting n as the matrix system size 
    according to the size of array b '''
    n = len(b)

    ''' Set condtion where if this is the 
    first step and there is no intial 
    value of x, set an arbitary 
    initial solution '''

    if not x:
        ''' Returns an arbitary array of 
        zeros in the shape of n as the
        intial solution for x '''
        x = np.zeros(n)
    
    # Calculation of the residual, r
    r = np.dot(A, x) - b

    ''' The initial value of the conjugate 
    step direction is the negative of 
    the residual '''
    s = -r

    ''' Finding the dot product between the 
    residual and residual and the residual 
    to be used in the calculations to find 
    the beta value '''
    r_dotProduct = np.dot(r, r)

    # Set the maximum number of iterations as 2*n
    for i in range(2*n):
        print('Iteration: ', i)
        ''' A_dot_s is the dot product between the given matrix, 
        A and the conjugate direction of the step, s '''
        A_dot_s = np.dot(A, s)

        ''' Calculates beta, the weight of the step to be 
        taken each iteration '''
        beta_numerator = np.dot(np.transpose(s), r)
        beta_denomenator =(np.dot(np.transpose(s), A_dot_s))
        beta = -(beta_numerator/beta_denomenator)

        ''' The new estimate of x is the last estimate of x + 
        (the weight constant, beta) * (conjugate direction, s) '''
        x += beta * s

        ''' The new residual value is the previous residual value - 
        (weight of the step to be taken beta) * (matrix A) * 
        (conjugate direction s) '''
        r += beta * A_dot_s

        ''' Find the dot product of the new residual value, 
        r by it's transpose to be used to calculate theta '''
        rNext_dotProduct = np.dot(np.transpose(r), r)
        
        ''' Theta is the constant corresponding to the previous 
        step's conjugate step direction, s '''
        theta = rNext_dotProduct / r_dotProduct

        ''' The new residual value for the next step, r becomes 
        the r of the current step '''
        r_dotProduct = rNext_dotProduct

        ''' If the residual dot product is less thant the 
        predetermined error term, stop doing iterations, you 
        have found the optimal solution, x'''
        if rNext_dotProduct < 1e-5:

            # Print the number of iterations
            print('The optimal solution for x was found after this many iterations: ', i)
            break

        # Find next step's conjugate step direction, s
        s = theta * s - r    
    return x

if __name__ == '__main__':
    # Here is the execution for question 3
    if question3==1:
        A_row1 = [7, 3, -1, 2]
        A_row2 = [3, 8, 1, -4]
        A_row3 = [-1, 1, 4, -1]
        A_row4 = [2, -4, -1, 6]
        A_combined = [A_row1, A_row2, A_row3, A_row4]

        A = np.array(A_combined)
        b = np.array([-1, 0, -3, 1])

        print('The given linear system of equations: ', '\n', 'A= ', '\n', A)
        print('b= ', '\n', b, '\n')

        # Finding solution using the conjugate gradient method
        print('''First find the optimal solution for x using the formulas specified in the conjugate gradient method: ''')
        t1_start = time.process_time()
        x_using_formulas = conjugate_gradient(A, b)
        t1_finish = time.process_time()
        print(x_using_formulas)

        print('Time it took to do the conjugate gradient method: ', (t1_finish-t1_start), '\n')

        # Finding solution using the numpy package
        print('''Finally, find the exact optimal solution for x using the linlag.solve module from the numpy package: ''')
        t2_start = time.process_time()
        x_using_imported_package = np.linalg.solve(A, b)
        t2_finish = time.process_time()
        print(x_using_imported_package)

        # Finding absolute and relative error
        abs_error = la.norm(x_using_formulas -  x_using_imported_package)
        print("Absolute error: " + str(abs_error))
        rel_error = (la.norm(x_using_formulas -  x_using_imported_package)) / la.norm(x_using_imported_package)
        print("Percent error:" + "{:10.4f}".format(rel_error))
        print("")

        print('Time it took to do the numpy package: ', (t2_finish-t2_start), '\n')
        print('--------------------------------------------------------------------')

    # Here is the execution for question 10a
    if question10a==1:
        # A is mxm matrix
        time_list = []

        for m in range(8,12):
            print('Matrix Size: ', m)
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
            
            # Setting a random x solution to be able to generate a b to evaluate a the method
            randomx = random.sample(range(m), int(m))
            print('Set a random vector x: ', randomx)
            b = np.matmul(A, randomx)

            print('The given linear system of equations: ', '\n', 'A= ', '\n', A)
            print('b= ', '\n', b, '\n')

            # Finding solution using the conjugate gradient method
            print('This was the initial random solution given: ', randomx, '\n')
            print('''First find the optimal solution for x using the formulas specified in the conjugate gradient method: ''')
            t1_start = time.process_time()
            x_using_formulas = conjugate_gradient(A, b)
            t1_finish = time.process_time()
            print(x_using_formulas)
            print('Time it took to do the conjugate gradient method: ', (t1_finish-t1_start), '\n')
            
            # Find the absolute and relative error
            abs_error = la.norm(x_using_formulas - randomx)
            print("Absolute error: " + str(abs_error))
            rel_error = (la.norm(x_using_formulas - randomx)) / la.norm(randomx)
            print("Percent error:" + "{:10.4f}".format(rel_error))
            print("")

            # Append the times per size matrix into the empty list
            time_list.append(t1_finish-t1_start)
            print('--------------------------------------------------------------------')

        print('Here is the final list of processing times as the matrix gets larger: \n', time_list)

    # Here is the execution for question 10b
    if question10b==1:
        # A is mxm matrix
        time_list = []

        for m in range(8,12):
            print('Matrix Size ', m)
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

            # Setting a random x solution to be able to generate a b to evaluate a the method            
            randomx = random.sample(range(m), int(m))
            print('Set a random vector x: ', randomx)
            b = np.matmul(A, randomx)

            print('The given linear system of equations: ', '\n', 'A= ', '\n', A)
            print('b= ', '\n', b, '\n')

            # Finding solution using the conjugate gradient method
            print('This was the initial random solution given: ', randomx, '\n')
            print('First find the optimal solution for x using the formulas specified in the conjugate gradient method: ')
            t1_start = time.process_time()
            x_using_formulas = conjugate_gradient(A, b)
            t1_finish = time.process_time()
            print(x_using_formulas)
            print('Time it took to do the conjugate gradient method: ', (t1_finish-t1_start), '\n')


            abs_error = la.norm(x_using_formulas - randomx)
            print("Absolute error: " + str(abs_error))
            rel_error = (la.norm(x_using_formulas - randomx)) / la.norm(randomx)
            print("Percent error:" + "{:10.4f}".format(rel_error))
            print("")

            # Append the times per size matrix into the empty list
            time_list.append(t1_finish-t1_start)
            print('--------------------------------------------------------------------')

        print('Here is the final list of processing times as the matrix gets larger: \n', time_list)