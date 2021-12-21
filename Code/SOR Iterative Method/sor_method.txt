'''
  Computer Project II - Solution of Linear Equations
  Andrew Roig - MAP 4384 Fall 2020
'''

import time
import math
import numpy as np
from numpy import linalg as la
import scipy.sparse
from scipy.sparse.linalg import spsolve


# Global arrays used for 3rd iteration data for project report purposes
# Not printing them at time of approximation because want to print them together at the end
# and would also have to pass extraneous arguments/parameters either to the function call
# or to the main script or change when/where the exact solution is stored/computed
# Using w=1.4 for the report table because need to pick one, its over-relaxed and it's the one
# the book mentions for exercise #3
# [0-exercise 3, 1-exercise 10a, 2-exercise 10b]
# One result for exercise #3
# Four results for exercise #10 part a and part b
# The two solution arrays are arrays of vectors
array_of_3rd_iters_exact_solutions = np.zeros(9, dtype=object)
array_of_3rd_iters_solutions = np.zeros(9, dtype=object)
array_of_3rd_iters_time = np.zeros(9)
flag = 0
flag_for_exact = 0



def sor_method(A_matrix, b_vector, x_vector, relax_factor, size):
  global array_of_3rd_iters_solutions
  global array_of_3rd_iters_time
  global flag

  # Max Iterations set to a high ceiling to allow convergence
  k_max = 5000
  epsilon = 0.5 * pow(10,-4)

  total_time = 0
  time_temp = time.process_time()

  # Main Loop - Loop through k_max iterations
  for k in range(k_max):
    
    # y_vector is copy of current unaltered x_vector, thus it is xi
    y_vector = np.copy(x_vector)

    # Loop through all rows of the coefficient matrix A
    for i in range(size):
      sum = b_vector[i]
      diag = A_matrix[i,i]

      for j in range(i):
        sum -= A_matrix[i,j] * x_vector[j]
      for j in range(i+1,size):
        sum -= A_matrix[i,j] * x_vector[j]

      x_vector[i] = sum / diag
      x_vector[i] = relax_factor * x_vector[i] + (1.0 - relax_factor) * y_vector[i]

    # Stop gap for report to get the data at 3rd iteration for report table save it in a global variable
    if (k == 2 and relax_factor == 1.4):
      total_time += time.process_time() - time_temp
      array_of_3rd_iters_solutions[flag] = x_vector
      array_of_3rd_iters_time[flag] = total_time
      flag += 1

    
    # Convergence Check
    # Test if the magnitude(norm) of the difference between xi+1 - xi is sufficiently small
    # That is, the required precision prescribed by the used epsilon value
    if np.linalg.norm(x_vector - y_vector) < epsilon:
      return k
  return k

  


if __name__ in "__main__":

  '''
    ***Exercise 8.4.3 Setup and Data***
  '''
  time_entire = 0
  time_entire_temp = time.process_time()

  # Size of the square matrix nxn
  size = 4

  # Array for plotting
  plot_result = []

  # Array of Relaxation Factor's to compare
  # Convergence is 0 < w < 2
  # Fails to converge outside the relaxation range (0,2)
  relax_factor = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])

  # Initial Data from computer exercise 8.4
  b_vector = np.array([-1,0,-3,1])
  A_matrix = np.array([[7,3,-1,2],[3,8,1,-4],[-1,1,4,-1],[2,-4,-1,6]])

  # Just numpy option to make printing to stdout more readable
  np.set_printoptions(linewidth=160)

  print("")
  method = "SOR Method - Computer Exercise 8.4.3"
  print(method)
  for x in method:
    print('-', end = '')
  print("")

  exact_solution = np.linalg.solve(A_matrix, b_vector)
  array_of_3rd_iters_exact_solutions[0] = exact_solution
  flag_for_exact += 1

  # Step through all relax_factors, call sor_method and print the results
  for i in range(len(relax_factor)):
    total_time = 0
    # Starting solution vector, the standard initial guess is the zero vector
    x_vector = np.zeros(size)
    time_temp = time.process_time()
    # SOR method call. The iterative solution vector is pass-by-reference so only returns iterations
    result = sor_method(A_matrix, b_vector, x_vector, relax_factor[i], size)
    total_time += time.process_time() - time_temp

    print("Iterations: " + str(result) + "\t" + " Relaxation Factor: " + str(relax_factor[i]))
    print("Solution: " + str(x_vector))

    abs_error = (la.norm(x_vector) - la.norm(exact_solution))
    print("Absolute error: " + str(abs_error))
    rel_error = (la.norm(x_vector)-la.norm(exact_solution)) / la.norm(exact_solution)
    print("Relative error: " + str(rel_error))
    print("Time to compute: " + str(total_time) + " seconds")
    print("")
  
  print("--------")
  print("")


  '''
     Exercise C.E. 8.4.10 - Sparse Matrix Analysis
     Variation that utilizes the same sparse matrices supplied but uses
     a randomly generated b vector to analyze how the solution changes
     as the sparse matrix grows in size
  '''
  method = "SOR Method - Computer Exercise 8.4.10"
  print(method)
  for x in method:
    print('-', end = '')
  print("")

  '''
    ***Part A Setup***
  '''
  print("Part A")
  print("")
  # Initial size of sparse matrix m x m
  m = 8
  # Main loop - Grow sparse matrix on each loop upto 16 x 16
  while m <= 11:
    # Setup Diagonals
    diagonals = [[1 for _ in range(m-2)],
                [-4 for _ in range(m-1)],
                [6 for _ in range(m)],
                [-4 for _ in range(m-1)],
                [1 for _ in range(m-2)]]

    # Generate A
    A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()

    # Initialize the nonrepeating values in the corners
    A[0, 0:3] = [12., -6., (4/3)]
    A[-1, -3:] = [(4/3), -6., 12.]
    condition_number = la.cond(A)
    print(str(m)+ "x"+ str(m) + " sparse matrix" + "\t" + " Condition Number: " + str(condition_number))
    print(A)
    print("")
    # Create Random Exact solution x_vector
    exact_solution = np.random.rand(m)

    # Third Iteration vector dump
    array_of_3rd_iters_exact_solutions[flag_for_exact] = exact_solution
    flag_for_exact += 1
    print("")
    
    # Get the b_vector by multiplying A by x , i.e. b = Ax
    b_vector = np.matmul(A, exact_solution)

    # Step through all relax_factors, call sor_method and print the results
    for i in range(len(relax_factor)):
      total_time = 0
      # Starting solution vector, the standard initial guess is the zero vector
      # Is pass by reference so don't need to return the x_vector back
      x_vector = np.zeros(m)
      time_temp = time.process_time()
      result = sor_method(A, b_vector, x_vector, relax_factor[i], m)
      total_time += time.process_time() - time_temp
      print("Iterations: " + str(result) + "\t" + " Relaxation factor: " + str(relax_factor[i]))
      print("Solution: " + str(x_vector))

      abs_error = la.norm(x_vector - exact_solution)
      print("Absolute error: " + str(abs_error))
      rel_error = (la.norm(x_vector - exact_solution)) / la.norm(exact_solution)
      print("Relative error: " + str(rel_error))
      print("Time to compute: " + str(total_time) + " seconds")
      print("")

    m += 1
    print("")

  '''
    *** Part B Setup ***
  '''
  print("Part B")
  print("")
  # Initial size of sparse matrix mxm
  m = 8
  # Main loop - Grow sparse matrix on each loop
  while m <= 11:
    # Setup Diagonals
    diagonals = [[1 for _ in range(m-2)],
                [-4 for _ in range(m-1)],
                [6 for _ in range(m)],
                [-4 for _ in range(m-1)],
                [1 for _ in range(m-2)]]

    # Generate A
    A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()

    # Initialize the nonrepeating values in the corners and in 2nd to last row
    A[0, 0:3] = [12., -6., (4/3)]
    A[-2, -4:] = [1, -(93/25), (111/25), -(43/25)]   
    A[-1, -3:] = [(12/25), (24/25), (12/25)]
    condition_number = la.cond(A)
    print(str(m)+ "x"+ str(m) + " sparse matrix" + "\t" + " Condition Number: " + str(condition_number))
    print(A)


    # Create Random Exact solution x_vector
    exact_solution = np.random.rand(m)

    # Third Iteration vector dump
    array_of_3rd_iters_exact_solutions[flag_for_exact] = exact_solution
    flag_for_exact += 1

    # Get the b_vector by multiplying A by x , i.e. b = Ax
    b_vector = np.matmul(A, exact_solution)
    # Step through all relax_factors, call sor_method and print the results
    for i in range(len(relax_factor)):
      total_time = 0

      # Starting solution vector, the standard initial guess is the zero vector
      # Is pass by reference so don't need to return the x_vector back
      x_vector = np.zeros(m)
      time_temp = time.process_time()
      result = sor_method(A, b_vector, x_vector, relax_factor[i], m)
      total_time += time.process_time() - time_temp

      print("Iterations: " + str(result) + "\t" + " Relaxation factor: " + str(relax_factor[i]))
      print("Solution: " + str(x_vector))

      abs_error = la.norm(x_vector - exact_solution)
      print("Absolute error: " + str(abs_error))
      rel_error = (la.norm(x_vector - exact_solution)) / la.norm(exact_solution)
      print("Relative error: " + str(rel_error))
      print("Time to compute: " + str(total_time) + " seconds")
      print("")
      print("")

    m += 1
    print("")

  '''
    Just alot of lines of fluff to print out the third iteration data for report
  '''
  str_for_pretty_print = "Third Iteration Results"
  print(str_for_pretty_print)
  for char in str_for_pretty_print:
    print('~', end = '')
  print("")
  for i in range(9):
    if(i==0):
      print("Exercise 8.4.3")
      for char in str_for_pretty_print:
        print('-', end = '')
      print("")
    if(i == 0):
      abs_error = la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])
      print("Absolute error: " + str(abs_error))
      rel_error = (la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])) / la.norm(array_of_3rd_iters_exact_solutions[i])
      print("Relative error: " + str(rel_error))
      print("Time to compute: " + str(array_of_3rd_iters_time[i]) + " seconds")
    if (i == 1):
      print("Exercise 8.4.10 Part a")
      for char in str_for_pretty_print:
        print('-', end = '')
      print("")
    if(i in range(1,5)):
      if(i == 1):
        print("8x8 Sparse Matrix")
      if(i == 2):
        print("9x9 Sparse Matrix")
      if(i == 3):
        print("10x10 Sparse Matrix")
      if(i == 4):
        print("11x11 Sparse Matrix")
      abs_error = la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])
      print("Absolute error: " + str(abs_error))
      rel_error = (la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])) / la.norm(array_of_3rd_iters_exact_solutions[i])
      print("Relative error: " + str(rel_error))
      print("Time to compute: " + str(array_of_3rd_iters_time[i]) + " seconds")
    if(i == 5):
      print("Exercise 8.4.10 Part b")
      for char in str_for_pretty_print:
        print('-', end = '')
      print("")
    if(i in range(5,9)):
      if(i == 5):
        print("8x8 Sparse Matrix")
      if(i == 6):
        print("9x9 Sparse Matrix")
      if(i == 7):
        print("10x10 Sparse Matrix")
      if(i == 8):
        print("11x11 Sparse Matrix")
      abs_error = la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])
      print("Absolute error: " + str(abs_error))
      rel_error = (la.norm(array_of_3rd_iters_solutions[i] - array_of_3rd_iters_exact_solutions[i])) / la.norm(array_of_3rd_iters_exact_solutions[i])
      print("Relative error: " + str(rel_error))
      print("Time to compute: " + str(array_of_3rd_iters_time[i]) + " seconds")
    print("")
  for char in str_for_pretty_print:
    print('*', end = '')
  print("")
  print("Note: All data supplied for the project's method comparisons and charts is w = 1.4")
  # Just an extraneous fun measurement because it fell like eons rerunning the code everytime
  time_entire += time.process_time() - time_entire_temp
  print("Script runtime: " + str(time_entire ) + " seconds")

  