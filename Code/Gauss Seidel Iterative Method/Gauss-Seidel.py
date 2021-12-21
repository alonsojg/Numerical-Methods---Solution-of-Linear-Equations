#!/usr/bin/env python
# coding: utf-8

# # Gauss-Seidel

# In[7]:


import time
import numpy as np
import random
import scipy.sparse
from scipy.sparse.linalg import spsolve

# Generate b, then use scipy.sparse.linalg.spsolve to find x

'''Defining our function as seidel which takes 3 
   arguments an A matrix, Solution and B matrix ''' 

def seidel(a, x ,b): 
    #Finding length of a(n)        
    n = len(a)                    
    # for loop for n times as to calculate x, y , z 
    for j in range(0, n):         
        # temp variable d to store b[j] 
        d = b[j]
        # to calculate respective xi, yi, zi 
        for i in range(0, n):      
            if(j != i): 
                d-=a[j][i] * x[i] 
        # updating the value of our solution         
        x[j] = d / a[j][j] 
    # returning our updated solution            
    return x 

# '''Here is the sample input for Q.CE 3 from 8.4'''

# initial solution depending on n(here n=4)  
x = np.array([0.0,0.0,0.0,0.0])
a = np.array([[7,3,-1,2],[3,8,1,-4],[-1,1,4,-1],[2,-4,-1,6]])
b = np.array([-1,0,-3,1])
xPrime = np.array([-1,1,-1,1])

# relative error total and rel_sol values to stop while loop at certain error margin
reltol = 1.0e-4
rel_sol = 1.0

# list of errors
errors = []

#iteration counter
i = 0

'''Here we actually iterate through solution
through the algorithm as it stumbles closer
and closer to the actual solution of Q.CE 3 from 8.4'''
print("-"*25+"Q.CE 3 from 8.4"+"-"*25)
tic=time.time()
#loop run for m times depending on the error value magnitude.
while (rel_sol > reltol):
    print('Iteration # %s\t Value of x: %s' % (i, x))
    i+=1
    #print each time the updated solution
    tempA = x - 0
    x = seidel(a, x, b)
    rel_sol = np.abs(np.max(tempA-x))
    errors.append(rel_sol)
    

toc=time.time()
print('Actual Solution:',xPrime)
condition_number = np.linalg.cond(A)
print("Condition Number: " + "{:10.8f}".format(condition_number))
print("\nGauss-Seidel solver time:\t\t{0:0.10f} seconds.".format(toc-tic))
abs_error = np.linalg.norm(xPrime - x)
print("Absolute error: " + str(abs_error))
rel_error = (np.linalg.norm(xPrime - x)) / np.linalg.norm(x)
print("Percent error:" + "{:10.4f}".format(rel_error * 100))
print("")
print("\n\n\n")

print("*"*90)

'''Here is the sample input for Q.CE 3 from 8.4'''
print("-"*25+"Q.CE 10 from 8.4"+"-"*25)

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


    '''We are creating a randomized x vector so that we may
    multiply Ax and get a b'''
    # create randomized x vector
    xtemp = random.sample(range(100), int(m))
    b = np.matmul(A,xtemp)

    # our intitial guess
    x = np.array(np.zeros(m))

    # relative error total and rel_sol values to stop while loop at certain error margin
    rel_sol = 1.0

    # list of errors
    errors = []

    #iteration counter
    i = 0

    '''Here we actually iterate through solution
    through the algorithm as it stumbles closer
    and closer to the actual solution of Q.CE 10 from 8.4'''
#     print(A)
#     print(b)
    
    tic=time.time()
    toc = 0
    # loop run for m times depending on the error value magnitude.
    while (rel_sol > reltol):
        print('Iteration # %s\t Value of x: %s' % (i, x))
        i+=1
        
        if i == 3:
            toc=time.time()
            
        # print each time the updated solution
        tempA = x - 0
        x = seidel(A, x, b)
        rel_sol = np.abs(np.max(tempA-x))
        errors.append(rel_sol)


    print("For {}x{} Matrix".format(m,m))
    print('Actual Solution:',xtemp)
    condition_number = np.linalg.cond(A)
    print("Condition Number: " + "{:10.8f}".format(condition_number))
    print("\nGauss-Seidel solver time:\t\t{0:0.10f} seconds.".format(toc-tic))
    abs_error = np.linalg.norm(x - xtemp)
    print("Absolute error: " + str(abs_error))
    rel_error = (np.linalg.norm(x - xtemp)) / np.linalg.norm(xtemp)
    print("Percent error:" + "{:10.4f}".format(rel_error * 100))
    print("")
    print("\n\n\n")

print("*"*90)
    
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


    '''We are creating a randomized x vector so that we may
    multiply Ax and get a b'''
    # create randomized x vector
    xtemp = random.sample(range(100), int(m))
    b = np.matmul(A,xtemp)

    # our intitial guess
    x = np.array(np.zeros(m))

    # relative error total and rel_sol values to stop while loop at certain error margin
    reltol = 1.0e-4
    rel_sol = 1.0

    # list of errors
    errors = []

    #iteration counter
    i = 0

    '''Here we actually iterate through solution
    through the algorithm as it stumbles closer
    and closer to the actual solution of Q.CE 10 from 8.4'''
    
#     print(A)
#     print(b)
    tic=time.time()
    toc = 0
    # loop run for m times depending on the error value magnitude.
    while (rel_sol > reltol):

        print('Iteration # %s\t Value of x: %s' % (i, x))
        i+=1

        
        if i == 3:
            toc=time.time()
        # print each time the updated solution
        tempA = x - 0
        x = seidel(A, x, b)
        rel_sol = np.abs(np.max(tempA-x))
        errors.append(rel_sol)

#     print('Iteration # %s\t Value of x: %s' % (i, x))
    print('Actual Solution:',xtemp)
    print("\nGauss-Seidel solver time:\t\t{0:0.10f} seconds.".format(toc-tic))
    condition_number = np.linalg.cond(A)
    print("Condition Number: " + "{:10.8f}".format(condition_number))
    abs_error = np.linalg.norm(x - xtemp)
    print("Absolute error: " + str(abs_error))
    rel_error = (np.linalg.norm(x - xtemp)) / np.linalg.norm(xtemp)
    print("Percent error:" + "{:10.4f}".format(rel_error * 100))
    print("")
    print("\n\n\n")

print("*"*90)

