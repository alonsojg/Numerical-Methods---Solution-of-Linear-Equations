import numpy as np
import scipy.sparse
import time

def gauss(n, A, b):
    """Solve Ax=b using gaussian elimination with scaled partial pivoting

    A, b, must be numpy arrays
    """
    x = np.zeros(n)
    s = np.zeros(n)
    l = [i for i in range(n)]
    n = n-1
    A = A.copy()
    b = b.copy()

    for i in range(0, n+1):
        l[i] = i
        for j in range(0, n+1):
            s[i] = max(s[i], np.abs(A[i,j]))
    for k in range(0, n):
        r_max = 0
        for i in range(k, n+1):
            r = np.abs(A[l[i],k] / s[l[i]])
            if r > r_max:
                r_max = r
                j = i

        l[j], l[k] = l[k], l[j]
        for i in range(k+1, n+1):
            A[l[i],k] = A[l[i], k]/A[l[k],k]
            for j in range(k+1, n+1):
                A[l[i],j] = A[l[i],j] - A[l[i],k] * A[l[k],j]

    for k in range(0, n):
        for i in range(k+1, n+1):
            b[l[i]] = b[l[i]] - A[l[i],k] * b[l[k]]

    x[n] = b[l[n]]/A[l[n],n]
    for i in range(n-1, -1, -1):
        sum = b[l[i]]
        for j in range(i+1, n+1):
            sum = sum - A[l[i],j]*x[j]
        x[i] = sum/(A[l[i],i])
    return x

# Verify validity of method
m = 10
n_correct = 0
n_tries = 100
for _ in range(n_tries):
    A = np.random.rand(m, m)
    b = np.random.rand(m)

    ans = np.linalg.solve(A,b)
    ans2 = gauss(m, A, b)
    if np.allclose(ans, ans2):
        n_correct += 1

print(f"random systems solved correctly: {n_correct}/{n_tries}")


A = np.array([[ 7, 3,-1, 2],
              [ 3, 8, 1,-4],
              [-1, 1, 4,-1],
              [ 2,-4,-1, 6]])

b = np.array([-1, 0, -3, 1])

print('Q3 Solution:', gauss(4, A, b))
print('Q3 Expected:', np.linalg.solve(A,b))

time_temp = time.process_time()
gauss(4, A, b)
total_time = time.process_time() - time_temp
print('time:', total_time)
print('Q3 error:', np.linalg.norm(np.linalg.solve(A,b) - gauss(4, A, b)))


m = 10 # A is mxm matrix
# Choose Diagonals
diagonals = [[1 for _ in range(m-2)],
             [-4 for _ in range(m-1)],
             [6 for _ in range(m)],
             [-4 for _ in range(m-1)],
             [1 for _ in range(m-2)]]
# Generate A
A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()
# Set different values at corners
A[0, 0:3] = [12., -6., (4/3)]
A[-1, -3:] = [(4/3), -6., 12.]

error = 0
for _ in range(n_tries):
    x = np.random.rand(m)
    b = A @ x

    ans = np.linalg.solve(A,b)
    ans2 = np.nan_to_num(gauss(m, A, b))
    error += np.square(ans - ans2).mean()
error = error/100
print('Q10A avg MSE:', error)



A[-2, -3:] = [-(93/25), (111/25), -(43/25)]
A[-1, -3:] = [(12/25), (24/25), (12/25)]


error = 0
for _ in range(n_tries):
    x = np.random.rand(m)
    b = A @ x

    ans = np.linalg.solve(A,b)
    ans2 = np.nan_to_num(gauss(m, A, b))
    error += np.square(ans - ans2).mean()
error = error/100
print('Q10B avg MSE:', error)



error = 0
for _ in range(n_tries):
    x = np.random.rand(m)
    b = A @ x

    ans = np.linalg.solve(A,b)
    ans2 = np.nan_to_num(gauss(m, A, b))
    error += np.square(ans - ans2).mean()
error = error/100



# Output:
# random systems solved correctly: 100/100
# Q3 Solution: [-0.16578947 -0.22105263 -0.76315789 -0.05263158]
# Q3 Expected: [-1.  1. -1.  1.]
# Q3 error: [0.83421053 1.22105263 0.23684211 1.05263158]
# Q10A avg MSE: 0.3401405809958578
# Q10B avg MSE: 0.33736794920692



# Q 10 stats
for m in range(8, 12):
    # Choose Diagonals
    diagonals = [[1 for _ in range(m-2)],
                 [-4 for _ in range(m-1)],
                 [6 for _ in range(m)],
                 [-4 for _ in range(m-1)],
                 [1 for _ in range(m-2)]]
    # Generate A
    A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()
    # Set different values at corners
    A[0, 0:3] = [12., -6., (4/3)]
    A[-1, -3:] = [(4/3), -6., 12.]

    error = 0
    rel_err = 0
    total_time = 0
    for _ in range(100):
        x = np.random.rand(m)
        b = A @ x

        time_temp = time.process_time()
        ans = np.nan_to_num(gauss(m, A, b))
        total_time += time.process_time() - time_temp
        error += np.linalg.norm(x - ans)
        rel_err += np.linalg.norm(x - ans) / np.linalg.norm(x)
    error = error/100
    rel_err = rel_err/100
    avg_time = total_time/100

    print(f'm = {m} Avg error 10A: {error}')
    print(f'm = {m} Avg rel error 10A: {rel_err}')
    print(f'm = {m} Avg time elapsed 10A: {avg_time}')

# Q 10 stats
for m in range(8, 12):
    # Choose Diagonals
    diagonals = [[1 for _ in range(m-2)],
                 [-4 for _ in range(m-1)],
                 [6 for _ in range(m)],
                 [-4 for _ in range(m-1)],
                 [1 for _ in range(m-2)]]
    # Generate A
    A = scipy.sparse.diags(diagonals, [-2, -1, 0, 1, 2]).toarray()
    # Set different values at corners
    A[0, 0:3] = [12., -6., (4/3)]
    A[-2, -3:] = [-(93/25), (111/25), -(43/25)]
    A[-1, -3:] = [(12/25), (24/25), (12/25)]

    error = 0
    rel_err = 0
    total_time = 0
    for _ in range(100):
        x = np.random.rand(m)
        b = A @ x

        time_temp = time.process_time()
        ans = np.nan_to_num(gauss(m, A, b))
        total_time += time.process_time() - time_temp
        error += np.linalg.norm(x - ans)
        rel_err += np.linalg.norm(x - ans) / np.linalg.norm(x)
    error = error/100
    rel_err = rel_err/100
    avg_time = total_time/100
    print(f'm = {m} Avg error 10B: {error}')
    print(f'm = {m} Avg rel error 10B: {rel_err}')
    print(f'm = {m} Avg time elapsed 10B: {avg_time}')
