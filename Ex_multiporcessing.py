from multiprocessing import Pool
import numpy as np
import time

def splitJobs(num_jobs, num_core):
	pool = []
	each = num_jobs//num_core
	more = num_jobs - (num_core * each)

	print(f"each = {each}")
	print(f"more = {more}")

	for i in range(more):
		pool.append([(each+1)*i, (each+1)*(i+1)])
	s = (each+1) * more
	
	for i in range(more, num_core, 1):
		pool.append([s, s+each])
		s += each
	return pool

def matrixMul(mat):
	return np.dot(mat[0], mat[1])

if __name__ == "__main__":
	p = Pool()
	dim = 1000
	n = dim * dim
	num_core = 8

	A = np.arange(n).reshape((dim, dim))
	B = np.arange(n).reshape((dim, dim))

	start_time = time.time()
	C = A @ B

	mid_time = time.time()
	print(f"time taken by A @ B = {mid_time - start_time} seconds")

	print('[#] Multiprocessing')

	jobs = splitJobs(dim, num_core)
	mat = []
	for i in range(num_core):
		mat.append([A[jobs[i][0]:jobs[i][1], :], B])
	
	result = p.map(matrixMul, mat)
	C2 = np.vstack((result[0], result[1]))

	for i in range(2, num_core, 1):
		C2 = np.vstack((C2, result[i]))
	
	end_time = time.time()
	print(f"time taken by multiprocessing = {end_time - mid_time} seconds")


