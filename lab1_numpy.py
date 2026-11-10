import numpy as np
import time
from scipy import sparse

print("--- ЗАДАЧА 1 ---")
arr_f32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
print(f"Sum float32: {arr_f32.sum()}, Expected: 0.6, Diff: {abs(arr_f32.sum() - 0.6)}")
print(f"Epsilon float32: {np.finfo(np.float32).eps}")
print(f"Epsilon float64: {np.finfo(np.float64).eps}")
val_f32 = np.float32(1.0) + np.float32(1e-8) - np.float32(1.0)
val_f64 = np.float64(1.0) + np.float64(1e-8) - np.float64(1.0)
print(f"Catastrophic cancellation float32: {val_f32}, float64: {val_f64}")

print("\n--- ЗАДАЧА 2 ---")
A = np.arange(20).reshape(4, 5) - 10
print("2.1 Second row:", A[1, :])
print("2.1 Column 3:", A[:, 3])
print("2.1 Subarray:\n", A[1:4, 2:5])
print("2.2 Fancy indexing:\n", A[np.ix_([0, 2, 3], [1, 4])])
A[A < 0] = 0
print("2.3 After logical indexing:\n", A)
B = np.ones((4, 1))
C = np.ones((1, 5)) * 2
print("2.4 Broadcasting:\n", B + C)

print("\n--- ЗАДАЧА 3 ---")
filename = 'test_array.dat'
mmap = np.memmap(filename, dtype='float64', mode='w+', shape=(1000, 1000))
mmap[:] = np.random.rand(1000, 1000)
mmap.flush()
row = mmap[0, :]
col = mmap[:, 0]
print(f"3.2 Read row shape: {row.shape}, col shape: {col.shape}")

def sum_row_major(m):
    total = 0.0
    for i in range(m.shape[0]):
        total += np.sum(m[i, :])
    return total

def sum_col_major(m):
    total = 0.0
    for j in range(m.shape[1]):
        total += np.sum(m[:, j])
    return total

start = time.perf_counter()
sum_row_major(mmap)
time_row = time.perf_counter() - start

start = time.perf_counter()
sum_col_major(mmap)
time_col = time.perf_counter() - start
print(f"3.3 Row-major time: {time_row:.4f}s, Col-major time: {time_col:.4f}s")

print("\n--- ЗАДАЧА 4 ---")
S_csr = sparse.random(5, 5, density=0.2, format='csr')
print("4.1 CSR matrix:\n", S_csr)
S_coo = S_csr.tocoo()
S_csc = S_csr.tocsc()
print("4.2 COO matrix:\n", S_coo)
dense_vec = np.ones(5)
print("4.3 Sparse * dense vector:", S_csr.dot(dense_vec))

S_large_csr = sparse.random(1000, 1000, density=0.01, format='csr')
D_large = S_large_csr.toarray()
vec_large = np.ones(1000)

start = time.perf_counter()
S_large_csr.dot(vec_large)
time_sparse = time.perf_counter() - start

start = time.perf_counter()
D_large.dot(vec_large)
time_dense = time.perf_counter() - start
print(f"4.4 Sparse time: {time_sparse:.6f}s, Dense time: {time_dense:.6f}s")

print("\n--- ЗАДАЧА 5 ---")
try:
    import sum_of_squares_cy
    
    test_arr = np.random.rand(10_000_000)
    
    def sum_of_squares_py(arr):
        total = 0.0
        for x in arr:
            total += x * x
        return total
        
    start = time.perf_counter()
    sum_of_squares_py(test_arr)
    py_time = time.perf_counter() - start
    
    start = time.perf_counter()
    sum_of_squares_cy.sum_of_squares_cy(test_arr)
    cy_time = time.perf_counter() - start
    
    print(f"5.3 Python time: {py_time:.4f}s, Cython time: {cy_time:.4f}s, Speedup: {py_time / cy_time:.1f}x")
except ImportError:
    print("5.3 Cython module not compiled yet.")