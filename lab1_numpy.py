import numpy as np
import time
from scipy import sparse

print("=" * 60)
print("ЗАДАЧА 1: Числа с плавающей точкой")
print("=" * 60)

arr_f32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
print(f"\n  Сумма float32 [0.1, 0.2, 0.3]: {arr_f32.sum()}")
print(f"  Ожидаемое значение: 0.6")
print(f"  Разница: {abs(arr_f32.sum() - 0.6):.2e}")

print(f"\n  Машинное эпсилон:")
print(f"    float32: {np.finfo(np.float32).eps:.2e}")
print(f"    float64: {np.finfo(np.float64).eps:.2e}")

val_f32 = np.float32(1.0) + np.float32(1e-8) - np.float32(1.0)
val_f64 = np.float64(1.0) + np.float64(1e-8) - np.float64(1.0)
print(f"\n  Катастрофическая отмена (1.0 + 1e-8) - 1.0:")
print(f"    float32: {val_f32} ← потеря точности!")
print(f"    float64: {val_f64}")

print("\n" + "=" * 60)
print("ЗАДАЧА 2: Индексация, срезы и broadcasting")
print("=" * 60)

A = np.arange(20).reshape(4, 5) - 10
print(f"\n  Исходный массив A (4×5):\n{A}")

print(f"\n  2.1 Вторая строка: {A[1, :]}")
print(f"  2.1 Столбец с индексом 3: {A[:, 3]}")
print(f"  2.1 Подмассив [1:4, 2:5]:\n{A[1:4, 2:5]}")

print(f"\n  2.2 Fancy-индексация [0,2,3]×[1,4]:\n{A[np.ix_([0, 2, 3], [1, 4])]}")

A[A < 0] = 0
print(f"\n  2.3 После замены отрицательных на 0:\n{A}")

B = np.ones((4, 1))
C = np.ones((1, 5)) * 2
print(f"\n  2.4 Broadcasting B(4,1) + C(1,5):\n{B + C}")

print("\n" + "=" * 60)
print("ЗАДАЧА 3: Memory-mapped файлы и оптимизация кэша")
print("=" * 60)

filename = 'test_array.dat'
mmap = np.memmap(filename, dtype='float64', mode='w+', shape=(1000, 1000))
mmap[:] = np.random.rand(1000, 1000)
mmap.flush()

row = mmap[0, :]
col = mmap[:, 0]
print(f"\n  3.2 Прочитано без загрузки всего массива:")
print(f"    Строка: форма {row.shape}")
print(f"    Столбец: форма {col.shape}")

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

print(f"\n  3.3 Время обхода:")
print(f"    По строкам (row-major): {time_row:.4f} сек")
print(f"    По столбцам (col-major): {time_col:.4f} сек")
print(f"    Разница: {time_col/time_row:.1f}x")
print(f"\n  Объяснение: row-major быстрее из-за локальности кэша")

print("\n" + "=" * 60)
print("ЗАДАЧА 4: Разреженные матрицы")
print("=" * 60)

S_csr = sparse.random(5, 5, density=0.2, format='csr')
print(f"\n  4.1 Разреженная матрица CSR (5×5, density=0.2):\n{S_csr}")

S_coo = S_csr.tocoo()
S_csc = S_csr.tocsc()
print(f"\n  4.2 Преобразования:")
print(f"    COO: {S_coo.nnz} ненулевых элементов")
print(f"    CSC: {S_csc.nnz} ненулевых элементов")

dense_vec = np.ones(5)
result = S_csr.dot(dense_vec)
print(f"\n  4.3 Умножение CSR на вектор:\n{result}")

S_large_csr = sparse.random(1000, 1000, density=0.01, format='csr')
D_large = S_large_csr.toarray()
vec_large = np.ones(1000)

start = time.perf_counter()
S_large_csr.dot(vec_large)
time_sparse = time.perf_counter() - start

start = time.perf_counter()
D_large.dot(vec_large)
time_dense = time.perf_counter() - start

print(f"\n  4.4 Сравнение времени (1000×1000, density=0.01):")
print(f"    Разреженная (CSR): {time_sparse:.6f} сек")
print(f"    Плотная: {time_dense:.6f} сек")
print(f"    Ускорение: {time_dense/time_sparse:.1f}x")

print("\n" + "=" * 60)
print("ЗАДАЧА 5: Cython")
print("=" * 60)

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

    print(f"\n  Массив: 10⁷ элементов")
    print(f"  Python (цикл): {py_time:.4f} сек")
    print(f"  Cython: {cy_time:.4f} сек")
    print(f"  Ускорение: {py_time/cy_time:.1f}x")
except ImportError:
    print("\n  Модуль Cython не скомпилирован")
    print("  Выполните: python setup.py build_ext --inplace")

print("\n" + "=" * 60)