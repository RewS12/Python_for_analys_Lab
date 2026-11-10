import numpy as np
cimport numpy as cnp

def sum_of_squares_cy(cnp.ndarray[cnp.float64_t, ndim=1] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef double total = 0.0
    for i in range(n):
        total += arr[i] * arr[i]
    return total