import numpy as np
from cython import boundscheck, wraparound
from cpython.array cimport array, clone


@boundscheck(False)
@wraparound(False)
cdef double trace(double[:, :] m, int n) nogil:
    cdef:
        double res = 0.0
        int i
    for i in range(n):
        res += m[i, i]
    return res


@boundscheck(False)
@wraparound(False)
cdef double vectors_dot_prod(double[:] x, double[:] y, int n) nogil:
    cdef:
        double res = 0.0
        int i
    for i in range(n):
        res += x[i] * y[i]
    return res


@boundscheck(False)
@wraparound(False)
cdef double[:] matrix_vector_mult(double[:, :] mat, double[:] vec, int rows, int cols):
    cdef:
        int i
        array[double] result, template = array('d')
    result = clone(template, rows, zero=False)
    for i in range(rows):
        result[i] = vectors_dot_prod(mat[i], vec, cols)
    return result


@boundscheck(False)
cdef double _2x2_det(double[:, :] m) nogil:
    return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]


@boundscheck(False)
cdef double _3x3_det(double[:, :] m) nogil:
    return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) \
         - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) \
         + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])


@boundscheck(False)
cdef double[:, :] _3x3_inv(double[:, :] m):
    cdef:
        double det = _3x3_det(m)
        double[:, :] inv = np.empty((3, 3), dtype=np.double)
    inv[0, 0] = (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) / det
    inv[0, 1] = (m[2, 1] * m[0, 2] - m[0, 1] * m[2, 2]) / det
    inv[0, 2] = (m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2]) / det
    inv[1, 0] = (m[2, 0] * m[1, 2] - m[1, 0] * m[2, 2]) / det
    inv[1, 1] = (m[0, 0] * m[2, 2] - m[2, 0] * m[0, 2]) / det
    inv[1, 2] = (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]) / det
    inv[2, 0] = (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]) / det
    inv[2, 1] = (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]) / det
    inv[2, 2] = (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) / det
    return inv


@boundscheck(False)
@wraparound(False)
cdef double[:, :] matrix_mult(double[:, :] m1, double[:, :] m2, int rows, int cols, int n):
    cdef:
        int i, j, k
        double[:, :] product = np.empty((rows, cols), dtype=np.double)
    for i in range(rows):
        for j in range(cols):
            product[i][j] = 0.0
            for k in range(n):
                product[i][j] += m1[i][k] * m2[k][j]
    return product


@boundscheck(False)
@wraparound(False)
cdef bint check_orthogonal(double[:, :] m, int rows, int cols, double tol):
    cdef:
        int i = 0, j = 0
        double[:, :] product = matrix_mult(m, m.T, rows, rows, cols)
    for i in range(rows):
        for j in range(rows):
            if i == j:
                if abs(product[i][j] - 1.0) > tol:
                    break
            else:
                if abs(product[i][j]) > tol:
                    break
        if j != rows - 1:
            break
    if i != rows - 1:
        return False
    return True
