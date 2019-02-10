import numpy as np
from cython import boundscheck, wraparound
from cpython.array cimport array, clone
from scipy.linalg.cython_lapack cimport dsyevd


@boundscheck(False)
@wraparound(False)
cpdef double trace(double[:, :] m, int n):
    cdef:
        double res = 0.0
        int i
    for i in range(n):
        res += m[i, i]
    return res


@boundscheck(False)
@wraparound(False)
cpdef double vectors_dot_prod(double[:] x, double[:] y, int n):
    cdef:
        double res = 0.0
        int i
    for i in range(n):
        res += x[i] * y[i]
    return res


@boundscheck(False)
@wraparound(False)
cpdef double[:] matrix_vector_mult(double[:, :] mat, double[:] vec, int rows, int cols):
    cdef:
        int i
        array[double] result, template = array('d')
    result = clone(template, rows, zero=False)
    for i in range(rows):
        result[i] = vectors_dot_prod(mat[i], vec, cols)
    return result


cpdef double _2x2_det(double[:, :] m):
    return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]


cpdef double _3x3_det(double[:, :] m):
    return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) \
         - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) \
         + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])


cpdef double[:, :] _3x3_inv(double[:, :] m):
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
cpdef double[:, :] matrix_mult(double[:, :] m1, double[:, :] m2, int rows, int cols, int n):
    cdef:
        int i, j, k
        double s = 0.0
        double[:, :] product = np.empty((rows, cols), dtype=np.double)
    for i in range(rows):
        for j in range(cols):
            for k in range(n):
                s += m1[i][k] * m2[k][j]
            product[i][j] = s
            s = 0.0
    return product


@boundscheck(False)
@wraparound(False)
cpdef bint check_orthogonal(double[:, :] m, int rows, int cols, double tol):
    cdef:
        int c, d, k
        double s = 0.0
        double[:, :] product = matrix_mult(m, m.T, rows, rows, cols)
    for c in range(rows):
        for d in range(rows):
            if c == d:
                if abs(product[c][d] - 1.0) > tol:
                    break
            else:
                if abs(product[c][d]) > tol:
                    break
        if d != rows - 1:
            break
    if c != rows - 1:
        return False
    return True


cpdef double[:] decomp(double[:, :] m):
    cdef:
        array[double] quadruple, w, work, template = array('d')
        array[int] iwork, itemplate = array('i')
        double[:, :] k_m
        int n = 4, lwork = 57, liwork = 23, info
        char L = b'L', J = b'V'
    quadruple = clone(template, 4, zero=False)
    w = clone(template, 4, zero=False)
    work = clone(template, lwork, zero=False)
    iwork = clone(itemplate, liwork, zero=False)
    k_m = np.array([[m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]],
                    [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]],
                    [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
                    [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3
    print(np.asarray(k_m))
    w_n, v_n = np.linalg.eigh(k_m)
    print(w_n)
    print(v_n)
    dsyevd(&J, &L, &n, &k_m[0, 0], &n, &w[0], &work[0], &lwork, &iwork[0], &liwork, &info)
    print(np.asarray(k_m.T))
    print(np.asarray(w))
    print(info)
    print(np.asarray(work))
    print(np.asarray(iwork))
    q = v_n[[3, 0, 1, 2], np.argmax(w_n)]
    quadruple[0] = q[0]
    quadruple[1] = q[1]
    quadruple[2] = q[2]
    quadruple[3] = q[3]
    return quadruple
