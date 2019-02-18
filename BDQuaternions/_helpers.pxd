cdef double trace(double[:, :] m, int n) nogil
cdef double vectors_dot_prod(double[:] x, double[:] y, int n) nogil
cdef double[:] matrix_vector_mult(double[:, :] mat, double[:] vec, int rows, int cols)
cdef double[:, :] matrix_mult(double[:, :] m1, double[:, :] m2, int rows, int cols, int n)
cdef double _2x2_det(double[:, :] m) nogil
cdef double _3x3_det(double[:, :] m) nogil
cdef double[:, :] _3x3_inv(double[:, :] m)
cdef bint check_orthogonal(double[:, :] m, int rows, int cols, double tol)
