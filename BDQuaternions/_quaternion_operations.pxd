cpdef double[:] mul(double[:] q1, double[:] q2)
cpdef double norm(double[:] quadruple)

cpdef double[:, :] real_matrix(double[:] q)
cpdef double complex[:, :] complex_matrix(double[:] q)

cpdef double[:, :] quaternion_to_rotation_matrix(double[:] q)
cpdef double[:] quaternion_from_rotation_matrix(double[:, :] m)

cpdef double[:] exp(double[:] q)
cpdef double[:] log(double[:] q)
