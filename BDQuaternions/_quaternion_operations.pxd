cpdef mul(double[:] q1, double[:] q2)
cpdef norm(double[:] quadruple)

cpdef real_matrix(double[:] q)
cpdef complex_matrix(double[:] q)

cpdef quaternion_to_rotation_matrix(double[:] q)
cpdef quaternion_from_rotation_matrix(double[:, :] m)

cpdef exp(double[:] q)
cpdef log(double[:] q)
