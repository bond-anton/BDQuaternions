cdef class Quaternion(object):
    cdef:
        double[:] __quadruple

    cpdef double scalar_part(self)
    cpdef double[:] vector_part(self)
    cdef double[:] __conjugate(self)
    cpdef Quaternion conjugate(self)

    cpdef double norm(self)
    cpdef double distance(self, Quaternion other)
    cpdef Quaternion versor(self)
    cpdef Quaternion reciprocal(self)

    cpdef double[:, :] real_matrix(self)
    cpdef double complex[:, :] complex_matrix(self)
