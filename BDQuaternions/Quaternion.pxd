cdef class Quaternion(object):
    cdef:
        double[:] __quadruple

    cpdef double scalar_part(self)
    cdef __vector_part(self)
    cdef __conjugate(self)
    cpdef Quaternion conjugate(self)

    cpdef norm(self)
    cpdef distance(self, Quaternion other)
    cpdef versor(self)
    cpdef reciprocal(self)

    cpdef real_matrix(self)
    cpdef complex_matrix(self)
