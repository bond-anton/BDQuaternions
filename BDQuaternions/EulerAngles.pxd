from .EulerAnglesConventions cimport Convention
from .Rotation cimport Rotation


cdef class EulerAngles(object):
    cdef:
        double[:] __euler_angles
        Convention __convention

    cdef double __reduce_angle(self, double angle, bint center=*, bint half=*)
    cdef double[:] __reduce_euler_angles(self, double[:] euler_angles)
    cpdef void to_parent_convention(self)
    cpdef double[:, :] rotation_matrix(self)
    cpdef void from_rotation_matrix(self, double[:, :] m, Convention convention)
    cpdef void change_convention(self, Convention new_convention)

    cdef __to_quaternion(self)
    cpdef Rotation to_quaternion(self)

    cdef void __from_quaternion(self, double[:] quadruple, Convention convention)
    cpdef void from_quaternion(self, Rotation quaternion, Convention convention)
