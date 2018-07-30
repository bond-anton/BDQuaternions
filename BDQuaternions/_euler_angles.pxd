from BDQuaternions._euler_angles_conventions cimport Convention

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
