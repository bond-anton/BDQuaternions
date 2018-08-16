from .UnitQuaternion cimport UnitQuaternion
from .EulerAnglesConventions cimport Convention
from .EulerAngles cimport EulerAngles


cdef class Rotation(UnitQuaternion):
    cdef Convention __euler_angles_convention

    cpdef Rotation conjugate(self)
    cpdef Rotation reciprocal(self)

    cpdef rotate(self, xyz)
