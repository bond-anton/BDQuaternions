from .Quaternion cimport Quaternion


cdef class UnitQuaternion(Quaternion):

    cpdef UnitQuaternion conjugate(self)
