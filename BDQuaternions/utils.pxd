from .Quaternion cimport Quaternion
from .UnitQuaternion cimport UnitQuaternion
from .Rotation cimport Rotation

cpdef Rotation random_rotation()
cpdef UnitQuaternion random_unit_quaternion()
cpdef Quaternion random_quaternion(double quadruple_norm=*)
