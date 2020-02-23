import numbers
import numpy as np

from libc.float cimport DBL_MIN

from .Quaternion cimport Quaternion
from ._quaternion_operations cimport mul, norm


cdef class UnitQuaternion(Quaternion):
    """
    Sub-class of Quaternion to deal with unit quaternions (of norm == 1)
    """

    def __init__(self, double[:] quadruple=np.array([1, 0, 0, 0], dtype=np.double)):
        assert np.allclose(norm(quadruple), 1.0)
        super(UnitQuaternion, self).__init__(quadruple)

    cpdef UnitQuaternion conjugate(self):
        """
        Calculates conjugate for the Unit Quaternion
        :return: Unit Quaternion which is conjugate of current unit quaternion
        """
        return UnitQuaternion(self.__conjugate())

    cpdef UnitQuaternion reciprocal(self):
        """
        for Unit quaternion reciprocal is equal to conjugate
        :return: Unit quaternion reciprocal to current quaternion
        """
        return self.conjugate()

    def __mul__(x, y):
        if isinstance(x, UnitQuaternion) and isinstance(y, UnitQuaternion):
            return UnitQuaternion(mul(x.quadruple, y.quadruple))
        elif isinstance(x, UnitQuaternion) and isinstance(y, Quaternion):
            return Quaternion(mul(x.quadruple, y.quadruple))
        elif isinstance(x, Quaternion) and isinstance(y, UnitQuaternion):
            return Quaternion(mul(x.quadruple, y.quadruple))
        elif isinstance(x, UnitQuaternion) and isinstance(y, numbers.Number):
            if abs(float(y) - 1) < 4 * DBL_MIN:
                return UnitQuaternion(x.quadruple)
            elif abs(float(y) + 1) < 4 * DBL_MIN:
                return UnitQuaternion(-1 * x.quadruple)
            else:
                return Quaternion(x.quadruple) * y
        elif isinstance(x, numbers.Number) and isinstance(y, UnitQuaternion):
            if abs(float(x) - 1) < 4 * DBL_MIN:
                return UnitQuaternion(y.quadruple)
            elif abs(float(x) + 1) < 4 * DBL_MIN:
                return UnitQuaternion(-1 * y.quadruple)
            else:
                return Quaternion(y.quadruple) * x
        else:
            return NotImplemented
