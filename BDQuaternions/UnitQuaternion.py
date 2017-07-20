from __future__ import division, print_function
import numbers
import numpy as np

from BDQuaternions._quaternion_operations import check_quadruple, mul

from BDQuaternions import Quaternion


class UnitQuaternion(Quaternion):
    """
    Sub-class of Quaternion to deal with unit quaternions (of norm == 1)
    """

    def __init__(self, quadruple=None):
        if quadruple is None:
            quadruple = [1, 0, 0, 0]
        quadruple = check_quadruple(quadruple)
        assert np.allclose(np.sum(quadruple ** 2), 1.0)
        super(UnitQuaternion, self).__init__(quadruple)

    def conjugate(self):
        """
        Calculates conjugate for the Unit Quaternion
        :return: Unit Quaternion which is conjugate of current unit quaternion
        """
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return UnitQuaternion(quadruple)

    def reciprocal(self):
        """
        for Unit quaternion reciprocal is equal to conjugate
        :return: Unit quaternion reciprocal to current quaternion
        """
        return self.conjugate()

    def __mul__(self, other):
        if isinstance(other, UnitQuaternion):
            return UnitQuaternion(mul(self.quadruple, other.quadruple))
        elif isinstance(other, Quaternion):
            return Quaternion(mul(self.quadruple, other.quadruple))
        elif isinstance(other, numbers.Number):
            if np.allclose(abs(float(other)), 1):
                return UnitQuaternion(self.quadruple)
            else:
                return Quaternion(self.quadruple) * other
        else:
            raise ValueError('Unit quaternion can be multiplied only by another quaternion or by number')

    def __rmul__(self, other):
        if isinstance(other, UnitQuaternion):
            return other * self
        elif isinstance(other, Quaternion):
            return Quaternion(mul(other.quadruple, self.quadruple))
        elif isinstance(other, numbers.Number):
            if np.allclose(abs(float(other)), 1):
                return UnitQuaternion(self.quadruple)
            else:
                return other * Quaternion(self.quadruple)
        else:
            raise ValueError('Unit quaternion can be multiplied only by another quaternion or by number')
