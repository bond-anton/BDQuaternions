from __future__ import division, print_function
import numbers
import numpy as np

from Quaternions import Quaternion


class UnitQuaternion(Quaternion):

    def __init__(self, quadruple=None):
        if quadruple is None:
            quadruple = np.array([1, 0, 0, 0])
        quadruple = np.array(quadruple, dtype=np.float)
        assert np.allclose(np.sum(quadruple ** 2), 1.0)
        super(UnitQuaternion, self).__init__(quadruple)

    def conjugate(self):
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return UnitQuaternion(quadruple)

    def __mul__(self, other):
        if isinstance(other, UnitQuaternion):
            q1 = self.quadruple
            q2 = other.quadruple
            quadruple = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                                  q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                                  q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                                  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
            return UnitQuaternion(quadruple)
        elif isinstance(other, Quaternion):
            return Quaternion(self.quadruple) * other
        elif isinstance(other, numbers.Number):
            if np.allclose(abs(float(other)), 1):
                q2 = np.array([float(other), 0, 0, 0])
                return self * UnitQuaternion(q2)
            else:
                return Quaternion(self.quadruple) * other
        else:
            raise ValueError('Unit quaternion can be multiplied only by another quaternion or by number')

    def __rmul__(self, other):
        if isinstance(other, UnitQuaternion):
            return other * self
        elif isinstance(other, Quaternion):
            return other * Quaternion(self.quadruple)
        elif isinstance(other, numbers.Number):
            if np.allclose(abs(float(other)), 1):
                q2 = np.array([float(other), 0, 0, 0])
                return UnitQuaternion(q2) * self
            else:
                return other * Quaternion(self.quadruple)
        else:
            raise ValueError('Unit quaternion can be multiplied only by another quaternion or by number')
