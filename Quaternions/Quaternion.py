from __future__ import division, print_function
import numpy as np
import numbers


class Quaternion(object):

    def __init__(self, quadruple=np.array([0, 0, 0, 1])):
        self._quadruple = None
        self._set_quadruple(quadruple)

    def __str__(self):
        return str(self._quadruple)

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return np.allclose(self.quadruple, other.quadruple)
        else:
            raise ValueError('Only another quaternion can be compared to given quaternion')

    def _set_quadruple(self, quadruple):
        quadruple = np.array(quadruple, dtype=np.float)
        if quadruple.size != 4:
            raise ValueError('Quadruple must have exactly 4 elements')
        self._quadruple = quadruple

    def _get_quadruple(self):
        return self._quadruple

    quadruple = property(_get_quadruple, _set_quadruple)

    def scalar_part(self):
        return self._quadruple[0]

    def vector_part(self):
        return self._quadruple[1:]

    def conjugate(self):
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return Quaternion(quadruple)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q1 = self.quadruple
            q2 = other.quadruple
            quadruple = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                                  q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                                  q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                                  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
            return Quaternion(quadruple)
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return self * Quaternion(q2)
        else:
            raise ValueError('Quaternion can be multiplied only by another quaternion')

    def __rmul__(self, other):
        if isinstance(other, Quaternion):
            return other * self
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return Quaternion(q2) * self
        else:
            raise ValueError('Quaternion can be multiplied only by another quaternion or number')

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.quadruple + other.quadruple)
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return self + Quaternion(q2)
        else:
            raise ValueError('Only another quaternion or number can be added to quaternion')

    def __radd__(self, other):
        if isinstance(other, Quaternion):
            return other + self
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return Quaternion(q2) + self
        else:
            raise ValueError('Only another quaternion or number can be added to quaternion')

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return self + (-1 * other)
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return self - Quaternion(q2)
        else:
            raise ValueError('Only another quaternion or number be subtracted from quaternion')

    def __rsub__(self, other):
        if isinstance(other, Quaternion):
            return other - self
        elif isinstance(other, numbers.Number):
            q2 = np.array([float(other), 0, 0, 0])
            return Quaternion(q2) - self
        else:
            raise ValueError('Only another quaternion or number be subtracted from quaternion')

    def norm(self):
        return np.sqrt(np.sum(self.quadruple * self.quadruple))

    def distance(self, other):
        if isinstance(other, Quaternion):
            return (self - other).norm()
        else:
            raise ValueError('Only another quaternion can be subtracted from quaternion')

    def versor(self):
        if not np.allclose(self.quadruple, np.zeros(4)):
            return 1 / self.norm() * self
        else:
            raise ZeroDivisionError('Zero quaternion has no versor')

    def reciprocal(self):
        if not np.allclose(self.quadruple, np.zeros(4)):
            return 1 / (self.norm() ** 2) * self.conjugate()
        else:
            raise ZeroDivisionError('Zero quaternion has no reciprocal')

    def __div__(self, other):
        if isinstance(other, Quaternion):
            raise ValueError('The notation p/q is ambiguous for two quaternions p and q. \
             Please use explicit form p * q.reciprocal or q.reciprocal * p')
        elif isinstance(other, numbers.Number):
            return self * (1 / float(other))
        else:
            raise ValueError('Quaternion can be divided only by number')

    def __rdiv__(self, other):
        if isinstance(other, Quaternion):
            raise ValueError('The notation p/q is ambiguous for two quaternions p and q. \
             Please use explicit form p * q.reciprocal or q.reciprocal * p')
        elif isinstance(other, numbers.Number):
            return self.reciprocal() * float(other)
        else:
            raise ValueError('Quaternion can be divided only by number')

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def _get_polar(self):
        if not np.allclose(self.norm(), 0.0):
            a = self.scalar_part()
            v = self.vector_part()
            v_norm = np.sqrt(np.sum(v * v))
            if not np.allclose(v_norm, 0.0):
                n_hat = v / v_norm
            else:
                n_hat = np.zeros(3)
            theta = np.arccos(a / self.norm())
            return self.norm(), n_hat, theta
        else:
            return 0, np.zeros(3), 0

    def _set_polar(self, (norm, n_hat, theta)):
        assert norm >= 0
        assert np.allclose(np.sqrt(np.sum(n_hat * n_hat)), 1.0)
        a = norm * np.cos(theta)
        v = n_hat * norm * np.sin(theta)
        self._set_quadruple(np.hstack((a, v)))

    polar = property(_get_polar, _set_polar)

    def __pow__(self, power):
        if isinstance(power, numbers.Number):
            norm, n_hat, theta = self.polar
            result = Quaternion()
            result.polar = (norm ** power, n_hat, theta * power)
            return result
        else:
            raise ValueError('Quaternions can be raised only into real-value power')

    def real_matrix(self):
        a, b, c, d = self.quadruple
        return np.array([[a, b, c, d],
                         [-b, a, -d, c],
                         [-c, d, a, -b],
                         [-d, -c, b, a]])

    def complex_matrix(self):
        a, b, c, d = self.quadruple
        return np.array([[a + b * 1j, c + d * 1j],
                         [-c + d * 1j, a - b * 1j]])
