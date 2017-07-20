from __future__ import division, print_function
import numpy as np
import numbers

from BDQuaternions._quaternion_operations import check_quadruple, mul, norm, real_matrix, complex_matrix


class Quaternion(object):
    """
    General quaternion object
    """

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
        self._quadruple = check_quadruple(quadruple)

    def _get_quadruple(self):
        return self._quadruple

    """
    Property to get/set quaternion' quadruple
    """
    quadruple = property(_get_quadruple, _set_quadruple)

    def scalar_part(self):
        """
        Calculates scalar part of the Quaternion
        :return: scalar part of the Quaternion
        """
        return self._quadruple[0]

    def vector_part(self):
        """
        Calculates vector part of the Quaternion
        :return: vector part of the Quaternion
        """
        return self._quadruple[1:]

    def conjugate(self):
        """
        Calculates conjugate for the Quaternion
        :return: Quaternion which is conjugate of current quaternion
        """
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return Quaternion(quadruple)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(mul(self.quadruple, other.quadruple))
        elif isinstance(other, numbers.Number):
            return Quaternion(mul(self.quadruple, [float(other), 0, 0, 0]))
        else:
            raise ValueError('Quaternion can be multiplied only by another quaternion')

    def __rmul__(self, other):
        if isinstance(other, Quaternion):
            return other * self
        elif isinstance(other, numbers.Number):
            return Quaternion(mul([float(other), 0, 0, 0], self.quadruple))
        else:
            raise ValueError('Quaternion can be multiplied only by another quaternion or number')

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.quadruple + other.quadruple)
        elif isinstance(other, numbers.Number):
            return Quaternion(self.quadruple + np.array([float(other), 0, 0, 0]))
        else:
            raise ValueError('Only another quaternion or number can be added to quaternion')

    def __radd__(self, other):
        if isinstance(other, Quaternion):
            return other + self
        elif isinstance(other, numbers.Number):
            return Quaternion(np.array([float(other), 0, 0, 0]) + self.quadruple)
        else:
            raise ValueError('Only another quaternion or number can be added to quaternion')

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return self + (-1 * other)
        elif isinstance(other, numbers.Number):
            return Quaternion(self.quadruple - np.array([float(other), 0, 0, 0]))
        else:
            raise ValueError('Only another quaternion or number be subtracted from quaternion')

    def __rsub__(self, other):
        if isinstance(other, Quaternion):
            return other - self
        elif isinstance(other, numbers.Number):
            return Quaternion(np.array([float(other), 0, 0, 0]) - self.quadruple)
        else:
            raise ValueError('Only another quaternion or number be subtracted from quaternion')

    def norm(self):
        """
        Calculates the norm of the Quaternion
        :return: norm of Quaternion
        """
        return norm(self.quadruple)

    def distance(self, other):
        """
        Calculates distance between two quaternions
        :param other: other Quaternion
        :return: distance to other Quaternion
        """
        if isinstance(other, Quaternion):
            return (self - other).norm()
        else:
            raise ValueError('Only another quaternion can be subtracted from quaternion')

    def versor(self):
        """
        Return versor for current quaternion
        :return: Quaternion which is versor for the given quaternion
        """
        if not np.allclose(self.quadruple, np.zeros(4)):
            return 1 / self.norm() * self
        else:
            raise ZeroDivisionError('Zero quaternion has no versor')

    def reciprocal(self):
        """
        Return quaternion reciprocal to given
        """
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
        if not np.allclose(self.norm(), [0.0]):
            a = self.scalar_part()
            v = self.vector_part()
            v_norm = np.sqrt(np.sum(v * v))
            if not np.allclose(v_norm, [0.0]):
                n_hat = v / v_norm
            else:
                n_hat = np.zeros(3)
            theta = np.arccos(a / self.norm())
            return self.norm(), n_hat, theta
        else:
            return 0, np.zeros(3), 0

    def _set_polar(self, polar_components):
        q_norm, n_hat, theta = polar_components
        n_hat = np.array(n_hat, dtype=np.float)
        assert q_norm >= 0
        assert np.allclose(np.sqrt(np.sum(n_hat * n_hat)), [1.0])
        a = q_norm * np.cos(theta)
        v = n_hat * q_norm * np.sin(theta)
        self._set_quadruple(np.hstack((a, v)))

    """
    Property to get/set quaternion using polar notation
    """
    polar = property(_get_polar, _set_polar)

    def __pow__(self, power):
        if isinstance(power, numbers.Number):
            q_norm, n_hat, theta = self.polar
            result = Quaternion()
            result.polar = (q_norm ** power, n_hat, theta * power)
            return result
        else:
            raise ValueError('BDQuaternions can be raised only into real-value power')

    def real_matrix(self):
        """
        Calculates real 4x4 matrix representation of the quaternion
        :return: 4x4 real numpy array matrix
        """
        return real_matrix(self.quadruple)

    def complex_matrix(self):
        """
        Calculates complex 2x2 matrix representation of the quaternion
        :return: 2x2 complex numpy array matrix
        """
        return complex_matrix(self.quadruple)
