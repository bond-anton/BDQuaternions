from __future__ import division, print_function
import numpy as np
import numbers

from cpython.object cimport Py_EQ, Py_NE

from ._quaternion_operations cimport mul, norm, real_matrix, complex_matrix


cdef class Quaternion(object):
    """
    General quaternion object
    """

    def __init__(self, double[:] quadruple=np.array([0, 0, 0, 1], dtype=np.double)):
        self.__quadruple = np.empty(4, dtype=np.double)
        self.__quadruple[0] = quadruple[0]
        self.__quadruple[1] = quadruple[1]
        self.__quadruple[2] = quadruple[2]
        self.__quadruple[3] = quadruple[3]

    def __str__(self):
        return str(np.asarray(self.__quadruple))

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Quaternion) and isinstance(y, Quaternion):
                return np.allclose(x.quadruple, y.quadruple)
            return False
        elif op == Py_NE:
            if isinstance(x, Quaternion) and isinstance(y, Quaternion):
                return not np.allclose(x.quadruple, y.quadruple)
            return True
        else:
            return False


    @property
    def quadruple(self):
        return np.asarray(self.__quadruple)

    @quadruple.setter
    def quadruple(self, double[:] quadruple):
        self.__quadruple[0] = quadruple[0]
        self.__quadruple[1] = quadruple[1]
        self.__quadruple[2] = quadruple[2]
        self.__quadruple[3] = quadruple[3]

    cpdef double scalar_part(self):
        """
        Calculates scalar part of the Quaternion
        :return: scalar part of the Quaternion
        """
        return self.__quadruple[0]

    cdef __vector_part(self):
        return self.__quadruple[1:]

    def vector_part(self):
        """
        Calculates vector part of the Quaternion
        :return: vector part of the Quaternion
        """
        return np.asarray(self.__vector_part())

    cpdef Quaternion conjugate(self):
        """
        Calculates conjugate for the Quaternion
        :return: Quaternion which is conjugate of current quaternion
        """
        cdef:
            double[:] quadruple = np.empty(4, dtype=np.double)
            double[:] v_part = self.__vector_part()
        quadruple[0] = self.scalar_part()
        quadruple[1] = -v_part[0]
        quadruple[2] = -v_part[1]
        quadruple[3] = -v_part[2]
        return Quaternion(quadruple)

    def __mul__(x, y):
        if isinstance(x, Quaternion) and isinstance(y, Quaternion):
            return Quaternion(mul(x.quadruple, y.quadruple))
        elif isinstance(x, Quaternion) and isinstance(y, numbers.Number):
            return Quaternion(mul(x.quadruple, np.array([np.double(y), 0, 0, 0], dtype=np.double)))
        elif isinstance(x, numbers.Number) and isinstance(y, Quaternion):
            return Quaternion(mul(np.array([np.double(x), 0, 0, 0], dtype=np.double), y.quadruple))
        else:
            return NotImplemented

    def __add__(x, y):
        if isinstance(x, Quaternion) and isinstance(y, Quaternion):
            return Quaternion(x.quadruple + y.quadruple)
        elif isinstance(x, Quaternion) and isinstance(y, numbers.Number):
            return Quaternion(x.quadruple + np.array([np.double(y), 0, 0, 0]))
        elif isinstance(x, numbers.Number) and isinstance(y, Quaternion):
            return Quaternion(y.quadruple + np.array([np.double(x), 0, 0, 0]))
        else:
            return NotImplemented

    def __sub__(x, y):
        if isinstance(x, Quaternion) and isinstance(y, Quaternion):
            return Quaternion(x.quadruple - y.quadruple)
        elif isinstance(x, Quaternion) and isinstance(y, numbers.Number):
            return Quaternion(x.quadruple - np.array([np.double(y), 0, 0, 0]))
        elif isinstance(x, numbers.Number) and isinstance(y, Quaternion):
            return Quaternion(np.array([np.double(x), 0, 0, 0]) - y.quadruple)
        else:
            return NotImplemented

    cpdef norm(self):
        """
        Calculates the norm of the Quaternion
        :return: norm of Quaternion
        """
        return norm(self.__quadruple)

    cpdef distance(self, Quaternion other):
        """
        Calculates distance between two quaternions
        :param other: other Quaternion
        :return: distance to other Quaternion
        """
        return (self - other).norm()

    cpdef versor(self):
        """
        Return versor for current quaternion
        :return: Quaternion which is versor for the given quaternion
        """
        return 1 / self.norm() * self

    cpdef reciprocal(self):
        """
        Return quaternion reciprocal to given
        """
        return 1 / (self.norm() ** 2) * self.conjugate()

    def __div__(x, y):
        if isinstance(x, Quaternion) and isinstance(y, Quaternion):
            return NotImplemented
        elif isinstance(x, Quaternion) and isinstance(y, numbers.Number):
            return x * (1.0 / np.double(y))
        elif isinstance(x, numbers.Number) and isinstance(y, Quaternion):
            return y.reciprocal() * np.double(x)
        else:
            return NotImplemented

    def __truediv__(x, y):
        if isinstance(x, Quaternion) and isinstance(y, Quaternion):
            return NotImplemented
        elif isinstance(x, Quaternion) and isinstance(y, numbers.Number):
            return x * (1.0 / np.double(y))
        elif isinstance(x, numbers.Number) and isinstance(y, Quaternion):
            return y.reciprocal() * np.double(x)
        else:
            return NotImplemented


    """
        Property to get/set quaternion using polar notation
    """
    @property
    def polar(self):
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

    @polar.setter
    def polar(self, polar_components):
        q_norm, n_hat, theta = polar_components
        n_hat = np.array(n_hat, dtype=np.float)
        assert q_norm >= 0
        assert np.allclose(np.sqrt(np.sum(n_hat * n_hat)), [1.0])
        a = q_norm * np.cos(theta)
        v = n_hat * q_norm * np.sin(theta)
        self.__quadruple = np.hstack((a, v))

    def __pow__(x, power, modulo):
        if isinstance(x, Quaternion) and isinstance(power, numbers.Number):
            q_norm, n_hat, theta = x.polar
            result = Quaternion(np.array([0, 0, 0, 1], dtype=np.double))
            result.polar = (q_norm ** power, n_hat, theta * power)
            return result
        else:
            return NotImplemented

    cpdef real_matrix(self):
        """
        Calculates real 4x4 matrix representation of the quaternion
        :return: 4x4 real numpy array matrix
        """
        return real_matrix(self.__quadruple)

    cpdef complex_matrix(self):
        """
        Calculates complex 2x2 matrix representation of the quaternion
        :return: 2x2 complex numpy array matrix
        """
        return complex_matrix(self.__quadruple)
