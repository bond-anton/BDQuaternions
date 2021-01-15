import numpy as np
import numbers

from cython import boundscheck, wraparound

from libc.math cimport sqrt, cos, sin, acos
from cpython.array cimport array, clone
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

    def __repr__(self):
        return str(self)

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
            return NotImplemented


    @property
    def quadruple(self):
        return np.asarray(self.__quadruple)

    @quadruple.setter
    def quadruple(self, double[:] quadruple):
        self.__quadruple[0] = quadruple[0]
        self.__quadruple[1] = quadruple[1]
        self.__quadruple[2] = quadruple[2]
        self.__quadruple[3] = quadruple[3]

    @wraparound(False)
    @boundscheck(False)
    cpdef double scalar_part(self):
        """
        Calculates scalar part of the Quaternion
        :return: scalar part of the Quaternion
        """
        return self.__quadruple[0]

    @wraparound(False)
    @boundscheck(False)
    cpdef double[:] vector_part(self):
        """
        Calculates vector part of the Quaternion
        :return: vector part of the Quaternion
        """
        return self.__quadruple[1:4]

    @wraparound(False)
    @boundscheck(False)
    cdef double[:] __conjugate(self):
        cdef:
            double[:] v_part = self.vector_part()
            array[double] quadruple, template = array('d')
        quadruple = clone(template, 4, zero=False)
        quadruple[0] = self.scalar_part()
        quadruple[1] = -v_part[0]
        quadruple[2] = -v_part[1]
        quadruple[3] = -v_part[2]
        return quadruple

    cpdef Quaternion conjugate(self):
        """
        Calculates conjugate for the Quaternion
        :return: Quaternion which is conjugate of current quaternion
        """
        return Quaternion(self.__conjugate())

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

    cpdef double _norm(self):
        """
        Calculates the norm of the Quaternion
        :return: norm of Quaternion
        """
        return norm(self.__quadruple)

    @property
    def norm(self):
        return self._norm()

    cpdef double distance(self, Quaternion other):
        """
        Calculates distance between two quaternions
        :param other: other Quaternion
        :return: distance to other Quaternion
        """
        return (self - other)._norm()

    cpdef Quaternion versor(self):
        """
        Return versor for current quaternion
        :return: Quaternion which is versor for the given quaternion
        """
        return 1 / self._norm() * self

    cpdef Quaternion reciprocal(self):
        """
        Return quaternion reciprocal to given
        """
        return 1 / (self._norm() ** 2) * self.conjugate()

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
        cdef:
            double a, v_norm, theta
            double[:] v
            array[double] n_hat, template = array('d')
        n_hat = clone(template, 3, zero=False)
        if not np.allclose(self._norm(), [0.0]):
            a = self.scalar_part()
            v = self.vector_part()
            v_norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if v_norm > 0.0:
                n_hat[0] = v[0] / v_norm
                n_hat[1] = v[1] / v_norm
                n_hat[2] = v[2] / v_norm
            else:
                n_hat[0] = 0.0
                n_hat[1] = 0.0
                n_hat[2] = 0.0
            theta = acos(a / self._norm())
            return self._norm(), n_hat, theta
        else:
            n_hat[0] = 0.0
            n_hat[1] = 0.0
            n_hat[2] = 0.0
            return 0.0, n_hat, 0.0

    @polar.setter
    def polar(self, polar_components):
        q_norm, n_hat, theta = polar_components
        n_hat = np.array(n_hat, dtype=np.double)
        assert q_norm >= 0
        a = q_norm * cos(theta)
        v = n_hat * q_norm * sin(theta)
        self.__quadruple = np.hstack((a, v))

    def __pow__(x, power, modulo):
        if isinstance(x, Quaternion) and isinstance(power, numbers.Number):
            q_norm, n_hat, theta = x.polar
            result = Quaternion(np.array([0, 0, 0, 1], dtype=np.double))
            result.polar = (q_norm ** power, n_hat, theta * power)
            return result
        else:
            return NotImplemented

    cpdef double[:, :] real_matrix(self):
        """
        Calculates real 4x4 matrix representation of the quaternion
        :return: 4x4 real numpy array matrix
        """
        return real_matrix(self.__quadruple)

    cpdef double complex[:, :] complex_matrix(self):
        """
        Calculates complex 2x2 matrix representation of the quaternion
        :return: 2x2 complex numpy array matrix
        """
        return complex_matrix(self.__quadruple)
