from __future__ import division, print_function
import numbers
import numpy as np

from cpython.object cimport Py_EQ, Py_NE
from libc.float cimport DBL_MIN

from .Quaternion cimport Quaternion
from ._quaternion_operations cimport mul, quaternion_to_rotation_matrix, quaternion_from_rotation_matrix
from .UnitQuaternion cimport UnitQuaternion
from .EulerAnglesConventions cimport Conventions, Convention
from .EulerAngles cimport EulerAngles


conventions = Conventions()


cdef class Rotation(UnitQuaternion):
    """
    Rotation is the special class on top of UnitQuaternion dealing with 3D rotations
    Rotation restricts quaternion binary operations to ones allowed for rotations group
    """

    def __init__(self, double[:] quadruple=np.array([1, 0, 0, 0], dtype=np.double),
                 Convention euler_angles_convention=conventions.get_convention('Bunge')):
        self.__euler_angles_convention = euler_angles_convention
        super(Rotation, self).__init__(quadruple)

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Rotation) and isinstance(y, Rotation):
                return np.allclose(x.quadruple, y.quadruple) or np.allclose(x.quadruple, -y.quadruple)
            return False
        elif op == Py_NE:
            if isinstance(x, Rotation) and isinstance(y, Rotation):
                return not (np.allclose(x.quadruple, y.quadruple) or np.allclose(x.quadruple, -y.quadruple))
            return True
        else:
            return NotImplemented

    cpdef Rotation conjugate(self):
        """
        Calculates conjugate for the Rotation quaternion
        :return: Rotation quaternion which is conjugate of current quaternion
        """
        return Rotation(self.__conjugate(), euler_angles_convention=self.__euler_angles_convention)

    cpdef Rotation reciprocal(self):
        """
        for Unit quaternion reciprocal is equal to conjugate
        :return: Rotation quaternion reciprocal to current quaternion
        """
        return self.conjugate()

    """
    roation matrix representation of Rotation quaternion
    rotation_matrix is a get/set property
    """

    @property
    def rotation_matrix(self):
        return quaternion_to_rotation_matrix(self.quadruple)

    @rotation_matrix.setter
    def rotation_matrix(self, m):
        self.__quadruple = quaternion_from_rotation_matrix(m)

    """
    axis and angle representation of Rotation quaternion
    axis_angle is a get/set property
    """

    @property
    def axis_angle(self):
        _, axis, theta = self.polar
        return axis, theta * 2

    @axis_angle.setter
    def axis_angle(self, axis_angle_components):
        axis, theta = axis_angle_components
        axis = np.array(axis, dtype=np.float)
        axis_norm = np.sqrt(np.sum(axis * axis))
        if axis_norm > 0:
            axis /= axis_norm
        self.polar = 1, axis, theta / 2

    def __str__(self):
        information = 'Rotation quaternion: ' + str(self.quadruple) + '\n'
        information += 'Euler angles: %s\n' % self.euler_angles_convention.description + '\n'
        information += str(self.euler_angles) + '\n'
        information += 'rotation matrix:\n'
        information += str(np.asarray(self.rotation_matrix)) + '\n'
        information += 'rotation axis, angle:\n'
        information += str(self.axis_angle) + '\n'
        return information

    """
        euler angles convention get/set property
    """

    @property
    def euler_angles_convention(self):
        return self.__euler_angles_convention

    @euler_angles_convention.setter
    def euler_angles_convention(self, euler_angles_convention):
        if isinstance(euler_angles_convention, Convention):
            self.__euler_angles_convention = euler_angles_convention
        else:
            convention = conventions.get_convention(str(euler_angles_convention))
            self.__euler_angles_convention = convention

    """
        Euler angles representation of Rotation quaternion
        euler_angles is a get/set property
    """

    @property
    def euler_angles(self):
        cdef EulerAngles ea
        ea = EulerAngles(np.zeros(3, dtype=np.double), convention=self.__euler_angles_convention)
        ea.from_quaternion(self, self.__euler_angles_convention)
        return ea

    @euler_angles.setter
    def euler_angles(self, EulerAngles ea):
        cdef Rotation r
        r = ea.to_quaternion()
        self.__quadruple = r.__quadruple
        self.__euler_angles_convention = r.__euler_angles_convention

    def __add__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented

    def __mul__(x, y):
        if isinstance(x, Rotation) and isinstance(y, Rotation):
            quadruple = mul(x.quadruple, y.quadruple)
            return Rotation(quadruple, euler_angles_convention=x.euler_angles_convention)
        elif isinstance(x, Rotation) and isinstance(y, UnitQuaternion):
            quadruple = mul(x.quadruple, y.quadruple)
            return Rotation(quadruple, euler_angles_convention=x.euler_angles_convention)
        elif isinstance(x, UnitQuaternion) and isinstance(y, Rotation):
            quadruple = mul(x.quadruple, y.quadruple)
            return Rotation(quadruple, euler_angles_convention=y.euler_angles_convention)
        elif isinstance(x, Rotation) and isinstance(y, Quaternion):
            quadruple = mul(x.quadruple, y.quadruple)
            return Quaternion(quadruple)
        elif isinstance(x, UnitQuaternion) and isinstance(y, Rotation):
            quadruple = mul(x.quadruple, y.quadruple)
            return Quaternion(quadruple)
        elif isinstance(x, Rotation) and isinstance(y, numbers.Number):
            if abs(float(y) - 1) < 4 * DBL_MIN:
                return Rotation(x.quadruple, euler_angles_convention=x.euler_angles_convention)
            elif abs(float(y) + 1) < 4 * DBL_MIN:
                return Rotation(-1 * x.quadruple, euler_angles_convention=x.euler_angles_convention)
            else:
                return Quaternion(x.quadruple) * y
        elif isinstance(x, numbers.Number) and isinstance(y, Rotation):
            if abs(float(x) - 1) < 4 * DBL_MIN:
                return Rotation(y.quadruple, euler_angles_convention=y.euler_angles_convention)
            elif abs(float(x) + 1) < 4 * DBL_MIN:
                return Rotation(-1 * y.quadruple, euler_angles_convention=y.euler_angles_convention)
            else:
                return Quaternion(y.quadruple) * x
        else:
            return NotImplemented

    cpdef double[:, :] rotate(self, double[:, :] xyz):
        """
        Apply rotation to vector or array of vectors
        :param xyz: vector or array of vectors
        :return: rotated vector or array of vectors
        """
        return np.dot(self.rotation_matrix, xyz.T).T
