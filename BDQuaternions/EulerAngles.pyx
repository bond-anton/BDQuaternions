from __future__ import division, print_function
import numpy as np

from cython import boundscheck, wraparound


from libc.math cimport sin, cos, atan2, sqrt, M_PI
from libc.float cimport DBL_MIN
from .EulerAnglesConventions cimport Convention
from ._quaternion_operations cimport quaternion_to_rotation_matrix, quaternion_from_rotation_matrix
from .Rotation cimport Rotation

"""
Euler angles conversion algorithms after Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222

All angles are in radians by default
"""

cdef class EulerAngles(object):

    def __init__(self, double[:] euler_angles, Convention convention):
        self.__euler_angles = self.__reduce_euler_angles(euler_angles)
        self.__convention = convention

    cdef double __reduce_angle(self, double angle, bint center=True, bint half=False):
        """
        Adjusts rotation angle to be in the range [-2*pi; 2*pi]
        :param angle: angle or array-like of input angle
        :param center: if True (default) adjust angle to be within [-pi; pi]
        :param half: if True (default False) adjust angle to be within [0; pi]
        :return: reduced angle or array of reduced angles
        """
        cdef:
            double reduced_angle
        if angle > 2 * M_PI:
            reduced_angle = angle - 2 * M_PI * (angle // (2 * M_PI))
        elif angle < -2 * M_PI:
            reduced_angle = angle + 2 * M_PI * (abs(angle) // (2 * M_PI))
        else:
            reduced_angle = angle
        if center:
            if reduced_angle < -M_PI:
                reduced_angle += 2 * M_PI
            elif reduced_angle > M_PI:
                reduced_angle -= 2 * M_PI
        if half:
            if reduced_angle < 0:
                reduced_angle = -reduced_angle
        return reduced_angle

    @boundscheck(False)
    @wraparound(False)
    cdef double[:] __reduce_euler_angles(self, double[:] euler_angles):
        cdef reduced_angles = np.empty(3, dtype=np.double)
        reduced_angles[0] = self.__reduce_angle(euler_angles[0], center=True, half=False)
        reduced_angles[1] = self.__reduce_angle(euler_angles[1], center=True, half=False)
        reduced_angles[2] = self.__reduce_angle(euler_angles[2], center=True, half=False)
        return reduced_angles

    @property
    def euler_angles(self):
        return self.__euler_angles

    @euler_angles.setter
    def euler_angles(self, double[:] euler_angles):
        self.__euler_angles = self.__reduce_euler_angles(euler_angles)

    @property
    def convention(self):
        return self.__convention

    def __str__(self):
        cdef:
            int i
            str label
        label = 'Euler angles (' + self.convention.label + ' convention)\n'
        for i in range(3):
            label += self.convention.axes_labels[i] + ': %2.2f\n' % self.euler_angles[i]
        return label[:-1]

    cpdef void to_parent_convention(self):
        """
        Theoretically derived conventions can be nested endless.
        This function will bring the angles to the highest level parent convention.
        """
        if self.__convention.__parent != self.__convention:
            self.__euler_angles = self.__reduce_euler_angles(self.__convention.to_parent(self.__euler_angles))
            self.__convention = self.__convention.__parent

    @boundscheck(False)
    @wraparound(False)
    cpdef double[:, :] rotation_matrix(self):
        """
        Convert Euler angles to rotation matrix
        :return: 3x3 rotation matrix as numpy array of floats
        """
        cdef:
            double[:] euler_angles = np.empty(3, dtype=np.double)
            Convention parent_convention = self.__convention
            int inner_axis, parity, repetition, frame
            int i, j, k
            double ci, si, cj, sj, ck, sk, cc, ss, cs, sc
            double[:, :] m = np.empty((3,3), dtype=np.double)
        euler_angles[0] = self.__euler_angles[0]
        euler_angles[1] = self.__euler_angles[1]
        euler_angles[2] = self.__euler_angles[2]
        while parent_convention.__parent != parent_convention:
            euler_angles = parent_convention.to_parent(euler_angles)
            parent_convention = parent_convention.__parent
        inner_axis, parity, repetition, frame = parent_convention.__code
        i = parent_convention.__euler_safe_axis[inner_axis]
        j = parent_convention.__euler_next_axis[i + parity]
        k = parent_convention.__euler_next_axis[i - parity + 1]
        if frame:
            euler_angles[0], euler_angles[2] = euler_angles[2], euler_angles[0]
        if parity:
            euler_angles[0], euler_angles[1], euler_angles[2] = -euler_angles[0], -euler_angles[1], -euler_angles[2]
        ci = cos(euler_angles[0])
        si = sin(euler_angles[0])
        cj = cos(euler_angles[1])
        sj = sin(euler_angles[1])
        ck = cos(euler_angles[2])
        sk = sin(euler_angles[2])
        cc = ci * ck
        ss = si * sk
        cs = ci * sk
        sc = si * ck

        if repetition:
            m[i, i] = cj
            m[i, j] = si * sj
            m[i, k] = ci * sj
            m[j, i] = sj * sk
            m[j, j] = cc - cj * ss
            m[j, k] = -sc - cj * cs
            m[k, i] = -sj * ck
            m[k, j] = cs + cj * sc
            m[k, k] = cj * cc - ss
        else:
            m[i, i] = cj * ck
            m[i, j] = sj * sc - cs
            m[i, k] = sj * cc + ss
            m[j, i] = cj * sk
            m[j, j] = sj * ss + cc
            m[j, k] = sj * cs - sc
            m[k, i] = -sj
            m[k, j] = si * cj
            m[k, k] = ci * cj
        return m

    @boundscheck(False)
    @wraparound(False)
    cpdef void from_rotation_matrix(self, double[:, :] m, Convention convention):
        """
        convert rotation matrix to Euler angles
        :param m: 3x3 rotation matrix
        :param convention: Euler angles convention
        :return: ax, ay, az three Euler angles
        """
        cdef:
            double[:] euler_angles = np.zeros(3, dtype=np.double)
            Convention parent_convention = convention
            Convention current_convention = convention
            int inner_axis, parity, repetition, frame
            int i, j, k
            double ci, si, cj, sj, ck, sk, sy, cy, ax, ay, az
        while parent_convention.__parent != parent_convention:
            parent_convention = parent_convention.__parent
        inner_axis, parity, repetition, frame = parent_convention.__code
        i = parent_convention.__euler_safe_axis[inner_axis]
        j = parent_convention.__euler_next_axis[i + parity]
        k = parent_convention.__euler_next_axis[i - parity + 1]
        if repetition:
            sy = sqrt(m[i, j] * m[i, j] + m[i, k] * m[i, k])
            if sy > DBL_MIN * 4:
                ax = atan2(m[i, j], m[i, k])
                ay = atan2(sy, m[i, i])
                az = atan2(m[j, i], -m[k, i])
            else:
                ax = atan2(-m[j, k], m[j, j])
                ay = atan2(sy, m[i, i])
                az = 0.0
        else:
            cy = sqrt(m[i, i] * m[i, i] + m[j, i] * m[j, i])
            if cy > DBL_MIN * 4:
                ax = atan2(m[k, j], m[k, k])
                ay = atan2(-m[k, i], cy)
                az = atan2(m[j, i], m[i, i])
            else:
                ax = atan2(-m[j, k], m[j, j])
                ay = atan2(-m[k, i], cy)
                az = 0.0
        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        euler_angles[0] = ax
        euler_angles[1] = ay
        euler_angles[2] = az
        while parent_convention != convention:
            while current_convention.__parent != parent_convention:
                current_convention = current_convention.__parent
            euler_angles = current_convention.from_parent(euler_angles)
            parent_convention = current_convention
            current_convention = convention
        self.__euler_angles = euler_angles
        self.__convention = current_convention


    cpdef void change_convention(self, Convention new_convention):
        """
        Convert Euler angles from given to new convention
        :param new_convention: dict describing new Euler angles convention
        :return: ax, ay, az three Euler angles
        """
        cdef double[:, :] m
        m = self.rotation_matrix()
        self.from_rotation_matrix(m, new_convention)


    cdef __to_quaternion(self):
        """
        Convert Euler angles to quaternion
        :return: Quaternion as an array of for floats [w*1, x*i, y*j, z*k]
        """
        cdef:
            double[:, :] matrix
        matrix = self.rotation_matrix()
        return quaternion_from_rotation_matrix(matrix)

    cpdef Rotation to_quaternion(self):
        return Rotation(self.__to_quaternion(), euler_angles_convention=self.__convention)


    cdef void __from_quaternion(self, double[:] quadruple, Convention convention):
        """
        Convert Quaternion to Euler angles
        :param quadruple: Quaternion as an array of for floats [w*1, x*i, y*j, z*k]
        :param convention: dict describing Euler angles convention
        """
        cdef:
            double[:, :] matrix
        matrix = quaternion_to_rotation_matrix(quadruple)
        self.from_rotation_matrix(matrix, convention)


    cpdef void from_quaternion(self, Rotation quaternion, Convention convention):
        self.__from_quaternion(quaternion.quadruple, convention)
