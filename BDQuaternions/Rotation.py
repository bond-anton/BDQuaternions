from __future__ import division, print_function
import numpy as np

from BDQuaternions._quaternion_operations import mul, norm,\
    quaternion_to_rotation_matrix, quaternion_from_rotation_matrix
# from BDQuaternions._euler_angles import check_euler_angles_convention,\
#     euler_angles_from_quaternion, euler_angles_to_quaternion

from BDQuaternions import UnitQuaternion


class Rotation(UnitQuaternion):
    """
    Rotation is the special class on top of UnitQuaternion dealing with 3D rotations
    Rotation restricts quaternion binary operations to ones allowed for rotations group
    """

    def __init__(self, quadruple=None, euler_angles_convention=None):
        self.__euler_angles_convention = None
        self.__euler_angles = None
        if quadruple is None:
            quadruple = [1, 0, 0, 0]
        quadruple = quadruple
        assert np.allclose(norm(quadruple), [1.0])
        super(Rotation, self).__init__(quadruple)
        self.euler_angles_convention = euler_angles_convention

    def __eq__(self, other):
        if isinstance(other, Rotation):
            return np.allclose(self.quadruple, other.quadruple) or np.allclose(self.quadruple, -other.quadruple)
        else:
            raise ValueError('Only another quaternion can be compared to given quaternion')

    def conjugate(self):
        """
        Calculates conjugate for the Rotation quaternion
        :return: Rotation quaternion which is conjugate of current quaternion
        """
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return Rotation(quadruple, euler_angles_convention=self.euler_angles_convention['title'])

    def reciprocal(self):
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
        self.quadruple = quaternion_from_rotation_matrix(m)

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
        information += 'Euler angles: %s\n' % self.euler_angles_convention['description'] + '\n'
        information += str(self.euler_angles) + '\n'
        information += 'rotation matrix:\n'
        information += str(self.rotation_matrix) + '\n'
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
        self.__euler_angles_convention = check_euler_angles_convention(euler_angles_convention)

    """
        Euler angles representation of Rotation quaternion
        euler_angles is a get/set property
    """

    @property
    def euler_angles(self):
        return euler_angles_from_quaternion(self.quadruple, self.euler_angles_convention)

    @euler_angles.setter
    def euler_angles(self, euler_angles_components):
        ai, aj, ak = euler_angles_components
        quadruple = euler_angles_to_quaternion(ai, aj, ak, self.euler_angles_convention)
        self.quadruple = quadruple

    def __add__(self, other):
        raise TypeError('Wrong operation for rotations \'+\'.')

    def __radd__(self, other):
        raise TypeError('Wrong operation for rotations \'+\'.')

    def __sub__(self, other):
        raise TypeError('Wrong operation for rotations \'-\'.')

    def __rsub__(self, other):
        raise TypeError('Wrong operation for rotations \'-\'.')

    def __mul__(self, other):
        if isinstance(other, Rotation):
            quadruple = mul(self.quadruple, other.quadruple)
            return Rotation(quadruple, euler_angles_convention=self.euler_angles_convention['title'])
        else:
            raise ValueError('Rotation can be multiplied only by another rotation')

    def __rmul__(self, other):
        if isinstance(other, Rotation):
            return other * self
        else:
            raise ValueError('Rotation can be multiplied only by another rotation')

    def rotate(self, xyz):
        """
        Apply rotation to vector or array of vectors
        :param xyz: vector or array of vectors
        :return: rotated vector or array of vectors
        """
        xyz = np.array(xyz, dtype=np.float)
        if len(xyz.shape) == 2:
            if xyz.shape[1] != 3 and xyz.shape[0] == 3:
                xyz = xyz.T
            elif 3 not in xyz.shape:
                raise ValueError('Input must be a single point or an array of points coordinates with shape Nx3')
        elif len(xyz.shape) == 1 and xyz.size == 3:
            xyz = xyz.reshape(1, 3)
        else:
            raise ValueError('Input must be a single point or an array of points coordinates with shape Nx3')
        result = np.dot(self.rotation_matrix, xyz.T).T
        if xyz.size == 3:
            return result[0]
        else:
            return result
