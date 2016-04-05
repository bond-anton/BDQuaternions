from __future__ import division, print_function
import numbers
import numpy as np

from Quaternions import UnitQuaternion


class Rotation(UnitQuaternion):

    def __init__(self, quadruple=None, euler_angles_convention=None):
        self._euler_angles_convention = None
        self._euler_angles = None
        if quadruple is None:
            quadruple = np.array([1, 0, 0, 0])
        quadruple = np.array(quadruple, dtype=np.float)
        assert np.allclose(np.sum(quadruple ** 2), 1.0)
        super(Rotation, self).__init__(quadruple)
        self._set_euler_angles_convention(euler_angles_convention)

    def conjugate(self):
        quadruple = np.hstack((self.scalar_part(), -self.vector_part()))
        return Rotation(quadruple)

    def reciprocal(self):
        return self.conjugate()

    def _get_rotation_matrix(self):
        w, x, y, z = self.quadruple
        m = np.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
                      [2 * x * y - 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z + 2 * w * x],
                      [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x**2 - 2 * y**2]])
        return m

    def _set_rotation_matrix(self, m):
        det_m = np.linalg.det(m)
        if abs(1 - det_m ** 2) > 1e-6:
            raise ValueError('Not a rotation matrix. det M = %2.2g' % det_m)
        inv_m = np.linalg.inv(m)
        if np.allclose(det_m, 1.0) and np.allclose(m.T, inv_m):
            m = np.array(m, dtype=np.float)
            if m.shape != (3, 3):
                raise ValueError('3x3 rotation matrix expected, got' + str(m))
            t = np.trace(m)
            if t > 3 * np.finfo(float).eps:
                r = np.sqrt(1 + t)
                w = 0.5 * r
                s = 0.5 / r
                x = (m[2, 1] - m[1, 2]) * s
                y = (m[0, 2] - m[2, 0]) * s
                z = (m[1, 0] - m[0, 1]) * s
            elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
                r = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
                s = 0.5 / r
                w = (m[2, 1] - m[1, 2]) * s
                x = 0.5 * r
                y = (m[0, 1] + m[1, 0]) * s
                z = (m[2, 0] + m[0, 2]) * s
            elif m[1, 1] >= m[2, 2]:
                r = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
                s = 0.5 / r
                w = (m[2, 0] - m[0, 2]) * s
                x = (m[0, 1] + m[1, 0]) * s
                y = 0.5 * r
                z = (m[1, 2] + m[2, 1]) * s
            else:
                r = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
                s = 0.5 / r
                w = (m[0, 1] - m[1, 0]) * s
                x = (m[2, 0] + m[0, 2]) * s
                y = (m[1, 2] + m[2, 1]) * s
                z = 0.5 * r
            quadruple = np.array([w, x, y, z])
        else:
            print('Not a rotation matrix. det M = %2.2g' % det_m)
            k_m = np.array([[m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]],
                            [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]],
                            [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
                            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3
            w, v_n = np.linalg.eigh(k_m)
            quadruple = v_n[[3, 0, 1, 2], np.argmax(w)]
        self.quadruple = quadruple

    rotation_matrix = property(_get_rotation_matrix, _set_rotation_matrix)

    def _get_axis_angle(self):
        _, axis, theta = self.polar
        return axis, theta * 2

    def _set_axis_angle(self, (axis, theta)):
        self.polar = 1, axis, theta / 2

    axis_angle = property(_get_axis_angle, _set_axis_angle)

    def __str__(self):
        information = 'Rotation quaternion: ' + str(self.quadruple) + '\n'
        information += 'Orientation: %s:\n' % self.euler_angles_convention['description'] + '\n'
        #information += str(self.euler_angles) + '\n'
        information += 'rotation matrix:\n'
        information += str(self.rotation_matrix) + '\n'
        information += 'rotation axis, angle:\n'
        information += str(self.axis_angle) + '\n'
        return information

    def _set_euler_angles_convention(self, euler_angles_convention):
        conventions = {
            'Bunge': {'variants': ['bunge', 'zxz'],
                      'labels': ['phi1', 'Phi', 'phi2'],
                      'description': 'Bunge (phi1 Phi phi2) ZXZ convention'},
            'Matthies': {'variants': ['matthies', 'zyz', 'nfft', 'abg'],
                         'labels': ['alpha', 'beta', 'gamma'],
                         'description': 'Matthies (alpha beta gamma) ZYZ convention'},
            'Roe': {'variants': ['roe'],
                    'labels': ['Psi', 'Theta', 'Phi'],
                    'description': 'Roe (Psi, Theta, Phi) convention'},
            'Kocks': {'variants': ['kocks'],
                      'labels': ['Psi', 'Theta', 'phi'],
                      'description': 'Kocks (Psi Theta phi) convention'},
            'Canova': {'variants': ['canova'],
                       'labels': ['omega', 'Theta', 'phi'],
                       'description': 'Canova (omega, Theta, phi) convention'}
        }
        convention = conventions['Bunge']
        if euler_angles_convention is not None:
            match = False
            for key in conventions.keys():
                if str(euler_angles_convention).lower().strip() in conventions[key]['variants']:
                    convention = conventions[key]
                    match = True
                    break
            if not match:
                print('Convention: %s not found or not supported.' % euler_angles_convention)
                print('Falling back to Bunge convention.')
            elif 'bunge' not in convention['variants']:
                print('You asked to use %s' % convention['description'])
                print('Unfortunately it is not supported for now.')
                print('Falling back to Bunge convention.')
                convention = conventions['Bunge']
        self._euler_angles_convention = convention

    def _get_euler_angles_convention(self):
        return self._euler_angles_convention

    euler_angles_convention = property(_get_euler_angles_convention, _set_euler_angles_convention)

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
            q1 = self.quadruple
            q2 = other.quadruple
            quadruple = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                                  q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                                  q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                                  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
            return Rotation(quadruple)
        else:
            raise ValueError('Rotation can be multiplied only by another rotation')

    def __rmul__(self, other):
        if isinstance(other, Rotation):
            return other * Rotation
        else:
            raise ValueError('Rotation can be multiplied only by another rotation')
