from __future__ import division
import unittest
import numpy as np

from BDQuaternions._euler_angles import EulerAngles
from BDQuaternions._euler_angles_conventions import Conventions, Convention, Function


class TestEulerAngles(unittest.TestCase):

    def setUp(self):
        self.conventions = Conventions()

    def test_euler_angles(self):
        convention = self.conventions.get_convention(self.conventions.default_convention)
        ea_data = np.array([np.pi / 4, np.pi / 3, np.pi / 4])
        ea = EulerAngles(ea_data, convention)
        np.testing.assert_allclose(ea.euler_angles, ea_data)

    def test_euler_angles_to_matrix(self):
        convention_1 = self.conventions.get_convention('Roe')
        ea_data = np.deg2rad([0, 0, 0])
        ea = EulerAngles(ea_data, convention_1)
        m = ea.rotation_matrix()
        np.testing.assert_allclose(m, np.eye(3))

        ea_data = np.deg2rad([-47, 120, 90])
        ea = EulerAngles(ea_data, convention_1)
        m = ea.rotation_matrix()
        ea.from_rotation_matrix(m, convention_1)
        np.testing.assert_allclose(np.rad2deg(ea_data), np.rad2deg(ea.euler_angles))

        ea_data = np.deg2rad([0, 0, 90])
        ea = EulerAngles(ea_data, convention_1)
        m = ea.rotation_matrix()
        ea.from_rotation_matrix(m, convention_1)
        np.testing.assert_allclose(np.rad2deg(ea_data), np.rad2deg(ea.euler_angles))
        m2 = np.dot(m, m)
        ea.from_rotation_matrix(m2, convention_1)
        np.testing.assert_allclose(np.rad2deg([ea_data[0],
                                               ea_data[1],
                                               ea_data[2] * 2]),
                                   np.rad2deg([ea.euler_angles[0],
                                               ea.euler_angles[1],
                                               ea.euler_angles[2]]))
'''
    def test_euler_angles_to_rotation_matrix(self):

        class Canova2Synth(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = euler_angles[0] + 0
                result[1] = euler_angles[1]
                result[2] = euler_angles[2] + 0
                return result

        class Synth2Canova(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = euler_angles[0] - 0
                result[1] = euler_angles[1]
                result[2] = euler_angles[2] - 0
                return result

        synthetic_convention = Convention('Synthetic 1', 'XYZr', ['Phi', 'Theta', 'rho'], ['alpha', 'beta', 'gamma'],
                                          [2, 1, 0, 1], description='', parent='Kocks',
                                          to_parent=Synth2Canova(), from_parent=Canova2Synth())
        self.assertEqual(synthetic_convention.parent.label, 'Kocks')
        ea_data = np.array([np.pi * 7/8, np.pi / 8, -np.pi / 4])
        ea = EulerAngles(ea_data, synthetic_convention)
        m = ea.rotation_matrix()
        ea.from_rotation_matrix(m, synthetic_convention)
        np.testing.assert_allclose(ea.euler_angles, ea_data)
'''

'''
    def test_euler_angles_conventions_conversion(self):
        convention_1 = ea.check_euler_angles_convention('Bunge')
        convention_2 = ea.check_euler_angles_convention('Nautical')
        ai, aj, ak = np.deg2rad([30, 45, 70])
        ax, ay, az = ea.change_euler_angles_convention(ai, aj, ak, convention_1, convention_2)
        ai_1, aj_1, ak_1 = ea.change_euler_angles_convention(ax, ay, az, convention_2, convention_1)
        np.testing.assert_allclose(np.array([ai, aj, ak]), np.array([ai_1, aj_1, ak_1]))

    def test_euler_angles_to_matrix(self):
        convention_1 = ea.check_euler_angles_convention('Bunge')
        ai, aj, ak = np.deg2rad([0, 0, 0])
        m = ea.euler_angles_to_matrix(ai, aj, ak, convention_1)
        np.testing.assert_allclose(m, np.eye(3))
        ai, aj, ak = np.deg2rad([0, 0, 90])
        m = ea.euler_angles_to_matrix(ai, aj, ak, convention_1)
        ai_1, aj_1, ak_1 = ea.euler_angles_from_matrix(m, convention_1)
        np.testing.assert_allclose(np.rad2deg([ai_1, aj_1, ak_1]), np.rad2deg([ai, aj, ak]))
        m2 = np.dot(m, m)
        ai_1, aj_1, ak_1 = ea.euler_angles_from_matrix(m2, convention_1)
        np.testing.assert_allclose(np.rad2deg([ai_1, aj_1, ak_1]), np.rad2deg([ai, aj, ak * 2]))

    def test_euler_angles_to_quaternion(self):
        convention_1 = ea.check_euler_angles_convention('Bunge')
        ai, aj, ak = np.deg2rad([0, 0, 0])
        q = ea.euler_angles_to_quaternion(ai, aj, ak, convention_1)
        np.testing.assert_allclose(q, [1, 0, 0, 0])
        ai, aj, ak = np.deg2rad([0, 0, 90])
        q = ea.euler_angles_to_quaternion(ai, aj, ak, convention_1)
        ai_1, aj_1, ak_1 = ea.euler_angles_from_quaternion(q, convention_1)
        np.testing.assert_allclose(np.rad2deg([ai_1, aj_1, ak_1]), np.rad2deg([ai, aj, ak]))
        q2 = qo.mul(q, q)
        ai_1, aj_1, ak_1 = ea.euler_angles_from_quaternion(q2, convention_1)
        np.testing.assert_allclose(np.rad2deg([ai_1, aj_1, ak_1]), np.rad2deg([ai, aj, ak * 2]))
'''