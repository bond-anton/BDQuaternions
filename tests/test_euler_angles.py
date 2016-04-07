from __future__ import division
import unittest
import numpy as np

from Quaternions import _euler_angles as ea, _quaternion_operations as qo
from Quaternions._euler_angles_conventions import default_convention, derived_conventions


class TestEulerAngles(unittest.TestCase):

    def test_euler_angles_conventions(self):
        wrong_convention = 'asdfs434%%%ggg@'
        fallback_convention = ea.check_euler_angles_convention(wrong_convention)
        self.assertEqual(fallback_convention, ea.check_euler_angles_convention(default_convention))

    def test_euler_angles_nested_derived_conventions(self):
        Synthetic_convention = {'variants': ('synthetic 1', 'synthetic_1', 'synthetic1',),
                                'parent_convention': 'Canova',
                                'axes_labels': ('RD', 'TD', 'ND'),
                                'labels': ('omega_1', 'theta', 'phi_1'),
                                'from_parent': lambda omega, Theta, phi: (2 * omega, Theta + 1, 3 * phi),
                                'to_parent': lambda omega_1, theta, phi_1: (omega_1 / 2, theta - 1, phi_1 / 3),
                                'description': 'Synthetic (omega_1, theta, phi_1) convention for testing purposes'}
        convention_name = 'Synthetic 1'
        derived_conventions[convention_name] = Synthetic_convention
        convention = ea.check_euler_angles_convention(convention_name)
        ai, aj, ak = np.deg2rad([30, 45, 70])
        ax, ay, az, parent_convention = ea.euler_angles_in_parent_convention(ai, aj, ak, convention)
        self.assertEqual(parent_convention['title'], 'Roe')
        ai_1, aj_1, ak_1 = ea.euler_angles_from_parent_convention(ax, ay, az, convention)
        np.testing.assert_allclose(np.array([ai, aj, ak]), np.array([ai_1, aj_1, ak_1]))

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
