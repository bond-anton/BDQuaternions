from __future__ import division
import unittest
import numpy as np

from BDQuaternions.EulerAngles import EulerAngles
from BDQuaternions.EulerAnglesConventions import Conventions, Convention, Function


class TestEulerAngles(unittest.TestCase):

    def setUp(self):
        self.conventions = Conventions()
        self.conventions_list = ['XYZ', 'XZY', 'XYX', 'XZX',
                                 'YZX', 'YXZ', 'YXY', 'YZY',
                                 'ZXY', 'ZYX', 'ZXZ', 'ZYZ']

    def test_euler_angles(self):
        convention = self.conventions.get_convention(self.conventions.default_convention)
        ea_data = np.array([np.pi / 4, np.pi / 3, np.pi / 4])
        ea = EulerAngles(ea_data, convention)
        np.testing.assert_allclose(ea.euler_angles, ea_data)

    def test_euler_angles_to_rotation_matrix(self):
        for frame in ['s', 'r']:
            for axes in self.conventions_list:
                convention_1 = self.conventions.get_convention(axes + frame)
                print('CONV:', axes+frame)
                ea_data = np.deg2rad([0, 0, 0])
                ea = EulerAngles(ea_data, convention_1)
                m = ea.rotation_matrix()
                np.testing.assert_allclose(m, np.eye(3))

                for i in range(0, 360, 60):
                    for j in range(0, 360, 60):
                        for k in range(0, 360, 60):
                            ea_data = np.deg2rad([i - 180, j - 180, k - 180])
                            ea = EulerAngles(ea_data, convention_1)
                            np.testing.assert_allclose(np.rad2deg(ea_data), np.rad2deg(ea.euler_angles))
                            m = ea.rotation_matrix()
                            ea.from_rotation_matrix(m, convention_1)
                            ea_data2 = ea.euler_angles
                            m2 = ea.rotation_matrix()
                            np.testing.assert_allclose(m, m2, atol=1e-10)
                            ea.from_rotation_matrix(m2, convention_1)
                            ea_data3 = ea.euler_angles
                            np.testing.assert_allclose(np.rad2deg(ea_data2), np.rad2deg(ea_data3))
        convention_1 = self.conventions.get_convention('bunge')
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

    def test_euler_angles_derived_conventions(self):

        class Parent2Synth(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = euler_angles[0] + 0.32
                result[1] = euler_angles[1] * 2
                result[2] = euler_angles[2] + 0.75
                return result

        class Synth2Parent(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = euler_angles[0] - 0.32
                result[1] = euler_angles[1] * 2
                result[2] = euler_angles[2] - 0.75
                return result

        for parent_conv in ['Kocks', 'Canova']:
            synthetic_convention = Convention('Synthetic 1', 'ZYZr', ['Phi', 'Theta', 'rho'], ['alpha', 'beta', 'gamma'],
                                              [2, 1, 0, 1], description='', parent=parent_conv,
                                              to_parent=Synth2Parent(), from_parent=Parent2Synth())
            print('Parent convention:', parent_conv)
            self.assertEqual(synthetic_convention.parent.label, parent_conv)
            for i in range(0, 360, 60):
                for j in range(0, 360, 60):
                    for k in range(0, 360, 60):
                        ea_data = np.deg2rad([i - 180, j - 180, k - 180])
                        ea = EulerAngles(ea_data, synthetic_convention)
                        m = ea.rotation_matrix()
                        ea.from_rotation_matrix(m, synthetic_convention)
                        m2 = ea.rotation_matrix()
                        np.testing.assert_allclose(m, m2, atol=1e-10)

    def test_euler_angles_conventions_conversion(self):
        conventions_list = ['XYZ', 'XZY', 'XYX', 'XZX',
                            'YZX', 'YXZ', 'YXY', 'YZY',
                            'ZXY', 'ZYX', 'ZXZ', 'ZYZ',
                            'Nautical', 'Bunge', 'Rhoe', 'Matthies', 'Kocks', 'Canova']
        for convention_1_lbl in conventions_list:
            for convention_2_lbl in conventions_list:
                print(convention_1_lbl, '==>', convention_2_lbl)
                convention_1 = self.conventions.get_convention(convention_1_lbl)
                convention_2 = self.conventions.get_convention(convention_2_lbl)
                for i in range(0, 360, 60):
                    for j in range(0, 360, 60):
                        for k in range(0, 360, 60):
                            ea_data = np.deg2rad([i - 180, j - 180, k - 180])
                            ea = EulerAngles(ea_data, convention_1)
                            m = ea.rotation_matrix()
                            ea.change_convention(convention_2)
                            m2 = ea.rotation_matrix()
                            np.testing.assert_allclose(m, m2, atol=1e-10)
                            ea.change_convention(convention_1)
                            m3 = ea.rotation_matrix()
                            np.testing.assert_allclose(m, m3, atol=1e-10)


'''
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