from __future__ import division
import unittest
import numpy as np

from BDQuaternions.EulerAnglesConventions import Convention, Function, Conventions


class TestEulerAnglesConventions(unittest.TestCase):

    def setUp(self):
        self.conventions = Conventions()

    def test_euler_angles_conventions(self):
        wrong_convention = 'asdfs434%%%ggg@'
        fallback_convention = self.conventions.get_convention(wrong_convention)
        self.assertEqual(fallback_convention, self.conventions.get_convention(self.conventions.default_convention))
        self.assertTrue(self.conventions.check(self.conventions.default_convention))
        self.assertFalse(self.conventions.check(wrong_convention))

    def test_euler_angles_derived_convention(self):
        derived_convention = self.conventions.get_convention('Canova')
        self.assertEqual(derived_convention.parent, self.conventions.get_convention('Roe'))
        euler_angles = np.array([np.pi, np.pi / 3, np.pi / 4])
        euler_angles_p = derived_convention.to_parent(euler_angles)
        np.testing.assert_allclose(euler_angles_p, np.array([np.pi / 2 - np.pi, np.pi / 3, 3 * np.pi / 2 - np.pi / 4]))
        euler_angles_b = derived_convention.from_parent(euler_angles_p)
        np.testing.assert_allclose(euler_angles_b, np.array([np.pi, np.pi / 3, np.pi / 4]))

    def test_euler_angles_nested_derived_conventions(self):

        class Canova2Synth(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = 2 * euler_angles[0]
                result[1] = euler_angles[1] + 1
                result[2] = 3 * euler_angles[2]
                return result

        class Synth2Canova(Function):
            def evaluate(self, euler_angles):
                result = np.zeros(3, dtype=np.double)
                result[0] = euler_angles[0] / 2
                result[1] = euler_angles[1] - 1
                result[2] = euler_angles[2] / 3
                return result

        synthetic_convention = Convention('Synthetic 1', 'XYZr', ['Phi', 'Theta', 'rho'], ['alpha', 'beta', 'gamma'],
                                          [2, 1, 0, 1], description='', parent='Canova',
                                          to_parent=Synth2Canova(), from_parent=Canova2Synth())
        self.assertEqual(synthetic_convention.parent.label, 'Canova')
        synthetic_convention.print_convention_tree()
        euler_angles = np.array([np.pi, np.pi / 3, np.pi / 4])
        euler_angles_p = synthetic_convention.to_parent(euler_angles)
        np.testing.assert_allclose(euler_angles_p, np.array([np.pi / 2, np.pi / 3 - 1, np.pi / 12]))
        euler_angles_b = synthetic_convention.from_parent(euler_angles_p)
        np.testing.assert_allclose(euler_angles_b, np.array([np.pi, np.pi / 3, np.pi / 4]))
