from __future__ import division
import unittest
import numpy as np

from Quaternions import Quaternion
from Quaternions import functions as qf


class TestQuaternion(unittest.TestCase):

    def setUp(self):
        self.q1 = Quaternion(np.array([0, 0, 0, 1]))
        self.q2 = Quaternion(np.array([1, 2, 3, 4]))

    def test_quaternion_constructor(self):
        np.testing.assert_allclose(self.q1.quadruple, np.array([0, 0, 0, 1]))
        self.assertRaises(ValueError, Quaternion, np.array([0, 0, 0]))
        self.assertRaises(ValueError, Quaternion, 'xxx')

    def test_scalar_vector_part(self):
        np.testing.assert_allclose(self.q1.scalar_part(), self.q1.quadruple[0])
        np.testing.assert_allclose(self.q1.vector_part(), self.q1.quadruple[1:])

    def test_multiplication(self):
        r1 = self.q1.scalar_part()
        r2 = self.q2.scalar_part()
        v1 = self.q1.vector_part()
        v2 = self.q2.vector_part()
        r3 = r1 * r2 - np.dot(v1, v2)
        v3 = r1 * v2 + r2 * v1 + np.cross(v1, v2)
        quadruple = np.array(np.hstack((r3, v3)))
        np.testing.assert_allclose((self.q1 * self.q2).quadruple, quadruple)
        np.testing.assert_allclose((self.q1 * 3).quadruple, self.q1.quadruple * 3)
        np.testing.assert_allclose((3 * self.q1).quadruple, self.q1.quadruple * 3)

    def test_addition(self):
        np.testing.assert_allclose((self.q1 + self.q2).quadruple, np.array([1, 2, 3, 5]))
        np.testing.assert_allclose((self.q1 - self.q2).quadruple, np.array([-1, -2, -3, -3]))
        np.testing.assert_allclose((self.q1 - 3).quadruple, np.array([-3, 0, 0, 1]))

    def test_conjugation(self):
        np.testing.assert_allclose(self.q2.conjugate().quadruple, np.array([1, -2, -3, -4]))
        self.assertEqual((self.q1 * self.q2).conjugate(), self.q2.conjugate() * self.q1.conjugate())
        self.assertNotEqual((self.q1 * self.q2).conjugate(), self.q1.conjugate() * self.q2.conjugate())

    def test_norm(self):
        np.testing.assert_allclose(self.q1.norm(), 1.0)
        np.testing.assert_allclose((3 * self.q1).norm(), 3.0)
        np.testing.assert_allclose((self.q1 * self.q2).norm(), self.q1.norm() * self.q2.norm())
        np.testing.assert_allclose((self.q2 * self.q1).norm(), self.q1.norm() * self.q2.norm())

    def test_distance(self):
        self.assertEqual(self.q1.distance(self.q2), (self.q1 - self.q2).norm())
        self.assertEqual(self.q2.distance(self.q1), (self.q1 - self.q2).norm())
        self.assertEqual(self.q1.distance(self.q2), self.q2.distance(self.q1))
        self.assertEqual(self.q1.distance(self.q1), 0)

    def test_versor(self):
        np.testing.assert_allclose(self.q1.versor().norm(), 1)
        np.testing.assert_allclose(self.q2.versor().norm(), 1)
        self.assertRaises(ZeroDivisionError, Quaternion([0, 0, 0, 0]).versor)

    def test_reciprocal(self):
        self.assertRaises(ZeroDivisionError, Quaternion([0, 0, 0, 0]).reciprocal)
        self.assertEqual(self.q1.reciprocal(), self.q1.conjugate())
        self.assertEqual(self.q2.reciprocal(), self.q2.conjugate() * (1 / self.q2.norm() ** 2))

    def test_division(self):
        self.assertRaises(ValueError, self.q1.__div__, self.q2)
        self.assertRaises(ValueError, self.q1.__rdiv__, self.q2)
        self.assertEqual(self.q1 / 3, 1 / 3 * self.q1)
        self.assertEqual(3 / self.q1, 3 * self.q1.reciprocal())

    def test_exp_and_log(self):
        components_magnitude = 1
        my_quaternion = Quaternion((np.random.random(4) - 0.5) * 2 * components_magnitude)
        self.assertEqual(qf.exp(qf.log(my_quaternion)), my_quaternion)
        self.assertEqual(qf.log(qf.exp(my_quaternion)), my_quaternion)
