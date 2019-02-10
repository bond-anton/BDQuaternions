from __future__ import division, print_function
import sys
import unittest
import numpy as np

from BDQuaternions import Rotation
from BDQuaternions import Conventions, EulerAngles


class TestRotation(unittest.TestCase):

    def setUp(self):
        self.q1 = Rotation(np.array([1, 0, 0, 0], dtype=np.double))

    def test_rotation_constructor(self):
        np.testing.assert_allclose(self.q1.quadruple, np.array([1, 0, 0, 0]))
        self.assertRaises(ValueError, Rotation, np.array([0, 0, 0]))
        if sys.version_info > (3, 0):
            self.assertRaises(TypeError, Rotation, 'xxx')
        else:
            self.assertRaises(BufferError, Rotation, 'xxx')
        q2 = Rotation()
        self.assertEqual(self.q1, q2)
        self.assertFalse(self.q1 == 1)

    def test_str(self):
        print(str(self.q1))
        self.assertTrue(str(self.q1))

    def test_euler_angles(self):
        np.testing.assert_allclose(self.q1.euler_angles.euler_angles, np.zeros(3))

    def test_forbidden_math(self):
        with self.assertRaises(TypeError):
            _ = self.q1 + 3
        with self.assertRaises(TypeError):
            _ = self.q1 - 3

    def test_mul(self):
        self.assertEqual(self.q1 * self.q1, self.q1)
        with self.assertRaises(TypeError):
            _ = self.q1 * 'x'

    def test_axis_angle(self):
        axis, angle = self.q1.axis_angle
        np.testing.assert_allclose(axis, np.zeros(3))
        np.testing.assert_allclose(angle, np.zeros(1))
        self.q1.axis_angle = ([1, 1, 1], np.deg2rad(45))
        axis, angle = self.q1.axis_angle
        np.testing.assert_allclose(axis, np.ones(3) / np.sqrt(3))
        np.testing.assert_allclose(angle, np.deg2rad(45))

    def test_rotation_matrix(self):
        np.testing.assert_allclose(self.q1.rotation_matrix, np.eye(3))
        self.q1.axis_angle = ([1, 0, 0], np.deg2rad(45))
        m = self.q1.rotation_matrix
        v = np.array([0, 0, 1])
        v_r_m = np.dot(m, v)
        qv = Rotation(np.hstack([0.0, v]))
        v_r_q = (self.q1 * qv * self.q1.reciprocal()).quadruple[1:]
        np.testing.assert_allclose(v_r_m, v_r_q)

        m2 = np.dot(m, m)
        self.q1.rotation_matrix = m2
        v_r_m = np.dot(m2, v)
        v_r_q = (self.q1 * qv * self.q1.reciprocal()).quadruple[1:]
        np.testing.assert_allclose(v_r_m, v_r_q, atol=np.finfo(float).eps * 4)

        self.q1.rotation_matrix = m
        self.q1 = self.q1 * self.q1
        v_r_m = np.dot(m2, v)
        v_r_q = (self.q1 * qv * self.q1.reciprocal()).quadruple[1:]
        np.testing.assert_allclose(v_r_m, v_r_q, atol=np.finfo(float).eps * 4)

    def test_rotate(self):
        conventions = Conventions()
        np.testing.assert_allclose(self.q1.rotate(np.array([[1.0, 0.0, 0.0]])), np.array([[1, 0, 0]]))
        self.q1.axis_angle = ([0, 0, 1], np.pi / 2)
        self.q1.euler_angles_convention = conventions.get_convention('Bunge')
        self.q1.euler_angles = EulerAngles(np.array([np.pi/2, 0, 0]), self.q1.euler_angles_convention)
        np.testing.assert_allclose(self.q1.rotate(np.array([[1.0, 0.0, 0.0]])),
                                   np.array([[0.0, 1.0, 0.0]]), atol=np.finfo(float).eps * 4)
        with self.assertRaises(TypeError):
            self.q1.rotate([0, 1])
        with self.assertRaises(ValueError):
            self.q1.rotate(np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.double))
        with self.assertRaises(TypeError):
            self.q1.rotate('x')
