from __future__ import division
import unittest
import numpy as np

from Quaternions import Rotation


class TestRotation(unittest.TestCase):

    def setUp(self):
        self.q1 = Rotation(np.array([1, 0, 0, 0]))

    def test_rotation_constructor(self):
        np.testing.assert_allclose(self.q1.quadruple, np.array([1, 0, 0, 0]))
        self.assertRaises(ValueError, Rotation, np.array([0, 0, 0]))
        self.assertRaises(ValueError, Rotation, 'xxx')

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
