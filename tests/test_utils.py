from __future__ import division
import numpy as np

from BDQuaternions import utils as utl
from BDQuaternions import _quaternion_operations as qo

import unittest


class TestUtils(unittest.TestCase):

    def test_random_rotation(self):
        for _ in range(100):
            rot = utl.random_rotation()
            np.testing.assert_allclose(rot.norm(), 1.0)

    def test_random_unit_quaternion(self):
        for _ in range(100):
            uq = utl.random_unit_quaternion()
            np.testing.assert_allclose(uq.norm(), 1.0)

    def test_random_quaternion(self):
        for _ in range(100):
            q_norm = np.random.rand() * 10.0
            q = utl.random_quaternion(q_norm)
            np.testing.assert_allclose(q.norm(), q_norm)
        q_norm = 0.0
        q = utl.random_quaternion(q_norm)
        np.testing.assert_allclose(q.norm(), q_norm)

    def test_random_rotations_array(self):
        for _ in range(10):
            shape = (np.random.randint(10), np.random.randint(10))
            rots = utl.random_rotations_array(shape)
            np.testing.assert_allclose(rots.shape, shape)
            for rot in rots.ravel():
                np.testing.assert_allclose(rot.norm(), 1.0)

    def test_random_uq_array(self):
        for _ in range(10):
            shape = (np.random.randint(10), np.random.randint(10))
            uqs = utl.random_unit_quaternions_array(shape)
            np.testing.assert_allclose(uqs.shape, shape)
            for uq in uqs.ravel():
                np.testing.assert_allclose(uq.norm(), 1.0)

    def test_random_q_array(self):
        for _ in range(10):
            shape = (np.random.randint(10), np.random.randint(10))
            q_norm = np.random.rand() * 10.0
            qs = utl.random_quaternions_array(shape, q_norm)
            np.testing.assert_allclose(qs.shape, shape)
            for q in qs.ravel():
                np.testing.assert_allclose(q.norm(), q_norm)
