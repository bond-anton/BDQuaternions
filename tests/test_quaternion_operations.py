from __future__ import division
import warnings
import numpy as np

from BDQuaternions import _quaternion_operations as qo

import unittest


class TestQuaternionOperations(unittest.TestCase):

    def test_quaternion_norm(self):
        q = np.array([1, 0, 0, 0], dtype=np.double)
        np.testing.assert_allclose(qo.norm(q), [1])
        q = np.array([0, 0, 0, 0], dtype=np.double)
        np.testing.assert_allclose(qo.norm(q), [0])
        q = np.array([0, 0, 1, 0], dtype=np.double)
        np.testing.assert_allclose(qo.norm(q), [1])
        q = np.array([1, 0, 1, 0], dtype=np.double)
        np.testing.assert_allclose(qo.norm(q), [np.sqrt(2)])

    def test_quaternion_real_matrix(self):
        q = np.array([1, 0, 0, 0], dtype=np.double)
        np.testing.assert_allclose(qo.real_matrix(q), np.eye(4))
        q = np.array([5, 0, 0, 0], dtype=np.double)
        np.testing.assert_allclose(qo.real_matrix(q), np.eye(4) * 5)

    def test_quaternion_complex_matrix(self):
        q = np.array([1, 0, 0, 0], dtype=np.double)
        np.testing.assert_allclose(qo.complex_matrix(q), np.eye(2))

    def test_quaternion_to_rotation_matrix(self):
        for i in range(100):
            q = (np.random.random(4) - 0.5) * 2
            while qo.norm(q) == 0:
                q = np.random.random(4)
            q_n = q / qo.norm(q)
            np.testing.assert_allclose(qo.norm(q_n), [1])
            r_m = qo.quaternion_to_rotation_matrix(q)
            r_m_n = qo.quaternion_to_rotation_matrix(q_n)
            np.testing.assert_allclose(r_m, r_m_n)
            q_m = qo.quaternion_from_rotation_matrix(r_m)
            q_m_n = qo.quaternion_from_rotation_matrix(r_m_n)
            self.assertTrue(np.allclose(q_m, q_n) or np.allclose(q_m, -q_n))
            self.assertTrue(np.allclose(q_m_n, q_n) or np.allclose(q_m_n, -q_n))
        m = np.eye(4, 4) + 1
        with self.assertRaises(ValueError):
            qo.quaternion_from_rotation_matrix(m)
        m = np.random.randn(3, 3)
        d = np.linalg.det(m)
        if d < 0:
            m[:, 1] = -m[:, 1]
            d = -d
        m = (d**(-1 / 3)) * m
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            qo.quaternion_from_rotation_matrix(m)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('Not a rotation matrix' in str(w[-1].message))

    def test_log_and_exp(self):
        magnitude = 2
        q = (np.random.random(4) - 0.5) * 2 * magnitude
        exp_q = qo.exp(q)
        log_q = qo.log(q)
        np.testing.assert_allclose(qo.log(exp_q), q)
        np.testing.assert_allclose(qo.exp(log_q), q)
        q = np.hstack(([1], np.zeros(3)))
        np.testing.assert_allclose(qo.exp(q), np.array([np.exp(1), 0, 0, 0]))
        with self.assertRaises(ZeroDivisionError):
            qo.log(np.zeros(4))
