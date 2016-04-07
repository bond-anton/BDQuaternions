from __future__ import division
import unittest
import numpy as np

from Quaternions import _quaternion_operations as qo


class TestQuaternionOperations(unittest.TestCase):

    def test_quaternion_check(self):
        self.assertRaises(ValueError, qo.check_quadruple, ['a', 'b', 'c', 3])
        self.assertRaises(ValueError, qo.check_quadruple, [1, 2, 3])
        q = qo.check_quadruple([1, 2, 3, 4])
        np.testing.assert_allclose(q, [1, 2, 3, 4])

    def test_quaternion_norm(self):
        q = qo.check_quadruple([1, 0, 0, 0])
        np.testing.assert_allclose(qo.norm(q), [1])
        q = qo.check_quadruple([0, 0, 0, 0])
        np.testing.assert_allclose(qo.norm(q), [0])
        q = qo.check_quadruple([0, 0, 1, 0])
        np.testing.assert_allclose(qo.norm(q), [1])
        q = qo.check_quadruple([1, 0, 1, 0])
        np.testing.assert_allclose(qo.norm(q), [np.sqrt(2)])

    def test_quaternion_real_matrix(self):
        q = qo.check_quadruple([1, 0, 0, 0])
        np.testing.assert_allclose(qo.real_matrix(q), np.eye(4))
        q = qo.check_quadruple([5, 0, 0, 0])
        np.testing.assert_allclose(qo.real_matrix(q), np.eye(4) * 5)

    def test_quaternion_complex_matrix(self):
        q = qo.check_quadruple([1, 0, 0, 0])
        np.testing.assert_allclose(qo.complex_matrix(q), np.eye(2))

    def test_quaternion_to_rotation_matrix(self):
        for i in range(1):
            q = qo.check_quadruple((np.random.random(4) - 0.5) * 2)
            while qo.norm(q) == 0:
                q = qo.check_quadruple(np.random.random(4))
            q_n = q / qo.norm(q)
            np.testing.assert_allclose(qo.norm(q_n), [1])
            r_m = qo.quaternion_to_rotation_matrix(q)
            r_m_n = qo.quaternion_to_rotation_matrix(q_n)
            #np.testing.assert_allclose(r_m, r_m_n)
            #np.testing.assert_allclose(qo.quaternion_from_rotation_matrix(r_m), q_n)
            #np.testing.assert_allclose(qo.quaternion_from_rotation_matrix(r_m_n), q_n)

    def test_log_and_exp(self):
        q = qo.check_quadruple((np.random.random(4) - 0.5) * 2)
        exp_q = qo.exp(q)
        log_q = qo.log(q)
        np.testing.assert_allclose(qo.log(exp_q), q)
        np.testing.assert_allclose(qo.exp(log_q), q)
