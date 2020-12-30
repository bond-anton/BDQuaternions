import numpy as np

from BDQuaternions import Quaternion
from BDQuaternions.functions import exp, log

import unittest


def rand_q(magnitude=2):
    return Quaternion((np.random.random(4) - 0.5) * 2 * magnitude)


def check_log_exp_q(q=None):
    if q is None:
        q = rand_q(magnitude=2)
    exp_q = exp(q)
    log_q = log(q)
    return (np.array(log(exp_q)) == np.array(q)).all() and (np.array(exp(log_q)) == np.array(q)).all()


def check_log_exp_number(test_num=None):
    if test_num is None:
        magnitude = 2
        test_num = np.random.random(1)[0] * magnitude
    test_exp = exp(test_num)
    test_log = log(test_num)
    return np.allclose(log(test_exp), test_num) and np.allclose(exp(test_log), test_num)


class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_log_exp_q(self):
        self.assertTrue(check_log_exp_q())
        q = Quaternion(np.array([0, 0, 0, 0], dtype=np.double))
        self.assertTrue(check_log_exp_q(q))
        q = Quaternion(np.array([0, 0, 0, 1], dtype=np.double))
        self.assertTrue(check_log_exp_q(q))
        q = Quaternion(np.array([1, 0, 0, 0], dtype=np.double))
        self.assertTrue(check_log_exp_q(q))

    def test_log_exp_number(self):
        self.assertTrue(check_log_exp_number())

    def test_arrays(self):
        qs = check_log_exp_q([rand_q(magnitude=1) for _ in range(5)])
        self.assertTrue(qs)
        ns = check_log_exp_number([np.random.random(1)[0] * 5 for _ in range(5)])
        self.assertTrue(ns)

    def test_raises(self):
        with self.assertRaises(ValueError):
            exp('a')
        with self.assertRaises(ValueError):
            log('a')
