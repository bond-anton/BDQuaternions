import numpy as np

from BDQuaternions import Quaternion, UnitQuaternion
from BDQuaternions.functions import exp, log
from BDQuaternions.utils import random_unit_quaternion

import unittest


def check_log_exp_q(q=None):
    if q is None:
        q = random_unit_quaternion()
    exp_q = exp(q)
    log_q = log(q)
    log_exp_q = log(exp_q)
    exp_log_q = exp(log_q)
    log_exp_q_q = (np.array(log_exp_q) == np.array(q)).all()
    exp_log_q_q = (np.array(exp_log_q) == np.array(q)).all()
    return log_exp_q_q and exp_log_q_q


def check_log_exp_number(test_num=None):
    if test_num is None:
        magnitude = 1
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
        q = UnitQuaternion(np.array([0, 0, 0, 1], dtype=np.double))
        self.assertTrue(check_log_exp_q(q))
        q = UnitQuaternion(np.array([1, 0, 0, 0], dtype=np.double))
        self.assertTrue(check_log_exp_q(q))

    def test_log_exp_number(self):
        self.assertTrue(check_log_exp_number())

    def test_arrays(self):
        qs = check_log_exp_q([random_unit_quaternion() for _ in range(5)])
        self.assertTrue(qs)
        ns = check_log_exp_number([np.random.random(1)[0] for _ in range(5)])
        self.assertTrue(ns)

    def test_raises(self):
        with self.assertRaises(ValueError):
            exp('a')
        with self.assertRaises(ValueError):
            log('a')
