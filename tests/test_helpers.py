from __future__ import division
import numpy as np

from BDQuaternions import _helpers as hlp
from BDQuaternions import _quaternion_operations as qo

import unittest


class TestHelpers(unittest.TestCase):

    def test_2x2_det(self):
        for _ in range(1000):
            m = np.random.random((2, 2))
            d = hlp._2x2_det(m)
            np.testing.assert_allclose(d, np.linalg.det(m))

    def test_3x3_det(self):
        for _ in range(1000):
            m = np.random.random((3, 3))
            d = hlp._3x3_det(m)
            np.testing.assert_allclose(d, np.linalg.det(m))

    def test_v_dot(self):
        for _ in range(1000):
            n = np.random.randint(2, 256)
            v1 = np.random.random(n)
            v2 = np.random.random(n)
            v1v2 = hlp.vectors_dot_prod(v1, v2, n)
            np.testing.assert_allclose(v1v2, np.dot(v1, v2))

    def test_mv_dot(self):
        for _ in range(1000):
            r = np.random.randint(2, 256)
            c = np.random.randint(2, 256)
            v = np.random.random(c)
            m = np.random.random((r, c))
            mv = hlp.matrix_vector_mult(m, v, r, c)
            np.testing.assert_allclose(mv, np.dot(m, v))

    def test_3x3_inv(self):
        for _ in range(1000):
            m = np.random.random((3, 3))
            inv = hlp._3x3_inv(m)
            np.testing.assert_allclose(inv, np.linalg.inv(m))

    def test_matrix_mult(self):
        for _ in range(1000):
            r = np.random.randint(2, 256)
            c = np.random.randint(2, 256)
            n = np.random.randint(2, 256)
            m1 = np.random.random((r, n))
            m2 = np.random.random((n, c))
            mm = hlp.matrix_mult(m1, m2, r, c, n)
            np.testing.assert_allclose(mm, np.dot(m1, m2))

    def test_decomp(self):
        q = (np.random.random(4) - 0.5) * 2
        while qo.norm(q) == 0:
            q = np.random.random(4)
        q_n = q / qo.norm(q)
        np.testing.assert_allclose(qo.norm(q_n), [1])
        r_m = qo.quaternion_to_rotation_matrix(q)
        hlp.decomp(r_m)
