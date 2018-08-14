from __future__ import division, print_function
import warnings
import numpy as np

from libc.math cimport sqrt, cos, sin, acos
from libc.math cimport exp as c_exp
from libc.float cimport DBL_MIN


cpdef mul(double[:] q1, double[:] q2):
    """
    Multiplication of two quaternions
    :param q1: first quaternion as any iterable of four numbers
    :param q2: second quaternion as any iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        double[:] quadruple = np.empty(4, dtype=np.double)
    quadruple[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    quadruple[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    quadruple[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    quadruple[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return quadruple


cpdef norm(double[:] q):
    """
    Calculates norm of quaternion
    :param q: quaternion as an iterable of four numbers
    :return: the norm of the quaternion as float number
    """
    return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


cpdef real_matrix(double[:] q):
    """
    returns 4x4 real matrix representation of quaternion
    :param q: quaternion as an iterable of four numbers
    :return: 4x4 real matrix as numpy array
    """
    cdef:
        double a = q[0], b = q[1], c = q[2], d = q[3]
        double[:, :] result
    result = np.array([[ a,  b,  c,  d],
                       [-b,  a, -d,  c],
                       [-c,  d,  a, -b],
                       [-d, -c,  b,  a]])
    return result


cpdef complex_matrix(double[:] q):
    """
    returns 2x2 complex matrix representation of quaternion
    :param q: quaternion as an iterable of four numbers
    :return: 2x2 complex matrix as numpy array
    """
    cdef:
        double a = q[0], b = q[1], c = q[2], d = q[3]
        # complex double[:, :] result
    result = np.array([[ a + b * 1j, c + d * 1j],
                       [-c + d * 1j, a - b * 1j]])
    return result


cpdef quaternion_to_rotation_matrix(double[:] q):
    """
    Convert versor corresponding to given quaternion to rotation 3x3 matrix
    :param q: quaternion as an iterable of four numbers
    :return: 3x3 rotation matrix as numpy array
    """
    cdef:
        double n = norm(q)
        double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n
        double[:, :] m
    m = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                  [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
                  [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return m


cpdef quaternion_from_rotation_matrix(double[:, :] m):
    """
    Convert 3x3 rotation matrix to quaternion
    :param m: 3x3 rotation matrix as numpy array
    :return: quaternion as numpy array of four floats
    """
    cdef:
        double[:, :] inv_m, k_m
        double det_m, t, r, w, s, x, y, z
        double[:] quadruple, w_n
    det_m = np.linalg.det(m)
    if abs(1 - det_m ** 2) > 1e-6:
        raise ValueError('Not a rotation matrix. det M = %2.2g' % det_m)
    inv_m = np.linalg.inv(m)
    if np.allclose(det_m, [1.0]) and np.allclose(m.T, inv_m):
        t = np.trace(m)
        if t > 3 * DBL_MIN:
            r = sqrt(1 + t)
            w = 0.5 * r
            s = 0.5 / r
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            r = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            s = 0.5 / r
            w = (m[2, 1] - m[1, 2]) * s
            x = 0.5 * r
            y = (m[0, 1] + m[1, 0]) * s
            z = (m[2, 0] + m[0, 2]) * s
        elif m[1, 1] >= m[2, 2]:
            r = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            s = 0.5 / r
            w = (m[0, 2] - m[2, 0]) * s
            x = (m[0, 1] + m[1, 0]) * s
            y = 0.5 * r
            z = (m[1, 2] + m[2, 1]) * s
        else:
            r = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            s = 0.5 / r
            w = (m[1, 0] - m[0, 1]) * s
            x = (m[2, 0] + m[0, 2]) * s
            y = (m[1, 2] + m[2, 1]) * s
            z = 0.5 * r
        quadruple = np.array([w, x, y, z])
    else:
        warnings.warn('Not a rotation matrix. det M = %2.2g' % det_m)
        k_m = np.array([[m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]],
                        [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]],
                        [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
                        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3
        w_n, v_n = np.linalg.eigh(k_m)
        quadruple = v_n[[3, 0, 1, 2], np.argmax(w_n)]
    return quadruple


cpdef exp(double[:] q):
    """
    Calculates exp() function on quaternion
    :param q: quaternion as an iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        double v_norm, a = q[0], b = q[1], c = q[2], d = q[3]
        double[:] result = np.empty(4, dtype=np.double)
    v_norm = sqrt(b * b + c * c + d * d)
    if v_norm > 0.0:
        result[0] = c_exp(a) * cos(v_norm)
        result[1] = c_exp(a) * b / v_norm * sin(v_norm)
        result[2] = c_exp(a) * c / v_norm * sin(v_norm)
        result[3] = c_exp(a) * d / v_norm * sin(v_norm)
    else:
        result[0] = c_exp(a)
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
    return result


cpdef log(double[:] q):
    """
    Calculates log() function on quaternion
    :param q: quaternion as any iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        double q_norm, v_norm, a = q[0], b = q[1], c = q[2], d = q[3]
        double[:] result = np.empty(4, dtype=np.double)
    q_norm = norm(q)
    v_norm = sqrt(b * b + c * c + d * d)
    result[0] = np.log(q_norm)
    result[1] = b / v_norm * acos(a / q_norm)
    result[2] = c / v_norm * acos(a / q_norm)
    result[3] = d / v_norm * acos(a / q_norm)
    return result
