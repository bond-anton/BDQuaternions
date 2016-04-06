from __future__ import division, print_function
import numpy as np


def check_quadruple(quadruple):
    quadruple = np.array(quadruple, dtype=np.float)
    if quadruple.size != 4:
        raise ValueError('Quadruple must have exactly 4 elements')
    return quadruple


def mul(q1, q2):
    q1 = check_quadruple(q1)
    q2 = check_quadruple(q2)
    quadruple = np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                          q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                          q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                          q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
    return quadruple


def norm(quadruple):
    quadruple = check_quadruple(quadruple)
    return np.sqrt(np.sum(quadruple * quadruple))


def real_matrix(quadruple):
    quadruple = check_quadruple(quadruple)
    a, b, c, d = quadruple
    return np.array([[a, b, c, d],
                     [-b, a, -d, c],
                     [-c, d, a, -b],
                     [-d, -c, b, a]])


def complex_matrix(quadruple):
    quadruple = check_quadruple(quadruple)
    a, b, c, d = quadruple
    return np.array([[a + b * 1j, c + d * 1j],
                     [-c + d * 1j, a - b * 1j]])


def quaternion_to_rotation_matrix(quadruple):
    q = check_quadruple(quadruple)
    assert q.size == 4
    assert norm(q) > 0
    w, x, y, z = q / norm(q)
    m = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
                  [2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],
                  [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return m


def quaternion_from_rotation_matrix(matrix):
    m = np.array(matrix, dtype=np.float)[:3, :3]
    det_m = np.linalg.det(m)
    if abs(1 - det_m ** 2) > 1e-6:
        raise ValueError('Not a rotation matrix. det M = %2.2g' % det_m)
    inv_m = np.linalg.inv(m)
    if np.allclose(det_m, 1.0) and np.allclose(m.T, inv_m):
        m = np.array(m, dtype=np.float)
        if m.shape != (3, 3):
            raise ValueError('3x3 rotation matrix expected, got' + str(m))
        t = np.trace(m)
        if t > 3 * np.finfo(float).eps:
            r = np.sqrt(1 + t)
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
            w = (m[2, 0] - m[0, 2]) * s
            x = (m[0, 1] + m[1, 0]) * s
            y = 0.5 * r
            z = (m[1, 2] + m[2, 1]) * s
        else:
            r = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            s = 0.5 / r
            w = (m[0, 1] - m[1, 0]) * s
            x = (m[2, 0] + m[0, 2]) * s
            y = (m[1, 2] + m[2, 1]) * s
            z = 0.5 * r
        quadruple = np.array([w, x, y, z])
    else:
        print('Not a rotation matrix. det M = %2.2g' % det_m)
        k_m = np.array([[m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2]],
                        [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1], m[0, 2] - m[2, 0]],
                        [m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
                        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3
        w, v_n = np.linalg.eigh(k_m)
        quadruple = v_n[[3, 0, 1, 2], np.argmax(w)]
    return quadruple
