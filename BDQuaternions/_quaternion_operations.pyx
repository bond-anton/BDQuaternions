import warnings
import numpy as np

from cython import boundscheck, wraparound

from cpython.array cimport array, clone
from libc.math cimport sqrt, cos, sin, acos
from libc.math cimport exp as c_exp, log as c_log
from libc.float cimport DBL_MIN
from scipy.linalg.cython_lapack cimport dsyevd
from ._helpers cimport _3x3_det, trace, check_orthogonal


@wraparound(False)
@boundscheck(False)
cpdef double[:] mul(double[:] q1, double[:] q2):
    """
    Multiplication of two quaternions
    :param q1: first quaternion as any iterable of four numbers
    :param q2: second quaternion as any iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        array[double] quadruple, template = array('d')
    quadruple = clone(template, 4, zero=False)
    quadruple[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    quadruple[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    quadruple[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    quadruple[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return quadruple


@wraparound(False)
@boundscheck(False)
cpdef double norm(double[:] q) nogil:
    """
    Calculates norm of quaternion
    :param q: quaternion as an iterable of four numbers
    :return: the norm of the quaternion as float number
    """
    return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


@wraparound(False)
@boundscheck(False)
cpdef double[:, :] real_matrix(double[:] q):
    """
    returns 4x4 real matrix representation of quaternion
        [ a,  b,  c,  d]
        [-b,  a, -d,  c]
        [-c,  d,  a, -b]
        [-d, -c,  b,  a]
    :param q: quaternion as an iterable of four numbers
    :return: 4x4 real matrix as numpy array
    """
    cdef:
        double[:, :] m = np.empty((4, 4), dtype=np.double)
    m[0][0] =  q[0]
    m[0][1] =  q[1]
    m[0][2] =  q[2]
    m[0][3] =  q[3]
    m[1][0] = -q[1]
    m[1][1] =  q[0]
    m[1][2] = -q[3]
    m[1][3] =  q[2]
    m[2][0] = -q[2]
    m[2][1] =  q[3]
    m[2][2] =  q[0]
    m[2][3] = -q[1]
    m[3][0] = -q[3]
    m[3][1] = -q[2]
    m[3][2] =  q[1]
    m[3][3] =  q[0]
    return m


@wraparound(False)
@boundscheck(False)
cpdef double complex[:, :] complex_matrix(double[:] q):
    """
    returns 2x2 complex matrix representation of quaternion
        [ a + b * 1j, c + d * 1j]
        [-c + d * 1j, a - b * 1j]
    :param q: quaternion as an iterable of four numbers
    :return: 2x2 complex matrix as numpy array
    """
    cdef:
        double complex[:, :] cm = np.empty((2, 2), dtype=np.complex128)
    cm[0][0] = q[0] + q[1]* 1j
    cm[0][1] = q[2] + q[3] * 1j
    cm[1][0] = -q[2] + q[3] * 1j
    cm[1][1] = q[0] - q[1] * 1j
    return cm


@wraparound(False)
@boundscheck(False)
cpdef double[:, :] quaternion_to_rotation_matrix(double[:] q):
    """
    Convert versor corresponding to given quaternion to rotation 3x3 matrix
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y]
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x]
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
    :param q: quaternion as an iterable of four numbers
    :return: 3x3 rotation matrix as numpy array
    """
    cdef:
        double n = norm(q)
        double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n
        double[:, :] m = np.empty((3, 3), dtype=np.double)
    m[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    m[0, 1] = 2 * x * y - 2 * w * z
    m[0, 2] = 2 * x * z + 2 * w * y
    m[1, 0] = 2 * x * y + 2 * w * z
    m[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    m[1, 2] = 2 * y * z - 2 * w * x
    m[2, 0] = 2 * x * z - 2 * w * y
    m[2, 1] = 2 * y * z + 2 * w * x
    m[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2
    return m


@wraparound(False)
@boundscheck(False)
cpdef double[:] quaternion_from_rotation_matrix(double[:, :] m):
    """
    Convert 3x3 rotation matrix to quaternion
    :param m: 3x3 rotation matrix as numpy array
    :return: quaternion as numpy array of four floats
    """
    cdef:
        double det_m, t, r, w, s, x, y, z
        array[double] quadruple, template = array('d')
        int n = 4, lwork = 57, liwork = 23, info
        double k_m[4][4]
        double work[57]
        int iwork[23]
        double w_n[4]
        char L = b'L', J = b'V'
    quadruple = clone(template, 4, zero=False)
    det_m = _3x3_det(m)
    if abs(1 - det_m ** 2) > 1e-6:
        raise ValueError('Not a rotation matrix. det M = %2.2g' % det_m)
    if check_orthogonal(m, 3, 3, 1.0e-12) and abs(det_m - 1.0) < 1.0e-12:
        t = trace(m, 3)
        if t > 3 * DBL_MIN:
            r = sqrt(1 + t)
            w = 0.5 * r
            s = 0.5 / r
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            r = sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            s = 0.5 / r
            w = (m[2, 1] - m[1, 2]) * s
            x = 0.5 * r
            y = (m[0, 1] + m[1, 0]) * s
            z = (m[2, 0] + m[0, 2]) * s
        elif m[1, 1] >= m[2, 2]:
            r = sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            s = 0.5 / r
            w = (m[0, 2] - m[2, 0]) * s
            x = (m[0, 1] + m[1, 0]) * s
            y = 0.5 * r
            z = (m[1, 2] + m[2, 1]) * s
        else:
            r = sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            s = 0.5 / r
            w = (m[1, 0] - m[0, 1]) * s
            x = (m[2, 0] + m[0, 2]) * s
            y = (m[1, 2] + m[2, 1]) * s
            z = 0.5 * r
        quadruple[0] = w
        quadruple[1] = x
        quadruple[2] = y
        quadruple[3] = z
    else:
        warnings.warn('Not a rotation matrix. det M = %2.2g' % det_m)
        k_m[0][0] = (m[0, 0] - m[1, 1] - m[2, 2]) / 3.0
        k_m[0][1] = (m[0, 1] + m[1, 0]) / 3.0
        k_m[0][2] = (m[0, 2] + m[2, 0]) / 3.0
        k_m[0][3] = (m[2, 1] - m[1, 2]) / 3.0
        k_m[1][0] = (m[0, 1] + m[1, 0]) / 3.0
        k_m[1][1] = (m[1, 1] - m[0, 0] - m[2, 2]) / 3.0
        k_m[1][2] = (m[1, 2] + m[2, 1]) / 3.0
        k_m[1][3] = (m[0, 2] - m[2, 0]) / 3.0
        k_m[2][0] = (m[0, 2] + m[2, 0]) / 3.0
        k_m[2][1] = (m[1, 2] + m[2, 1]) / 3.0
        k_m[2][2] = (m[2, 2] - m[0, 0] - m[1, 1]) / 3.0
        k_m[2][3] = (m[1, 0] - m[0, 1]) / 3.0
        k_m[3][0] = (m[2, 1] - m[1, 2]) / 3.0
        k_m[3][1] = (m[0, 2] - m[2, 0]) / 3.0
        k_m[3][2] = (m[1, 0] - m[0, 1]) / 3.0
        k_m[3][3] = (m[0, 0] + m[1, 1] + m[2, 2]) / 3.0
        dsyevd(&J, &L, &n, &k_m[0][0], &n, &w_n[0], &work[0], &lwork, &iwork[0], &liwork, &info)
        quadruple[0] = k_m[3][3]
        quadruple[1] = k_m[3][0]
        quadruple[2] = k_m[3][1]
        quadruple[3] = k_m[3][2]
    return quadruple


@wraparound(False)
@boundscheck(False)
cpdef double[:] exp(double[:] q):
    """
    Calculates exp() function on quaternion
    :param q: quaternion as an iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        double v_norm, a1, sin_v_norm, a = q[0], b = q[1], c = q[2], d = q[3]
        array[double] result, template = array('d')
    result = clone(template, 4, zero=False)
    v_norm = sqrt(b * b + c * c + d * d)
    a1 = c_exp(a)
    if v_norm > 0.0:
        sin_v_norm = a1 * sin(v_norm) / v_norm
        result[0] = a1 * cos(v_norm)
        result[1] = b * sin_v_norm
        result[2] = c * sin_v_norm
        result[3] = d * sin_v_norm
    else:
        result[0] = a1
        result[1] = 0.0
        result[2] = 0.0
        result[3] = 0.0
    return result


@wraparound(False)
@boundscheck(False)
cpdef double[:] log(double[:] q):
    """
    Calculates log() function on quaternion
    :param q: quaternion as any iterable of four numbers
    :return: result quaternion as numpy array of four floats
    """
    cdef:
        double q_norm, v_norm, acos_a_q_norm, a = q[0], b = q[1], c = q[2], d = q[3]
        array[double] result, template = array('d')
    result = clone(template, 4, zero=False)
    q_norm = norm(q)
    v_norm = sqrt(b * b + c * c + d * d)
    result[0] = c_log(q_norm)
    if v_norm > 0.0:
        acos_a_q_norm = acos(a / q_norm) / v_norm
        result[1] = b * acos_a_q_norm
        result[2] = c * acos_a_q_norm
        result[3] = d * acos_a_q_norm
    return result
