from __future__ import division, print_function
import numpy as np

from _quaternion_operations import quaternion_to_rotation_matrix
from _euler_angles_conventions import euler_angles_codes, conventions, default_convention, euler_next_axis

"""
Euler angles conversion algorithms after Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222
"""


def check_euler_angles_convention(convention):
    euler_angles_convention = conventions[default_convention]
    if convention is not None:
        match = False
        for key in conventions.keys():
            if str(convention).lower().strip() in conventions[key]['variants']:
                euler_angles_convention = conventions[key]
                euler_angles_convention['title'] = key
                match = True
                break
        if not match:
            print('Convention: %s not found or not supported.' % convention)
            print('Falling back to default convention %s.' % default_convention)
            euler_angles_convention['title'] = default_convention
    euler_angles_convention['code'] = euler_angles_codes[euler_angles_convention['axes']]
    return euler_angles_convention


def euler_angles_to_matrix(ai, aj, ak, convention):
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = convention['code']
    i = inner_axis
    j = euler_next_axis[i + parity]
    k = euler_next_axis[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)

    m = np.identity(3)
    if repetition:
        m[i, i] = cj
        m[i, j] = si * sj
        m[i, k] = ci * sj
        m[j, i] = sj * sk
        m[j, j] = ci * ck - si * cj * sk
        m[j, k] = -si * ck - ci * cj * sk
        m[k, i] = -sj * ck
        m[k, j] = ci * sk + si * cj * ck
        m[k, k] = ci * cj * ck - si * sk
    else:
        m[i, i] = cj * ck
        m[i, j] = si * sj * ck - ci * sk
        m[i, k] = ci * sj * ck + si * sk
        m[j, i] = cj * sk
        m[j, j] = si * sj * sk + ci * ck
        m[j, k] = ci * sj * sk - si * ck
        m[k, i] = -sj
        m[k, j] = si * cj
        m[k, k] = ci * cj
    return m


def euler_angles_from_matrix(matrix, convention):
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = convention['code']
    i = inner_axis
    j = euler_next_axis[i + parity]
    k = euler_next_axis[i - parity + 1]
    m = np.array(matrix, dtype=np.float)[:3, :3]
    if repetition:
        sy = np.sqrt(m[i, j] * m[i, j] + m[i, k] * m[i, k])
        if sy > np.finfo(float).eps * 4:
            ax = np.arctan2(m[i, j], m[i, k])
            ay = np.arctan2(sy, m[i, i])
            az = np.arctan2(m[j, i], -m[k, i])
        else:
            ax = np.arctan2(-m[j, k], m[j, j])
            ay = np.arctan2(sy, m[i, i])
            az = 0.0
    else:
        cy = np.sqrt(m[i, i] * m[i, i] + m[j, i] * m[j, i])
        if cy > np.finfo(float).eps * 4:
            ax = np.arctan2(m[k, j], m[k, k])
            ay = np.arctan2(-m[k, i], cy)
            az = np.arctan2(m[j, i], m[i, i])
        else:
            ax = np.arctan2(-m[j, k], m[j, j])
            ay = np.arctan2(-m[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_angles_to_quaternion(ai, aj, ak, convention):
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = convention['code']
    i = inner_axis + 1
    j = euler_next_axis[i + parity - 1] + 1
    k = euler_next_axis[i - parity] + 1
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj
    ai /= 2
    aj /= 2
    ak /= 2
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)

    quadruple = np.zeros(4, dtype=np.float)
    if repetition:
        quadruple[0] = ci * cj * ck - si * cj * sk
        quadruple[i] = ci * cj * sk + si * cj * ck
        quadruple[j] = ci * sj * ck + si * sj * sk
        quadruple[k] = ci * sj * sk - si * sj * ck
    else:
        quadruple[0] = ci * cj * ck + si * sj * sk
        quadruple[i] = si * cj * ck - ci * sj * sk
        quadruple[j] = si * cj * sk + ci * sj * ck
        quadruple[k] = ci * cj * sk - si * sj * ck
    if parity:
        quadruple[j] *= -1.0
    return quadruple


def euler_angles_from_quaternion(quadruple, convention):
    matrix = quaternion_to_rotation_matrix(quadruple)
    return euler_angles_from_matrix(matrix, convention)
