from __future__ import division, print_function
import numpy as np

from Quaternions._quaternion_operations import quaternion_to_rotation_matrix
from Quaternions._euler_angles_conventions import general_conventions, special_conventions, derived_conventions,\
    default_convention, euler_angles_codes, euler_next_axis

"""
Euler angles conversion algorithms after Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222

All angles are in radians y default
"""


def check_euler_angles_convention(convention):
    """
    returns euler_angles convention as a dict
    :param convention: string with short form of convention e.g. 'XYZs' or 'Bunge'
    :return: dict describing convention
    """
    conventions = general_conventions.copy()
    conventions.update(special_conventions)
    euler_angles_convention = conventions[default_convention]
    if convention is not None:
        match = False
        # first we search requested convention in the dict of 'standard' conventions
        for key in conventions.keys():
            if str(convention).lower().strip() in conventions[key]['variants']:
                euler_angles_convention = conventions[key]
                euler_angles_convention['title'] = key
                euler_angles_convention['parent_convention'] = None
                euler_angles_convention['parent'] = None
                euler_angles_convention['code'] = euler_angles_codes[euler_angles_convention['axes']]
                match = True
                break
        # If not found we look through the dict of 'derived' conventions
        for key in derived_conventions.keys():
            if str(convention).lower().strip() in derived_conventions[key]['variants']:
                euler_angles_convention = derived_conventions[key]
                euler_angles_convention['title'] = key
                parent_convention = check_euler_angles_convention(euler_angles_convention['parent_convention'])
                euler_angles_convention['parent'] = parent_convention
                match = True
                break
        if not match:
            print('Convention: %s not found or not supported.' % convention)
            print('Falling back to default convention %s.' % default_convention)
            euler_angles_convention = check_euler_angles_convention(default_convention)
    else:
        euler_angles_convention = check_euler_angles_convention(default_convention)
    return euler_angles_convention


def print_euler_angles_conventions_tree(convention):
    """
    Theoretically derived conventions can be nested endless.
    This function prints out the tree beginning from highest level parent convention.
    :param convention: dict describing Euler angles convention
    """
    current_convention = convention
    flat_parent_list = []
    while current_convention['parent_convention'] is not None:
        flat_parent_list.append(current_convention)
        current_convention = current_convention['parent']
    flat_parent_list.append(current_convention)
    level = 0
    while flat_parent_list:
        current_convention = flat_parent_list.pop()
        print('-' * level + ' ' * (level > 0) + current_convention['title'])
        level += 1


def euler_angles_in_parent_convention(ai, aj, ak, convention):
    """
    Theoretically derived conventions can be nested endless.
    This function will bring the angles to the highest level parent convention.
    :param ai: first Euler angle
    :param aj: second Euler angle
    :param ak: third Euler angle
    :param convention: dict describing Euler angles convention
    :return: ax, ay, az, parent_convention - three Euler angles and the parent convention dict
    """
    current_convention = convention
    ax, ay, az = ai, aj, ak
    while current_convention['parent_convention'] is not None:
        ax, ay, az = current_convention['to_parent'](ax, ay, az)
        current_convention = current_convention['parent']
    return ax, ay, az, current_convention


def euler_angles_from_parent_convention(ai, aj, ak, convention):
    """
    Theoretically derived conventions can be nested endless.
    This function will bring the angles from the highest level parent convention to given.
    :param ai: first Euler angle in most parent convention
    :param aj: second Euler angle in most parent convention
    :param ak: third Euler angle in most parent convention
    :param convention: dict describing Euler angles convention
    :return: ax, ay, az - three Euler angles in specified convention
    """
    ax, ay, az = ai, aj, ak
    current_convention = convention
    flat_parent_list = []
    while current_convention['parent_convention'] is not None:
        flat_parent_list.append(current_convention)
        current_convention = current_convention['parent']
    while flat_parent_list:
        current_convention = flat_parent_list.pop()
        ax, ay, az = current_convention['from_parent'](ax, ay, az)
    return ax, ay, az


def euler_angles_to_matrix(ai, aj, ak, convention):
    """
    Convert Euler angles to rotation matrix
    :param ai: first Euler angle
    :param aj: second Euler angle
    :param ak: third Euler angle
    :param convention: dict describing Euler angles convention
    :return: 3x3 rotation matrix as numpy array of floats
    """
    ax, ay, az, parent_convention = euler_angles_in_parent_convention(ai, aj, ak, convention)
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = parent_convention['code']
    i = inner_axis
    j = euler_next_axis[i + parity]
    k = euler_next_axis[i - parity + 1]
    if frame:
        ax, az = az, ax
    if parity:
        ax, ay, az = -ax, -ay, -az
    ci = np.cos(ax)
    si = np.sin(ax)
    cj = np.cos(ay)
    sj = np.sin(ay)
    ck = np.cos(az)
    sk = np.sin(az)

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
    """
    convert rotation matrix to Euler angles
    :param matrix: 3x3 rotation matrix
    :param convention: dict describing Euler angles convention
    :return: ax, ay, az three Euler angles
    """
    ax, ay, az, parent_convention = euler_angles_in_parent_convention(0, 0, 0, convention)
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = parent_convention['code']
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
    return euler_angles_from_parent_convention(ax, ay, az, convention)


def euler_angles_to_quaternion(ai, aj, ak, convention):
    """
    Convert Euler angles to quaternion
    :param ai: first Euler angle
    :param aj: second Euler angle
    :param ak: third Euler angle
    :param convention: dict describing Euler angles convention
    :return: Quaternion as numpy array of for floats [w*1, x*i, y*j, z*k]
    """
    ax, ay, az, parent_convention = euler_angles_in_parent_convention(ai, aj, ak, convention)
    # the tuples in convention['code'] coding the inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
    # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
    inner_axis, parity, repetition, frame = parent_convention['code']
    i = inner_axis + 1
    j = euler_next_axis[i + parity - 1] + 1
    k = euler_next_axis[i - parity] + 1
    if frame:
        ax, az = az, ax
    if parity:
        ay = -ay
    ax /= 2
    ay /= 2
    az /= 2
    ci = np.cos(ax)
    si = np.sin(ax)
    cj = np.cos(ay)
    sj = np.sin(ay)
    ck = np.cos(az)
    sk = np.sin(az)

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
    """
    Convert Quaternion to Euler angles
    :param quadruple: Quaternion as numpy array of for floats [w*1, x*i, y*j, z*k]
    :param convention: dict describing Euler angles convention
    :return: ax, ay, az three Euler angles
    """
    matrix = quaternion_to_rotation_matrix(quadruple)
    return euler_angles_from_matrix(matrix, convention)


def change_euler_angles_convention(ai, aj, ak, convention, new_convention):
    """
    Convert Euler angles from given to new convention
    :param ai: first Euler angle
    :param aj: second Euler angle
    :param ak: third Euler angle
    :param convention: dict describing current Euler angles convention
    :param new_convention: dict describing new Euler angles convention
    :return: ax, ay, az three Euler angles
    """
    matrix = euler_angles_to_matrix(ai, aj, ak, convention)
    return euler_angles_from_matrix(matrix, new_convention)
