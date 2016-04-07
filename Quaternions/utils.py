from __future__ import division, print_function
import numbers
import numpy as np

from _quaternion_operations import norm

from Quaternions import Quaternion, UnitQuaternion, Rotation


def random_rotation():
    """
    Calculates random rotation quaternion
    :return: random Rotation
    """
    quadruple = np.array([1, 0, 0, 0], dtype=np.float)
    random_quadruple_norm = 1.0
    while True:
        quadruple = (np.random.random(4) - 0.5) * 2
        random_quadruple_norm = norm(quadruple)
        if random_quadruple_norm > 0:
            break
    quadruple /= random_quadruple_norm
    return Rotation(quadruple)


def random_unit_quaternion():
    """
    Calculates random unit quaternion
    :return: random UnitQuaternion
    """
    quadruple = np.array([1, 0, 0, 0], dtype=np.float)
    random_quadruple_norm = 1.0
    while True:
        quadruple = (np.random.random(4) - 0.5) * 2
        random_quadruple_norm = norm(quadruple)
        if random_quadruple_norm > 0:
            break
    quadruple /= random_quadruple_norm
    return UnitQuaternion(quadruple)


def random_quaternion(quadruple_norm=None):
    """
    Calculates random quaternion
    :return: random Quaternion
    """
    quadruple = np.array([1, 0, 0, 0], dtype=np.float)
    random_quadruple_norm = 1
    nonzero = False
    if isinstance(quadruple_norm, numbers.Number):
        nonzero = True
    while True:
        quadruple = (np.random.random(4) - 0.5) * 2
        random_quadruple_norm = norm(quadruple)
        if nonzero and random_quadruple_norm > 0:
            break
        else:
            break
    if nonzero:
        quadruple /= random_quadruple_norm
        quadruple *= float(quadruple_norm)
    return Quaternion(quadruple)


def random_rotations_array(shape):
    """
    Calculates random rotation quaternions array
    :return: random Rotation array of given shape
    """
    rotations = np.empty(shape, dtype=object)
    size = rotations.size
    rotations = rotations.ravel()
    for i in range(size):
        rotations[i] = random_rotation()
    return rotations.reshape(shape)


def random_unit_quaternions_array(shape):
    """
    Calculates random unit quaternions array
    :return: random UnitQuaternion array of given shape
    """
    unit_quaternions = np.empty(shape, dtype=object)
    size = unit_quaternions.size
    unit_quaternions = unit_quaternions.ravel()
    for i in range(size):
        unit_quaternions[i] = random_unit_quaternion()
    return unit_quaternions.reshape(shape)


def random_quaternions_array(shape, quadruple_norm=None):
    """
    Calculates random quaternions array
    :return: random Quaternion array of given shape
    """
    quaternions = np.empty(shape, dtype=object)
    size = quaternions.size
    quaternions = quaternions.ravel()
    for i in range(size):
        quaternions[i] = random_quaternion(quadruple_norm=quadruple_norm)
    return quaternions.reshape(shape)
