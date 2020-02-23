import numpy as np

from cython import boundscheck, wraparound

from cpython.array cimport array, clone
from libc.stdlib cimport rand, RAND_MAX
from ._quaternion_operations cimport norm
from .Quaternion cimport Quaternion
from .UnitQuaternion cimport UnitQuaternion
from .Rotation cimport Rotation


cpdef Rotation random_rotation():
    """
    Calculates random rotation quaternion
    :return: random Rotation
    """
    cdef:
        double random_quadruple_norm
        array[double] quadruple, template = array('d')
    quadruple = clone(template, 4, zero=False)
    while True:
        quadruple[0] = rand()
        quadruple[1] = rand()
        quadruple[2] = rand()
        quadruple[3] = rand()
        random_quadruple_norm = norm(quadruple)
        if random_quadruple_norm > 0:
            break
    quadruple[0] /= random_quadruple_norm
    quadruple[1] /= random_quadruple_norm
    quadruple[2] /= random_quadruple_norm
    quadruple[3] /= random_quadruple_norm
    return Rotation(quadruple)


cpdef UnitQuaternion random_unit_quaternion():
    """
    Calculates random unit quaternion
    :return: random UnitQuaternion
    """
    cdef:
        double random_quadruple_norm
        array[double] quadruple, template = array('d')
    quadruple = clone(template, 4, zero=False)
    while True:
        quadruple[0] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[1] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[2] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[3] = (rand() / RAND_MAX - 0.5) * 2
        random_quadruple_norm = norm(quadruple)
        if random_quadruple_norm > 0:
            break
    quadruple[0] /= random_quadruple_norm
    quadruple[1] /= random_quadruple_norm
    quadruple[2] /= random_quadruple_norm
    quadruple[3] /= random_quadruple_norm
    return UnitQuaternion(quadruple)


cpdef Quaternion random_quaternion(double quadruple_norm=1.0):
    """
    Calculates random quaternion
    :return: random Quaternion
    """
    cdef:
        double random_quadruple_norm
        array[double] quadruple, template = array('d')
    quadruple = clone(template, 4, zero=False)
    while True:
        quadruple[0] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[1] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[2] = (rand() / RAND_MAX - 0.5) * 2
        quadruple[3] = (rand() / RAND_MAX - 0.5) * 2
        random_quadruple_norm = norm(quadruple)
        if abs(quadruple_norm) > 0.0:
            if random_quadruple_norm > 0:
                break
        else:
            break
    if random_quadruple_norm > 0:
        quadruple[0] = quadruple[0] / random_quadruple_norm * abs(quadruple_norm)
        quadruple[1] = quadruple[1] / random_quadruple_norm * abs(quadruple_norm)
        quadruple[2] = quadruple[2] / random_quadruple_norm * abs(quadruple_norm)
        quadruple[3] = quadruple[3] / random_quadruple_norm * abs(quadruple_norm)
    return Quaternion(quadruple)


@boundscheck(False)
@wraparound(False)
cpdef random_rotations_array(shape):
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


@boundscheck(False)
@wraparound(False)
cpdef random_unit_quaternions_array(shape):
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


@boundscheck(False)
@wraparound(False)
cpdef random_quaternions_array(shape, double quadruple_norm=1.0):
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
