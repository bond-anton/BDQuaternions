from __future__ import division, print_function
import numbers
import numpy as np

from .Quaternion cimport Quaternion
from ._quaternion_operations cimport exp as q_exp, log as q_log


def exp(arg):
    """
    Calculate exponent function on quaternions and numbers
    :param arg: Quaternion, number or array of both or mix
    :return: exponent of Quaternion, number or array of both or mix
    """
    if isinstance(arg, Quaternion):
        return Quaternion(q_exp(arg.quadruple))
    elif isinstance(arg, numbers.Number):
        return np.exp(arg)
    elif isinstance(arg, (list, tuple, np.ndarray)):
        result = np.copy(np.array(arg))
        shape = result.shape
        size = result.size
        result = result.ravel()
        for i in range(size):
            result[i] = exp(result[i])
        return result.reshape(shape)
    else:
        raise ValueError('Not supported argument of type %s' % str(type(arg)))


def log(arg):
    """
    Calculate logarithm function on quaternions and numbers
    :param arg: Quaternion, number or array of both or mix
    :return: logarithm of Quaternion, number or array of both or mix
    """
    if isinstance(arg, Quaternion):
        return Quaternion(q_log(arg.quadruple))
    elif isinstance(arg, numbers.Number):
        return np.log(arg)
    elif isinstance(arg, (list, tuple, np.ndarray)):
        result = np.copy(np.array(arg))
        shape = result.shape
        size = result.size
        result = result.ravel()
        for i in range(size):
            result[i] = log(result[i])
        return result.reshape(shape)
    else:
        raise ValueError('Not supported argument of type %s' % str(type(arg)))
