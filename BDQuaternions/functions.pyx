import numbers
import numpy as np

from cython import boundscheck, wraparound

from .Quaternion cimport Quaternion
from ._quaternion_operations cimport exp as q_exp, log as q_log
from libc.math cimport exp as c_exp, log as c_log

@boundscheck(False)
@wraparound(False)
cpdef exp(arg):
    """
    Calculate exponent function on quaternions and numbers
    :param arg: Quaternion, number or array of both or mix
    :return: exponent of Quaternion, number or array of both or mix
    """
    cdef:
        int i, size
    if isinstance(arg, Quaternion):
        return Quaternion(q_exp(arg.quadruple))
    elif isinstance(arg, numbers.Number):
        return c_exp(arg)
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

@boundscheck(False)
@wraparound(False)
cpdef log(arg):
    """
    Calculate logarithm function on quaternions and numbers
    :param arg: Quaternion, number or array of both or mix
    :return: logarithm of Quaternion, number or array of both or mix
    """
    cdef:
        int i, size
    if isinstance(arg, Quaternion):
        return Quaternion(q_log(arg.quadruple))
    elif isinstance(arg, numbers.Number):
        return c_log(arg)
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
