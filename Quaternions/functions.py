from __future__ import division, print_function
import numbers
import numpy as np
from Quaternions import Quaternion


def exp(arg):
    if isinstance(arg, Quaternion):
        a = arg.scalar_part()
        v = arg.vector_part()
        v_norm = np.sqrt(np.sum(v * v))
        if not np.allclose(v_norm, 0.0):
            return Quaternion(np.hstack((np.exp(a) * np.cos(v_norm), np.exp(a) * v / v_norm * np.sin(v_norm))))
        else:
            return Quaternion(np.hstack((np.exp(a), np.zeros(3))))
    elif isinstance(arg, (np.ndarray, numbers.Number)):
        return np.exp(arg)
    else:
        raise ValueError('Not supported argument of type %s' % str(type(arg)))


def log(arg):
    if isinstance(arg, Quaternion):
        q_norm = arg.norm()
        if not np.allclose(q_norm, 0.0):
            a = arg.scalar_part()
            v = arg.vector_part()
            v_norm = np.sqrt(np.sum(v * v))
            quadruple = np.zeros(4)
            quadruple[0] = np.log(q_norm)
            if not np.allclose(v_norm, 0.0):
                quadruple[1:] = v / v_norm * np.arccos(a / q_norm)
            return Quaternion(quadruple)
        else:
            raise ValueError('Only nonzero-quaternions are supported by log function')
    elif isinstance(arg, (np.ndarray, numbers.Number)):
        return np.log(arg)
    else:
        raise ValueError('Not supported argument of type %s' % str(type(arg)))
