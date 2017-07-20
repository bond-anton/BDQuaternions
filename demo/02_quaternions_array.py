from __future__ import division, print_function
import numpy as np

from BDQuaternions import utils
from BDQuaternions import functions as qf

shape = (4, 4)
quaternions_array = utils.random_quaternions_array(shape, 5)
print(quaternions_array.shape)
print(qf.exp(qf.log(quaternions_array)) == quaternions_array + np.eye(min(shape)))
print(qf.log(qf.exp(quaternions_array)) == quaternions_array)

print(quaternions_array * utils.random_quaternion())
