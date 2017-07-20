from __future__ import division, print_function
import numpy as np

from BDQuaternions import Quaternion
from BDQuaternions import functions as qf

q1 = Quaternion(np.array([[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                          [[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                          [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]]))
print('Q1:')
print(q1)

print('Q1 norm:')
print(q1.norm())

q2 = Quaternion([1, 2, 3, 4])
print('Q2:')
print(q2)
print('Q2 norm:')
print(q2.norm())

print('Q1 + Q2:')
print(q1 + q2)

print('Q1 - Q2:')
print(q1 - q2)

print('Q1 * Q2:')
print(q1 * q2)

print('Q1 versor:')
print(q1.versor())
