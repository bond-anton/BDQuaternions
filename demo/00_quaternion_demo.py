from __future__ import division, print_function
import numpy as np

from BDQuaternions import Quaternion

q1 = Quaternion(np.array([0, 0, 0, 1], dtype=np.double))
print('Q1:')
print(q1)

print('Q1 norm:')
print(q1.norm())

q2 = Quaternion(np.array([1, 2, 3, 4], dtype=np.double))
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
