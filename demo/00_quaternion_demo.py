from __future__ import division, print_function
import numpy as np

from Quaternions import Quaternion
from Quaternions import functions as qt

q1 = Quaternion()
q2 = Quaternion(np.array([1, 2, 3, 4]))
print('Quaternions:', q1, q2)
print('Scalar part:', q1.scalar_part())
print('Vector part:', q1.vector_part())


print('q1* =', q1.conjugate())
print('q2* =', q2.conjugate())
print('q1 + q2 =', q1 + q2)
print('q1 - q2 =', q1 - q2)
print('q1 * q2 =', q1 * q2)
print('q1 == q2:', q1 == q2, '\tq1 == q1:', q1 == q1)
print('||q1|| =', q1.norm())
print('||q2|| =', q2.norm())
print('distance d(q1,q2) =', q1.distance(q2))
print('q1 versor =', q1.versor())
print('q2 versor =', q2.versor())
print('q1 reciprocal =', q1.reciprocal())
print('q2 reciprocal =', q2.reciprocal())
print('q1 / 3 =', q1 / 3)
print('3 / q2 =', 3 / q2)
print('q1 real matrix:\n', q1.real_matrix())
print('q1 complex matrix:\n', q1.complex_matrix())
print('q1 polar representation:', q1.polar)
print('q2 polar representation:', q2.polar)
print('q1^3 =', q1 ** 3)
print('q2^3 =', q2 ** 3)
print('exp(q1) =', qt.exp(q1))
print('exp(q2) =', qt.exp(q2))
print('log(q1) =', qt.log(q1))
print('log(q2) =', qt.log(q2))
print('exp(log(q1)) =', qt.exp(qt.log(q1)))
print('exp(log(q2)) =', qt.exp(qt.log(q2)))
