from __future__ import division, print_function
import numpy as np

from Quaternions import _quaternion_operations as qo

q = qo.check_quadruple((np.random.random(4) - 0.5) * 2)
while qo.norm(q) == 0:
    q = qo.check_quadruple(np.random.random(4))
q_n = q / qo.norm(q)

v = np.array([1, 1, 1])


np.testing.assert_allclose(qo.norm(q_n), [1])
r_m = qo.quaternion_to_rotation_matrix(q)
r_m_n = qo.quaternion_to_rotation_matrix(q_n)
print('Comparison of rot matrices: r_m == r_m_n: ', np.allclose(r_m, r_m_n))
q_m = qo.quaternion_from_rotation_matrix(r_m)
print('Comparison of quaternions from rot. matrix:', np.allclose(q_n, q_m))
# print(q_n - q_m)
print(q_n)
print(q_m)