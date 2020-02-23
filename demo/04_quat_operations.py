import numpy as np

from BDQuaternions import Rotation
from BDQuaternions import _quaternion_operations as qo


count = 0
while count < 5:
    q = (np.random.random(4) - 0.5) * 2
    while qo.norm(q) == 0:
        q = (np.random.random(4) - 0.5) * 2
    q_n = q / qo.norm(q)

    r_m = qo.quaternion_to_rotation_matrix(q)
    q_m = np.asarray(qo.quaternion_from_rotation_matrix(r_m))
    t = np.trace(r_m)
    if t <= 0 and r_m[1, 1] < r_m[2, 2]:
        print('Comparison of quaternions from rot. matrix:',
              np.allclose(q_n, q_m) or np.allclose(q_n, -q_m))
        # print(q_n - q_m)
        rotation_1 = Rotation(q_n)
        rotation_2 = Rotation(q_m)
        axis_1, angle_1 = rotation_1.axis_angle
        axis_2, angle_2 = rotation_2.axis_angle
        print('was:', q_n, axis_1, np.rad2deg(angle_1))
        print('now:', q_m, axis_2, np.rad2deg(angle_2))
        print(np.dot(axis_1, axis_2))
        count += 1
