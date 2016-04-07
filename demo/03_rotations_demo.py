from __future__ import division, print_function
import numpy as np
from Quaternions import Rotation
from Quaternions import _quaternion_operations as qo

rotation = Rotation()
rotation.euler_angles_convention = 'Bunge'
print(rotation)

rotation.axis_angle = ([1, 0, 0], np.deg2rad(90))
print(rotation)

rotation2 = Rotation(rotation.quadruple * np.array([-1, -1, -1, -1]))
rotation2.euler_angles_convention = 'Bunge'
print(rotation2, 3*np.pi/2)
print(rotation2 == rotation)

'''
m = rotation.rotation_matrix
v = np.array([0, 0, 1])
print('m*z = ', np.dot(m, v))
qv = Rotation(np.hstack([0, v]))
print('q*z*q\' = ', (rotation * qv * rotation.reciprocal()).quadruple[1:])

m2 = np.dot(m, m)
rotation.rotation_matrix = m2
print(rotation)
print('m2*z = ', np.dot(m2, v))
print('q*z*q\' = ', (rotation * qv * rotation.reciprocal()).quadruple[1:])
'''
