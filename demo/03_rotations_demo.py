from __future__ import division, print_function
import numpy as np
from Quaternions import Rotation, list_euler_angles_conventions

rotation = Rotation()
rotation.euler_angles_convention = 'Bunge'
print(rotation)

rotation.axis_angle = ([1, 0, 0], np.deg2rad(90))
print(rotation)

rotation2 = Rotation(rotation.quadruple * np.array([-1, -1, -1, -1]))
rotation2.euler_angles_convention = 'Bunge'
print(rotation2, 3*np.pi/2)
print(rotation2 == rotation)

print(list_euler_angles_conventions('special'))
print(list_euler_angles_conventions('derived'))
print(list_euler_angles_conventions(['special', 'derived']))
print(list_euler_angles_conventions('general'))
