from __future__ import division, print_function
import numpy as np
from Quaternions import Rotation

rotation = Rotation()
rotation.euler_angles_convention = 'abg'
print(rotation)

m = rotation.rotation_matrix
noise = np.random.random((3, 3)) * 1e-6
#print(m + noise)
rotation.rotation_matrix = m + noise
print(rotation)