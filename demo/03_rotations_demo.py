from __future__ import division, print_function
import numpy as np
from Quaternions import Rotation

rotation = Rotation()
rotation.euler_angles_convention = 'abg'
print(rotation)
