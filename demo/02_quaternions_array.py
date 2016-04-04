from __future__ import division, print_function
import numpy as np
from Quaternions import Quaternion

#quaternions_array = np.array([Quaternion((np.random.random(4) - 1) * 10) for i in range(10)], dtype=Quaternion)
quaternions_array = np.array([Quaternion(np.arange(4)) for i in range(3)], dtype=Quaternion) + 1
print((quaternions_array ** 2)[0])

