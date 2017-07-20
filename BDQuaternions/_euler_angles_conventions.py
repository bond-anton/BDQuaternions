from __future__ import division, print_function
import numpy as np

"""
description of rotations after Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222
"""

euler_next_axis = (1, 2, 0, 1)

# the tuples are for inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
# repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
euler_angles_codes = {
    # static frame
    'XYZs': (0, 0, 0, 0), 'XYXs': (0, 0, 1, 0), 'XZYs': (0, 1, 0, 0),
    'XZXs': (0, 1, 1, 0), 'YZXs': (1, 0, 0, 0), 'YZYs': (1, 0, 1, 0),
    'YXZs': (1, 1, 0, 0), 'YXYs': (1, 1, 1, 0), 'ZXYs': (2, 0, 0, 0),
    'ZXZs': (2, 0, 1, 0), 'ZYXs': (2, 1, 0, 0), 'ZYZs': (2, 1, 1, 0),
    # rotating frame
    'ZYXr': (0, 0, 0, 1), 'XYXr': (0, 0, 1, 1), 'YZXr': (0, 1, 0, 1),
    'XZXr': (0, 1, 1, 1), 'XZYr': (1, 0, 0, 1), 'YZYr': (1, 0, 1, 1),
    'ZXYr': (1, 1, 0, 1), 'YXYr': (1, 1, 1, 1), 'YXZr': (2, 0, 0, 1),
    'ZXZr': (2, 0, 1, 1), 'XYZr': (2, 1, 0, 1), 'ZYZr': (2, 1, 1, 1)}

default_convention = 'XYZs'

"""
Dictionary of Euler angles conventions
variants should be lower-cased possible list of synonyms of the convention
"""
special_conventions = {
    # special named conventions
    'Nautical': {'variants': ('nautical', 'aircraft', 'cardan'),
                 'axes': 'ZYXr',
                 'axes_labels': ('roll', 'pitch', 'yaw'),
                 'labels': ('yaw', 'pitch', 'roll'),
                 'description': 'Nautical (yaw pitch roll) ZYXr convention'},
    'Bunge': {'variants': ('bunge',),
              'axes': 'ZXZr',
              'axes_labels': ('X', 'Y', 'Z'),
              'labels': ('phi1', 'Phi', 'phi2'),
              'description': 'Bunge (phi1 Phi phi2) ZXZr convention'},
    'Matthies': {'variants': ('matthies', 'nfft', 'abg'),
                 'axes': 'ZYZr',
                 'axes_labels': ('X', 'Y', 'Z'),
                 'labels': ('alpha', 'beta', 'gamma'),
                 'description': 'Matthies (alpha beta gamma) ZYZr convention'},
    'Roe': {'variants': ('roe',),
            'axes': 'ZYZr',
            'axes_labels': ('RD', 'TD', 'ND'),
            'labels': ('Psi', 'Theta', 'Phi'),
            'description': 'Roe (Psi, Theta, Phi) RD,TD,ND convention (ZYZr)'},
}

general_conventions = {
    # General conventions in the axes names notation XYZr or ZXZs. r/s means rotating or static frame
    # static frame conventions
    'XYZs': {'variants': ('xyzs', 'sxyz'),
             'axes': 'XYZs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XYZ static frame convention'},
    'XYXs': {'variants': ('xyxs', 'sxyx'),
             'axes': 'XYXs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XYX static frame convention'},
    'XZYs': {'variants': ('xzys', 'sxzy'),
             'axes': 'XZYs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XZY static frame convention'},
    'XZXs': {'variants': ('xzxs', 'sxzx'),
             'axes': 'XZXs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XZX static frame convention'},
    'YZXs': {'variants': ('yzxs', 'syzx'),
             'axes': 'YZXs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YZX static frame convention'},
    'YZYs': {'variants': ('yzys', 'syzy'),
             'axes': 'YZYs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YZY static frame convention'},
    'YXZs': {'variants': ('yxzs', 'syxz'),
             'axes': 'YXZs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YXZ static frame convention'},
    'YXYs': {'variants': ('yxys', 'syxy'),
             'axes': 'YXYs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YXY static frame convention'},
    'ZXYs': {'variants': ('zxys', 'szxy'),
             'axes': 'ZXYs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZXY static frame convention'},
    'ZXZs': {'variants': ('zxzs', 'szxz'),
             'axes': 'ZXZs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZXZ static frame convention'},
    'ZYXs': {'variants': ('zyxs', 'szyx'),
             'axes': 'ZYXs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZYX static frame convention'},
    'ZYZs': {'variants': ('zyzs', 'szyz'),
             'axes': 'ZYZs',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZYZ static frame convention'},
    # rotating frame
    'ZYXr': {'variants': ('zyxr', 'rzyx'),
             'axes': 'ZYXr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZYX rotating frame convention'},
    'XYXr': {'variants': ('xyxr', 'rxyx'),
             'axes': 'XYXr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XYX rotating frame convention'},
    'YZXr': {'variants': ('yzxr', 'ryzx'),
             'axes': 'YZXr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YZX rotating frame convention'},
    'XZXr': {'variants': ('xzxr', 'rxzx'),
             'axes': 'XZXr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XZX rotating frame convention'},
    'XZYr': {'variants': ('xzyr', 'rxzy'),
             'axes': 'XZYr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XZY rotating frame convention'},
    'YZYr': {'variants': ('yzyr', 'ryzy'),
             'axes': 'YZYr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YZY rotating frame convention'},
    'ZXYr': {'variants': ('zxyr', 'rzxy'),
             'axes': 'ZXYr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZXY rotating frame convention'},
    'YXYr': {'variants': ('yxyr', 'ryxy'),
             'axes': 'YXYr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YXY rotating frame convention'},
    'YXZr': {'variants': ('yxzr', 'ryxz'),
             'axes': 'YXZr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'YXZ rotating frame convention'},
    'ZXZr': {'variants': ('zxzr', 'rzxz'),
             'axes': 'ZXZr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZXZ rotating frame convention'},
    'XYZr': {'variants': ('xyzr', 'rxyz'),
             'axes': 'XYZr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'XYZ rotating frame convention'},
    'ZYZr': {'variants': ('zyzr', 'rzyz'),
             'axes': 'ZYZr',
             'axes_labels': ('X', 'Y', 'Z'),
             'labels': ('theta_1', 'theta_2', 'theta_3'),
             'description': 'ZYZ rotating frame convention'}
}

"""
Dictionary of derived Euler angles conventions.
Those that use rotations with changing directions,
but can be derived from another 'standard' rotational Euler angles convention.
Variants should be lower-cased possible list of synonyms of the convention.
from_parent is the routine to convert from parent convention Euler angles to the described.
to_parent is the routine to convert to parent convention Euler angles from the described.
"""
derived_conventions = {
    # 'symmetrical angles' conventions used in crystallographic texture analysis
    'Kocks': {'variants': ('kocks',),
              'parent_convention': 'Roe',
              'axes_labels': ('RD', 'TD', 'ND'),
              'labels': ('Psi', 'Theta', 'phi'),
              'from_parent': lambda Psi, Theta, Phi: (Psi, Theta, np.pi - Phi),
              'to_parent': lambda Psi, Theta, phi: (Psi, Theta, np.pi - phi),
              'description': 'Kocks (Psi Theta phi) convention'},
    'Canova': {'variants': ('canova',),
               'parent_convention': 'Roe',
               'axes_labels': ('RD', 'TD', 'ND'),
               'labels': ('omega', 'Theta', 'phi'),
               'from_parent': lambda Psi, Theta, Phi: (np.pi / 2 - Psi, Theta, 3 * np.pi / 2 - Phi),
               'to_parent': lambda omega, Theta, phi: (np.pi / 2 - omega, Theta, 3 * np.pi / 2 - phi),
               'description': 'Canova (omega, Theta, phi) convention'}
}


def list_euler_angles_conventions(filter=None):
    """
    Function to get list of available conventions
    :param filter: None to select all or list of filtering strings: 'general', 'special', 'derived'
    :return: list of conventions
    """
    general = general_conventions.keys()
    special = special_conventions.keys()
    derived = derived_conventions.keys()
    if filter is None:
        final_list = general + special + derived
    elif isinstance(filter, str):
        if 'general' == filter.strip():
            final_list = general
        elif 'special' == filter.strip():
            final_list = special
        elif 'derived' == filter.strip():
            final_list = derived
        else:
            final_list = general + special + derived
            print('Expected list or any of known filter words [\'general\', \'special\', \'derived\'] or None.')
            print('Falling back to complete list')
    elif isinstance(filter, (list, tuple, np.ndarray)):
        final_list = []
        if 'general' in filter:
            final_list += general
        if 'special' in filter:
            final_list += special
        if 'derived' in filter:
            final_list += derived
        if not final_list:
            final_list = general + special + derived
            print('Expected list or any of known filter words [\'general\', \'special\', \'derived\'] or None.')
            print('Falling back to complete list')
    else:
        final_list = general + special + derived
        print('Expected list of known filter words [\'general\', \'special\', \'derived\'] or None.')
        print('Falling back to complete list')
    return final_list
