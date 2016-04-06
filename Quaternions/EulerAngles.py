from __future__ import division, print_function

from EulerAnglesConventions import euler_angles_codes, conventions, default_convention


def check_euler_angles_convention(convention):
    """
    Euler angles conversion algorithm by Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222
    """
    euler_angles_convention = conventions[default_convention]
    if convention is not None:
        match = False
        for key in conventions.keys():
            if str(convention).lower().strip() in conventions[key]['variants']:
                euler_angles_convention = conventions[key]
                euler_angles_convention['title'] = key
                match = True
                break
        if not match:
            print('Convention: %s not found or not supported.' % convention)
            print('Falling back to default convention %s.' % default_convention)
            euler_angles_convention['title'] = default_convention
    euler_angles_convention['code'] = euler_angles_codes[euler_angles_convention['axes']]
    return euler_angles_convention
