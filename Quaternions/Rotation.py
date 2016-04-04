from __future__ import division, print_function
import numbers
import numpy as np

from Quaternions import UnitQuaternion


class Rotation(UnitQuaternion):

    def __init__(self, quadruple=None, euler_angles_convention=None):
        self.euler_angles_convention = None
        self.set_euler_angles_convention(euler_angles_convention)
        if quadruple is None:
            quadruple = np.array([0, 0, 0, 1])
        quadruple = np.array(quadruple, dtype=np.float)
        assert np.allclose(np.sum(quadruple ** 2), 1.0)
        super(Rotation, self).__init__(quadruple)

    def set_euler_angles_convention(self, euler_angles_convention):
        conventions = {
            'Bunge': {'variants': ['bunge', 'zxz'],
                      'labels': ['phi1', 'Phi', 'phi2'],
                      'description': 'Bunge (phi1 Phi phi2) ZXZ convention'},
            'Matthies': {'variants': ['matthies', 'zyz', 'nfft', 'abg'],
                         'labels': ['alpha', 'beta', 'gamma'],
                         'description': 'Matthies (alpha beta gamma) ZYZ convention'},
            'Roe': {'variants': ['roe'],
                    'labels': ['Psi', 'Theta', 'Phi'],
                    'description': 'Roe (Psi, Theta, Phi) convention'},
            'Kocks': {'variants': ['kocks'],
                      'labels': ['Psi', 'Theta', 'phi'],
                      'description': 'Kocks (Psi Theta phi) convention'},
            'Canova': {'variants': ['canova'],
                       'labels': ['omega', 'Theta', 'phi'],
                       'description': 'Canova (omega, Theta, phi) convention'}
        }
        convention = conventions['Bunge']
        if euler_angles_convention is not None:
            match = False
            for key in conventions.keys():
                if str(euler_angles_convention).lower().strip() in conventions[key]['variants']:
                    convention = conventions[key]
                    match = True
                    break
            if not match:
                print('Convention: %s not found or not supported.' % euler_angles_convention)
                print('Falling back to Bunge convention.')
            elif 'bunge' in convention['variants']:
                print('You asked to use %s' % convention['description'])
                print('Unfortunately it is not supported for now.')
                print('Falling back to Bunge convention.')
                convention = conventions['Bunge']
        self.euler_angles_convention = convention

    def __add__(self, other):
        raise TypeError('Wrong operation for rotations \'+\'.')

    def __radd__(self, other):
        raise TypeError('Wrong operation for rotations \'+\'.')

    def __sub__(self, other):
        raise TypeError('Wrong operation for rotations \'-\'.')

    def __rsub__(self, other):
        raise TypeError('Wrong operation for rotations \'-\'.')
