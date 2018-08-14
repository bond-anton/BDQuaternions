from __future__ import division, print_function
import numpy as np

from cython import boundscheck, wraparound
from cpython.object cimport Py_EQ, Py_NE
from libc.math cimport M_PI


"""
description of rotations after Ken Shoemake in Graphics Gems IV (Academic Press, 1994), p. 222
"""

cdef class Conventions(object):

    def __init__(self):
        # the tuples are for inner axis (X - 0, Y - 1, Z - 2), parity (Even - 0, Odd - 1),
        # repetition (No - 0, Yes - 1), frame (0 - static; 1 - rotating frame)
        self.__euler_angles_codes = {
            # static frame
            'XYZs': (0, 0, 0, 0), 'XYXs': (0, 0, 1, 0), 'XZYs': (0, 1, 0, 0),
            'XZXs': (0, 1, 1, 0), 'YZXs': (1, 0, 0, 0), 'YZYs': (1, 0, 1, 0),
            'YXZs': (1, 1, 0, 0), 'YXYs': (1, 1, 1, 0), 'ZXYs': (2, 0, 0, 0),
            'ZXZs': (2, 0, 1, 0), 'ZYXs': (2, 1, 0, 0), 'ZYZs': (2, 1, 1, 0),
            # rotating frame
            'XYZr': (0, 0, 0, 1), 'XYXr': (0, 0, 1, 1), 'XZYr': (0, 1, 0, 1),
            'XZXr': (0, 1, 1, 1), 'YZXr': (1, 0, 0, 1), 'YZYr': (1, 0, 1, 1),
            'YXZr': (1, 1, 0, 1), 'YXYr': (1, 1, 1, 1), 'ZXYr': (2, 0, 0, 1),
            'ZXZr': (2, 0, 1, 1), 'ZYXr': (2, 1, 0, 1), 'ZYZr': (2, 1, 1, 1)
        }

        self.__default_convention = 'XYZs'

        """
        Dictionary of Euler angles conventions
        variants should be lower-cased possible list of synonyms of the convention
        """
        self.__special_conventions = {
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

        self.__general_conventions = {
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
        self.__derived_conventions = {
            # 'symmetrical angles' conventions used in crystallographic texture analysis
            'Kocks': {'variants': ('kocks',),
                      'parent_convention': 'Roe',
                      'axes_labels': ('RD', 'TD', 'ND'),
                      'labels': ('Psi', 'Theta', 'phi'),
                      'from_parent': Kocks2Roe(),
                      'to_parent': Kocks2Roe(),
                      'description': 'Kocks (Psi Theta phi) convention'},
            'Canova': {'variants': ('canova',),
                       'parent_convention': 'Roe',
                       'axes_labels': ('RD', 'TD', 'ND'),
                       'labels': ('omega', 'Theta', 'phi'),
                       'from_parent': Canova2Roe(),
                       'to_parent': Canova2Roe(),
                       'description': 'Canova (omega, Theta, phi) convention'}
        }

    @property
    def euler_angles_codes(self):
        return self.__euler_angles_codes

    @property
    def default_convention(self):
        return self.__default_convention

    @property
    def special_conventions(self):
        return self.__special_conventions

    @property
    def general_conventions(self):
        return self.__general_conventions

    @property
    def derived_conventions(self):
        return self.__derived_conventions

    def list_euler_angles_conventions(self, filter=None):
        """
        Function to get list of available conventions
        :param filter: None to select all or list of filtering strings: 'general', 'special', 'derived'
        :return: list of conventions
        """
        general = list(self.general_conventions.keys())
        special = list(self.special_conventions.keys())
        derived = list(self.derived_conventions.keys())
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

    @boundscheck(False)
    @wraparound(False)
    cpdef bint check(self, str convention):
        cdef:
            dict conventions
            str key
        conventions = dict(self.__general_conventions, **self.__special_conventions)
        # first we search requested convention in the dict of 'standard' conventions
        for key in conventions.keys():
            if convention.lower().strip() in conventions[key]['variants']:
                return True
        # If not found we look through the dict of 'derived' conventions
        for key in self.__derived_conventions.keys():
            if convention.lower().strip() in self.__derived_conventions[key]['variants']:
                return True
        return False

    @boundscheck(False)
    @wraparound(False)
    cdef dict __get_convention(self, str convention):
        cdef:
            dict conventions, euler_angles_convention, parent_convention
            bint match = False
            str key, parent_label
        conventions = dict(self.__general_conventions, **self.__special_conventions)
        euler_angles_convention = conventions[self.__default_convention]
        # first we search requested convention in the dict of 'standard' conventions
        for key in conventions.keys():
            if convention.lower().strip() in conventions[key]['variants']:
                euler_angles_convention = conventions[key]
                euler_angles_convention['label'] = key
                euler_angles_convention['parent_convention'] = ''
                euler_angles_convention['code'] = self.__euler_angles_codes[euler_angles_convention['axes']]
                euler_angles_convention['to_parent'] = Function()
                euler_angles_convention['from_parent'] = Function()
                match = True
                break
        # If not found we look through the dict of 'derived' conventions
        for key in self.__derived_conventions.keys():
            if convention.lower().strip() in self.__derived_conventions[key]['variants']:
                euler_angles_convention = self.__derived_conventions[key]
                euler_angles_convention['label'] = key
                parent_label = euler_angles_convention['parent_convention']
                while True:
                    parent_convention = self.__get_convention(parent_label)
                    if 'axes' in list(parent_convention.keys()):
                        euler_angles_convention['axes'] = parent_convention['axes']
                        euler_angles_convention['code'] = self.__euler_angles_codes[euler_angles_convention['axes']]
                        break
                match = True
                break
        if not match:
            euler_angles_convention = self.__get_convention(self.__default_convention)
        return euler_angles_convention

    @boundscheck(False)
    @wraparound(False)
    cpdef Convention get_convention(self, str convention):
        """
        returns euler_angles convention as a dict
        :param convention: string with short form of convention e.g. 'XYZs' or 'Bunge'
        :return: dict describing convention
        """
        cdef:
            dict euler_angles_convention
            bint match = False
            str key
        euler_angles_convention = self.__get_convention(convention)
        return Convention(euler_angles_convention['label'], euler_angles_convention['axes'],
                          list(euler_angles_convention['axes_labels']), list(euler_angles_convention['labels']),
                          list(euler_angles_convention['code']), euler_angles_convention['description'],
                          euler_angles_convention['parent_convention'],
                          euler_angles_convention['to_parent'],
                          euler_angles_convention['from_parent'])


cdef class Convention(object):

    def __init__(self, str label, str axes, list axes_labels, list angle_labels, list code, str description='',
                 str parent='', Function to_parent=Function(), Function from_parent=Function()):
        self.__euler_safe_axis = (0, 1, 2, 0)
        self.__euler_next_axis = (1, 2, 0, 1)
        self.__label = label
        self.__axes = axes
        self.__axes_labels = [str(axes_label) for axes_label in axes_labels]
        self.__angle_labels = [str(angle_label) for angle_label in angle_labels]
        self.__code = [int(code_i) for code_i in code]
        self.__description = description
        self.__conventions = Conventions()
        self.__to_parent = to_parent
        self.__from_parent = from_parent
        if self.__conventions.check(parent):
            self.__parent = self.__conventions.get_convention(parent)
        else:
            self.__parent = self

    @property
    def euler_next_axis(self):
        return self.__euler_next_axis

    @property
    def label(self):
        return self.__label

    @property
    def axes(self):
        return self.__axes

    @property
    def axes_labels(self):
        return self.__axes_labels

    @property
    def angle_labels(self):
        return self.__angle_labels

    @property
    def code(self):
        return self.__code

    @property
    def description(self):
        return self.__description

    @property
    def parent(self):
        return self.__parent

    def __richcmp__(x, y, int op):
        if op == Py_EQ:
            if isinstance(x, Convention) and isinstance(y, Convention):
                if x.label == y.label:
                    if x.code == y.code:
                        if x.axes == y.axes:
                            if x.axes_labels == y.axes_labels:
                                if x.angle_labels == y.angle_labels:
                                    return True
            return False
        elif op == Py_NE:
            if isinstance(x, Convention) and isinstance(y, Convention):
                if x.label == y.label:
                    if x.code == y.code:
                        if x.axes == y.axes:
                            if x.axes_labels == y.axes_labels:
                                if x.angle_labels == y.angle_labels:
                                    return False
            return True
        else:
            return False

    @property
    def parent(self):
        if self.__parent != self:
            return self.__parent
        else:
            return None

    cpdef double[:] to_parent(self, double[:] euler_angles):
        return self.__to_parent.evaluate(euler_angles)

    cpdef double[:] from_parent(self, double[:] euler_angles):
        return self.__from_parent.evaluate(euler_angles)

    @boundscheck(False)
    @wraparound(False)
    cpdef void print_convention_tree(self):
        """
        Theoretically derived conventions can be nested endless.
        This function prints out the tree beginning from highest level parent convention.
        """
        cdef:
            list flat_parent_list = []
            Convention current_convention = self
        while current_convention.parent:
            flat_parent_list.append(current_convention)
            current_convention = current_convention.parent
        flat_parent_list.append(current_convention)
        level = 0
        while flat_parent_list:
            current_convention = flat_parent_list.pop()
            print('-' * level + ' ' * (level > 0) + current_convention.label)
            level += 1


cdef class Function(object):
    cpdef double[:] evaluate(self, double[:] euler_angles):
        cdef double[:] result = np.zeros(3, dtype=np.double)
        result[0] = euler_angles[0]
        result[1] = euler_angles[1]
        result[2] = euler_angles[2]
        return result

cdef class Kocks2Roe(Function):
    cpdef double[:] evaluate(self, double[:] euler_angles):
        cdef double[:] result = np.zeros(3, dtype=np.double)
        result[0] = euler_angles[0]
        result[1] = euler_angles[1]
        result[2] = M_PI - euler_angles[2]
        return result

cdef class Canova2Roe(Function):
    cpdef double[:] evaluate(self, double[:] euler_angles):
        cdef double[:] result = np.zeros(3, dtype=np.double)
        result[0] = M_PI / 2 - euler_angles[0]
        result[1] = euler_angles[1]
        result[2] = 3 * M_PI / 2 - euler_angles[2]
        return result
