cdef class Convention(object):
    cdef:
        int[4] __euler_next_axis
        str __label
        str __axes
        list __axes_labels
        list __angle_labels
        int[4] __code
        str __description
        Convention __parent
        Function __to_parent
        Function __from_parent
        Conventions __conventions

    cpdef double[:] to_parent(self, double[:] euler_angles)
    cpdef double[:] from_parent(self, double[:] euler_angles)
    cpdef void print_convention_tree(self)


cdef class Conventions(object):
    cdef:
        dict __euler_angles_codes
        str __default_convention
        dict __special_conventions
        dict __general_conventions
        dict __derived_conventions

    cpdef bint check(self, str convention)
    cdef dict __get_convention(self, str convention)
    cpdef Convention get_convention(self, str convention)


cdef class Function(object):
    cpdef double[:] evaluate(self, double[:] euler_angles)
