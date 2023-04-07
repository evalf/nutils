"""
The backports module provides minimal fallback implementations for functions
that are introduced in versions of Python that are newer than the minimum
required version. Function behavour is equal to its Python counterpart only to
the extent that the Nutils code base requires it to be. As such,
implementations found in this module should not be relied upon as general
drop-on replacements.
"""

# Introduced in Python 3.8

try:
    from functools import cached_property
except ImportError:

    # Fallback implementation. Notable difference: no lock is used to prevent
    # race conditions in multi-threaded contexts.

    class cached_property:

        def __init__(self, func):
            self.func = func

        def __set_name__(self, owner, name):
            self.attrname = name

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            try:
                val = instance.__dict__[self.attrname]
            except KeyError:
                val = instance.__dict__[self.attrname] = self.func(instance)
            return val


try:
    from math import comb
except ImportError:

    # Fallback implementation. Notable difference: if k > n, this
    # implementation raises a ValueError rather than returning 0.

    import math, functools, operator

    def comb(n, k):
        a, b = sorted([k, n-k])
        numer = functools.reduce(operator.mul, range(1+b, 1+n), 1)
        denom = math.factorial(a)
        return numer // denom


# vim:sw=4:sts=4:et
