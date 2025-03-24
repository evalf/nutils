"""
The function module defines the :class:`Evaluable` class and derived objects,
commonly referred to as nutils functions. They represent mappings from a
:mod:`nutils.topology` onto Python space. The notabe class of :class:`Array`
objects map onto the space of Numpy arrays of predefined dimension and shape.
Most functions used in nutils applicatons are of this latter type, including the
geometry and function bases for analysis.

Nutils functions are essentially postponed python functions, stored in a tree
structure of input/output dependencies. Many :class:`Array` objects have
directly recognizable numpy equivalents, such as :class:`Sin` or
:class:`Inverse`. By not evaluating directly but merely stacking operations,
complex operations can be defined prior to entering a quadrature loop, allowing
for a higher level style programming. It also allows for automatic
differentiation and code optimization.

It is important to realize that nutils functions do not map for a physical
xy-domain but from a topology, where a point is characterized by the combination
of an element and its local coordinate. This is a natural fit for typical finite
element operations such as quadrature. Evaluation from physical coordinates is
possible only via inverting of the geometry function, which is a fundamentally
expensive and currently unsupported operation.
"""

import typing
if typing.TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    Protocol = object

from . import debug_flags, _util as util, types, numeric, cache, warnings, parallel, _pyast
from functools import cached_property
from ._graph import Node, RegularNode, DuplicatedLeafNode, InvisibleNode, Subgraph, TupleNode
from statistics import geometric_mean
import nutils_poly as poly
import numpy
import sys
import itertools
import functools
import operator
import inspect
import numbers
import builtins
import re
import types as builtin_types
import abc
import collections.abc
import math
import treelog as log
import time
import contextlib
import subprocess
import os
import multiprocessing
import hashlib
import linecache
from io import StringIO

graphviz = os.environ.get('NUTILS_GRAPHVIZ')

isevaluable = lambda arg: isinstance(arg, Evaluable)


def simplified(value: 'Evaluable'):
    assert isinstance(value, Evaluable), f'value={value!r}'
    return value.simplified


_array_dtypes = bool, int, float, complex
_array_dtype_to_kind = {bool: 'b', int: 'i', float: 'f', complex: 'c'}
asdtype = lambda arg: arg if any(arg is dtype for dtype in _array_dtypes) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]
Dtype = typing.Union[_array_dtypes]
_BlockId = typing.Tuple[int, ...]


def asarray(arg):
    if hasattr(type(arg), 'as_evaluable_array'):
        return arg.as_evaluable_array
    if _containsarray(arg):
        return stack(arg, axis=0)
    else:
        return constant(arg)


def _isindex(arg):
    return isinstance(arg, Array) and arg.ndim == 0 and arg.dtype == int and arg._intbounds[0] >= 0


def _equals_simplified(arg1: 'Array', arg2: 'Array'):
    'return True if certainly equal, False if certainly not equal, None otherwise'

    assert isinstance(arg1, Array), f'arg1={arg1!r}'
    assert isinstance(arg2, Array), f'arg2={arg2!r}'
    if arg1 is arg2:
        return True
    if arg1.dtype != arg2.dtype or arg1.ndim != arg2.ndim:
        return False
    arg1 = arg1.simplified
    arg2 = arg2.simplified
    if arg1 is arg2:
        return True
    if arg1.arguments != arg2.arguments:
        return False
    if isinstance(arg1, Constant) and isinstance(arg2, Constant):
        return False # values differ


_certainly_different = lambda a, b: _equals_simplified(a, b) is False
_certainly_equal = lambda a, b: _equals_simplified(a, b) is True
_any_certainly_different = lambda a, b: len(a) != len(b) or any(map(_certainly_different, a, b))
_all_certainly_equal = lambda a, b: len(a) == len(b) and all(map(_certainly_equal, a, b))


class ExpensiveEvaluationWarning(warnings.NutilsInefficiencyWarning):
    pass


class NotPolynomal(Exception):
    def __init__(self, array, argument):
        super().__init__(f'{array} is not polynomial in argument {argument.name!r}')


class Evaluable(types.DataClass):
    'Base class'

    dependencies = util.abstract_property()

    @staticmethod
    def evalf(*args):
        raise NotImplementedError('Evaluable derivatives should implement the evalf method')

    @cached_property
    def arguments(self):
        'a frozenset of all arguments of this evaluable'
        return frozenset().union(*(child.arguments for child in self.dependencies))

    @property
    def isconstant(self):
        return not self.arguments

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self in cache:
            return cache[self]
        args = tuple(arg._node(cache, subgraph, times, unique_loop_ids) for arg in self.dependencies)
        label = '\n'.join(filter(None, (type(self).__name__, self._node_details)))
        cache[self] = node = RegularNode(label, args, {}, (type(self).__name__, times[self]), subgraph)
        return node

    @property
    def _node_details(self):
        return ''

    def asciitree(self, richoutput=False):
        'string representation'

        return self._node({}, None, collections.defaultdict(_Stats), False).generate_asciitree(richoutput)

    def __str__(self):
        return self.__class__.__name__

    @cached_property
    def eval(self):
        '''Evaluate function on a specified element, point set.'''

        return compile(self, _simplify=False, _optimize=False, stats=False, cache_const_intermediates=False)

    @util.deep_replace_property
    def simplified(obj):
        retval = obj._simplified()
        if retval is None:
            return obj
        if isinstance(obj, Array):
            assert isinstance(retval, Array) and not _any_certainly_different(retval.shape, obj.shape) and retval.dtype == obj.dtype, '{} --simplify--> {}'.format(obj, retval)
        return retval

    def _simplified(self):
        return

    @cached_property
    def optimized_for_numpy(self):
        return self.simplified._optimized_for_numpy1

    @util.deep_replace_property
    def _optimized_for_numpy1(obj):
        retval = obj._optimized_for_numpy()
        if retval is None:
            return obj
        if isinstance(obj, Array):
            assert isinstance(retval, Array) and not _any_certainly_different(retval.shape, obj.shape), '{0}._optimized_for_numpy or {0}._simplified resulted in shape change'.format(type(obj).__name__)
        return retval

    def _optimized_for_numpy(self):
        return

    @cached_property
    def _loops(self):
        deps = util.IDSet()
        for arg in self.dependencies:
            deps |= arg._loops
        return deps.view()

    def _compile(self, builder: '_BlockTreeBuilder') -> _pyast.Expression:
        # Compiles the entire tree defined by this evaluable.
        #
        # Arguments must be compiled via `builder.compile`, not by directly
        # calling `Evaluable._compile`, because the former caches the compiled
        # evaluables.
        args = builder.compile(self.dependencies)
        expression = self._compile_expression(builder.get_evaluable_expr(self), *args)
        out = builder.get_variable_for_evaluable(self)
        builder.get_block_for_evaluable(self).assign_to(out, expression)
        return out

    def _compile_expression(self, py_self: _pyast.Variable, *args: _pyast.Expression):
        # Compiles this evaluable given compiled arguments.
        #
        # `py_self` is a variable that refers to `self`.

        # Instead of triggering an exception in `Evaluable.evalf` when running
        # the generated code, we raise an exception during compile time.
        if self.evalf is Evaluable.evalf:
            raise NotImplementedError

        return py_self.get_attr('evalf').call(*args)

    def argument_degree(self, argument):
        '''return the highest power of argument of self is polynomial,
        or raise NotPolynomal otherwise.'''
        # IMPORTANT: since we are tracking only the highest power, we cannot
        # ever lower a power, e.g. via division. To see this, consider the sum
        # of a 0th and 1st power, registered as degree 1. This would be lowered
        # via division to 0, thereby suggesting that the evaluable is constant
        # while in reality it is not polynomial.
        if argument not in self.arguments:
            return 0
        n = self._argument_degree(argument)
        if n is None:
            raise NotPolynomal(self, argument)
        return n

    def _argument_degree(self, argument):
        pass


class Tuple(Evaluable):

    items: typing.Tuple[Evaluable, ...]

    @property
    def dependencies(self):
        return self.items

    def _compile_expression(self, py_self, *items):
        return _pyast.Tuple(items)

    def __iter__(self):
        'iterate'

        return iter(self.items)

    def __len__(self):
        'length'

        return len(self.items)

    def __getitem__(self, item):
        'get item'

        return self.items[item]

    def __add__(self, other):
        'add'

        return Tuple(self.items + tuple(other))

    def __radd__(self, other):
        'add'

        return Tuple(tuple(other) + self.items)

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if (cached := cache.get(self)) is not None:
            return cached
        cache[self] = node = TupleNode(tuple(item._node(cache, subgraph, times, unique_loop_ids) for item in self.items), (type(self).__name__, times[self]), subgraph=subgraph)
        return node

    @property
    def _intbounds_tuple(self):
        return tuple(item._intbounds for item in self.items)


# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.


def add(arg0, *args):
    for arg1 in args:
        arg0 = Add(types.frozenmultiset(_numpy_align(arg0, arg1)))
    return arg0


def multiply(arg0, *args):
    for arg1 in args:
        arg0 = Multiply(types.frozenmultiset(_numpy_align(arg0, arg1)))
    return arg0


def sum(arg, axis=None):
    '''Sum array elements over a given axis.'''

    if axis is None:
        return Sum(arg)
    axes = (axis,) if numeric.isint(axis) else axis
    summed = Transpose.to_end(arg, *axes)
    for i in range(len(axes)):
        summed = Sum(summed)
    return summed


def product(arg, axis):
    return Product(Transpose.to_end(arg, axis))


def power(arg, n):
    arg, n = _numpy_align(arg, n)
    return Power(arg, n)


def dot(a, b, axes):
    '''
    Contract ``a`` and ``b`` along ``axes``.
    '''

    a, b = _numpy_align(a, b)
    if a.dtype == bool or b.dtype == bool:
        raise ValueError('The boolean dot product is not supported.')
    return multiply(a, b).sum(axes)


def conjugate(arg):
    arg = asarray(arg)
    if arg.dtype == complex:
        return Conjugate(arg)
    else:
        return arg


def real(arg):
    arg = asarray(arg)
    if arg.dtype == complex:
        return Real(arg)
    else:
        return arg


def imag(arg):
    arg = asarray(arg)
    if arg.dtype == complex:
        return Imag(arg)
    else:
        return zeros_like(arg)


def transpose(arg, trans):
    arg = asarray(arg)
    trans = tuple(i.__index__() for i in trans)
    if all(i == n for i, n in enumerate(trans)):
        return arg
    return Transpose(arg, trans)


def swapaxes(arg, axis1, axis2):
    arg = asarray(arg)
    trans = numpy.arange(arg.ndim)
    trans[axis1], trans[axis2] = trans[axis2], trans[axis1]
    return transpose(arg, trans)


def align(arg, where, shape):
    '''Align array to target shape.

    The align operation can be considered the opposite of transpose: instead of
    specifying for each axis of the return value the original position in the
    argument, align specifies for each axis of the argument the new position in
    the return value. In addition, the return value may be of higher dimension,
    with new axes being inserted according to the ``shape`` argument.

    Args
    ----
    arg : :class:`Array`
        Original array.
    where : :class:`tuple` of integers
        New axis positions.
    shape : :class:`tuple`
        Shape of the aligned array.

    Returns
    -------
    :class:`Array`
        The aligned array.
    '''

    where = list(where)
    for i, length in enumerate(shape):
        if i not in where:
            arg = InsertAxis(arg, length)
            where.append(i)
    if where != list(range(len(shape))):
        arg = Transpose(arg, util.untake(where))
    assert not _any_certainly_different(arg.shape, shape), f'arg.shape={arg.shape!r}, shape={shape!r}'
    return arg


def unalign(*args, naxes: int = None):
    '''Remove (joint) inserted axes.

    Given one or more array arguments, return the shortest common axis vector
    along with function arguments such that the original arrays can be
    recovered by :func:`align`. Axes beyond the first ``naxes`` are not
    considered for removal, keep their position (as seen from the right), and
    are not part of the common axis vector. Those axes should be added to the
    axis vector before calling :func:`align`.

    If ``naxes`` is ``None`` (the default), all arguments must have the same
    number of axes and ``naxes`` is set to this number.
    '''

    assert args
    if len(args) == 1 and naxes is None:
        return args[0]._unaligned
    if naxes is None:
        if any(arg.ndim != args[0].ndim for arg in args[1:]):
            raise ValueError('varying dimensions in unalign')
        naxes = args[0].ndim
    elif any(arg.ndim < naxes for arg in args):
        raise ValueError('one or more arguments have fewer axes than expected')
    nonins = functools.reduce(operator.or_, [set(arg._unaligned[1]) for arg in args]) & set(range(naxes))
    if len(nonins) == naxes:
        return (*args, tuple(range(naxes)))
    ret = []
    for arg in args:
        unaligned, where = arg._unaligned
        keep = tuple(range(naxes, arg.ndim))
        for i in sorted((nonins | set(keep)) - set(where)):
            unaligned = InsertAxis(unaligned, arg.shape[i])
            where += i,
        if not ret:  # first argument
            commonwhere = tuple(i for i in where if i < naxes)
        if where != commonwhere + keep:
            unaligned = transpose(unaligned, tuple(where.index(n) for n in commonwhere + keep))
        ret.append(unaligned)
    return (*ret, commonwhere)

# ARRAYS


def verify_sparse_chunks(func):
    if not debug_flags.sparse:
        return func
    def _assparse(self):
        chunks = func(self)
        assert isinstance(chunks, tuple)
        assert all(isinstance(chunk, tuple) for chunk in chunks)
        assert all(all(isinstance(item, Array) for item in chunk) for chunk in chunks)
        if self.ndim:
            for *indices, values in chunks:
                assert len(indices) == self.ndim
                assert all(idx.dtype == int for idx in indices)
                assert not any(_any_certainly_different(idx.shape, values.shape) for idx in indices)
        elif chunks:
            assert len(chunks) == 1
            chunk, = chunks
            assert len(chunk) == 1
            values, = chunk
            assert values.shape == ()
        return chunks
    return _assparse


class AsEvaluableArray(Protocol):
    'Protocol for conversion into an :class:`Array`.'

    @property
    def as_evaluable_array(self) -> 'Array':
        'Lower this object to a :class:`nutils.evaluable.Array`.'


class Array(Evaluable):
    '''
    Base class for array valued functions.

    Attributes
    ----------
    shape : :class:`tuple` of :class:`int`\\s
        The shape of this array function.
    ndim : :class:`int`
        The number of dimensions of this array array function.  Equal to
        ``len(shape)``.
    dtype : :class:`int`, :class:`float`
        The dtype of the array elements.
    '''

    __array_priority__ = 1.  # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

    shape = util.abstract_property()
    dtype = util.abstract_property()

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = item,
        if ... in item:
            iell = item.index(...)
            if ... in item[iell+1:]:
                raise IndexError('an index can have only a single ellipsis')
            # replace ellipsis by the appropriate number of slice(None)
            item = item[:iell] + (slice(None),)*(self.ndim-len(item)+1) + item[iell+1:]
        if len(item) > self.ndim:
            raise IndexError('too many indices for array')
        array = self
        for axis, it in reversed(tuple(enumerate(item))):
            array = get(array, axis, item=constant(it)) if numeric.isint(it) \
                else _takeslice(array, it, axis) if isinstance(it, slice) \
                else take(array, it, axis)
        return array

    def __bool__(self):
        return True

    def __len__(self):
        if self.ndim == 0:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    def __index__(self):
        try:
            index = self.__index
        except AttributeError:
            if self.ndim or self.dtype not in (int, bool) or not self.isconstant:
                raise TypeError('cannot convert {!r} to int'.format(self))
            index = self.__index = int(self.simplified.eval())
        return index

    T = property(lambda self: transpose(self, tuple(range(self.ndim-1, -1, -1))))

    __add__ = __radd__ = add
    __sub__ = lambda self, other: subtract(self, other)
    __rsub__ = lambda self, other: subtract(other, self)
    __mul__ = __rmul__ = multiply
    __truediv__ = lambda self, other: divide(self, other)
    __rtruediv__ = lambda self, other: divide(other, self)
    __pos__ = lambda self: self
    __neg__ = lambda self: negative(self)
    __pow__ = power
    __abs__ = lambda self: abs(self)
    __mod__ = lambda self, other: mod(self, other)
    __and__ = __rand__ = multiply
    __or__ = __ror__ = add
    __int__ = __index__
    __str__ = __repr__ = lambda self: '{}.{}<{}>'.format(type(self).__module__, type(self).__name__, self._shape_str(form=str))
    __inv__ = lambda self: LogicalNot(self) if self.dtype == bool else NotImplemented

    def _shape_str(self, form):
        prefix = shape = suffix = ''
        try:
            prefix = self.dtype.__name__[0] + ':'
            shape = ['?'] * self.ndim
            for i, n in enumerate(self.shape):
                if n.isconstant:
                    shape[i] = str(n.__index__())
            for i in set(range(self.ndim)) - set(self._unaligned[1]):
                shape[i] = f'({shape[i]})'
            for i, _ in self._inflations:
                shape[i] = f'~{shape[i]}'
            for axes in self._diagonals:
                for i in axes:
                    shape[i] = f'{shape[i]}/'
        except:
            suffix = '(e)'
        return prefix + ','.join(shape) + suffix

    sum = sum
    prod = product
    dot = dot
    swapaxes = swapaxes
    transpose = transpose
    choose = lambda self, choices: Choose(self, stack(choices, -1))
    conjugate = conjugate

    @property
    def real(self):
        return real(self)

    @property
    def imag(self):
        return imag(self)

    @property
    def assparse(self):
        if self.ndim:
            value_parts = []
            index_parts = []
            for flatindex, *indices, values in self._assparse:
                for n, index in zip(self.shape[1:], indices):
                    flatindex = flatindex * n + index
                value_parts.append(_flat(values))
                index_parts.append(_flat(flatindex))
            if value_parts:
                flatindex, inverse = unique(concatenate(index_parts), return_inverse=True)
                values = Inflate(concatenate(value_parts), inverse, flatindex.shape[0])
                indices = [flatindex]
                for n in reversed(self.shape[1:]):
                    indices[:1] = divmod(indices[0], n)
            else:
                indices = [zeros((constant(0),), int)] * self.ndim
                values = zeros((constant(0),), self.dtype)
        else: # scalar
            values = InsertAxis(self, constant(1))
            indices = ()
        return values, tuple(indices), self.shape

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        # Convert to a sequence of sparse COO arrays. The returned data is a tuple
        # of `(*indices, values)` tuples, where `values` is an `Array` with the
        # same dtype as `self`, but this is not enforced yet, and each index in
        # `indices` is an `Array` with dtype `int` and the exact same shape as
        # `values`. The length of `indices` equals `self.ndim`. In addition, if
        # `self` is 0d the length of `self._assparse` is at most one and the
        # `values` array must be 0d as well.
        #
        # The sparse data can be reassembled after evaluation by
        #
        #     dense = numpy.zeros(self.shape)
        #     for I0,...,Ik,V in self._assparse:
        #       for i0,...,ik,v in zip(I0.eval().ravel(),...,Ik.eval().ravel(),V.eval().ravel()):
        #         dense[i0,...,ik] = v

        indices = [prependaxes(appendaxes(Range(length), self.shape[i+1:]), self.shape[:i]) for i, length in enumerate(self.shape)]
        return (*indices, self),

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self in cache:
            return cache[self]
        args = tuple(arg._node(cache, subgraph, times, unique_loop_ids) for arg in self.dependencies)
        bounds = '[{},{}]'.format(*self._intbounds) if self.dtype == int else None
        label = '\n'.join(filter(None, (type(self).__name__, self._node_details, self._shape_str(form=repr), bounds)))
        cache[self] = node = RegularNode(label, args, {}, (type(self).__name__, times[self]), subgraph)
        return node

    # simplifications
    _multiply = lambda self, other: None
    _transpose = lambda self, axes: None
    _insertaxis = lambda self, axis, length: None
    _power = lambda self, n: None
    _add = lambda self, other: None
    _sum = lambda self, axis: None
    _take = lambda self, index, axis: None
    _rtake = lambda self, index, axis: None
    _determinant = lambda self, axis1, axis2: None
    _inverse = lambda self, axis1, axis2: None
    _takediag = lambda self, axis1, axis2: None
    _diagonalize = lambda self, axis: None
    _product = lambda self: None
    _sign = lambda self: None
    _eig = lambda self, symmetric: None
    _inflate = lambda self, dofmap, length, axis: None
    _rinflate = lambda self, func, length, axis: None
    _unravel = lambda self, axis, shape: None
    _ravel = lambda self, axis: None
    _loopsum = lambda self, loop_index: None  # NOTE: type of `loop_index` is `_LoopIndex`
    _real = lambda self: None
    _imag = lambda self: None
    _conjugate = lambda self: None

    @property
    def _unaligned(self):
        return self, tuple(range(self.ndim))

    _diagonals = ()
    _inflations = ()

    def _derivative(self, var, seen):
        if self.dtype in (bool, int) or var not in self.arguments:
            return Zeros(self.shape + var.shape, dtype=self.dtype)
        raise NotImplementedError('derivative not defined for {}'.format(self.__class__.__name__))

    @property
    def as_evaluable_array(self):
        'return self'

        return self

    @cached_property
    def _intbounds(self):
        # inclusive lower and upper bounds
        lower, upper = self._intbounds_impl()
        assert isinstance(lower, int) or lower == float('-inf')
        assert isinstance(upper, int) or upper == float('inf')
        assert lower <= upper
        return lower, upper

    def _intbounds_impl(self):
        if self.ndim == 0 and self.dtype == int and self.isconstant:
            value = self.__index__()
            return value, value
        return float('-inf'), float('inf')

    @property
    def _const_uniform(self):
        if self.dtype == int:
            lower, upper = self._intbounds
            return lower if lower == upper else None

    def _compile_with_out(self, builder, out, out_block_id, mode):
        # Compiles `self` and writes the result to `out`. It is the
        # responsibility of the caller to ensure that `out_block_id <=
        # builder.get_block_id(self)`. Valid values for mode are
        #
        #     'assign': results are assigned to `out`
        #     'iadd': results add added to `out`
        return NotImplemented


def assert_equal(a, b):
    return a if a == b else AssertEqual(a, b)


def assert_equal_tuple(A, B):
    assert len(A) == len(B)
    return tuple(map(assert_equal, A, B))


class AssertEqual(Array):
    'Confirm arrays equality at runtime'

    a: Array
    b: Array

    def __post_init__(self):
        assert not _certainly_different(self.a, self.b)

    @property
    def dtype(self):
        return self.a.dtype

    @cached_property
    def shape(self):
        return assert_equal_tuple(self.a.shape, self.b.shape)

    def _intbounds_impl(self):
        lowera, uppera = self.a._intbounds_impl()
        lowerb, upperb = self.b._intbounds_impl()
        return max(lowera, lowerb), min(uppera, upperb)

    def _simplified(self):
        # Canonicalize nested array equals to (((obj1 = obj2) = obj3) = obj4),
        # with all objects unique. We use the fact that self.a is already
        # simplified in descending only the left arm to gather all the equality
        # objects.
        left = []
        obj = self.a
        while isinstance(obj, AssertEqual):
            assert not isinstance(obj.b, AssertEqual)
            left.append(obj.b)
            obj = obj.a
        left.append(obj)
        # Again using the fact that self.b is simplified, we then descend its
        # left arm to stack new objects on top of self.a.
        retval = self.a
        obj = self.b
        while isinstance(obj, AssertEqual):
            assert not isinstance(obj.b, AssertEqual)
            if obj.b not in left:
                retval = AssertEqual(retval, obj.b)
            obj = obj.a
        if obj not in left:
            retval = AssertEqual(retval, obj)
        return retval

    @property
    def dependencies(self):
        return self.a, self.b

    @staticmethod
    def evalf(a, b):
        if a.shape != b.shape or (a != b).any():
            raise Exception('values are not equal')
        return a

    def _compile_expression(self, py_self, a, b):
        return _pyast.Variable('evaluable').get_attr('AssertEqual').get_attr('evalf').call(a, b)


class Orthonormal(Array):
    'make a vector orthonormal to a subspace'

    basis: Array
    vector: Array

    dtype = float

    def __post_init__(self):
        assert isinstance(self.basis, Array) and self.basis.ndim >= 2 and self.basis.dtype not in (bool, complex), f'basis={self.basis!r}'
        assert isinstance(self.vector, Array) and self.vector.ndim >= 1 and self.vector.dtype not in (bool, complex), f'vector={self.vector!r}'
        assert not _any_certainly_different(self.basis.shape[:-1], self.vector.shape)

    @property
    def dependencies(self):
        return self.basis, self.vector

    @cached_property
    def shape(self):
        return self.vector.shape

    def _simplified(self):
        if isunit(self.shape[-1]):
            return Sign(self.vector)
        basis, vector, where = unalign(self.basis, self.vector, naxes=self.ndim - 1)
        if len(where) < self.ndim - 1:
            return align(Orthonormal(basis, vector), (*where, self.ndim - 1), self.shape)

    @staticmethod
    def evalf(G, n):
        GG = numpy.einsum('...ki,...kj->...ij', G, G)
        v1 = numpy.einsum('...ij,...i->...j', G, n)
        v2 = numpy.linalg.solve(GG, v1[...,numpy.newaxis])[...,0] # NOTE: the newaxis/getitem dance is necessary since Numpy 2
        v3 = numpy.einsum('...ij,...j->...i', G, v2)
        return numeric.normalize(n - v3)

    def _compile_expression(self, py_self, G, n):
        return _pyast.Variable('evaluable').get_attr('Orthonormal').get_attr('evalf').call(G, n)

    def _derivative(self, var, seen):
        if isunit(self.shape[-1]):
            return zeros(self.shape + var.shape)

        # definitions:
        #
        # P := I - G (G^T G)^-1 G^T (orthogonal projector)
        # n := P v (orthogonal projection of v)
        # N := n / |n| (self: orthonormal projection of v)
        #
        # identities:
        #
        #   P^T = P          N^T N = 1
        #   P P = P          P N = N
        #   P G = P Q = 0    G^T N = Q^T N = 0
        #
        # derivatives:
        #
        # P' = Q P + P Q^T where Q := -G (G^T G)^-1 G'^T
        # n' = P' v + P v'
        #    = Q n + P (Q^T v + v')
        # N' = (I - N N^T) n' / |n|
        #    = (I - N N^T) (Q n / |n| + P (Q^T v + v') / |n|)
        #    = Q N + (P - N N^T) (Q^T v + v') / |n|

        G = self.basis
        invGG = inverse(einsum('Aki,Akj->Aij', G, G))

        Q = -einsum('Aim,Amn,AjnB->AijB', G, invGG, derivative(G, var, seen))
        QN = einsum('Ai,AjiB->AjB', self, Q)

        if _certainly_equal(G.shape[-1], G.shape[-2] - 1): # dim(kern(G)) = 1
            # In this situation, since N is a basis for the kernel of G, we
            # have the identity P == N N^T which cancels the entire second term
            # of N' along with any reference to v', reducing it to N' = Q N.
            return QN

        v = self.vector
        P = Diagonalize(ones(self.shape)) - einsum('Aim,Amn,Ajn->Aij', G, invGG, G)
        Z = P - einsum('Ai,Aj->Aij', self, self) # P - N N^T

        return QN + einsum('A,AiB->AiB',
            power(einsum('Ai,Aij,Aj->A', v, P, v), -.5),
            einsum('Aij,AjB->AiB', Z, einsum('Ai,AijB->AjB', v, Q) + derivative(v, var, seen)))


class Constant(Array):

    _value: types.arraydata

    dependencies = ()

    def __post_init__(self):
        assert isinstance(self._value, types.arraydata), f'value={self._value!r}'

    @cached_property
    def value(self):
        return numpy.asarray(self._value)

    @cached_property
    def dtype(self):
        return self._value.dtype

    @cached_property
    def shape(self):
        return tuple(constant(n) for n in self._value.shape)

    def _simplified(self):
        if not self.value.any(): # true if any axis is length 0
            return zeros_like(self)
        # At this point all axes are a least length 1
        for i, sh in enumerate(self.shape):
            pancake = iter(numpy.moveaxis(self.value, i, 0))
            first = next(pancake)
            if not first.ndim and numpy.all(self.value == first) or all(numpy.equal(first, other).all() for other in pancake):
                return insertaxis(constant(first), i, sh)
        # At this point all axes are a least length 2
        if self.ndim == 1 and self.dtype == int and self.value[-1] == self.value[0] + self.value.size - 1 and numpy.all(self.value[1:] > self.value[:-1]):
            r = Range(self.shape[0])
            if self.value[0]:
                r += self.value[0]
            return r

    def eval(self, /, **evalargs):
        return self.value

    def _compile(self, builder):
        # `self.value` is always a `numpy.ndarray`. If the array is 0d, we
        # convert the array to a `numpy.number` (`self.value[()]`), which
        # behaves like a `numpy.ndarray`, and has much faster implementations
        # for add, multiply etc.
        return builder.add_constant(self.value[()] if self.ndim == 0 else self.value)

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self.ndim:
            return super()._node(cache, subgraph, times, unique_loop_ids)
        elif self in cache:
            return cache[self]
        else:
            label = '{}'.format(self.value[()])
            if len(label) > 9:
                label = '~{:.2e}'.format(self.value[()])
            cache[self] = node = DuplicatedLeafNode(label, (type(self).__name__, times[self]))
            return node

    @cached_property
    def _isunit(self):
        return numpy.equal(self.value, 1).all()

    def _transpose(self, axes):
        return constant(self.value.transpose(axes))

    def _sum(self, axis):
        return constant((numpy.any if self.dtype == bool else numpy.sum)(self.value, axis))

    def _add(self, other):
        if isinstance(other, Constant):
            return constant(numpy.add(self.value, other.value))

    def _inverse(self, axis1, axis2):
        assert 0 <= axis1 < axis2 < self.ndim
        axes = (*range(axis1), *range(axis1+1, axis2), *range(axis2+1, self.ndim), axis1, axis2)
        value = numpy.transpose(self.value, axes)
        return constant(numpy.transpose(numeric.inv(value), util.untake(axes)))

    def _product(self):
        return constant((numpy.all if self.dtype == bool else numpy.prod)(self.value, -1))

    def _multiply(self, other):
        if self._isunit:
            return other
        if isinstance(other, Constant):
            return constant(numpy.multiply(self.value, other.value))

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        return constant(numpy.einsum('...kk->...k', numpy.transpose(self.value,
                                                                    list(range(axis1)) + list(range(axis1+1, axis2)) + list(range(axis2+1, self.ndim)) + [axis1, axis2])))

    def _take(self, index, axis):
        if isinstance(index, Constant):
            return constant(self.value.take(index.value, axis))

    def _power(self, n):
        if isinstance(n, Constant):
            return constant(numpy.power(self.value, n.value))

    def _eig(self, symmetric):
        eigval, eigvec = (numpy.linalg.eigh if symmetric else numpy.linalg.eig)(self.value)
        if not symmetric:
            eigval = eigval.astype(complex, copy=False)
            eigvec = eigvec.astype(complex, copy=False)
        return Tuple((constant(eigval), constant(eigvec)))

    def _sign(self):
        return constant(numpy.sign(self.value))

    def _unravel(self, axis, shape):
        shape = self.value.shape[:axis] + shape + self.value.shape[axis+1:]
        return constant(self.value.reshape(shape))

    def _determinant(self, axis1, axis2):
        value = numpy.transpose(self.value, tuple(i for i in range(self.ndim) if i != axis1 and i != axis2) + (axis1, axis2))
        return constant(numpy.linalg.det(value))

    def _intbounds_impl(self):
        if self.dtype == int and self.value.size:
            return int(self.value.min()), int(self.value.max())
        else:
            return super()._intbounds_impl()

    @property
    def _const_uniform(self):
        if self.ndim == 0:
            return self.dtype(self.value[()])


class InsertAxis(Array):

    func: Array
    length: Array

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func!r}'
        assert _isindex(self.length), f'length={self.length!r}'

    @property
    def dependencies(self):
        return self.func, self.length

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return (*self.func.shape, self.length)

    @property
    def _diagonals(self):
        return self.func._diagonals

    @cached_property
    def _inflations(self):
        return tuple((axis, types.frozendict((dofmap, InsertAxis(func, self.length)) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

    @cached_property
    def _unaligned(self):
        return self.func._unaligned

    def _simplified(self):
        if iszero(self.length):
            return zeros_like(self)
        return self.func._insertaxis(self.ndim-1, self.length)

    @staticmethod
    def evalf(func, length):
        if length == 1:
            return func[..., numpy.newaxis]
        try:
            return numpy.ndarray(buffer=func, dtype=func.dtype, shape=(*func.shape, length), strides=(*func.strides, 0))
        except ValueError:  # non-contiguous data
            return numpy.repeat(func[..., numpy.newaxis], length, -1)

    def _compile_expression(self, py_self, func, length):
        if isunit(self.length):
            return func.get_item(_pyast.Tuple((_pyast.Raw('...'), _pyast.Variable('numpy').get_attr('newaxis'))))
        else:
            return super()._compile_expression(py_self, func, length)

    def _derivative(self, var, seen):
        return insertaxis(derivative(self.func, var, seen), self.ndim-1, self.length)

    def _sum(self, i):
        if i == self.ndim - 1:
            return self.func if self.dtype == bool else self.func * astype(self.length, self.func.dtype)
        return InsertAxis(sum(self.func, i), self.length)

    def _product(self):
        return self.func if self.dtype == bool else self.func**astype(self.length, self.func.dtype)

    def _power(self, n):
        unaligned1, unaligned2, where = unalign(self, n)
        if len(where) != self.ndim:
            return align(unaligned1 ** unaligned2, where, self.shape)

    def _add(self, other):
        unaligned1, unaligned2, where = unalign(self, other)
        if len(where) != self.ndim:
            return align(unaligned1 + unaligned2, where, self.shape)

    def _multiply(self, other):
        unaligned1, unaligned2, where = unalign(self, other)
        if len(where) != self.ndim:
            return align(unaligned1 * unaligned2, where, self.shape)

    def _diagonalize(self, axis):
        if axis < self.ndim - 1:
            return insertaxis(diagonalize(self.func, axis, self.ndim - 1), self.ndim - 1, self.length)

    def _inflate(self, dofmap, length, axis):
        if axis + dofmap.ndim < self.ndim:
            return InsertAxis(_inflate(self.func, dofmap, length, axis), self.length)
        elif axis == self.ndim:
            return insertaxis(Inflate(self.func, dofmap, length), self.ndim - 1, self.length)

    def _insertaxis(self, axis, length):
        if axis == self.ndim - 1:
            return InsertAxis(InsertAxis(self.func, length), self.length)

    def _take(self, index, axis):
        if axis == self.ndim - 1:
            return appendaxes(self.func, index.shape)
        return InsertAxis(_take(self.func, index, axis), self.length)

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        if axis2 == self.ndim-1:
            return Transpose.to_end(self.func, axis1)
        else:
            return insertaxis(_takediag(self.func, axis1, axis2), self.ndim-3, self.length)

    def _unravel(self, axis, shape):
        if axis == self.ndim - 1:
            return InsertAxis(InsertAxis(self.func, shape[0]), shape[1])
        else:
            return InsertAxis(unravel(self.func, axis, shape), self.length)

    def _sign(self):
        return InsertAxis(Sign(self.func), self.length)

    def _determinant(self, axis1, axis2):
        if axis1 < self.ndim-1 and axis2 < self.ndim-1:
            return InsertAxis(determinant(self.func, (axis1, axis2)), self.length)

    def _inverse(self, axis1, axis2):
        if axis1 < self.ndim-1 and axis2 < self.ndim-1:
            return InsertAxis(inverse(self.func, (axis1, axis2)), self.length)
        # either axis1 or axis2 is inserted
        if self.length._intbounds[0] > 1: # matrix is at least 2x2
            return singular_like(self)

    def _loopsum(self, index):
        return InsertAxis(loop_sum(self.func, index), self.length)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return tuple((*(InsertAxis(idx, self.length) for idx in indices), prependaxes(Range(self.length), values.shape), InsertAxis(values, self.length)) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds

    @property
    def _const_uniform(self):
        return self.func._const_uniform

    def _argument_degree(self, argument):
        if argument not in self.length.arguments:
            return self.func.argument_degree(argument)


class Transpose(Array):

    func: Array
    axes: typing.Tuple[int, ...]

    def _end(array, *axes, post):
        ndim = array.ndim
        axes = [numeric.normdim(ndim, axis) for axis in axes]
        if all(a == b for a, b in enumerate(axes, start=ndim-len(axes))):
            return array
        trans = [i for i in range(ndim) if i not in axes]
        trans.extend(axes)
        return Transpose(array, post(trans))

    to_end = functools.partial(_end, post=tuple)
    from_end = functools.partial(_end, post=util.untake)

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func!r}'
        assert isinstance(self.axes, tuple) and all(isinstance(axis, int) for axis in self.axes), f'axes={self.axes!r}'
        assert sorted(self.axes) == list(range(self.func.ndim))
        assert self.axes != tuple(range(self.func.ndim))

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return tuple(self.func.shape[n] for n in self.axes)

    @cached_property
    def _diagonals(self):
        return tuple(frozenset(self._invaxes[i] for i in axes) for axes in self.func._diagonals)

    @cached_property
    def _inflations(self):
        return tuple((self._invaxes[axis], types.frozendict((dofmap, transpose(func, self._axes_for(dofmap.ndim, self._invaxes[axis]))) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

    @cached_property
    def _unaligned(self):
        unaligned, where = unalign(self.func)
        return unaligned, tuple(self._invaxes[i] for i in where)

    @cached_property
    def _invaxes(self):
        return util.untake(self.axes)

    def _simplified(self):
        return self.func._transpose(self.axes)

    def _compile_expression(self, py_self, func):
        axes = _pyast.Tuple(tuple(map(_pyast.LiteralInt, self.axes)))
        return _pyast.Variable('numpy').get_attr('transpose').call(func, axes)

    def _compile_with_out(self, builder, out, out_block_id, mode):
        invaxes = _pyast.Tuple(tuple(map(_pyast.LiteralInt, self._invaxes)))
        inv_trans_out = _pyast.Variable('numpy').get_attr('transpose').call(out, invaxes)
        builder.compile_with_out(self.func, inv_trans_out, out_block_id, mode)

    @property
    def _node_details(self):
        return ','.join(map(str, self.axes))

    def _transpose(self, axes):
        if axes == self._invaxes:
            # NOTE: While we could leave this particular simplification to be dealt
            # with by Transpose, the benefit of handling it directly is that _add and
            # _multiply can rely on _transpose for the right hand side without having
            # to separately account for the trivial case.
            return self.func
        newaxes = tuple(self.axes[i] for i in axes)
        return transpose(self.func, newaxes)

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        orig1, orig2 = sorted(self.axes[axis] for axis in [axis1, axis2])
        trytakediag = self.func._takediag(orig1, orig2)
        if trytakediag is not None:
            exclude_orig = [ax-(ax > orig1)-(ax > orig2) for ax in self.axes[:axis1] + self.axes[axis1+1:axis2] + self.axes[axis2+1:]]
            return transpose(trytakediag, (*exclude_orig, self.ndim-2))

    def _sum(self, i):
        axis = self.axes[i]
        trysum = self.func._sum(axis)
        if trysum is not None:
            axes = tuple(ax-(ax > axis) for ax in self.axes if ax != axis)
            return transpose(trysum, axes)

    def _derivative(self, var, seen):
        return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

    def _multiply(self, other):
        other_trans = other._transpose(self._invaxes)
        if other_trans is not None and not isinstance(other_trans, Transpose):
            # The second clause is to avoid infinite recursions; see
            # tests.test_evaluable.simplify.test_multiply_transpose.
            return Transpose(multiply(self.func, other_trans), self.axes)
        trymultiply = self.func._multiply(Transpose(other, self._invaxes))
        if trymultiply is not None:
            return Transpose(trymultiply, self.axes)

    def _add(self, other):
        other_trans = other._transpose(self._invaxes)
        if other_trans is not None and not isinstance(other_trans, Transpose):
            # The second clause is to avoid infinite recursions
            return Transpose(self.func + other_trans, self.axes)
        tryadd = self.func._add(Transpose(other, self._invaxes))
        if tryadd is not None:
            return Transpose(tryadd, self.axes)

    def _take(self, indices, axis):
        trytake = self.func._take(indices, self.axes[axis])
        if trytake is not None:
            return transpose(trytake, self._axes_for(indices.ndim, axis))

    def _axes_for(self, ndim, axis):
        funcaxis = self.axes[axis]
        axes = [ax+(ax > funcaxis)*(ndim-1) for ax in self.axes if ax != funcaxis]
        axes[axis:axis] = range(funcaxis, funcaxis + ndim)
        return tuple(axes)

    def _power(self, n):
        n_trans = Transpose(n, self._invaxes)
        return Transpose(Power(self.func, n_trans), self.axes)

    def _sign(self):
        return Transpose(Sign(self.func), self.axes)

    def _unravel(self, axis, shape):
        orig_axis = self.axes[axis]
        tryunravel = self.func._unravel(orig_axis, shape)
        if tryunravel is not None:
            axes = [ax + (ax > orig_axis) for ax in self.axes]
            axes.insert(axis+1, orig_axis+1)
            return transpose(tryunravel, tuple(axes))

    def _product(self):
        if self.axes[-1] == self.ndim-1:
            return Transpose(Product(self.func), self.axes[:-1])

    def _determinant(self, axis1, axis2):
        orig1, orig2 = self.axes[axis1], self.axes[axis2]
        trydet = self.func._determinant(orig1, orig2)
        if trydet:
            axes = tuple(ax-(ax > orig1)-(ax > orig2) for ax in self.axes if ax != orig1 and ax != orig2)
            return transpose(trydet, axes)

    def _inverse(self, axis1, axis2):
        tryinv = self.func._inverse(self.axes[axis1], self.axes[axis2])
        if tryinv:
            return Transpose(tryinv, self.axes)

    def _ravel(self, axis):
        if self.axes[axis] == self.ndim-2 and self.axes[axis+1] == self.ndim-1:
            return Transpose(Ravel(self.func), self.axes[:-1])

    def _inflate(self, dofmap, length, axis):
        i = self.axes[axis] if dofmap.ndim else self.func.ndim
        if self.axes[axis:axis+dofmap.ndim] == tuple(range(i, i+dofmap.ndim)):
            tryinflate = self.func._inflate(dofmap, length, i)
            if tryinflate is not None:
                axes = [ax-(ax > i)*(dofmap.ndim-1) for ax in self.axes]
                axes[axis:axis+dofmap.ndim] = i,
                return transpose(tryinflate, tuple(axes))

    def _diagonalize(self, axis):
        trydiagonalize = self.func._diagonalize(self.axes[axis])
        if trydiagonalize is not None:
            return Transpose(trydiagonalize, self.axes + (self.ndim,))

    def _insertaxis(self, axis, length):
        return Transpose(InsertAxis(self.func, length), self.axes[:axis] + (self.ndim,) + self.axes[axis:])

    def _loopsum(self, index):
        return Transpose(loop_sum(self.func, index), self.axes)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return tuple((*(indices[i] for i in self.axes), values) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds

    @property
    def _const_uniform(self):
        return self.func._const_uniform

    def _optimized_for_numpy(self):
        if isinstance(self.func, Transpose):
            return transpose(self.func.func, [self.func.axes[i] for i in self.axes])
        if isinstance(self.func, Assemble):
            offsets = numpy.cumsum([0] + [index.ndim for index in self.func.indices])
            axes = [n for axis in self.axes for n in range(offsets[axis], offsets[axis+1])]
            return Assemble(transpose(self.func.func, axes), tuple(self.func.indices[i] for i in self.axes), self.shape)

    def _argument_degree(self, argument):
        return self.func.argument_degree(argument)


class Product(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-1]

    def _compile_expression(self, py_self, func):
        return _pyast.Variable('numpy').get_attr('all' if self.dtype == bool else 'prod').call(func, axis=_pyast.LiteralInt(-1))

    def _simplified(self):
        if isunit(self.func.shape[-1]):
            return get(self.func, self.ndim, constant(0))
        return self.func._product()

    def _derivative(self, var, seen):
        grad = derivative(self.func, var, seen)
        funcs = Product(insertaxis(self.func, -2, self.func.shape[-1]) + Diagonalize(astype(1, self.func.dtype) - self.func))  # replace diagonal entries by 1
        return einsum('Ai,AiB->AB', funcs, grad)

    def _take(self, indices, axis):
        return Product(_take(self.func, indices, axis))

    def _takediag(self, axis1, axis2):
        return product(_takediag(self.func, axis1, axis2), self.ndim-2)


class Inverse(Array):
    '''
    Matrix inverse of ``func`` over the last two axes.  All other axes are
    treated element-wise.
    '''

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.dtype in (float, complex) and self.func.ndim >= 2 and not _certainly_different(self.func.shape[-1], self.func.shape[-2]), f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape

    def _simplified(self):
        if isunit(self.func.shape[-1]):
            return reciprocal(self.func)
        if iszero(self.func.shape[-1]):
            return singular_like(self)
        result = self.func._inverse(self.ndim-2, self.ndim-1)
        if result is not None:
            return result

    def _compile_expression(self, py_self, mat):
        return _pyast.Variable('numeric').get_attr('inv').call(mat)

    def _derivative(self, var, seen):
        return -einsum('Aij,AjkB,Akl->AilB', self, derivative(self.func, var, seen), self)

    def _eig(self, symmetric):
        eigval, eigvec = Eig(self.func, symmetric)
        return Tuple((reciprocal(eigval), eigvec))

    def _determinant(self, axis1, axis2):
        if sorted([axis1, axis2]) == [self.ndim-2, self.ndim-1]:
            return reciprocal(Determinant(self.func))

    def _take(self, indices, axis):
        if axis < self.ndim - 2:
            return Inverse(_take(self.func, indices, axis))

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        if axis2 < self.ndim-2:
            return inverse(_takediag(self.func, axis1, axis2), (self.ndim-4, self.ndim-3))

    def _unravel(self, axis, shape):
        if axis < self.ndim-2:
            return Inverse(unravel(self.func, axis, shape))


class Determinant(Array):

    func: Array

    def __post_init__(self):
        assert isarray(self.func) and self.func.dtype in (float, complex) and self.func.ndim >= 2 and not _certainly_different(self.func.shape[-1], self.func.shape[-2])

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-2]

    def _simplified(self):
        result = self.func._determinant(self.ndim, self.ndim+1)
        if result is not None:
            return result
        if isunit(self.func.shape[-1]):
            return Take(Take(self.func, zeros((), int)), zeros((), int))

    def _compile_expression(self, py_self, array):
        return _pyast.Variable('numpy').get_attr('linalg').get_attr('det').call(array)

    def _derivative(self, var, seen):
        return einsum('A,Aji,AijB->AB', self, inverse(self.func), derivative(self.func, var, seen))

    def _take(self, index, axis):
        return Determinant(_take(self.func, index, axis))

    def _takediag(self, axis1, axis2):
        return determinant(_takediag(self.func, axis1, axis2), (self.ndim-2, self.ndim-1))


class Multiply(Array):

    funcs: types.frozenmultiset

    def __post_init__(self):
        assert isinstance(self.funcs, types.frozenmultiset), f'funcs={self.funcs!r}'
        func1, func2 = self.funcs
        assert not _any_certainly_different(func1.shape, func2.shape) and func1.dtype == func2.dtype, 'Multiply({}, {})'.format(func1, func2)

    @property
    def dependencies(self):
        return tuple(self.funcs)

    @cached_property
    def dtype(self):
        func1, func2 = self.funcs
        assert func1.dtype == func2.dtype
        return func1.dtype

    @cached_property
    def shape(self):
        func1, func2 = self.funcs
        return assert_equal_tuple(func1.shape, func2.shape)

    @property
    def _factors(self):
        for func in self.funcs:
            if isinstance(func, Multiply):
                yield from func._factors
            else:
                yield func

    def _simplified(self):
        factors = tuple(self._factors)
        for j, fj in enumerate(factors):
            if fj._const_uniform == 1:
                return multiply(*factors[:j], *factors[j+1:])
            for i, parts in fj._inflations:
                return util.sum(_inflate(multiply(f, *(_take(fi, dofmap, i) for fi in factors[:j] + factors[j+1:])), dofmap, self.shape[i], i) for dofmap, f in parts.items())
            for axis1, axis2, *other in map(sorted, fj._diagonals):
                return diagonalize(multiply(*(takediag(f, axis1, axis2) for f in factors)), axis1, axis2)
            for i, fi in enumerate(factors[:j]):
                if self.dtype == bool and fi == fj:
                    return multiply(*factors[:j], *factors[j+1:])
                unaligned1, unaligned2, where = unalign(fi, fj)
                fij = align(unaligned1 * unaligned2, where, self.shape) if len(where) != self.ndim \
                    else fi._multiply(fj) or fj._multiply(fi)
                if fij:
                    return multiply(*factors[:i], *factors[i+1:j], *factors[j+1:], fij)

    def _optimized_for_numpy(self):
        if self.dtype == bool:
            return None
        factors = tuple(self._factors)
        for i, fi in enumerate(factors):
            if fi._const_uniform == -1:
                return Negative(multiply(*factors[:i], *factors[i+1:]))
            if fi.dtype != complex and Sign(fi) in factors:
                i, j = sorted([i, factors.index(Sign(fi))])
                return multiply(*factors[:i], *factors[i+1:j], *factors[j+1:], Absolute(fi))
        if self.ndim:
            r = tuple(range(self.ndim))
            return Einsum(tuple(self.funcs), (r, r), r)

    def _compile_expression(self, py_self, func1, func2):
        return _pyast.BinOp(func1, '*', func2)

    def _sum(self, axis):
        factors = tuple(self._factors)
        for i, fi in enumerate(factors):
            unaligned, where = unalign(fi)
            if axis not in where:
                summed = sum(multiply(*factors[:i], *factors[i+1:]), axis)
                return summed * align(unaligned, [i-(i > axis) for i in where], summed.shape)

    def _add(self, other):
        factors = list(self._factors)
        other_factors = []
        common = []
        for f in other._factors if isinstance(other, Multiply) else [other]:
            if f in factors:
                factors.remove(f)
                common.append(f)
            else:
                other_factors.append(f)
        if not common:
            return
        if factors and other_factors:
            return multiply(*common) * add(multiply(*factors), multiply(*other_factors))
        nz = factors or other_factors
        if not nz: # self equals other (up to factor ordering)
            return self if self.dtype == bool else self * astype(2, self.dtype)
        if self.dtype != bool and len(nz) == 1 and tuple(nz)[0]._const_uniform == -1:
            # Since the subtraction x - y is stored as x + -1 * y, this handles
            # the simplification of x - x to 0. While we could alternatively
            # simplify all x + a * x to (a + 1) * x, capturing a == -1 as a
            # special case via Constant._add, it is not obvious that this is in
            # all situations an improvement.
            return zeros_like(self)

    def _determinant(self, axis1, axis2):
        axis1, axis2 = sorted([axis1, axis2])
        factors = tuple(self._factors)
        if all(isunit(self.shape[axis]) for axis in (axis1, axis2)):
            return multiply(*[determinant(f, (axis1, axis2)) for f in factors])
        for i, fi in enumerate(factors):
            unaligned, where = unalign(fi)
            if axis1 not in where and axis2 not in where:
                det = determinant(multiply(*factors[:i], *factors[i+1:]), (axis1, axis2))
                scale = align(unaligned**astype(self.shape[axis1], unaligned.dtype), [i-(i > axis1)-(i > axis2) for i in where if i not in (axis1, axis2)], det.shape)
                return det * scale

    def _product(self):
        return multiply(*[Product(f) for f in self._factors])

    def _derivative(self, var, seen):
        func1, func2 = self.funcs
        return einsum('A,AB->AB', func1, derivative(func2, var, seen)) \
            + einsum('A,AB->AB', func2, derivative(func1, var, seen))

    def _takediag(self, axis1, axis2):
        return multiply(*[_takediag(f, axis1, axis2) for f in self._factors])

    def _take(self, index, axis):
        return multiply(*[_take(f, index, axis) for f in self._factors])

    def _sign(self):
        return multiply(*[Sign(f) for f in self._factors])

    def _unravel(self, axis, shape):
        return multiply(*[unravel(f, axis, shape) for f in self._factors])

    def _inverse(self, axis1, axis2):
        factors = tuple(self._factors)
        for i, fi in enumerate(factors):
            if set(unalign(fi)[1]).isdisjoint((axis1, axis2)):
                inv = inverse(multiply(*factors[:i], *factors[i+1:]), (axis1, axis2))
                return divide(inv, fi)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        if self.dtype == bool:
            return super()._assparse
        # First we collect the clusters of factors that have no real (i.e. not
        # inserted) axes in common with the other clusters, and store them in
        # uninserted form.
        clusters = []
        for f in self._factors:
            uninserted, where = unalign(f)
            for i in reversed(range(len(clusters))):
                if set(where) & set(clusters[i][1]):
                    w = where
                    unins_, w_ = clusters.pop(i)
                    where = numpy.union1d(w, w_)
                    shape = tuple(self.shape[i] for i in where)
                    uninserted = align(uninserted, numpy.searchsorted(where, w), shape) * align(unins_, numpy.searchsorted(where, w_), shape)
            clusters.append((uninserted, where))
        # If there is only one cluster we fall back on the default
        # implementation.
        if len(clusters) == 1:
            return super()._assparse
        # If there are two or more clusters we write the product of additions
        # as an addition of products.
        uninserteds, wheres = zip(*clusters)
        sparse = []
        for items in itertools.product(*[u._assparse for u in uninserteds]):
            shape = util.sum(f.shape for *ind, f in items)
            indices = [None] * self.ndim
            factors = []
            a = 0
            for where, (*ind, f) in zip(wheres, items):
                b = a + f.ndim
                r = numpy.arange(a, b)
                for i, indi in zip(where, ind):
                    indices[i] = align(indi, r, shape)
                factors.append(align(f, r, shape))
                a = b
            sparse.append((*indices, multiply(*factors)))
        return tuple(sparse)

    def _intbounds_impl(self):
        func1, func2 = self.funcs
        extrema = [b1 and b2 and b1 * b2 for b1 in func1._intbounds for b2 in func2._intbounds]
        return min(extrema), max(extrema)

    def _argument_degree(self, argument):
        func1, func2 = self.funcs
        return func1.argument_degree(argument) + func2.argument_degree(argument)


class Add(Array):

    funcs: types.frozenmultiset

    def __post_init__(self):
        assert isinstance(self.funcs, types.frozenmultiset) and len(self.funcs) == 2, f'funcs={self.funcs!r}'
        func1, func2 = self.funcs
        assert not _any_certainly_different(func1.shape, func2.shape) and func1.dtype == func2.dtype, 'Add({}, {})'.format(func1, func2)

    @property
    def dependencies(self):
        return tuple(self.funcs)

    @cached_property
    def dtype(self):
        func1, func2 = self.funcs
        assert func1.dtype == func2.dtype
        return func1.dtype

    @cached_property
    def shape(self):
        func1, func2 = self.funcs
        return assert_equal_tuple(func1.shape, func2.shape)

    @cached_property
    def _inflations(self):
        if self.dtype == bool:
            return ()
        func1, func2 = self.funcs
        func2_inflations = dict(func2._inflations)
        inflations = []
        for axis, parts1 in func1._inflations:
            if axis not in func2_inflations:
                continue
            parts2 = func2_inflations[axis]
            jointparts = parts1 | parts2
            if (len(parts1) < len(jointparts) and len(parts2) < len(jointparts)  # neither set is a subset of the other; total may be dense
                    and isinstance(self.shape[axis], Constant) and all(isinstance(dofmap, Constant) for dofmap in jointparts)):
                mask = numpy.zeros(self.shape[axis].value, dtype=bool)
                for dofmap in jointparts:
                    mask[dofmap.value] = True
                if mask.all():  # axis adds up to dense
                    continue
            # fix overlap by concatenating values for common keys
            jointparts.update((dofmap, part1 + part2) for dofmap, part1 in parts1.items() if (part2 := parts2.get(dofmap)) is not None)
            inflations.append((axis, types.frozendict(jointparts)))
        return tuple(inflations)

    @property
    def _terms(self):
        for func in self.funcs:
            if isinstance(func, Add):
                yield from func._terms
            else:
                yield func

    def _simplified(self):
        terms = tuple(self._terms)
        for j, fj in enumerate(terms):
            for i, fi in enumerate(terms[:j]):
                if self.dtype == bool and fi == fj:
                    return add(*terms[:j], *terms[j+1:])
                diags = [sorted(axesi & axesj)[:2] for axesi in fi._diagonals for axesj in fj._diagonals if len(axesi & axesj) >= 2]
                unaligned1, unaligned2, where = unalign(fi, fj)
                fij = diagonalize(takediag(fi, *diags[0]) + takediag(fj, *diags[0]), *diags[0]) if diags \
                    else align(unaligned1 + unaligned2, where, self.shape) if len(where) != self.ndim \
                    else fi._add(fj) or fj._add(fi)
                if fij:
                    return add(*terms[:i], *terms[i+1:j], *terms[j+1:], fij)
        # NOTE: While it is tempting to use the _inflations attribute to push
        # additions through common inflations, doing so may result in infinite
        # recursion in case two or more axes are inflated. This mechanism is
        # illustrated in the following schematic, in which <I> and <J> represent
        # inflations along axis 1 and <K> and <L> inflations along axis 2:
        #
        #        A   B   C   D   E   F   G   H
        #       <I> <J> <I> <J> <I> <J> <I> <J>
        #  .--    \+/     \+/     \+/     \+/   <--.
        #  |       \__<K>__/       \__<L>__/       |
        #  |           \_______+_______/           |
        #  |                                       |
        #  |     A   E   C   G   B   F   D   H     |
        #  |    <K> <L> <K> <L> <K> <L> <K> <L>    |
        #  '-->   \+/     \+/     \+/     \+/    --'
        #          \__<I>__/       \__<J>__/
        #              \_______+_______/
        #
        # We instead rely on Inflate._add to handle this situation.

    def _compile(self, builder):
        if any(builder.ndependents[func] == 1 and type(func)._compile_with_out != Array._compile_with_out for func in self.funcs):
            out, out_block_id = builder.new_empty_array_for_evaluable(self)
            self._compile_with_out(builder, out, out_block_id, 'assign')
            return out
        else:
            return super()._compile(builder)

    def _compile_expression(self, py_self, func1, func2):
        return _pyast.BinOp(func1, '+', func2)

    def _compile_with_out(self, builder, out, out_block_id, mode):
        assert mode in ('iadd', 'assign')
        if mode == 'assign':
            builder.get_block_for_evaluable(self, block_id=out_block_id, comment='zero').array_fill_zeros(out)
        for func in self.funcs:
            builder.compile_with_out(func, out, out_block_id, 'iadd')

    def _sum(self, axis):
        return add(*[sum(f, axis) for f in self._terms])

    def _derivative(self, var, seen):
        return add(*[derivative(f, var, seen) for f in self._terms])

    def _takediag(self, axis1, axis2):
        return add(*[_takediag(f, axis1, axis2) for f in self._terms])

    def _take(self, index, axis):
        return add(*[_take(f, index, axis) for f in self._terms])

    def _unravel(self, axis, shape):
        return add(*[unravel(f, axis, shape) for f in self._terms])

    def _loopsum(self, index):
        dep = []
        indep = []
        for f in self._terms:
            (dep if index in f.arguments else indep).append(f)
        if indep:
            return add(*indep) * astype(index.length, self.dtype) + loop_sum(add(*dep), index)

    def _multiply(self, other):
        f_other = [f._multiply(other) or other._multiply(f) or (f._inflations or f._diagonals) and f * other for f in self._terms]
        if all(f_other):
            # NOTE: As this operation is the precise opposite of Multiply._add, there
            # appears to be a great risk of recursion. However, since both factors
            # are sparse, we can be certain that subsequent simpifications will
            # irreversibly process the new terms before reaching this point.
            return add(*f_other)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        if self.dtype == bool:
            return super()._assparse
        else:
            return _gathersparsechunks(itertools.chain(*[f._assparse for f in self._terms]))

    def _intbounds_impl(self):
        lowers, uppers = zip(*[f._intbounds for f in self._terms])
        return builtins.sum(lowers), builtins.sum(uppers)

    def _argument_degree(self, argument):
        func1, func2 = self.funcs
        return max(func1.argument_degree(argument), func2.argument_degree(argument))


class Einsum(Array):

    args: typing.Tuple[Array, ...]
    args_idx: typing.Tuple[typing.Tuple[int, ...], ...]
    out_idx: typing.Tuple[int, ...]

    def __post_init__(self):
        assert isinstance(self.args, tuple) and all(isinstance(arg, Array) for arg in self.args), f'arg={arg!r}'
        assert isinstance(self.args_idx, tuple) and all(isinstance(arg_idx, tuple) and all(isinstance(n, int) for n in arg_idx) for arg_idx in self.args_idx), f'args_idx={self.args_idx!r}'
        assert isinstance(self.out_idx, tuple) and all(isinstance(n, int) for n in self.out_idx) and len(self.out_idx) == len(set(self.out_idx)), f'out_idx={self.out_idx!r}'
        assert len(self.args_idx) == len(self.args) and all(len(idx) == arg.ndim for idx, arg in zip(self.args_idx, self.args)), f'len(args_idx)={len(self.args_idx)}, len(args)={len(self.args)}'
        dtype = self.args[0].dtype
        if dtype == bool or any(arg.dtype != dtype for arg in self.args[1:]):
            raise ValueError('Inconsistent or invalid dtypes.')
        lengths = {}
        for idx, arg in zip(self.args_idx, self.args):
            for i, length in zip(idx, arg.shape):
                n = lengths.get(i)
                lengths[i] = length if n is None else assert_equal(length, n)
        try:
            self.shape = tuple(lengths[i] for i in self.out_idx)
        except KeyError(e):
            raise ValueError(f'Output axis {e} is not listed in any of the arguments.')

    @cached_property
    def _einsumfmt(self):
        return ','.join(''.join(chr(97+i) for i in idx) for idx in self.args_idx) + '->' + ''.join(chr(97+i) for i in self.out_idx)

    @property
    def dependencies(self):
        return self.args

    @cached_property
    def dtype(self):
        return self.args[0].dtype

    def _compile_expression(self, py_self, *args):
        return _pyast.Variable('numpy').get_attr('einsum').call(_pyast.LiteralStr(self._einsumfmt), *args)

    @property
    def _node_details(self):
        return self._einsumfmt

    def _optimized_for_numpy(self):
        for i, arg in enumerate(self.args):
            if isinstance(arg, Transpose):  # absorb `Transpose`
                idx = util.untake(arg.axes, self.args_idx[i])
            elif isinstance(arg, InsertAxis) and any(self.args_idx[i][-1] in arg_idx for arg_idx in self.args_idx[:i] + self.args_idx[i+1:]):
                idx = self.args_idx[i][:-1]
            else:
                continue
            return Einsum(self.args[:i]+(arg.func,)+self.args[i+1:], self.args_idx[:i]+(idx,)+self.args_idx[i+1:], self.out_idx)


class Sum(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func!r}'

    @cached_property
    def evalf(self):
        return functools.partial(numpy.any if self.func.dtype == bool else numpy.sum, axis=-1)

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-1]

    def _compile_expression(self, py_self, func):
        return _pyast.Variable('numpy').get_attr('any' if self.dtype == bool else 'sum').call(func, axis=_pyast.LiteralInt(-1))

    def _simplified(self):
        if isunit(self.func.shape[-1]):
            return Take(self.func, constant(0))
        return self.func._sum(self.ndim)

    def _optimized_for_numpy(self):
        func = self.func
        axes = list(range(func.ndim))
        while isinstance(func, Transpose):
            axes = [func.axes[i] for i in axes]
            func = func.func
        if isinstance(func, Einsum):
            rmaxis = axes[-1]
            axes = [i-(i>rmaxis) for i in axes[:-1]]
            return transpose(Einsum(func.args, func.args_idx, func.out_idx[:rmaxis] + func.out_idx[rmaxis+1:]), axes)

    def _sum(self, axis):
        trysum = self.func._sum(axis)
        if trysum is not None:
            return Sum(trysum)

    def _derivative(self, var, seen):
        return sum(derivative(self.func, var, seen), self.ndim)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        if self.dtype == bool:
            return super()._assparse
        chunks = []
        for *indices, _rmidx, values in self.func._assparse:
            if self.ndim == 0:
                nsum = values.ndim
            else:
                *indices, where = unalign(*indices)
                values = transpose(values, where + tuple(i for i in range(values.ndim) if i not in where))
                nsum = values.ndim - len(where)
            for i in range(nsum):
                values = Sum(values)
            chunks.append((*indices, values))
        return _gathersparsechunks(chunks)

    def _intbounds_impl(self):
        lower_func, upper_func = self.func._intbounds
        lower_length, upper_length = self.func.shape[-1]._intbounds
        if upper_length == 0:
            return 0, 0
        elif lower_length == 0:
            return min(0, lower_func * upper_length), max(0, upper_func * upper_length)
        else:
            return min(lower_func * lower_length, lower_func * upper_length), max(upper_func * lower_length, upper_func * upper_length)

    def _take(self, index, axis):
        return Sum(_take(self.func, index, axis))

    def _takediag(self, axis1, axis2):
        return sum(_takediag(self.func, axis1, axis2), -2)

    def _argument_degree(self, argument):
        return self.func.argument_degree(argument)


class TakeDiag(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim >= 2 and not _certainly_different(*self.func.shape[-2:]), f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-1]

    def _simplified(self):
        if isunit(self.shape[-1]):
            return Take(self.func, constant(0))
        return self.func._takediag(self.ndim-1, self.ndim)

    def _optimized_for_numpy(self):
        func = self.func
        axes = list(range(func.ndim))
        while isinstance(func, Transpose):
            axes = [func.axes[i] for i in axes]
            func = func.func
        if isinstance(func, Einsum):
            axis, rmaxis = axes[-2:]
            args_idx = tuple(tuple(func.out_idx[axis] if i == func.out_idx[rmaxis] else i for i in idx) for idx in func.args_idx)
            axes = [i-(i>rmaxis) for i in axes[:-1]]
            return transpose(Einsum(func.args, args_idx, func.out_idx[:rmaxis] + func.out_idx[rmaxis+1:]), axes)

    def _compile_expression(self, py_self, arr):
        return _pyast.Variable('numpy').get_attr('einsum').call(_pyast.LiteralStr('...kk->...k'), arr)

    def _derivative(self, var, seen):
        return takediag(derivative(self.func, var, seen), self.ndim-1, self.ndim)

    def _take(self, index, axis):
        if axis < self.ndim - 1 and (simple := self.func._take(index, axis)):
            return TakeDiag(simple)

    def _takediag(self, axis1, axis2):
        if axis1 < self.ndim-1 and axis2 < self.ndim-1 and (simple := self.func._takediag(axis1, axis2)):
            return takediag(simple, -3, -2)

    def _sum(self, axis):
        if axis != self.ndim - 1 and (simple := self.func._sum(axis)):
            return TakeDiag(simple)

    def _intbounds_impl(self):
        return self.func._intbounds


class Take(Array):

    func: Array
    indices: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim > 0, f'func={self.func!r}'
        assert isinstance(self.indices, Array) and self.indices.dtype == int, f'indices={self.indices!r}'

    @property
    def dependencies(self):
        return self.func, self.indices

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-1] + self.indices.shape

    def _simplified(self):
        if any(iszero(n) for n in self.indices.shape):
            return zeros_like(self)
        unaligned, where = unalign(self.indices)
        if len(where) < self.indices.ndim:
            n = self.func.ndim-1
            return align(Take(self.func, unaligned), (*range(n), *(n+i for i in where)), self.shape)
        trytake = self.func._take(self.indices, self.func.ndim-1) or \
            self.indices._rtake(self.func, self.func.ndim-1)
        if trytake:
            return trytake
        for axis, parts in self.func._inflations:
            if axis == self.func.ndim - 1:
                return util.sum(Inflate(func, dofmap, self.func.shape[-1])._take(self.indices, self.func.ndim - 1) for dofmap, func in parts.items())

    def _optimized_for_numpy(self):
        if isinstance(self.indices, Range):
            return _TakeSlice(self.func, self.indices.length, constant(0))
        if self.indices.ndim == 1 and isinstance(self.indices, Add) and len(self.indices.funcs) == 2:
            for a, b in self.indices.funcs, tuple(self.indices.funcs)[::-1]:
                if isinstance(a, Range) and isinstance(b, InsertAxis) and _isindex(b.func):
                    return _TakeSlice(self.func, a.length, b.func)

    def _compile_expression(self, py_self, arr, indices):
        return _pyast.Variable('numpy').get_attr('take').call(arr, indices, axis=_pyast.LiteralInt(-1))

    def _derivative(self, var, seen):
        return _take(derivative(self.func, var, seen), self.indices, self.func.ndim-1)

    def _take(self, index, axis):
        if axis >= self.func.ndim-1:
            return Take(self.func, _take(self.indices, index, axis-self.func.ndim+1))
        trytake = self.func._take(index, axis)
        if trytake is not None:
            return Take(trytake, self.indices)

    def _takediag(self, axis1, axis2):
        if axis1 < self.func.ndim-1 and axis2 < self.func.ndim-1:
            return _take(_takediag(self.func, axis1, axis2), self.indices, self.func.ndim-3)

    def _sum(self, axis):
        if axis < self.func.ndim - 1 and (simple := self.func._sum(axis)):
            return Take(simple, self.indices)

    def _intbounds_impl(self):
        return self.func._intbounds

    def _argument_degree(self, argument):
        if argument not in self.indices.arguments:
            return self.func.argument_degree(argument)


class _TakeSlice(Array):
    # To be used by `_optimized_for_numpy` only.

    func: Array
    length: Array
    offset: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim > 0, f'func={self.func!r}'
        assert _isindex(self.length), f'length={self.length!r}'
        assert _isindex(self.offset), f'offset={self.offset!r}'

    @property
    def dependencies(self):
        return self.func, self.length, self.offset

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape[:-1] + (self.length,)

    def _compile_expression(self, py_self, arr, length, offset):
        slices = [_pyast.Variable('slice').call(_pyast.Variable('None'))] * (self.ndim - 1)
        slices.append(_pyast.Variable('slice').call(offset, _pyast.BinOp(offset, '+', length)))
        return arr.get_item(_pyast.Tuple(tuple(slices)))

    def _intbounds_impl(self):
        return self.func._intbounds


class Power(Array):

    func: Array
    power: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.dtype != bool, f'func={self.func!r}'
        assert isinstance(self.power, Array) and (self.power.dtype in (float, complex) or self.power.dtype == int and self.power._intbounds[0] >= 0), f'power={self.power!r}'
        assert not _any_certainly_different(self.func.shape, self.power.shape) and self.func.dtype == self.power.dtype, 'Power({}, {})'.format(self.func, self.power)

    @property
    def dependencies(self):
        return self.func, self.power

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape

    def _simplified(self):
        if iszero(self.power):
            return ones_like(self)
        p = self.power._const_uniform
        if p == 1:
            return self.func
        elif p == 2:
            return self.func * self.func
        else:
            return self.func._power(self.power)

    def _optimized_for_numpy(self):
        p = self.power._const_uniform
        if p == -1:
            return Reciprocal(self.func)
        elif p == -2:
            return Reciprocal(self.func * self.func)

    def _compile_expression(self, py_self, func, power):
        return _pyast.Variable('numpy').get_attr('power').call(func, power)

    def _derivative(self, var, seen):
        if isinstance(self.power, Constant):
            p = self.power.value.copy()
            p -= p != 0 # exclude zero powers from decrement to avoid potential division by zero errors
            return einsum('A,A,AB->AB', self.power, power(self.func, p), derivative(self.func, var, seen))
        # self = func**power
        # ln self = power * ln func
        # self` / self = power` * ln func + power * func` / func
        # self` = power` * ln func * self + power * func` * func**(power-1)
        return einsum('A,A,AB->AB', self.power, power(self.func, self.power - astype(1, self.power.dtype)), derivative(self.func, var, seen)) \
            + einsum('A,A,AB->AB', ln(self.func), self, derivative(self.power, var, seen))

    def _power(self, n):
        if self.dtype == complex or n.dtype == complex:
            return
        func = self.func
        newpower = multiply(self.power, n)
        if iszero(self.power % astype(2, self.power.dtype)) and not iszero(newpower % astype(2, newpower.dtype)):
            func = abs(func)
        return Power(func, newpower)

    def _takediag(self, axis1, axis2):
        return Power(_takediag(self.func, axis1, axis2), _takediag(self.power, axis1, axis2))

    def _take(self, index, axis):
        return Power(_take(self.func, index, axis), _take(self.power, index, axis))

    def _unravel(self, axis, shape):
        return Power(unravel(self.func, axis, shape), unravel(self.power, axis, shape))

    def _argument_degree(self, argument):
        power, _ = unalign(self.power.simplified)
        while isinstance(power, Cast):
            power = power.arg
        if argument not in self.power.arguments and not power.ndim and isinstance(power, Constant) and power.value >= 0 and int(power.value) == power.value:
            return self.func.argument_degree(argument) * int(power.value)


class Pointwise(Array):
    '''
    Abstract base class for pointwise array functions.
    '''

    deriv = None
    parameters = ()
    dependencies = util.abstract_property()

    def __post_init__(self):
        self.shape = self.dependencies[0].shape
        for dep in self.dependencies[1:]:
            self.shape = assert_equal_tuple(self.shape, dep.shape)

    def _newargs(self, *args):
        '''
        Reinstantiate self with different arguments. Parameters are preserved,
        as these are considered part of the type.
        '''

        return self.__class__(*args, *self.parameters)

    @classmethod
    def outer(cls, *args):
        '''Alternative constructor that outer-aligns the arguments.

        The output shape of this pointwise function is the sum of all shapes of its
        arguments. When called with multiple arguments, the first argument will be
        appended with singleton axes to match the output shape, the second argument
        will be prepended with as many singleton axes as the dimension of the
        original first argument and appended to match the output shape, and so
        forth and so on.
        '''

        args = tuple(map(asarray, args))
        shape = builtins.sum((arg.shape for arg in args), ())
        offsets = numpy.cumsum([0]+[arg.ndim for arg in args])
        return cls(*(prependaxes(appendaxes(arg, shape[r:]), shape[:l]) for arg, l, r in zip(args, offsets[:-1], offsets[1:])))

    def _simplified(self):
        if len(self.dependencies) == 1 and isinstance(self.dependencies[0], Transpose):
            arg, = self.dependencies
            return Transpose(self._newargs(arg.func), arg.axes)
        *uninserted, where = unalign(*self.dependencies)
        if len(where) != self.ndim:
            return align(self._newargs(*uninserted), where, self.shape)

    def _derivative(self, var, seen):
        if self.dtype == complex or var.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        elif self.deriv is not None:
            return util.sum(einsum('A,AB->AB', deriv(*self.dependencies, *self.parameters), derivative(arg, var, seen)) for arg, deriv in zip(self.dependencies, self.deriv))
        else:
            return super()._derivative(var, seen)

    def _takediag(self, axis1, axis2):
        return self._newargs(*[_takediag(arg, axis1, axis2) for arg in self.dependencies])

    def _take(self, index, axis):
        return self._newargs(*[_take(arg, index, axis) for arg in self.dependencies])

    def _unravel(self, axis, shape):
        return self._newargs(*[unravel(arg, axis, shape) for arg in self.dependencies])


class Holomorphic(Pointwise):
    '''
    Abstract base class for holomorphic array functions.
    '''

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    @cached_property
    def dtype(self):
        if self.arg.dtype not in (float, complex):
            raise ValueError(f'{self.__class__.__name__} is not defined for arguments of dtype {self.arg.dtype}')
        return self.arg.dtype

    def _derivative(self, var, seen):
        if self.deriv is not None:
            return util.sum(einsum('A,AB->AB', deriv(*self.dependencies, *self.parameters), derivative(arg, var, seen)) for arg, deriv in zip(self.dependencies, self.deriv))
        else:
            return super()._derivative(var, seen)


class Reciprocal(Holomorphic):

    def _compile_expression(self, py_self, value):
        return _pyast.Variable('numpy').get_attr('reciprocal').call(value)


class Negative(Holomorphic):

    def _compile_expression(self, py_self, value):
        return _pyast.UnaryOp('-', value)

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T == bool:
            raise ValueError('boolean values cannot be negated')
        return T

    def _intbounds_impl(self):
        lower, upper = self.arg._intbounds
        return -upper, -lower


class FloorDivide(Pointwise):

    dividend: Array
    divisor: Array

    @property
    def dependencies(self):
        return self.dividend, self.divisor

    def _compile_expression(self, py_self, dividend, divisor):
        return _pyast.BinOp(dividend, '//', divisor)

    @cached_property
    def dtype(self):
        dtype = self.dividend.dtype
        if self.divisor.dtype != dtype:
            raise ValueError(f'All arguments must have the same dtype but got {dividend} and {divisor}.')
        if dtype == bool:
            raise ValueError(f'The boolean floor division is not supported.')
        return dtype

    def _intbounds_impl(self):
        lower, upper = self.dividend._intbounds
        divisor_lower, divisor_upper = self.divisor._intbounds
        if divisor_upper < 0:
            divisor_lower, divisor_upper = -divisor_upper, -divisor_lower
            lower, upper = -upper, -lower
        elif divisor_lower <= 0:
            # The divisor range includes zero.
            return float('-inf'), float('inf')
        # `divisor_lower` is always finite and positive. `divisor_upper` may be
        # `float('inf')` in which case the floordiv of a finite `lower` or
        # `upper` with `divisor_upper` gives a float `0.0` or `-1.0`. To
        # prevent the float, we bound `divisor_upper` by the dividend.
        if isinstance(lower, int):
            lower //= divisor_lower if lower <= 0 else min(lower + 1, divisor_upper)
        if isinstance(upper, int):
            upper //= divisor_lower if upper >= 0 else min(1 - upper, divisor_upper)
        return lower, upper


class Absolute(Pointwise):

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    def _compile_expression(self, py_self, value):
        return _pyast.Variable('numpy').get_attr('absolute').call(value)

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T == bool:
            raise ValueError('The boolean absolute value is not implemented.')
        return float if T == complex else T

    def _intbounds_impl(self):
        lower, upper = self.arg._intbounds
        extrema = builtins.abs(lower), builtins.abs(upper)
        if lower <= 0 and upper >= 0:
            return 0, max(extrema)
        else:
            return min(extrema), max(extrema)


class Cos(Holomorphic):
    'Cosine, element-wise.'
    deriv = lambda x: -Sin(x),

    def _simplified(self):
        if iszero(self.arg):
            return ones(self.shape, dtype=self.dtype)
        return super()._simplified()

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('cos').call(x)


class Sin(Holomorphic):
    'Sine, element-wise.'
    deriv = Cos,

    def _simplified(self):
        if iszero(self.arg):
            return zeros(self.shape, dtype=self.dtype)
        return super()._simplified()

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('sin').call(x)


class Tan(Holomorphic):
    'Tangent, element-wise.'
    deriv = lambda x: Cos(x)**astype(-2, x.dtype),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('tan').call(x)


class ArcSin(Holomorphic):
    'Inverse sine, element-wise.'
    deriv = lambda x: reciprocal(sqrt(astype(1, x.dtype)-x**astype(2, x.dtype))),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('arcsin').call(x)


class ArcCos(Holomorphic):
    'Inverse cosine, element-wise.'
    deriv = lambda x: -reciprocal(sqrt(astype(1, x.dtype)-x**astype(2, x.dtype))),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('arccos').call(x)


class ArcTan(Holomorphic):
    'Inverse tangent, element-wise.'
    deriv = lambda x: reciprocal(astype(1, x.dtype)+x**astype(2, x.dtype)),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('arctan').call(x)


class Sinc(Holomorphic):

    n: int = 0

    @property
    def parameters(self):
        return self.n,

    deriv = lambda x, n: Sinc(x, n=n+1),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numeric').get_attr('sinc').call(x, _pyast.LiteralInt(self.n))


class CosH(Holomorphic):
    'Hyperbolic cosine, element-wise.'
    deriv = lambda x: SinH(x),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('cosh').call(x)


class SinH(Holomorphic):
    'Hyperbolic sine, element-wise.'
    deriv = CosH,

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('sinh').call(x)


class TanH(Holomorphic):
    'Hyperbolic tangent, element-wise.'
    deriv = lambda x: astype(1, x.dtype) - TanH(x)**astype(2, x.dtype),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('tanh').call(x)


class ArcTanH(Holomorphic):
    'Inverse hyperbolic tangent, element-wise.'
    deriv = lambda x: reciprocal(astype(1, x.dtype)-x**astype(2, x.dtype)),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('arctanh').call(x)


class Exp(Holomorphic):
    deriv = lambda x: Exp(x),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('exp').call(x)


class Log(Holomorphic):
    deriv = lambda x: reciprocal(x),

    def _compile_expression(self, py_self, x):
        return _pyast.Variable('numpy').get_attr('log').call(x)


class Mod(Pointwise):

    dividend: Array
    divisor: Array

    @property
    def dependencies(self):
        return self.dividend, self.divisor

    def _compile_expression(self, py_self, dividend, divisor):
        return _pyast.BinOp(dividend, '%', divisor)

    @cached_property
    def dtype(self):
        dtype = self.dividend.dtype
        if self.divisor.dtype != dtype:
            raise ValueError(f'All arguments must have the same dtype but got {dividend} and {divisor}.')
        if dtype == bool:
            raise ValueError(f'The boolean floor division is not supported.')
        if dtype == complex:
            raise ValueError(f'The complex floor division is not supported.')
        return dtype

    def _intbounds_impl(self):
        lower_divisor, upper_divisor = self.divisor._intbounds
        if lower_divisor > 0:
            lower_dividend, upper_dividend = self.dividend._intbounds
            if 0 <= lower_dividend and upper_dividend < lower_divisor:
                return lower_dividend, upper_dividend
            else:
                return 0, upper_divisor - 1
        else:
            return super()._intbounds_impl()

    def _simplified(self):
        lower_divisor, upper_divisor = self.divisor._intbounds
        if lower_divisor > 0:
            lower_dividend, upper_dividend = self.dividend._intbounds
            if 0 <= lower_dividend and upper_dividend < lower_divisor:
                return self.dividend
        return super()._simplified()


class ArcTan2(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    deriv = lambda x, y: y / (x**astype(2, x.dtype) + y**astype(2, x.dtype)), lambda x, y: -x / (x**astype(2, x.dtype) + y**astype(2, x.dtype))

    def _compile_expression(self, py_self, x, y):
        return _pyast.Variable('numpy').get_attr('arctan2').call(x, y)

    @cached_property
    def dtype(self):
        if self.x.dtype == complex or self.y.dtype == complex:
            raise ValueError('arctan2 is not defined for complex numbers')
        return float


class Greater(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    def _compile_expression(self, py_self, x, y):
        return _pyast.Variable('numpy').get_attr('greater').call(x, y)

    @cached_property
    def dtype(self):
        dtype = self.x.dtype
        if self.y.dtype != dtype:
            raise ValueError('Cannot compare different dtypes.')
        elif dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        elif dtype == bool:
            raise ValueError('Use logical operators to compare booleans.')
        return bool


class Equal(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    def _compile_expression(self, py_self, x, y):
        return _pyast.Variable('numpy').get_attr('equal').call(x, y)

    @cached_property
    def dtype(self):
        if self.x.dtype != self.y.dtype:
            raise ValueError('Cannot compare different dtypes.')
        return bool

    def _simplified(self):
        if self.x == self.y:
            return ones(self.shape, bool)
        if self.ndim == 2:
            u1, w1 = unalign(self.x)
            u2, w2 = unalign(self.y)
            if u1.ndim == u2.ndim == 1 and u1 == u2 and w1 != w2 and isinstance(u1, Range):
                # NOTE: Once we introduce isunique we can relax the Range bound
                return Diagonalize(ones(u1.shape, bool))
        return super()._simplified()


class Less(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    def _compile_expression(self, py_self, x, y):
        return _pyast.Variable('numpy').get_attr('less').call(x, y)

    @cached_property
    def dtype(self):
        dtype = self.x.dtype
        if self.y.dtype != dtype:
            raise ValueError('Cannot compare different dtypes.')
        elif dtype == complex:
            raise ValueError('Complex numbers have no total order.')
        elif dtype == bool:
            raise ValueError('Use logical operators to compare booleans.')
        return bool


class LogicalNot(Pointwise):

    x: Array

    @property
    def dependencies(self):
        return self.x,

    evalf = staticmethod(numpy.logical_not)

    @cached_property
    def dtype(self):
        if self.x.dtype != bool:
            raise ValueError(f'Expected a boolean but got {T}.')
        return bool

    def _simplified(self):
        if isinstance(self.x, LogicalNot):
            return self.x.x
        return super()._simplified()


class Minimum(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    evalf = staticmethod(numpy.minimum)
    deriv = lambda x, y: .5 - .5 * Sign(x - y), lambda x, y: .5 + .5 * Sign(x - y)

    @cached_property
    def dtype(self):
        T1 = self.x.dtype
        T2 = self.y.dtype
        if T1 == complex or T2 == complex:
            raise ValueError('Complex numbers have no total order.')
        return float if float in (T1, T2) else int if int in (T1, T2) else bool

    def _simplified(self):
        if self.dtype == int:
            lower1, upper1 = self.x._intbounds
            lower2, upper2 = self.y._intbounds
            if upper1 <= lower2:
                return self.x
            elif upper2 <= lower1:
                return self.y
        return super()._simplified()

    def _intbounds_impl(self):
        lower1, upper1 = self.x._intbounds
        lower2, upper2 = self.y._intbounds
        return min(lower1, lower2), min(upper1, upper2)


class Maximum(Pointwise):

    x: Array
    y: Array

    @property
    def dependencies(self):
        return self.x, self.y

    evalf = staticmethod(numpy.maximum)
    deriv = lambda x, y: .5 + .5 * Sign(x - y), lambda x, y: .5 - .5 * Sign(x - y)

    @cached_property
    def dtype(self):
        T1 = self.x.dtype
        T2 = self.y.dtype
        if T1 == complex or T2 == complex:
            raise ValueError('Complex numbers have no total order.')
        return float if float in (T1, T2) else int if int in (T1, T2) else bool

    def _simplified(self):
        if self.dtype == int:
            lower1, upper1 = self.x._intbounds
            lower2, upper2 = self.y._intbounds
            if upper2 <= lower1:
                return self.x
            elif upper1 <= lower2:
                return self.y
        return super()._simplified()

    def _intbounds_impl(self):
        lower1, upper1 = self.x._intbounds
        lower2, upper2 = self.y._intbounds
        return max(lower1, lower2), max(upper1, upper2)


class Conjugate(Pointwise):

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    evalf = staticmethod(numpy.conjugate)

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != complex:
            raise ValueError(f'Conjugate is not defined for arguments of type {T}')
        return complex

    def _simplified(self):
        retval = self.arg._conjugate()
        if retval is not None:
            return retval
        return super()._simplified()


class Real(Pointwise):

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    evalf = staticmethod(numpy.real)

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != complex:
            raise ValueError(f'Real is not defined for arguments of type {T}')
        return float

    def _simplified(self):
        retval = self.arg._real()
        if retval is not None:
            return retval
        return super()._simplified()


class Imag(Pointwise):

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    evalf = staticmethod(numpy.imag)

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != complex:
            raise ValueError(f'Real is not defined for arguments of type {T}')
        return float

    def _simplified(self):
        retval = self.arg._imag()
        if retval is not None:
            return retval
        return super()._simplified()


class Cast(Pointwise):

    arg: Array

    @property
    def dependencies(self):
        return self.arg,

    @property
    def dependencies(self):
        return self.arg,

    def _compile_expression(self, py_self, arg):
        return _pyast.Variable('numpy').get_attr('array').call(arg, dtype=_pyast.Variable(self.dtype.__name__))

    def _simplified(self):
        if iszero(self.arg):
            return zeros_like(self)
        for axis, parts in self.arg._inflations:
            return util.sum(_inflate(self._newargs(func), dofmap, self.shape[axis], axis) for dofmap, func in parts.items())
        return super()._simplified()

    def _intbounds_impl(self):
        if self.arg.dtype == bool:
            return 0, 1
        else:
            return self.arg._intbounds

    @property
    def _const_uniform(self):
        value = self.arg._const_uniform
        if value is not None:
            return self.dtype(value)


class BoolToInt(Cast):

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != bool:
            raise TypeError(f'Expected an array with dtype bool but got {T.__name__}.')
        return int


class IntToFloat(Cast):

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != int:
            raise TypeError(f'Expected an array with dtype int but got {T.__name__}.')
        return float

    def _add(self, other):
        if isinstance(other, __class__):
            return self._newargs(self.arg + other.arg)

    def _multiply(self, other):
        if isinstance(other, __class__):
            return self._newargs(self.arg * other.arg)

    def _sum(self, axis):
        return self._newargs(sum(self.arg, axis))

    def _product(self):
        return self._newargs(product(self.arg, -1))

    def _sign(self):
        assert self.dtype != complex
        return self._newargs(sign(self.arg))

    def _derivative(self, var, seen):
        return Zeros(self.shape + var.shape, dtype=self.dtype)


class FloatToComplex(Cast):

    @cached_property
    def dtype(self):
        T = self.arg.dtype
        if T != float:
            raise TypeError(f'Expected an array with dtype float but got {T.__name__}.')
        return complex

    def _add(self, other):
        if isinstance(other, __class__):
            return self._newargs(self.arg + other.arg)

    def _multiply(self, other):
        if isinstance(other, __class__):
            return self._newargs(self.arg * other.arg)

    def _sum(self, axis):
        return self._newargs(sum(self.arg, axis))

    def _product(self):
        return self._newargs(product(self.arg, -1))

    def _real(self):
        return self.arg

    def _imag(self):
        return zeros_like(self.arg)

    def _conjugate(self):
        return self

    def _derivative(self, var, seen):
        if var.dtype == complex:
            raise ValueError('The complex derivative does not exist.')
        return FloatToComplex(derivative(self.arg, var, seen))


def astype(arg, dtype):
    arg = asarray(arg)
    if arg.dtype == bool and dtype != bool:
        arg = BoolToInt(arg)
    if arg.dtype == int and dtype != int:
        arg = IntToFloat(arg)
    if arg.dtype == float and dtype != float:
        arg = FloatToComplex(arg)
    if arg.dtype != dtype:
        raise TypeError('Downcasting is forbidden.')
    return arg


class Sign(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.dtype != complex, f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape

    def _simplified(self):
        return self.func._sign()

    evalf = staticmethod(numpy.sign)

    def _takediag(self, axis1, axis2):
        return Sign(_takediag(self.func, axis1, axis2))

    def _take(self, index, axis):
        return Sign(_take(self.func, index, axis))

    def _sign(self):
        return self

    def _unravel(self, axis, shape):
        return Sign(unravel(self.func, axis, shape))

    def _derivative(self, var, seen):
        return Zeros(self.shape + var.shape, dtype=self.dtype)

    def _intbounds_impl(self):
        lower, upper = self.func._intbounds
        return int(numpy.sign(lower)), int(numpy.sign(upper))


class Sampled(Array):
    '''Basis-like identity operator.

    Basis-like function that for every point evaluates to a partition of unity
    of a predefined set based on the selected interpolation scheme.

    Args
    ----
    points : 2d :class:`Array`
        Present point coordinates.
    target : 2d :class:`Array`
        Elementwise constant that evaluates to the target point coordinates.
    interpolation : :class:`str`
        Interpolation scheme to map points to target: "none" or "nearest".
    '''

    points: Array
    target: Array
    interpolation: str

    dtype = float

    def __post_init__(self):
        assert isinstance(self.points, Array) and self.points.ndim == 2, f'points={self.points!r}'
        assert isinstance(self.target, Array) and self.target.ndim == 2, f'target={self.target!r}'
        assert self.points.shape[1] == self.target.shape[1]

    @property
    def evalf(self):
        return self.evalf_methods[self.interpolation]

    @property
    def dependencies(self):
        return self.points, self.target

    @cached_property
    def shape(self):
        return self.points.shape[0], self.target.shape[0]

    def evalf_none(points, target):
        if points.shape != target.shape or not numpy.equal(points, target).all():
            raise ValueError('points do not correspond to the target sample; consider using "nearest" interpolation if this is desired')
        return numpy.eye(len(points))

    def evalf_nearest(points, target):
        nearest = numpy.linalg.norm(points[:,numpy.newaxis,:] - target[numpy.newaxis,:,:], axis=2).argmin(axis=1)
        return numpy.eye(len(target))[nearest]

    evalf_methods = dict(none=evalf_none, nearest=evalf_nearest)


def Elemwise(data: typing.Tuple[types.arraydata, ...], index: Array, dtype: Dtype):
    assert isinstance(data, tuple) and all(isinstance(d, types.arraydata) and d.dtype == dtype for d in data), f'data={data!r}'
    assert isinstance(index, Array) and index.ndim == 0 and index.dtype == int, f'index={index!r}'
    unique, indices = util.unique(data)
    if len(unique) == 1:
        return Constant(unique[0])
    # Create shape from data and index, rather than unique and the modified
    # index, in order to avoid potential shape inconsistencies later on.
    shapes = numpy.array([d.shape for d in data])
    shape = [Take(constant(s), index) for s in shapes.T]
    if len(unique) < len(data):
        index = Take(constant(indices), index)
    # Move all axes with constant shape to the left and ravel the remainder.
    is_constant = numpy.all(shapes[1:] == shapes[0], axis=0)
    const_axes = tuple(is_constant.nonzero()[0])
    var_axes = tuple((~is_constant).nonzero()[0])
    raveled = [numpy.transpose(d, const_axes + var_axes).reshape(*shapes[0, const_axes], -1) for d in unique]
    # Concatenate the raveled axis, take slices, unravel and reorder the axes to
    # the original position.
    concat = constant(numpy.concatenate(raveled, axis=-1))
    if not var_axes:
        return Take(concat, index)
    offset = Take(_SizesToOffsets(asarray([d.shape[-1] for d in raveled])), index)
    ravelshape = [shape[i] for i in var_axes]
    elemwise = unravel(Take(concat, Range(util.product(ravelshape)) + offset), -1, ravelshape)
    return Transpose.from_end(elemwise, *var_axes)


class Eig(Evaluable):

    func: Array
    symmetric: bool = False

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim >= 2 and not _certainly_different(self.func.shape[-1], self.func.shape[-2]), f'func={self.func!r}'
        assert isinstance(self.symmetric, bool), f'symmetric={self.symmetric!r}'

    @property
    def _w_dtype(self):
        return float if self.symmetric else complex

    @property
    def _vt_dtype(self):
        return float if self.symmetric and self.func.dtype != complex else complex

    @property
    def dependencies(self):
        return self.func,

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            shape = self.func.shape[:-1]
            dtype = self._w_dtype
        elif index == 1:
            shape=self.func.shape
            dtype=self._vt_dtype
        else:
            raise IndexError
        return ArrayFromTuple(self, index=index, shape=shape, dtype=dtype)

    def _simplified(self):
        return self.func._eig(self.symmetric)

    def evalf(self, arr):
        w, vt = (numpy.linalg.eigh if self.symmetric else numpy.linalg.eig)(arr)
        w = w.astype(self._w_dtype, copy=False)
        vt = vt.astype(self._vt_dtype, copy=False)
        return (w, vt)


class ArrayFromTuple(Array):

    arrays: Evaluable
    index: int
    shape: typing.Tuple[Array, ...]
    dtype: Dtype

    def __post_init__(self):
        assert isinstance(self.arrays, Evaluable), f'arrays={self.arrays!r}'
        assert isinstance(self.index, int), f'index={self.index!r}'

    @property
    def dependencies(self):
        return self.arrays,

    def _simplified(self):
        if isinstance(self.arrays, Tuple):
            # This allows the self.arrays evaluable to simplify itself into a
            # Tuple and its components be exposed to the function tree.
            return self.arrays[self.index]

    def _compile_expression(self, py_self, arrays):
        return arrays.get_item(_pyast.LiteralInt(self.index))

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self in cache:
            return cache[self]
        node = self.arrays._node(cache, subgraph, times, unique_loop_ids)
        if isinstance(node, TupleNode):
            node = node[self.index]
        cache[self] = node
        return node

    def _intbounds_impl(self):
        intbounds = getattr(self.arrays, '_intbounds_tuple', None)
        return intbounds[self.index] if intbounds else (float('-inf'), float('inf'))


class Singular(Array):

    dtype: Dtype

    shape = ()
    dependencies = ()

    def __post_init__(self):
        assert self.dtype in (float, complex), 'Singular dtype must be float or complex'

    def evalf(self):
        warnings.warn('singular matrix', RuntimeWarning)
        return numpy.array(numpy.nan, self.dtype)


class Zeros(Array):
    'zero'

    shape: typing.Tuple[Array, ...]
    dtype: Dtype

    @property
    def dependencies(self):
        return self.shape

    @cached_property
    def _unaligned(self):
        return Zeros((), self.dtype), ()

    def _compile_expression(self, py_self, *shape):
        return _pyast.Variable('numpy').get_attr('zeros').call(_pyast.Tuple(shape), dtype=_pyast.Variable(self.dtype.__name__))

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self.ndim:
            return super()._node(cache, subgraph, times, unique_loop_ids)
        elif self in cache:
            return cache[self]
        else:
            cache[self] = node = DuplicatedLeafNode('0', (type(self).__name__, times[self]))
            return node

    def _add(self, other):
        return other

    def _multiply(self, other):
        return self

    def _diagonalize(self, axis):
        return Zeros(self.shape+(self.shape[axis],), dtype=self.dtype)

    def _sum(self, axis):
        return Zeros(self.shape[:axis] + self.shape[axis+1:], dtype=self.dtype)

    def _transpose(self, axes):
        shape = tuple(self.shape[n] for n in axes)
        return Zeros(shape, dtype=self.dtype)

    def _insertaxis(self, axis, length):
        return Zeros(self.shape[:axis]+(length,)+self.shape[axis:], self.dtype)

    def _takediag(self, axis1, axis2):
        return Zeros(self.shape[:axis1]+self.shape[axis1+1:axis2]+self.shape[axis2+1:self.ndim]+(self.shape[axis1],), dtype=self.dtype)

    def _take(self, index, axis):
        return Zeros(self.shape[:axis] + index.shape + self.shape[axis+1:], dtype=self.dtype)

    def _inflate(self, dofmap, length, axis):
        return Zeros(self.shape[:axis] + (length,) + self.shape[axis+dofmap.ndim:], dtype=self.dtype)

    def _unravel(self, axis, shape):
        shape = self.shape[:axis] + shape + self.shape[axis+1:]
        return Zeros(shape, dtype=self.dtype)

    def _ravel(self, axis):
        return Zeros(self.shape[:axis] + (self.shape[axis]*self.shape[axis+1],) + self.shape[axis+2:], self.dtype)

    def _determinant(self, axis1, axis2):
        assert axis1 != axis2
        length = self.shape[axis1]
        assert length == self.shape[axis2]
        i, j = sorted([axis1, axis2])
        shape = (*self.shape[:i], *self.shape[i+1:j], *self.shape[j+1:])
        dtype = complex if self.dtype == complex else float
        if iszero(length):
            return ones(shape, dtype)
        else:
            return Zeros(shape, dtype)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return ()

    def _intbounds_impl(self):
        return 0, 0

    def _inverse(self, axis1, axis2):
        return singular_like(self)


class Inflate(Array):

    func: Array
    dofmap: Array
    length: Array

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func!r}'
        assert isinstance(self.dofmap, Array), f'dofmap={self.dofmap!r}'
        assert _isindex(self.length), f'length={self.length!r}'
        assert not _any_certainly_different(self.func.shape[self.func.ndim-self.dofmap.ndim:], self.dofmap.shape), f'func.shape={self.func.shape!r}, dofmap.shape={self.dofmap.shape!r}'

    @property
    def dependencies(self):
        return self.func, self.dofmap, self.length

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return *self.func.shape[:self.func.ndim-self.dofmap.ndim], self.length

    @cached_property
    def _diagonals(self):
        return tuple(axes for axes in self.func._diagonals if all(axis < self.ndim-1 for axis in axes))

    @cached_property
    def _inflations(self):
        inflations = [(self.ndim-1, types.frozendict({self.dofmap: self.func}))]
        for axis, parts in self.func._inflations:
            inflations.append((axis, types.frozendict((dofmap, Inflate(func, self.dofmap, self.length)) for dofmap, func in parts.items())))
        return tuple(inflations)

    def _simplified(self):
        for axis in range(self.dofmap.ndim):
            if isunit(self.dofmap.shape[axis]):
                return Inflate(_take(self.func, constant(0), self.func.ndim-self.dofmap.ndim+axis), _take(self.dofmap, constant(0), axis), self.length)
        for axis, parts in self.func._inflations:
            i = axis - (self.ndim-1)
            if i >= 0:
                return util.sum(Inflate(f, _take(self.dofmap, ind, i), self.length) for ind, f in parts.items())
        if self.dofmap.ndim == 0 and iszero(self.dofmap) and isunit(self.length):
            return InsertAxis(self.func, constant(1))
        return self.func._inflate(self.dofmap, self.length, self.ndim-1) \
            or self.dofmap._rinflate(self.func, self.length, self.ndim-1)

    def _optimized_for_numpy(self):
        indices = [Range(n) for n in self.shape[:-1]] + [self.dofmap]
        return Assemble(self.func, tuple(indices), self.shape)

    def evalf(self, array, indices, length):
        assert indices.ndim == self.dofmap.ndim
        assert length.ndim == 0
        if not self.dofmap.isconstant and int(length) > indices.size:
            warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
        shape = *array.shape[:array.ndim-indices.ndim], length
        return numeric.accumulate(array, (slice(None),)*(self.ndim-1)+(indices,), shape)

    def _compile_with_out(self, builder, out, out_block_id, mode):
        assert mode in ('iadd', 'assign')
        if mode == 'assign':
            builder.get_block_for_evaluable(self, block_id=out_block_id, comment='zero').array_fill_zeros(out)
        indices = _pyast.Tuple((_pyast.Variable('slice').call(_pyast.Variable('None')),)*(self.ndim-1) + (builder.compile(self.dofmap),))
        values = builder.compile(self.func)
        builder.get_block_for_evaluable(self).array_add_at(out, indices, values)

    def _inflate(self, dofmap, length, axis):
        if dofmap.ndim == 0 and dofmap == self.dofmap and length == self.length:
            return diagonalize(self, -1, axis)

    def _derivative(self, var, seen):
        return _inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.ndim-1)

    def _multiply(self, other):
        return Inflate(multiply(self.func, Take(other, self.dofmap)), self.dofmap, self.length)

    def _add(self, other):
        if isinstance(other, Inflate) and self.dofmap == other.dofmap:
            return Inflate(add(self.func, other.func), self.dofmap, self.length)

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        if axis2 == self.ndim-1:
            func = _take(self.func, self.dofmap, axis1)
            for i in range(self.dofmap.ndim):
                func = _takediag(func, axis1, axis2+self.dofmap.ndim-1-i)
            return Inflate(func, self.dofmap, self.length)
        else:
            return _inflate(_takediag(self.func, axis1, axis2), self.dofmap, self.length, self.ndim-3)

    def _take(self, index, axis):
        if axis != self.ndim-1:
            return Inflate(_take(self.func, index, axis), self.dofmap, self.length)
        newindex, newdofmap = SwapInflateTake(self.dofmap, index)
        if self.dofmap.ndim:
            func = self.func
            for i in range(self.dofmap.ndim-1):
                func = Ravel(func)
            intersection = Take(func, newindex)
        else:  # kronecker; newindex is all zeros (but of varying length)
            intersection = InsertAxis(self.func, newindex.shape[0])
        if index.ndim:
            swapped = unravel(Inflate(intersection, newdofmap, util.product(index.shape)), -1, index.shape)
        else:  # get; newdofmap is all zeros (but of varying length)
            swapped = Sum(intersection)
        return swapped

    def _diagonalize(self, axis):
        if axis != self.ndim-1:
            return _inflate(diagonalize(self.func, axis), self.dofmap, self.length, self.ndim-1)

    def _sum(self, axis):
        if axis == self.ndim-1:
            func = self.func
            for i in range(self.dofmap.ndim):
                func = Sum(func)
            return func
        return Inflate(sum(self.func, axis), self.dofmap, self.length)

    def _unravel(self, axis, shape):
        if axis != self.ndim-1:
            return Inflate(unravel(self.func, axis, shape), self.dofmap, self.length)

    def _sign(self):
        if isinstance(self.dofmap, Constant) and _isunique(self.dofmap.value):
            return Inflate(Sign(self.func), self.dofmap, self.length)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        chunks = []
        flat_dofmap = _flat(self.dofmap)
        keep_dim = self.func.ndim - self.dofmap.ndim
        strides = (1, *itertools.accumulate(self.dofmap.shape[:0:-1], operator.mul))[::-1]
        for *indices, values in self.func._assparse:
            if self.dofmap.ndim:
                inflate_indices = Take(flat_dofmap, functools.reduce(operator.add, map(operator.mul, indices[keep_dim:], strides)))
            else:
                inflate_indices = appendaxes(self.dofmap, values.shape)
            chunks.append((*indices[:keep_dim], inflate_indices, values))
        return tuple(chunks)

    def _intbounds_impl(self):
        lower, upper = self.func._intbounds
        return min(lower, 0), max(upper, 0)

    def _argument_degree(self, argument):
        if argument not in self.dofmap.arguments and argument not in self.length.arguments:
            return self.func.argument_degree(argument)


class SwapInflateTake(Evaluable):

    inflateidx: Array
    takeidx: Array

    @property
    def dependencies(self):
        return self.inflateidx, self.takeidx

    def _simplified(self):
        if isinstance(self.inflateidx, Constant) and isinstance(self.takeidx, Constant):
            return Tuple(tuple(map(constant, self.evalf(self.inflateidx.value, self.takeidx.value))))

    def __iter__(self):
        shape = ArrayFromTuple(self, index=2, shape=(), dtype=int),
        return (ArrayFromTuple(self, index=index, shape=shape, dtype=int) for index in range(2))

    @staticmethod
    def evalf(inflateidx, takeidx):
        uniqueinflate = _isunique(inflateidx)
        uniquetake = _isunique(takeidx)
        unique = uniqueinflate and uniquetake
        # If both indices are unique (i.e. they do not contain duplicates) then the
        # take and inflate operations can simply be restricted to the intersection,
        # with the the location of the intersection in the original index vectors
        # being the new indices for the swapped operations.
        intersection, subinflate, subtake = numpy.intersect1d(inflateidx, takeidx, return_indices=True, assume_unique=unique)
        if unique:
            return subinflate, subtake, numpy.array(len(intersection))
        # Otherwise, while still limiting the operations to the intersection, we
        # need to add the appropriate duplications on either side. The easiest way
        # to do this is to form the permutation matrix A for take (may contain
        # multiple items per column) and B for inflate (may contain several items
        # per row) and take the product AB for the combined operation. To then
        # decompose AB into the equivalent take followed by inflate we can simply
        # take the two index vectors from AB.nonzero() and form CD = AB. The
        # algorithm below does precisely this without forming AB explicitly.
        newinflate = []
        newtake = []
        for k, n in enumerate(intersection):
            for i in [subtake[k]] if uniquetake else numpy.equal(takeidx.ravel(), n).nonzero()[0]:
                for j in [subinflate[k]] if uniqueinflate else numpy.equal(inflateidx.ravel(), n).nonzero()[0]:
                    newinflate.append(i)
                    newtake.append(j)
        return numpy.array(newtake, dtype=int), numpy.array(newinflate, dtype=int), numpy.array(len(newtake), dtype=int)

    @property
    def _intbounds_tuple(self):
        return ((0, float('inf')),) * 3


class Assemble(Array):

    func: Array
    indices: typing.Tuple[Array, ...]
    shape: typing.Tuple[Array, ...]

    def __post_init__(self):
        assert len(self.indices) == len(self.shape)
        assert builtins.sum(index.ndim for index in self.indices) == self.func.ndim

    @property
    def dtype(self):
        return self.func.dtype

    @property
    def dependencies(self):
        return self.func, *self.indices, *self.shape

    @staticmethod
    def evalf(func, *args):
        n = len(args) // 2
        indices = args[:n]
        shape = args[n:]
        reshaped_indices = tuple(index.reshape((1,)*offset + index.shape + (1,)*(func.ndim-index.ndim-offset))
            for offset, index in zip(util.cumsum(index.ndim for index in indices), indices))
        return numeric.accumulate(func, reshaped_indices, shape)

    def _compile_with_out(self, builder, out, out_block_id, mode):
        # Compiles to an assignment (or in place addition) of the form:
        #
        #     out[index1, index2, ...] = func.transpose(trans)
        #
        # The order of the indices is unchanged, but ranges are converted to
        # slices and any other indices (dubbed 'advanced') reshaped in similar
        # fashion to numpy._ix to index the cross product. The right hand side
        # is transposed if a slice operation separates two advanced indices to
        # match Numpy's advanced indexing rules.
        assert mode in ('iadd', 'assign')
        if mode == 'assign':
            builder.get_block_for_evaluable(self, block_id=out_block_id, comment='zero').array_fill_zeros(out)
        advanced_ndim = builtins.sum(index.ndim for index in self.indices if not isinstance(index, Range))
        compiled_indices = []
        trans = [] # axes of func corresponding to advanced indices
        i = 0
        for index in self.indices:
            j = i + index.ndim
            if isinstance(index, Range):
                n = builder.compile(index.shape[0])
                compiled_index = _pyast.Variable('slice').call(n)
            else:
                prefix = len(trans)
                trans.extend(range(i, j))
                suffix = advanced_ndim - len(trans)
                compiled_index = builder.compile(index)
                if index.ndim < advanced_ndim:
                    compiled_index = compiled_index.get_item(_pyast.Raw(','.join(['None'] * prefix + [':'] * index.ndim + ['None'] * suffix)))
            compiled_indices.append(compiled_index)
            i = j
        assert i == self.func.ndim
        assert len(trans) == advanced_ndim
        compiled_func = builder.compile(self.func)
        if advanced_ndim > 1 and trans[-1] - trans[0] != advanced_ndim - 1: # trans is noncontiguous
            # see https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
            trans.extend(i for i, index in enumerate(self.indices) if isinstance(index, Range))
            compiled_func = compiled_func.get_attr('transpose').call(*[_pyast.LiteralInt(i) for i in trans])
        builder.get_block_for_evaluable(self).array_add_at(out, _pyast.Tuple(tuple(compiled_indices)), compiled_func)

    def _optimized_for_numpy(self):
        if isinstance(self.func, Assemble):
            indices = list(self.indices)
            # We aim to merge the indices from the nested Assemble operations
            # if they are separable, i.e. preceded by or following on full
            # slices, by replacing Range instances in indices by the
            # corresponding index from self.func.
            for i, index in enumerate(self.func.indices):
                if index != Range(self.func.shape[i]): # non-trivial index of self.func
                    # we need to account for the different axis numberings
                    # between self and self.func to find the right insertion
                    # point.
                    ax1 = 0 # axis of self.func
                    ax2 = 0 # axis of self
                    while ax1 < i: # find ax1, ax2 corresponding to i
                        ax1 += indices[ax2].ndim
                        ax2 += 1
                    if ax1 != i or ax2 >= self.ndim or indices[ax2] != Range(self.shape[ax2]):
                        # Any nontrivial nesting scenario would have been
                        # handled by Inflate if possible, so we simply bail out
                        # at the first sign of difficulty.
                        return
                    indices[ax2] = index # merge!
            return Assemble(self.func.func, tuple(indices), self.shape)


class Diagonalize(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim > 0, f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return *self.func.shape, self.func.shape[-1]

    @cached_property
    def _diagonals(self):
        diagonals = [frozenset([self.ndim-2, self.ndim-1])]
        for axes in self.func._diagonals:
            if axes & diagonals[0]:
                diagonals[0] |= axes
            else:
                diagonals.append(axes)
        return tuple(diagonals)

    @property
    def _inflations(self):
        return tuple((axis, types.frozendict((dofmap, Diagonalize(func)) for dofmap, func in parts.items()))
                     for axis, parts in self.func._inflations
                     if axis < self.ndim-2)

    def _simplified(self):
        if self.shape[-1] == 1:
            return InsertAxis(self.func, 1)
        return self.func._diagonalize(self.ndim-2)

    def _compile(self, builder):
        out, out_block_id = builder.new_empty_array_for_evaluable(self)
        self._compile_with_out(builder, out, out_block_id, mode='assign')
        return out

    def _compile_with_out(self, builder, out, out_block_id, mode):
        out_diag = _pyast.Variable('numpy').get_attr('einsum').call(_pyast.LiteralStr('...ii->...i'), out)
        if mode == 'assign':
            builder.get_block_for_evaluable(self, block_id=out_block_id, comment='zero').array_fill_zeros(out)
        builder.compile_with_out(self.func, out_diag, out_block_id, mode)

    def _derivative(self, var, seen):
        return diagonalize(derivative(self.func, var, seen), self.ndim-2, self.ndim-1)

    def _inverse(self, axis1, axis2):
        if sorted([axis1, axis2]) == [self.ndim-2, self.ndim-1]:
            return Diagonalize(reciprocal(self.func))

    def _determinant(self, axis1, axis2):
        if sorted([axis1, axis2]) == [self.ndim-2, self.ndim-1]:
            return Product(self.func)
        elif axis1 < self.ndim-2 and axis2 < self.ndim-2:
            return Diagonalize(determinant(self.func, (axis1, axis2)))

    def _sum(self, axis):
        if axis >= self.ndim - 2:
            return self.func
        return Diagonalize(sum(self.func, axis))

    def _takediag(self, axis1, axis2):
        if axis1 == self.ndim-2:  # axis2 == self.ndim-1
            return self.func
        elif axis2 >= self.ndim-2:
            return diagonalize(_takediag(self.func, axis1, self.ndim-2), self.ndim-3, self.ndim-2)
        else:
            return diagonalize(_takediag(self.func, axis1, axis2), self.ndim-4, self.ndim-3)

    def _take(self, index, axis):
        if axis < self.ndim - 2:
            return Diagonalize(_take(self.func, index, axis))
        func = _take(self.func, index, self.ndim-2)
        for i in range(index.ndim):
            func = diagonalize(func, self.ndim-2+i)
        return _inflate(func, index, self.func.shape[-1], self.ndim-2 if axis == self.ndim-1 else self.ndim-2+index.ndim)

    def _unravel(self, axis, shape):
        if axis >= self.ndim - 2:
            diag = diagonalize(diagonalize(Unravel(self.func, *shape), self.ndim-2, self.ndim), self.ndim-1, self.ndim+1)
            return ravel(diag, self.ndim if axis == self.ndim-2 else self.ndim-2)
        else:
            return Diagonalize(unravel(self.func, axis, shape))

    def _sign(self):
        return Diagonalize(Sign(self.func))

    def _product(self):
        if numeric.isint(self.shape[-1]) and self.shape[-1] > 1:
            return Zeros(self.shape[:-1], dtype=self.dtype)

    def _loopsum(self, index):
        return Diagonalize(loop_sum(self.func, index))

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return tuple((*indices, indices[-1], values) for *indices, values in self.func._assparse)

    def _argument_degree(self, argument):
        return self.func.argument_degree(argument)


class Guard(Array):
    'bar all simplifications'

    fun: Array

    def __post_init__(self):
        assert isinstance(self.fun, Array), f'fun={self.fun!r}'

    @property
    def dependencies(self):
        return self.fun,

    @cached_property
    def dtype(self):
        return self.fun.dtype

    @cached_property
    def shape(self):
        return self.fun.shape

    @property
    def isconstant(self):
        return False  # avoid simplifications based on fun being constant

    def _compile(self, builder):
        return builder.compile(self.fun)

    def _derivative(self, var, seen):
        return Guard(derivative(self.fun, var, seen))


class Find(Array):
    'indices of boolean index vector'

    where: Array

    dtype = int

    def __post_init__(self):
        assert isarray(self.where) and self.where.ndim == 1 and self.where.dtype == bool

    @property
    def dependencies(self):
        return self.where,

    @cached_property
    def shape(self):
        return Sum(BoolToInt(self.where)),

    def _compile_expression(self, py_self, where):
        return _pyast.Variable('numpy').get_attr('nonzero').call(where).get_item(_pyast.LiteralInt(0))

    def _simplified(self):
        if isinstance(self.where, Constant):
            indices, = self.where.value.nonzero()
            return constant(indices)


class DerivativeTargetBase(Array):
    'base class for derivative targets'

    @property
    def isconstant(self):
        return False


class WithDerivative(Array):
    '''Wrap the given function and define the derivative to a target.

    The wrapper is typically used together with a virtual derivative target like
    :class:`IdentifierDerivativeTarget`. The wrapper is removed in the simplified
    form.

    Parameters
    ----------
    func : :class:`Array`
        The function to wrap.
    var : :class:`DerivativeTargetBase`
        The derivative target.
    derivative : :class:`Array`
        The derivative with shape ``func.shape + var.shape``.

    See Also
    --------
    :class:`IdentifierDerivativeTarget` : a virtual derivative target
    '''

    func: Array
    var: DerivativeTargetBase
    derivative: Array

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return self.func.shape

    @property
    def arguments(self):
        return self.func.arguments | {self.var}

    def _compile(self, builder):
        return builder.compile(self.func)

    def _derivative(self, var: DerivativeTargetBase, seen) -> Array:
        if var == self.var:
            return self.derivative
        else:
            return derivative(self.func, var, seen)

    def _simplified(self) -> Array:
        return self.func


class Argument(DerivativeTargetBase):
    '''Array argument, to be substituted before evaluation.

    The :class:`Argument` is an :class:`Array` with a known shape, but whose
    values are to be defined later, before evaluation, e.g. using
    :func:`replace_arguments`.

    It is possible to take the derivative of an :class:`Array` to an
    :class:`Argument`:

    >>> from nutils import evaluable
    >>> a = evaluable.Argument('x', ())
    >>> b = evaluable.Argument('y', ())
    >>> f = a**3. + b**2.
    >>> evaluable.derivative(f, b).simplified == 2.*b
    True

    Args
    ----
    name : :class:`str`
        The Identifier of this argument.
    shape : :class:`tuple` of :class:`int`\\s
        The shape of this argument.
    '''

    name: str
    shape: typing.Tuple[Array, ...]
    dtype: Dtype = float

    def __post_init__(self):
        assert isinstance(self.name, str), f'name={self.name!r}'
        assert isinstance(self.shape, tuple) and all(_isindex(n) for n in self.shape), f'shape={self.shape!r}'

    @property
    def dependencies(self):
        return self.shape

    def _compile(self, builder):
        shape = builder.compile(self.shape)
        out = builder.get_variable_for_evaluable(self)
        block = builder.get_block_for_evaluable(self)
        block.assign_to(out, _pyast.Variable('numpy').get_attr('asarray').call(builder.get_argument(self.name), dtype=_pyast.Variable(self.dtype.__name__)))
        block.if_(_pyast.BinOp(shape, '!=', out.get_attr('shape'))).raise_(
            _pyast.Variable('ValueError').call(
                _pyast.LiteralStr('argument {!r} has the wrong shape: expected {}, got {}').get_attr('format').call(
                    _pyast.LiteralStr(self.name),
                    shape,
                    out.get_attr('shape'),
                ),
            ),
        )
        return out

    def _derivative(self, var, seen):
        if isinstance(var, Argument) and var.name == self.name and self.dtype in (float, complex):
            result = ones(self.shape, self.dtype)
            for i, sh in enumerate(self.shape):
                result = diagonalize(result, i, i+self.ndim)
            return result
        else:
            return zeros(self.shape+var.shape)

    def __str__(self):
        return '{} {!r} <{}>'.format(self.__class__.__name__, self.name, self._shape_str(form=str))

    @cached_property
    def eval(self):
        '''Evaluate function on a specified element, point set.'''

        return compile(self, _simplify=False, _optimize=False, stats=False, cache_const_intermediates=True)

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self in cache:
            return cache[self]
        else:
            label = '\n'.join(filter(None, (type(self).__name__, self.name, self._shape_str(form=repr))))
            cache[self] = node = DuplicatedLeafNode(label, (type(self).__name__, times[self]))
            return node

    @property
    def arguments(self):
        return frozenset({self})

    def _argument_degree(self, argument):
        assert self == argument
        return 1


class IdentifierDerivativeTarget(DerivativeTargetBase):
    '''Virtual derivative target distinguished by an identifier.

    Parameters
    ----------
    identifier : hashable :class:`object`
        The identifier for this derivative target.
    shape : :class:`tuple` of :class:`Array` or :class:`int`
        The shape of this derivative target.

    See Also
    --------
    :class:`WithDerivative` : :class:`Array` wrapper with additional derivative
    '''

    identifier: typing.Any
    shape: typing.Tuple[Array, ...]

    dtype = float
    dependencies = ()

    def _compile(self, builder):
        raise Exception(f'{type(self).__name__} cannot be evaluated')


class Ravel(Array):

    func: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim >= 2, f'func={self.func!r}'

    @property
    def dependencies(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return *self.func.shape[:-2], self.func.shape[-2] * self.func.shape[-1]

    @cached_property
    def _inflations(self):
        inflations = []
        stride = self.func.shape[-1]
        n = None
        for axis, old_parts in self.func._inflations:
            if axis == self.ndim - 1 and n is None:
                n = self.func.shape[-1]
                inflations.append((self.ndim - 1, types.frozendict((RavelIndex(dofmap, Range(n), *self.func.shape[-2:]), func) for dofmap, func in old_parts.items())))
            elif axis == self.ndim and n is None:
                n = self.func.shape[-2]
                inflations.append((self.ndim - 1, types.frozendict((RavelIndex(Range(n), dofmap, *self.func.shape[-2:]), func) for dofmap, func in old_parts.items())))
            elif axis < self.ndim - 1:
                inflations.append((axis, types.frozendict((dofmap, Ravel(func)) for dofmap, func in old_parts.items())))
        return tuple(inflations)

    def _simplified(self):
        if isunit(self.func.shape[-2]):
            return get(self.func, -2, constant(0))
        if isunit(self.func.shape[-1]):
            return get(self.func, -1, constant(0))
        return self.func._ravel(self.ndim-1)

    @staticmethod
    def evalf(f):
        return f.reshape(f.shape[:-2] + (f.shape[-2]*f.shape[-1],))

    def _multiply(self, other):
        if isinstance(other, Ravel) and _all_certainly_equal(other.func.shape[-2:], self.func.shape[-2:]):
            return Ravel(multiply(self.func, other.func))
        return Ravel(multiply(self.func, Unravel(other, *self.func.shape[-2:])))

    def _add(self, other):
        return Ravel(self.func + Unravel(other, *self.func.shape[-2:]))

    def _sum(self, axis):
        if axis == self.ndim-1:
            return Sum(Sum(self.func))
        return Ravel(sum(self.func, axis))

    def _derivative(self, var, seen):
        return ravel(derivative(self.func, var, seen), axis=self.ndim-1)

    def _takediag(self, axis1, axis2):
        assert axis1 < axis2
        if axis2 <= self.ndim-2:
            return ravel(_takediag(self.func, axis1, axis2), self.ndim-3)
        else:
            unraveled = unravel(self.func, axis1, self.func.shape[-2:])
            return Ravel(_takediag(_takediag(unraveled, axis1, -2), axis1, -2))

    def _take(self, index, axis):
        if axis != self.ndim-1:
            return Ravel(_take(self.func, index, axis))

    def _rtake(self, func, axis):
        if self.ndim == 1:
            return Ravel(Take(func, self.func))

    def _unravel(self, axis, shape):
        if axis != self.ndim-1:
            return Ravel(unravel(self.func, axis, shape))
        elif _all_certainly_equal(shape, self.func.shape[-2:]):
            return self.func

    def _inflate(self, dofmap, length, axis):
        if axis < self.ndim-dofmap.ndim:
            return Ravel(_inflate(self.func, dofmap, length, axis))
        elif dofmap.ndim == 0:
            return ravel(Inflate(self.func, dofmap, length), self.ndim-1)
        else:
            return _inflate(self.func, Unravel(dofmap, *self.func.shape[-2:]), length, axis)

    def _diagonalize(self, axis):
        if axis != self.ndim-1:
            return ravel(diagonalize(self.func, axis), self.ndim-1)

    def _insertaxis(self, axis, length):
        return ravel(insertaxis(self.func, axis+(axis == self.ndim), length), self.ndim-(axis == self.ndim))

    def _power(self, n):
        return Ravel(Power(self.func, Unravel(n, *self.func.shape[-2:])))

    def _sign(self):
        return Ravel(Sign(self.func))

    def _product(self):
        return Product(Product(self.func))

    def _loopsum(self, index):
        return Ravel(loop_sum(self.func, index))

    @property
    def _unaligned(self):
        unaligned, where = unalign(self.func, naxes=self.ndim - 1)
        return Ravel(unaligned), (*where, self.ndim - 1)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return tuple((*indices[:-2], indices[-2]*self.func.shape[-1]+indices[-1], values) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds_impl()

    def _argument_degree(self, argument):
        return self.func.argument_degree(argument)


class Unravel(Array):

    func: Array
    sh1: Array
    sh2: Array

    def __post_init__(self):
        assert isinstance(self.func, Array) and self.func.ndim > 0, f'func={self.func!r}'
        assert _isindex(self.sh1), f'sh1={self.sh1!r}'
        assert _isindex(self.sh2), f'sh2={self.sh2!r}'
        assert not _certainly_different(self.func.shape[-1], self.sh1 * self.sh2)

    @property
    def dependencies(self):
        return self.func, self.sh1, self.sh2

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return *self.func.shape[:-1], self.sh1, self.sh2

    def _simplified(self):
        if isunit(self.shape[-2]):
            return insertaxis(self.func, self.ndim-2, constant(1))
        if isunit(self.shape[-1]):
            return insertaxis(self.func, self.ndim-1, constant(1))
        return self.func._unravel(self.ndim-2, self.shape[-2:])

    def _derivative(self, var, seen):
        return unravel(derivative(self.func, var, seen), axis=self.ndim-2, shape=self.shape[-2:])

    @staticmethod
    def evalf(f, sh1, sh2):
        return f.reshape(f.shape[:-1] + (sh1, sh2))

    def _takediag(self, axis1, axis2):
        if axis2 < self.ndim-2:
            return unravel(_takediag(self.func, axis1, axis2), self.ndim-4, self.shape[-2:])

    def _take(self, index, axis):
        if axis < self.ndim - 2:
            return Unravel(_take(self.func, index, axis), *self.shape[-2:])

    def _sum(self, axis):
        if axis < self.ndim - 2:
            return Unravel(sum(self.func, axis), *self.shape[-2:])

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        return tuple((*indices[:-1], *divmod(indices[-1], appendaxes(self.shape[-1], values.shape)), values) for *indices, values in self.func._assparse)

    def _argument_degree(self, argument):
        if argument not in self.sh1.arguments and argument not in self.sh2.arguments:
            return self.func.argument_degree(argument)


class RavelIndex(Array):

    ia: Array
    ib: Array
    na: Array
    nb: Array

    dtype = int

    def __post_init__(self):
        assert isinstance(self.ia, Array), f'ia={self.ia!r}'
        assert isinstance(self.ib, Array), f'ib={self.ib!r}'
        assert _isindex(self.na), f'na={self.na!r}'
        assert _isindex(self.nb), f'nb={self.nb!r}'

    @cached_property
    def _length(self):
        return self.na * self.nb

    @property
    def dependencies(self):
        return self.ia, self.ib, self.nb

    @cached_property
    def shape(self):
        return self.ia.shape + self.ib.shape

    @staticmethod
    def evalf(ia, ib, nb):
        return ia[(...,)+(numpy.newaxis,)*ib.ndim] * nb + ib

    def _take(self, index, axis):
        if axis < self.ia.ndim:
            return RavelIndex(_take(self.ia, index, axis), self.ib, self.na, self.nb)
        else:
            return RavelIndex(self.ia, _take(self.ib, index, axis - self.ia.ndim), self.na, self.nb)

    def _rtake(self, func, axis):
        if _certainly_equal(func.shape[axis], self._length):
            return _take(_take(unravel(func, axis, (self.na, self.nb)), self.ib, axis+1), self.ia, axis)

    def _rinflate(self, func, length, axis):
        if _certainly_equal(length, self._length):
            return Ravel(Inflate(_inflate(func, self.ia, self.na, func.ndim - self.ndim), self.ib, self.nb))

    def _unravel(self, axis, shape):
        if axis < self.ia.ndim:
            return RavelIndex(unravel(self.ia, axis, shape), self.ib, self.na, self.nb)
        else:
            return RavelIndex(self.ia, unravel(self.ib, axis-self.ia.ndim, shape), self.na, self.nb)

    def _intbounds_impl(self):
        nbmin, nbmax = self.nb._intbounds
        iamin, iamax = self.ia._intbounds
        ibmin, ibmax = self.ib._intbounds
        return iamin * nbmin + ibmin, (iamax and nbmax and iamax * nbmax) + ibmax


class Range(Array):

    length: Array

    dtype = int

    def __post_init__(self):
        assert _isindex(self.length), f'length={self.length!r}'

    @property
    def dependencies(self):
        return self.length,

    @property
    def shape(self):
        return self.length,

    def _take(self, index, axis):
        return InRange(index, self.length)

    def _unravel(self, axis, shape):
        if len(shape) == 2:
            return RavelIndex(Range(shape[0]), Range(shape[1]), shape[0], shape[1])

    def _rtake(self, func, axis):
        if _certainly_equal(self.length, func.shape[axis]):
            return func

    def _rinflate(self, func, length, axis):
        if length == self.length:
            return func

    evalf = staticmethod(numpy.arange)

    def _intbounds_impl(self):
        lower, upper = self.length._intbounds
        assert lower >= 0
        return 0, max(0, upper - 1)


class InRange(Array):

    index: Array
    length: Array

    dtype = int

    def __post_init__(self):
        assert isinstance(self.index, Array) and self.index.dtype == int, f'index={self.index!r}'
        assert isinstance(self.length, Array) and self.length.dtype == int and self.length.ndim == 0, f'length={self.length!r}'

    @property
    def dependencies(self):
        return self.index, self.length

    @cached_property
    def shape(self):
        return self.index.shape

    @staticmethod
    def evalf(index, length):
        assert index.size == 0 or 0 <= index.min() and index.max() < length
        return index

    def _simplified(self):
        lower_length, upper_length = self.length._intbounds
        lower_index, upper_index = self.index._intbounds
        if 0 <= lower_index <= upper_index < lower_length:
            return self.index

    def _intbounds_impl(self):
        lower_index, upper_index = self.index._intbounds
        lower_length, upper_length = self.length._intbounds
        upper = min(upper_index, max(0, upper_length - 1))
        return max(0, min(lower_index, upper)), upper


class Polyval(Array):
    '''Evaluate a polynomial

    The polynomials are of the form

    .. math:: Σ_{k ∈ ℤ^n | Σ_i k_i ≤ p} c_k ∏_i x_i^(k_i)

    where :math:`c` is a vector of coefficients, :math:`x` a vector of
    :math:`n` variables and :math:`p` a nonnegative integer degree. The
    coefficients are assumed to be in reverse [lexicographic order]: the
    coefficient for powers :math:`j ∈ ℤ^n` comes before the coefficient for
    powers :math:`k ∈ ℤ^n / {j}` iff :math:`j_i > k_i`, where :math:`i =
    max_l(j_l ≠ k_l)`, the index of the *last* non-matching power.

    Args
    ----
    coeffs : :class:`Array`
        Array of coefficients where the last axis is treated as the
        coefficients axes. All remaining axes are treated pointwise.
    points : :class:`Array`
        Array of values where the last axis is treated as the variables axis.
        All remaining axes are treated pointwise.
    '''

    coeffs: Array
    points: Array

    dtype = float

    def __post_init__(self):
        assert isinstance(self.coeffs, Array) and self.coeffs.dtype == float and self.coeffs.ndim >= 1, f'coeffs={self.coeffs!r}'
        assert isinstance(self.points, Array) and self.points.dtype == float and self.points.ndim >= 1 and self.points.shape[-1].isconstant, f'points={self.points!r}'

    @cached_property
    def points_ndim(self):
        return int(self.points.shape[-1])

    @property
    def dependencies(self):
        return self.coeffs, self.points

    @cached_property
    def shape(self):
        return self.points.shape[:-1] + self.coeffs.shape[:-1]

    evalf = staticmethod(poly.eval_outer)

    def _derivative(self, var, seen):
        if self.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        dpoints = einsum('ABi,AiD->ABD', Polyval(PolyGrad(self.coeffs, self.points_ndim), self.points), derivative(self.points, var, seen), A=self.points.ndim-1)
        dcoeffs = Transpose.from_end(Polyval(Transpose.to_end(derivative(self.coeffs, var, seen), *range(self.coeffs.ndim)), self.points), *range(self.points.ndim-1, self.ndim))
        return dpoints + dcoeffs

    def _take(self, index, axis):
        if axis < self.points.ndim - 1:
            return Polyval(self.coeffs, _take(self.points, index, axis))
        elif axis < self.ndim:
            return Polyval(_take(self.coeffs, index, axis - self.points.ndim + 1), self.points)

    def _simplified(self):
        ncoeffs_lower, ncoeffs_upper = self.coeffs.shape[-1]._intbounds
        if iszero(self.coeffs):
            return zeros_like(self)
        elif isunit(self.coeffs.shape[-1]):
            return prependaxes(get(self.coeffs, -1, constant(0)), self.points.shape[:-1])
        points, where_points = unalign(self.points, naxes=self.points.ndim - 1)
        coeffs, where_coeffs = unalign(self.coeffs, naxes=self.coeffs.ndim - 1)
        if len(where_points) + len(where_coeffs) < self.ndim:
            where = *where_points, *(axis + self.points.ndim - 1 for axis in where_coeffs)
            return align(Polyval(coeffs, points), where, self.shape)


class PolyDegree(Array):
    '''Returns the degree of a polynomial given the number of coefficients and number of variables

    Args
    ----
    ncoeffs : :class:`Array`
        The number of coefficients of the polynomial.
    nvars : :class:`int`
        The number of variables of the polynomial.

    Notes
    -----

    See :class:`Polyval` for a definition of the polynomial.
    '''

    ncoeffs: Array
    nvars: int

    dtype = int
    shape = ()

    def __post_init__(self):
        assert isinstance(self.ncoeffs, Array) and self.ncoeffs.ndim == 0 and self.ncoeffs.dtype == int, 'ncoeffs={self.ncoeffs!r}'
        assert isinstance(self.nvars, int) and self.nvars >= 0, 'nvars={self.nvars!r}'

    @property
    def dependencies(self):
        return self.ncoeffs,

    def _compile_expression(self, py_self, ncoeffs):
        ncoeffs = ncoeffs.get_attr('__index__').call()
        degree = _pyast.Variable('poly').get_attr('degree').call(_pyast.LiteralInt(self.nvars), ncoeffs)
        return _pyast.Variable('numpy').get_attr('int_').call(degree)

    def _intbounds_impl(self):
        lower, upper = self.ncoeffs._intbounds
        try:
            lower = poly.degree(self.nvars, lower)
        except:
            lower = 0
        try:
            upper = poly.degree(self.nvars, upper)
        except:
            upper = float('inf')
        return lower, upper


class PolyNCoeffs(Array):
    '''Returns the number of coefficients for a polynomial of given degree and number of variables

    Args
    ----
    nvars : :class:`int`
        The number of variables of the polynomial.
    degree : :class:`Array`
        The degree of the polynomial.

    Notes
    -----

    See :class:`Polyval` for a definition of the polynomial.
    '''

    nvars: int
    degree: Array

    dtype = int
    shape = ()

    def __post_init__(self):
        assert isinstance(self.degree, Array) and self.degree.ndim == 0 and self.degree.dtype == int, f'degree={self.degree!r}'
        assert isinstance(self.nvars, int) and self.nvars >= 0, 'nvars={self.nvars!r}'

    @property
    def dependencies(self):
        return self.degree,

    def _compile_expression(self, py_self, degree):
        degree = degree.get_attr('__index__').call()
        ncoeffs = _pyast.Variable('poly').get_attr('ncoeffs').call(_pyast.LiteralInt(self.nvars), degree)
        return _pyast.Variable('numpy').get_attr('int_').call(ncoeffs)

    def _intbounds_impl(self):
        lower, upper = self.degree._intbounds
        if isinstance(lower, int) and lower >= 0:
            lower = poly.ncoeffs(self.nvars, lower)
        else:
            lower = 0
        if isinstance(upper, int) and upper >= 0:
            upper = poly.ncoeffs(self.nvars, upper)
        else:
            upper = float('inf')
        return lower, upper


class PolyMul(Array):
    '''Compute the coefficients for the product of two polynomials

    Return the coefficients such that calling :class:`Polyval` on this result
    is equal to the product of :class:`Polyval` called on the individual arrays
    of coefficients (with the appropriate selection of the variables as
    described by parameter ``vars``).

    Args
    ----
    coeffs_left : :class:`Array`
        The coefficients for the left operand. The last axis is treated as the
        coefficients axis.
    coeffs_right : :class:`Array`
        The coefficients for the right operand. The last axis is treated as the
        coefficients axis.
    vars : :class:`tuple` of ``nutils_poly.MulVar``
        For each variable of this product, ``var`` defines if the variable
        exists in the left polynomial, the right or both.

    Notes
    -----

    See :class:`Polyval` for a definition of the polynomial.
    '''

    coeffs_left: Array
    coeffs_right: Array
    vars: typing.Tuple[poly.MulVar, ...]

    dtype = float

    def __post_init__(self):
        assert isinstance(self.coeffs_left, Array) and self.coeffs_left.ndim >= 1 and self.coeffs_left.dtype == float, f'coeffs_left={self.coeffs_left!r}'
        assert isinstance(self.coeffs_right, Array) and self.coeffs_right.ndim >= 1 and self.coeffs_right.dtype == float, f'coeffs_right={self.coeffs_right!r}'
        assert not _any_certainly_different(self.coeffs_left.shape[:-1], self.coeffs_right.shape[:-1]), 'PolyMul({}, {})'.format(self.coeffs_left, self.coeffs_right)

    @cached_property
    def degree_left(self):
        return PolyDegree(self.coeffs_left.shape[-1], builtins.sum(var != poly.MulVar.Right for var in self.vars))

    @cached_property
    def degree_right(self):
        return PolyDegree(self.coeffs_right.shape[-1], builtins.sum(var != poly.MulVar.Left for var in self.vars))

    @property
    def dependencies(self):
        return self.coeffs_left, self.coeffs_right

    @cached_property
    def shape(self):
        ncoeffs = PolyNCoeffs(len(self.vars), self.degree_left + self.degree_right)
        return *self.coeffs_left.shape[:-1], ncoeffs

    @cached_property
    def evalf(self):
        try:
            degree_left = self.degree_left.__index__()
            degree_right = self.degree_right.__index__()
        except TypeError as e:
            return functools.partial(poly.mul, vars=self.vars)
        else:
            return poly.MulPlan(self.vars, degree_left, degree_right)

    def _simplified(self):
        if iszero(self.coeffs_left) or iszero(self.coeffs_right):
            return zeros_like(self)

    def _takediag(self, axis1, axis2):
        if axis1 < self.ndim - 1 and axis2 < self.ndim - 1:
            coeffs_left = Transpose.to_end(_takediag(self.coeffs_left, axis1, axis2), -2)
            coeffs_right = Transpose.to_end(_takediag(self.coeffs_right, axis1, axis2), -2)
            return Transpose.to_end(PolyMul(coeffs_left, coeffs_right, self.vars), -2)

    def _take(self, index, axis):
        if axis < self.ndim - 1:
            return PolyMul(_take(self.coeffs_left, index, axis), _take(self.coeffs_right, index, axis), self.vars)

    def _unravel(self, axis, shape):
        if axis < self.ndim - 1:
            return PolyMul(unravel(self.coeffs_left, axis, shape), unravel(self.coeffs_right, axis, shape), self.vars)


class PolyGrad(Array):
    '''Compute the coefficients for the gradient of a polynomial

    The last two axes of this array are the axis of variables and the axis of
    coefficients.

    Args
    ----
    coeffs : :class:`Array`
        The coefficients of the polynomial to compute the gradient for. The
        last axis is treated as the coefficients axis.
    nvars : :class:`int`
        The number of variables of the polynomial.

    Notes
    -----

    See :class:`Polyval` for a definition of the polynomial.
    '''

    coeffs: Array
    nvars: int

    dtype = float

    def __post_init__(self):
        assert isinstance(self.coeffs, Array) and self.coeffs.dtype == float and self.coeffs.ndim >= 1, f'coeffs={self.coeffs!r}'
        assert isinstance(self.nvars, int) and self.nvars >= 0, f'nvars={self.nvars!r}'

    @cached_property
    def degree(self):
        return PolyDegree(self.coeffs.shape[-1], self.nvars)

    @property
    def dependencies(self):
        return self.coeffs,

    @cached_property
    def shape(self):
        ncoeffs = PolyNCoeffs(self.nvars, Maximum(constant(0), self.degree - constant(1)))
        return *self.coeffs.shape[:-1], constant(self.nvars), ncoeffs

    @cached_property
    def evalf(self):
        try:
            degree = self.degree.__index__()
        except TypeError as e:
            return functools.partial(poly.grad, nvars=self.nvars)
        else:
            return poly.GradPlan(self.nvars, degree)

    def _simplified(self):
        if iszero(self.coeffs) or iszero(self.degree):
            return zeros_like(self)
        elif isunit(self.degree):
            return InsertAxis(Take(self.coeffs, constant(self.nvars - 1) - Range(constant(self.nvars))), constant(1))

    def _takediag(self, axis1, axis2):
        if axis1 < self.ndim - 2 and axis2 < self.ndim - 2:
            coeffs = Transpose.to_end(_takediag(self.coeffs, axis1, axis2), -2)
            return Transpose.from_end(PolyGrad(coeffs, self.nvars), -3, -2)

    def _take(self, index, axis):
        if axis < self.ndim - 2:
            return PolyGrad(_take(self.coeffs, index, axis), self.nvars)

    def _unravel(self, axis, shape):
        if axis < self.ndim - 2:
            return PolyGrad(unravel(self.coeffs, axis, shape), self.nvars)


class Legendre(Array):
    '''Series of Legendre polynomial up to and including the given degree.

    Parameters
    ---------
    x : :class:`Array`
        The coordinates to evaluate the series at.
    degree : :class:`int`
        The degree of the last polynomial of the series.
    '''

    x: Array
    degree: int

    dtype = float

    def __post_init__(self):
        assert isinstance(self.x, Array) and self.x.dtype == float, f'x={self.x!r}'
        assert isinstance(self.degree, int) and self.degree >= 0, f'degree={self.degree!r}'

    @property
    def dependencies(self):
        return self.x,

    @cached_property
    def shape(self):
        return *self.x.shape, constant(self.degree+1)

    def evalf(self, x: numpy.ndarray) -> numpy.ndarray:
        P = numpy.empty((*x.shape, self.degree+1), dtype=float)
        P[..., 0] = 1
        if self.degree:
            P[..., 1] = x
        for i in range(2, self.degree+1):
            P[..., i] = (2-1/i)*P[..., 1]*P[..., i-1] - (1-1/i)*P[..., i-2]
        return P

    def _derivative(self, var, seen):
        if self.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        d = numpy.zeros((self.degree+1,)*2, dtype=int)
        for i in range(self.degree+1):
            d[i, i+1::2] = 2*i+1
        dself = einsum('Ai,ij->Aj', self, astype(d, self.dtype))
        return einsum('Ai,AB->AiB', dself, derivative(self.x, var, seen))

    def _simplified(self):
        unaligned, where = unalign(self.x)
        if where != tuple(range(self.x.ndim)):
            return align(Legendre(unaligned, self.degree), (*where, self.ndim-1), self.shape)

    def _takediag(self, axis1, axis2):
        if axis1 < self.ndim - 1 and axis2 < self.ndim - 1:
            return Transpose.to_end(Legendre(_takediag(self.x, axis1, axis2), self.degree), -2)

    def _take(self, index, axis):
        if axis < self.ndim - 1:
            return Legendre(_take(self.x, index, axis), self.degree)

    def _unravel(self, axis, shape):
        if axis < self.ndim - 1:
            return Legendre(unravel(self.x, axis, shape), self.degree)


class Choose(Array):
    '''Function equivalent of :func:`numpy.choose`.'''

    index: Array
    choices: Array

    def __post_init__(self):
        assert isinstance(self.index, Array) and self.index.dtype == int, f'index={self.index!r}'
        assert isinstance(self.choices, Array), f'choices={self.choices!r}'
        assert not _any_certainly_different(self.choices.shape[:-1], self.index.shape)

    @property
    def dependencies(self):
        return self.index, self.choices

    @cached_property
    def dtype(self):
        return self.choices.dtype

    @cached_property
    def shape(self):
        return self.index.shape

    def _compile_expression(self, py_self, index, choices):
        choices = _pyast.Variable('numpy').get_attr('moveaxis').call(choices, _pyast.LiteralInt(-1), _pyast.LiteralInt(0))
        return _pyast.Variable('numpy').get_attr('choose').call(index, choices)

    def _derivative(self, var, seen):
        return Choose(appendaxes(self.index, var.shape), Transpose.to_end(derivative(self.choices, var, seen), self.ndim))

    def _simplified(self):
        choices, where = unalign(self.choices)
        if self.ndim not in where:
            return align(choices, where, self.shape)
        index, choices, where = unalign(self.index, self.choices, naxes=self.ndim)
        if len(where) < self.ndim:
            return align(Choose(index, choices), where, self.shape)

    def _multiply(self, other):
        if isinstance(other, Choose) and self.index == other.index:
            return Choose(self.index, self.choices * other.choices)

    def _get(self, i, item):
        return Choose(get(self.index, i, item), get(self.choices, i, item))

    def _sum(self, axis):
        unaligned, where = unalign(self.index)
        if axis not in where:
            index = align(unaligned, [i-(i > axis) for i in where], self.shape[:axis]+self.shape[axis+1:])
            return Choose(index, sum(self.choices, axis))

    def _take(self, index, axis):
        return Choose(_take(self.index, index, axis), _take(self.choices, index, axis))

    def _takediag(self, axis, rmaxis):
        return Choose(takediag(self.index, axis, rmaxis), takediag(self.choices, axis, rmaxis))

    def _product(self):
        unaligned, where = unalign(self.index)
        if self.ndim-1 not in where:
            index = align(unaligned, where, self.shape[:-1])
            return Choose(index, product(self.choices, self.ndim-1))


class NormDim(Array):

    length: Array
    index: Array

    def __post_init__(self):
        assert isinstance(self.length, Array) and self.length.dtype == int, f'length={self.length!r}'
        assert isinstance(self.index, Array) and self.index.dtype == int, f'index={self.index!r}'
        assert not _any_certainly_different(self.length.shape, self.index.shape)
        # The following corner cases makes the assertion fail, hence we can only
        # assert the bounds if the arrays are guaranteed to be unempty:
        #
        #     Take(func, NormDim(func.shape[-1], Range(0) + func.shape[-1]))
        if all(n._intbounds[0] > 0 for n in self.index.shape):
            assert -self.length._intbounds[1] <= self.index._intbounds[0] and self.index._intbounds[1] <= self.length._intbounds[1] - 1

    @property
    def dependencies(self):
        return self.length, self.index

    @cached_property
    def dtype(self):
        return self.index.dtype

    @cached_property
    def shape(self):
        return self.index.shape

    @staticmethod
    def evalf(length, index):
        assert length.shape == index.shape
        assert length.dtype.kind == 'i'
        assert index.dtype.kind == 'i'
        result = numpy.empty(index.shape, dtype=int)
        for i in numpy.ndindex(index.shape):
            result[i] = numeric.normdim(length[i], index[i])
        return result

    def _simplified(self):
        lower_length, upper_length = self.length._intbounds
        lower_index, upper_index = self.index._intbounds
        if 0 <= lower_index and upper_index < lower_length:
            return self.index
        if isinstance(lower_length, int) and lower_length == upper_length and -lower_length <= lower_index and upper_index < 0:
            return self.index + lower_length
        if isinstance(self.length, Constant) and isinstance(self.index, Constant):
            return constant(self.evalf(self.length.value, self.index.value))

    def _intbounds_impl(self):
        lower_length, upper_length = self.length._intbounds
        lower_index, upper_index = self.index._intbounds
        if lower_index >= 0:
            return min(lower_index, upper_length - 1), min(upper_index, upper_length - 1)
        elif upper_index < 0 and isinstance(lower_length, int) and lower_length == upper_length:
            return max(lower_index + lower_length, 0), max(upper_index + lower_length, 0)
        else:
            return 0, upper_length - 1


class TransformCoords(Array):
    '''Transform coordinates from one coordinate system to another (spatial part)

    Args
    ----
    target : :class:`nutils.transformseq.Transforms`, optional
        The target coordinate system. If `None` the target is root coordinate
        system.
    source : :class:`nutils.transformseq.Transforms`
        The source coordinate system.
    index : scalar, integer :class:`Array`
        The index part of the source coordinates.
    coords : :class:`Array`
        The spatial part of the source coordinates.
    '''

    target: typing.Optional['transformseq.Transforms']
    source: 'transformseq.Transforms'
    index: Array
    coords: Array

    dtype = float

    def __post_init__(self):
        if self.target is not None and self.target.todims != self.source.todims:
            raise ValueError('the source and target sequences have different todims')
        if self.index.dtype != int or self.index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        if self.coords.dtype != float or self.coords.ndim == 0:
            raise ValueError('argument `coords` must be a real-valued array with at least one axis')
        if _certainly_different(self.coords.shape[-1], constant(self.source.fromdims)):
            raise ValueError('the last axis of argument `coords` must match the `fromdims` of the `source` transform chains sequence')

    @property
    def dependencies(self):
        return self.index, self.coords

    @cached_property
    def shape(self):
        target_dim = self.source.todims if self.target is None else self.target.fromdims
        return *self.coords.shape[:-1], constant(target_dim)

    def evalf(self, index, coords):
        chain = self.source[index.__index__()]
        if self.target is not None:
            _, chain = self.target.index_with_tail(chain)
        return functools.reduce(lambda c, t: t.apply(c), reversed(chain), coords)

    def _derivative(self, var, seen):
        linear = TransformLinear(self.target, self.source, self.index)
        dcoords = derivative(self.coords, var, seen)
        return einsum('ij,AjB->AiB', linear, dcoords, A=self.coords.ndim - 1, B=var.ndim)

    def _simplified(self):
        from nutils.transformseq import MaskedTransforms
        if self.target == self.source:
            return self.coords
        if isinstance(self.source, MaskedTransforms):
            index = Take(constant(self.source._indices), self.index)
            return TransformCoords(self.target, self.source._parent, index, self.coords)
        cax = self.ndim - 1
        coords, where = unalign(self.coords, naxes=cax)
        if len(where) < cax:
            return align(TransformCoords(self.target, self.source, self.index, coords), (*where, cax), self.shape)


class TransformIndex(Array):
    '''Transform coordinates from one coordinate system to another (index part)

    Args
    ----
    target : :class:`nutils.transformseq.Transforms`
        The target coordinate system.
    source : :class:`nutils.transformseq.Transforms`
        The source coordinate system.
    index : scalar, integer :class:`Array`
        The index part of the source coordinates.
    '''

    target: 'transformseq.Transforms'
    source: 'transformseq.Transforms'
    index: Array

    dtype = int
    shape = ()

    def __post_init__(self):
        if self.target.todims != self.source.todims:
            raise ValueError('the source and target sequences have different todims')
        if self.index.dtype != int or self.index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')

    @property
    def dependencies(self):
        return self.index,

    def evalf(self, index):
        index, _ = self.target.index_with_tail(self.source[index.__index__()])
        return numpy.array(index)

    def _intbounds_impl(self):
        return 0, len(self.target) - 1

    def _simplified(self):
        from nutils.transformseq import MaskedTransforms
        if self.target == self.source:
            return self.index
        if isinstance(self.source, MaskedTransforms):
            index = Take(constant(self.source._indices), self.index)
            return TransformIndex(self.target, self.source._parent, index)


class TransformLinear(Array):
    '''Linear part of a coordinate transformation

    Args
    ----
    target : :class:`nutils.transformseq.Transforms`, optional
        The target coordinate system. If `None` the target is the root
        coordinate system.
    source : :class:`nutils.transformseq.Transforms`
        The source coordinate system.
    index : scalar, integer :class:`Array`
        The index part of the source coordinates.
    '''

    target: typing.Optional['transformseq.Transforms']
    source: 'transformseq.Transforms'
    index: Array

    dtype = float

    def __post_init__(self):
        if self.target is not None and self.target.todims != self.source.todims:
            raise ValueError('the source and target sequences have different todims')
        if self.index.dtype != int or self.index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')

    @property
    def dependencies(self):
        return self.index,

    @cached_property
    def shape(self):
        target_dim = self.source.todims if self.target is None else self.target.fromdims
        return constant(target_dim), constant(self.source.fromdims)

    def evalf(self, index):
        chain = self.source[index.__index__()]
        if self.target is not None:
            _, chain = self.target.index_with_tail(chain)
        if chain:
            return functools.reduce(lambda r, i: i @ r, (item.linear for item in reversed(chain)))
        else:
            return numpy.eye(self.source.fromdims)

    def _simplified(self):
        from nutils.transformseq import MaskedTransforms
        if self.target == self.source:
            return diagonalize(ones((constant(self.source.fromdims),), dtype=float))
        if isinstance(self.source, MaskedTransforms):
            index = Take(constant(self.source._indices), self.index)
            return TransformLinear(self.target, self.source._parent, index)
        if self.source._linear_is_constant and (self.target is None or self.target._linear_is_constant):
            return constant(self.evalf(0))


class TransformBasis(Array):
    '''Vector basis for the root and a source coordinate system

    The columns of this matrix form a vector basis for the space of root
    coordinates. The first `n` vectors also span the space of source
    coordinates mapped to the root, where `n` is the dimension of the source
    coordinate system. The remainder is *not* a span of the complement space in
    general.

    No additional properties are guaranteed beyond the above. In particular, if
    the source coordinate system has the same dimension as the root, the
    basis is *not necessarily* the same as ``TransformLinear(None, source,
    index)``.

    Args
    ----
    source : :class:`nutils.transformseq.Transforms`
        The source coordinate system.
    index : scalar, integer :class:`Array`
        The index part of the source coordinates.
    '''

    source: 'transformseq.Transforms'
    index: Array

    dtype = float

    def __post_init__(self):
        if self.index.dtype != int or self.index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')

    @property
    def dependencies(self):
        return self.index,

    @cached_property
    def shape(self):
        return constant(self.source.todims), constant(self.source.todims)

    def evalf(self, index):
        chain = self.source[index.__index__()]
        linear = numpy.eye(self.source.fromdims)
        for item in reversed(chain):
            linear = item.linear @ linear
            assert item.fromdims <= item.todims <= item.fromdims + 1
            if item.todims == item.fromdims + 1:
                linear = numpy.concatenate([linear, item.ext[:, numpy.newaxis]], axis=1)
        assert linear.shape == (self.source.todims, self.source.todims)
        return linear

    def _simplified(self):
        from nutils.transformseq import MaskedTransforms
        if self.source.todims == self.source.fromdims:
            # Since we only guarantee that the basis spans the space of source
            # coordinates mapped to the root and the map is a bijection (every
            # `Transform` is assumed to be injective), we can return the unit
            # vectors here.
            return diagonalize(ones((self.source.fromdims,), dtype=float))
        if isinstance(self.source, MaskedTransforms):
            index = Take(constant(self.source._indices), self.index)
            return TransformBasis(self.source._parent, index)
        if self.source._linear_is_constant:
            return constant(self.evalf(0))


class _LoopId(types.Singleton):

    def __init__(self, id):
        self.id = id

    def __str__(self):
        return str(self.id)


class _LoopIndex(Array):

    loop_id: _LoopId
    length: Array

    dtype = int
    shape = ()

    def __post_init__(self):
        assert isinstance(self.loop_id, _LoopId), f'loop_id={self.loop_id!r}'
        assert _isindex(self.length), f'length={self.length!r}'

    @property
    def dependencies(self):
        return self.length,

    def __str__(self):
        try:
            length = self.length.__index__()
        except:
            length = '?'
        return f'LoopIndex({self.loop_id}, length={length})'

    def _compile(self, builder):
        raise ValueError(f'`_LoopIndex` outside `Loop` with corresponding `_LoopId`: {self.loop_id}.')

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if self in cache:
            return cache[self]
        cache[self] = node = RegularNode(f'LoopIndex {self.loop_id}', (), dict(length=self.length._node(cache, subgraph, times, unique_loop_ids)), (type(self).__name__, _Stats()), subgraph)
        return node

    def _intbounds_impl(self):
        lower_length, upper_length = self.length._intbounds
        return 0, max(0, upper_length - 1)

    def _simplified(self):
        if isunit(self.length):
            return Zeros((), int)

    @property
    def arguments(self):
        return frozenset({self})


class Loop(Array):
    '''Base class for evaluable loops.

    Subclasses must implement

    *   method ``evalf_loop_init(init_arg)`` and
    *   method ``evalf_loop_body(output, body_arg)``.
    '''

    loop_id: _LoopId
    length: Array

    init_args = util.abstract_property()
    body_args = util.abstract_property()

    def __post_init__(self):
        assert isinstance(self.loop_id, _LoopId), f'loop_id={loop_id!r}'
        assert isinstance(self.length, Array), f'length={length!r}'
        if any(self.index in arg.arguments for arg in self.init_args):
            raise ValueError('the loop initialization arguments must not depend on the index')

    @property
    def index(self):
        return _LoopIndex(self.loop_id, self.length)

    @property
    def dependencies(self):
        return self.length, *self.init_args, *self.body_args

    def _node(self, cache, subgraph, times, unique_loop_ids):
        if (cached := cache.get(self)) is not None:
            return cached

        # To prevent drawing descendents that do not depend on `self.index`
        # inside the subgraph for this loop, we populate the `cache` with
        # descendents that do no depend on this loop's index or indices of
        # nested loops (the `inside_indices`).
        stack = [(func, frozenset({self.index})) for func in [self.length, *self.init_args, *self.body_args]]
        while stack:
            func, inside_indices = stack.pop()
            if inside_indices.isdisjoint(frozenset(func.arguments)):
                func._node(cache, subgraph, times, unique_loop_ids)
            else:
                if isinstance(func, Loop):
                    inside_indices = inside_indices | frozenset({func.index})
                stack.extend([(dep, inside_indices) for dep in func.dependencies])

        if unique_loop_ids:
            loopcache = cache
            loopgraph = cache.setdefault(('subgraph', self.loop_id), Subgraph('Loop', subgraph))
            looptimes = times
        else:
            loopcache = cache.copy()
            loopcache.pop(self.index, None)
            loopgraph = Subgraph('Loop', subgraph)
            looptimes = times.get(self, collections.defaultdict(_Stats))
        cache[self] = node = self._node_loop_body(loopcache, loopgraph, looptimes, unique_loop_ids)
        return node

    @cached_property
    def arguments(self):
        return super().arguments - frozenset({self.index})

    @property
    def _loops(self):
        return (util.IDSet([self]) | super()._loops).view()


class LoopSum(Loop):

    func: Array
    shape: typing.Tuple[Array, ...]

    def __post_init__(self):
        assert isinstance(self.loop_id, _LoopId), f'loop_id={self.loop_id!r}'
        assert isinstance(self.length, Array), f'length={self.length!r}'
        assert isinstance(self.func, Array) and self.func.dtype != bool, f'func={self.func!r}'
        assert self.func.ndim == len(self.shape)
        super().__post_init__()

    @property
    def init_args(self):
        return self.shape

    @property
    def body_args(self):
        return self.func,

    @cached_property
    def dtype(self):
        return self.func.dtype

    def _compile(self, builder):
        out, out_block_id = builder.new_empty_array_for_evaluable(self)
        # `out_block_id` is always at the beginning the scope this evaluable
        # belongs to (out_block_id = (*builder.get_block_id(self)[:-1], 0), so
        # we're not reaching `NotImplemented` in `self._compile_with_out`.
        self._compile_with_out(builder, out, out_block_id, mode='assign')
        return out

    def _compile_with_out(self, builder, out, out_block_id, mode):
        assert mode in ('iadd', 'assign')
        if out_block_id > builder.get_block_id(self.index):
            # The loop body comes before the definition of `out`.
            return NotImplemented
        if mode == 'assign':
            builder.get_block_for_evaluable(self, block_id=out_block_id, comment='zero').array_fill_zeros(out)
        index_block_id = builder.get_block_id(self.index)
        body_block_id = builtins.max(index_block_id, builder.get_block_id(self.func))
        assert body_block_id[:-1] == index_block_id[:-1]
        builder.compile_with_out(self.func, out, body_block_id, 'iadd')

    def _derivative(self, var, seen):
        return loop_sum(derivative(self.func, var, seen), self.index)

    def _node_loop_body(self, cache, subgraph, times, unique_loop_ids):
        if (cached := cache.get(self)) is not None:
            return cached
        kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times, unique_loop_ids) for i, n in enumerate(self.shape)}
        kwargs['func'] = self.func._node(cache, subgraph, times, unique_loop_ids)
        cache[self] = node = RegularNode(f'LoopSum {self.loop_id}', (), kwargs, (type(self).__name__, times[self]), subgraph)
        return node

    def _simplified(self):
        if iszero(self.func):
            return zeros_like(self)
        elif self.index not in self.func.arguments:
            return self.func * astype(self.index.length, self.func.dtype)
        for axis, parts in self.func._inflations:
            if not any(self.index in dofmap.arguments for dofmap in parts):
                return util.sum(_inflate(loop_sum(func, self.index), dofmap, self.shape[axis], axis) for dofmap, func in parts.items())
        return self.func._loopsum(self.index)

    def _takediag(self, axis1, axis2):
        return loop_sum(_takediag(self.func, axis1, axis2), self.index)

    def _take(self, index, axis):
        return loop_sum(_take(self.func, index, axis), self.index)

    def _unravel(self, axis, shape):
        return loop_sum(unravel(self.func, axis, shape), self.index)

    def _sum(self, axis):
        return loop_sum(sum(self.func, axis), self.index)

    def _add(self, other):
        if isinstance(other, LoopSum) and other.index == self.index:
            return loop_sum(self.func + other.func, self.index)

    def _multiply(self, other):
        # If `other` depends on `self.index`, e.g. because `self` is the inner
        # loop of two nested `LoopSum`s over the same index, then we should not
        # move `other` inside this loop.
        if self.index not in other.arguments:
            return loop_sum(self.func * other, self.index)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        chunks = []
        for *elem_indices, elem_values in self.func._assparse:
            if self.ndim == 0:
                values = loop_concatenate(InsertAxis(elem_values, constant(1)), self.index)
                while values.ndim:
                    values = Sum(values)
                chunks.append((values,))
            else:
                if elem_values.ndim == 0:
                    *elem_indices, elem_values = (InsertAxis(arr, constant(1)) for arr in (*elem_indices, elem_values))
                else:
                    # minimize ravels by transposing all variable length axes to the end
                    variable = tuple(i for i, n in enumerate(elem_values.shape) if self.index in n.arguments)
                    *elem_indices, elem_values = (Transpose.to_end(arr, *variable) for arr in (*elem_indices, elem_values))
                    for i in variable[:-1]:
                        *elem_indices, elem_values = map(Ravel, (*elem_indices, elem_values))
                    assert all(self.index not in n.arguments for n in elem_values.shape[:-1])
                chunks.append(tuple(loop_concatenate(arr, self.index) for arr in (*elem_indices, elem_values)))
        return tuple(chunks)

    def _argument_degree(self, argument):
        if argument not in self.length.arguments:
            return self.func.argument_degree(argument)


class _SizesToOffsets(Array):

    sizes: Array

    dtype = int

    def __post_init__(self):
        assert self.sizes.ndim == 1
        assert self.sizes.dtype == int
        assert self.sizes._intbounds[0] >= 0

    @property
    def dependencies(self):
        return self.sizes,

    @cached_property
    def shape(self):
        return self.sizes.shape[0]+1,

    @staticmethod
    def evalf(sizes):
        return numpy.cumsum([0, *sizes])

    def _simplified(self):
        unaligned, where = unalign(self.sizes)
        if not where:
            return Range(self.shape[0]) * appendaxes(unaligned, self.shape[:1])

    def _intbounds_impl(self):
        n = self.sizes.shape[0]._intbounds[1]
        m = self.sizes._intbounds[1]
        return 0, (0 if n == 0 or m == 0 else n * m)


class LoopConcatenate(Loop):

    func: Array
    start: Array
    stop: Array
    concat_length: Array

    def __post_init__(self):
        assert isinstance(self.func, Array), f'func={self.func}'
        assert _isindex(self.start), f'start={self.start}'
        assert _isindex(self.stop), f'stop={self.stop}'
        assert _isindex(self.concat_length), f'concat_length={self.concat_length}'
        if not self.func.ndim:
            raise ValueError('expected an array with at least one axis')
        super().__post_init__()

    @cached_property
    def shape(self):
        return *self.func.shape[:-1], self.concat_length

    @property
    def init_args(self):
        return self.shape

    @property
    def body_args(self):
        return self.func, self.start, self.stop

    @cached_property
    def dtype(self):
        return self.func.dtype

    @cached_property
    def shape(self):
        return *self.func.shape[:-1], self.concat_length

    def _compile(self, builder):
        out, out_block_id = builder.new_empty_array_for_evaluable(self)
        # `out_block_id` is always at the beginning the scope this evaluable
        # belongs to (out_block_id = (*builder.get_block_id(self)[:-1], 0), so
        # we're not reaching `NotImplemented` in `self._compile_with_out`.
        self._compile_with_out(builder, out, out_block_id, mode='assign')
        return out

    def _compile_with_out(self, builder, out, out_block_id, mode):
        if out_block_id > builder.get_block_id(self.index):
            # The loop body comes before the definition of `out`.
            return NotImplemented
        start, stop = builder.compile((self.start, self.stop))
        index_block_id = builder.get_block_id(self.index)
        body_block_id = builtins.max((index_block_id, *map(builder.get_block_id, (self.start, self.stop))))
        assert body_block_id[:-1] == index_block_id[:-1]
        out_slice = out.get_item(_pyast.Tuple((_pyast.Raw('...'), _pyast.Variable('slice').call(start, stop))))
        builder.compile_with_out(self.func, out_slice, body_block_id, mode)

    def _derivative(self, var, seen):
        return Transpose.from_end(loop_concatenate(Transpose.to_end(derivative(self.func, var, seen), self.ndim-1), self.index), self.ndim-1)

    def _node_loop_body(self, cache, subgraph, times, unique_loop_ids):
        if (cached := cache.get(self)) is not None:
            return cached
        kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times, unique_loop_ids) for i, n in enumerate(self.shape)}
        kwargs['start'] = self.start._node(cache, subgraph, times, unique_loop_ids)
        kwargs['stop'] = self.stop._node(cache, subgraph, times, unique_loop_ids)
        kwargs['func'] = self.func._node(cache, subgraph, times, unique_loop_ids)
        cache[self] = node = RegularNode(f'LoopConcatenate {self.loop_id}', (), kwargs, (type(self).__name__, times[self]), subgraph)
        return node

    def _simplified(self):
        if iszero(self.func):
            return zeros_like(self)
        elif self.index not in self.func.arguments:
            return Ravel(Transpose.from_end(InsertAxis(self.func, self.index.length), -2))
        unaligned, where = unalign(self.func)
        reinserted_unit = False
        if self.ndim-1 not in where:
            # reinsert concatenation axis, at unit length if possible so we can
            # insert the remainder outside of the loop
            n = self.func.shape[-1]
            if self.index not in n.arguments and not isunit(n):
                n = constant(1)
                reinserted_unit = True
            unaligned = InsertAxis(unaligned, n)
            where += self.ndim-1,
        elif where[-1] != self.ndim-1:
            # bring concatenation axis to the end
            axis = where.index(self.ndim-1)
            unaligned = Transpose.to_end(unaligned, axis)
            where = (*where[:axis], *where[axis+1:], self.ndim-1)
        f = loop_concatenate(unaligned, self.index)
        if reinserted_unit:
            # last axis was reinserted at unit length AND it was not unit length
            # originally - if it was unit length originally then we proceed only if
            # there are other insertions to promote, otherwise we'd get a recursion.
            f = Ravel(InsertAxis(f, self.func.shape[-1]))
        elif len(where) == self.ndim:
            return
        return align(f, where, self.shape)

    def _takediag(self, axis1, axis2):
        if axis1 < self.ndim-1 and axis2 < self.ndim-1:
            return Transpose.from_end(loop_concatenate(Transpose.to_end(_takediag(self.func, axis1, axis2), -2), self.index), -2)

    def _take(self, index, axis):
        if axis < self.ndim-1:
            return loop_concatenate(_take(self.func, index, axis), self.index)

    def _unravel(self, axis, shape):
        if axis < self.ndim-1:
            return loop_concatenate(unravel(self.func, axis, shape), self.index)

    @cached_property
    @verify_sparse_chunks
    def _assparse(self):
        chunks = []
        for *indices, last_index, values in self.func._assparse:
            last_index = last_index + prependaxes(self.start, last_index.shape)
            chunks.append(tuple(loop_concatenate(_flat(arr), self.index) for arr in (*indices, last_index, values)))
        return tuple(chunks)

    def _intbounds_impl(self):
        return self.func._intbounds

    def _argument_degree(self, argument):
        if argument not in self.start.arguments and argument not in self.stop.arguments and argument not in self.concat_length.arguments:
            return self.func.argument_degree(argument)


class SearchSorted(Array):
    '''Find index of evaluable array into sorted numpy array.'''

    arg: Array
    array: Array
    sorter: typing.Optional[Array]
    side: str

    dtype = int

    def __post_init__(self):
        assert isinstance(self.arg, Array), f'arg={self.arg!r}'
        assert isinstance(self.array, Array) and self.array.ndim == 1, f'array={self.array!r}'
        assert self.side in ('left', 'right'), f'side={self.side!r}'
        assert self.sorter is None or isinstance(self.sorter, Array) and self.sorter.dtype == int and not _any_certainly_different(self.sorter.shape, self.array.shape), f'sorter={self.sorter!r}'

    @property
    def dependencies(self):
        if self.sorter is None:
            return self.arg, self.array
        else:
            return self.arg, self.array, self.sorter

    @cached_property
    def shape(self):
        return self.arg.shape

    def _compile_expression(self, py_self, arg, array, sorter=None):
        opt_args = {}
        if sorter is not None:
            opt_args['sorter'] = sorter
        index = _pyast.Variable('numpy').get_attr('searchsorted').call(array, arg, side=_pyast.LiteralStr(self.side), **opt_args)
        # on some platforms (windows) searchsorted does not return indices as
        # numpy.dtype(int), so we type cast it for consistency
        return index.get_attr('astype').call(_pyast.Variable('int'), copy=_pyast.LiteralBool(False))

    def _intbounds_impl(self):
        return 0, self.array.shape[0]._intbounds[1]

    def _takediag(self, axis1, axis2):
        return SearchSorted(_takediag(self.arg, axis1, axis2), array=self.array, side=self.side, sorter=self.sorter)

    def _take(self, index, axis):
        return SearchSorted(_take(self.arg, index, axis), array=self.array, side=self.side, sorter=self.sorter)

    def _unravel(self, axis, shape):
        return SearchSorted(unravel(self.arg, axis, shape), array=self.array, side=self.side, sorter=self.sorter)


class ArgSort(Array):

    array: Array

    dtype = int

    def __post_init__(self):
        assert self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def dependencies(self):
        return self.array,

    def evalf(self, array):
        index = numpy.argsort(array, -1, kind='stable')
        # on some platforms (windows) argsort does not return indices as
        # numpy.dtype(int), so we type cast it for consistency
        return index.astype(int, copy=False)


class UniqueMask(Array):

    sorted_array: Array

    dtype = bool

    def __post_init__(self):
        assert self.sorted_array.ndim == 1

    @property
    def shape(self):
        return self.sorted_array.shape

    @property
    def dependencies(self):
        return self.sorted_array,

    def evalf(self, sorted_array):
        mask = numpy.empty(sorted_array.shape, dtype=bool)
        mask[:1] = True
        numpy.not_equal(sorted_array[1:], sorted_array[:-1], out=mask[1:])
        return mask


class UniqueInverse(Array):

    unique_mask: Array
    sorter: Array

    dtype = int

    def __post_init__(self):
        assert self.unique_mask.dtype == bool and self.unique_mask.ndim == 1
        assert self.sorter.dtype == int and self.sorter.ndim == 1
        assert not _any_certainly_different(self.unique_mask.shape, self.sorter.shape), (self.unique_mask, self.sorter)

    @property
    def shape(self):
        return self.sorter.shape

    @property
    def dependencies(self):
        return self.unique_mask, self.sorter

    def evalf(self, unique_mask, sorter):
        inverse = numpy.empty_like(sorter)
        inverse[sorter] = numpy.cumsum(unique_mask)
        inverse -= 1
        return inverse


def unique(array, return_index=False, return_inverse=False):
    sorter = ArgSort(array)
    mask = UniqueMask(Take(array, sorter))
    index = Take(sorter, Find(mask))
    unique = Take(array, index)
    inverse = UniqueInverse(mask, sorter)
    return (unique, index, inverse)[slice(0, 2+return_inverse, 2-return_index) if return_inverse or return_index else 0]


class CompressIndices(Array):

    indices: Array
    length: Array

    dtype = int
    ndim = 1

    def __post_init__(self):
        assert self.indices.dtype == int and self.indices.ndim == 1
        assert self.length.dtype == int and self.length.ndim == 0

    @property
    def shape(self):
        return self.length + 1,

    @property
    def dependencies(self):
        return self.indices, self.length

    @staticmethod
    def evalf(indices, length):
        return numeric.compress_indices(indices, length)


def as_csr(array):
    assert array.ndim == 2
    values, (rowidx, colidx), (nrows, ncols) = array.simplified.assparse
    return values, CompressIndices(rowidx, nrows), colidx, ncols


@util.shallow_replace
def zero_all_arguments(value):
    '''Replace all function arguments by zeros.'''

    if isinstance(value, Argument):
        return zeros_like(value)


class Monomial(Array):
    '''Helper object for factor.

    Performs a sparse tensor multiplication, without summation, and returns the
    result as a dense vector; inflation and reshape is the responsibility of
    factor. The factors of the multiplication are ``values`` and all the
    ``args``, which are scattered into values via the ``indices``. The
    ``powers`` argument contains multiplicities, for the following reason.

    With reference to the example of the factor doc string, derivative will
    generate an evaluable of the form array'(arg) darg = darray_darg(0) darg +
    .5 arg d2array_darg2 darg + .5 darg d2array_darg2 arg. By Schwarz's theorem
    d2array_darg2 is symmetric, and the latter two terms are equal. However,
    since the row and column indices of d2array_darg2 differ, we cannot detect
    this equality but rather need to embed the information explicitly.

    In this situation, the ``powers`` argument contains the value 2 to indicate
    that its position is symmetric with the next, and the first integral can
    therefore be doubled. With that, the derivative takes the desired form of
    array'(arg) == darray_darg(0) + d2array_darg2 arg.'''

    values: Array
    args: typing.Tuple[Array, ...]
    indices: typing.Tuple[typing.Tuple[Array, ...], ...]
    powers: typing.Tuple[int]

    def __post_init__(self):
        assert isinstance(self.values, Array) and self.values.ndim == 1, self.values.shape
        assert len(self.args) == len(self.indices) == len(self.powers)
        assert all(isinstance(index, Array) and index.dtype == int and not _any_certainly_different(index.shape, self.values.shape) for indices in self.indices for index in indices), (self.values.shape, self.indices)
        assert all(len(indices) == arg.ndim for arg, indices in zip(self.args, self.indices)), (len(self.indices), self.args)
        assert all(power == 1 or power > 1 and self.powers[i+1] == power - 1 and self.args[i+1] == self.args[i] for i, power in enumerate(self.powers)), self.powers

    def _simplified(self):
        if not self.args:
            return self.values

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    @cached_property
    def dependencies(self):
        return self.values, *self.args, *itertools.chain.from_iterable(self.indices)

    def _compile(self, builder):
        values = builder.compile(self.values)
        args = builder.compile(self.args)
        indices = builder.compile(self.indices)
        block = builder.get_block_for_evaluable(self)
        out = builder.get_variable_for_evaluable(self)
        block.assign_to(out, _pyast.Variable('numpy').get_attr('array').call(values, copy=_pyast.LiteralBool(True)))
        for arg, index in zip(args, indices):
            block.array_imul(out, arg.get_item(index))
        return out

    def _derivative(self, var, seen):
        if not iszero(derivative(self.values, var, seen)):
            raise NotImplementedError
        deriv = Zeros(self.shape + var.shape, self.dtype)
        iarg = 0
        while iarg < len(self.args):
            arg = self.args[iarg]
            m = Monomial(self.values,
                self.args[:iarg] + self.args[iarg+1:],
                self.indices[:iarg] + self.indices[iarg+1:],
                self.powers[:iarg] + self.powers[iarg+1:])
            if arg.ndim:
                *indices, ravel_index = self.indices[iarg]
                *lengths, ravel_length = arg.shape
                while indices:
                    ravel_index += indices.pop() * ravel_length
                    ravel_length *= lengths.pop()
                m = unravel(Inflate(Diagonalize(m), ravel_index, ravel_length), -1, arg.shape)
            m = einsum('aB,BC->aC', m, derivative(arg, var, seen))
            power = self.powers[iarg]
            if power > 1:
                m *= m.dtype(power)
            deriv += m
            iarg += power
        assert iarg == len(self.args)
        return deriv

    def _argument_degree(self, argument):
        return self.values.argument_degree(argument) + builtins.sum(arg.argument_degree(argument) for arg in self.args)


@log.withcontext
def factor(array):
    '''Convert array to a sparse polynomial.

    This function forms the equivalent polynomial of an evaluable array, of
    which the coefficients are already evaluated, with the aim to speed up
    repeated evaluations in e.g. time or Newton loops.

    For example, if ``array`` is a quadratic function of ``arg``, then the
    ``factor`` function returns the equivalent Taylor series::
    
        array(arg) -> array(0) + darray_darg(0) arg + .5 arg d2array_darg2 arg

    The coefficients ``array(0)``, ``darray_darg(0)`` and ``d2array_darg2``
    are evaluated as sparse tensors. As such, the new representation retains
    all sparsity of the original.'''

    array = array.as_evaluable_array.simplified
    log.info(f'analysing function of {", ".join(arg.name for arg in array.arguments) or "no arguments"}')

    # PREPARATION. We construct the equivalent polynomial to the input array,
    # parameterized by the m_coeffs (monomial coefficients) and m_args
    # (monomial arguments) lists. For every index i, m_coeffs[i] is an an
    # evaluable array of shape array.shape (the shape of the calling argument)
    # plus all the shapes of m_args[i], in order, where every element of
    # m_args[i] is an Argument object. The same argument may be repeated to
    # form higher powers. The polynomial (not formed) would result from
    # contracting the arguments in m_args[i] with the corresponding axes of
    # m_coeffs[i], and adding the resulting monomials.

    m_coeffs = []
    m_args = []
    queue = [((), array)]
    degree = {arg: array.argument_degree(arg) for arg in array.arguments}
    for args, func in queue:
        func = func.simplified
        zeroed = zero_all_arguments(func).simplified
        if not iszero(zeroed):
            m_args.append(args)
            m_coeffs.append(zeroed.assparse)
        for arg in func.arguments: # as m_args grows, fewer arguments will remain in func
            # We keep only m_args that are alphabetically ordered to
            # deduplicate permutations of the same argument set:
            if not args or arg.name >= args[-1].name:
                n = args.count(arg) + 1 # new monomial power of arg
                assert n <= degree[arg]
                queue.append(((*args, arg), derivative(func, arg) / float(n)))

    log.info(f'constructing sparse polynomial', ' + '.join(' '.join([f'C{i+1}'] +
        [f'{arg.name}^{n}' if n > 1 else arg.name for arg, n in collections.Counter(args).items()]) for i, args in enumerate(m_args)))

    # EVALUATION. We now form the polynomial, by accumulating evaluable
    # monomials in which the coefficients are evaluated. To this end we
    # evaluate the items of m_coeffs in sparse COO form, and contract each
    # result with the relevant arguments from m_args. The sparse contraction is
    # performed by first taking from the (flattened) argument the (raveled)
    # sparse indices, and contracting the resulting vector with the relevant
    # axis of the sparse values array.

    polynomial = []
    nvals = 0

    for args, (values, indices, shape) in log.iter.fraction('monomial', m_args, eval_once(m_coeffs)):
        if not values.all(): # prune zeros
            nz, = values.nonzero()
            values = values[nz]
            indices = [index[nz] for index in indices]

        log.info(f'{len(values):,} coefficients for {shape or "scalar"} array ({100*len(values)/numpy.prod(shape):.1f}% full)')

        if not len(values):
            continue

        indexmap = map(constant, indices[array.ndim:])
        monomial = Monomial(constant(values), args,
            indices=tuple(tuple(next(indexmap) for _ in range(arg.ndim)) for arg in args),
            powers=tuple(args[i:].count(arg) for i, arg in enumerate(args)))

        if array.ndim == 0:
            term = Sum(monomial)
        else:
            index = constant(numpy.ravel_multi_index(indices[:array.ndim], shape[:array.ndim]))
            size = constant(numpy.prod(shape[:array.ndim], dtype=int))
            term = unravel(Inflate(monomial, index, size), -1, array.shape)

        polynomial.append(term)
        nvals += len(values)

    log.info(f'factored function contains {nvals:,} coefficients ({nvals>>17:,}MB)')
    return util.sum(polynomial) if polynomial else zeros_like(array)


# AUXILIARY FUNCTIONS (FOR INTERNAL USE)


_ascending = lambda arg: numpy.greater(numpy.diff(arg), 0).all()


def _gatherblocks(blocks):
    return tuple((ind, util.sum(funcs)) for ind, funcs in util.gather(blocks))


def _gathersparsechunks(chunks):
    return tuple((*ind, util.sum(funcs)) for ind, funcs in util.gather((tuple(ind), func) for *ind, func in chunks))


def _numpy_align(a, b):
    '''check shape consistency and inflate scalars'''

    a = asarray(a)
    b = asarray(b)
    if not a.ndim:
        return _inflate_scalar(a, b.shape), b
    if not b.ndim:
        return a, _inflate_scalar(b, a.shape)
    if not _any_certainly_different(a.shape, b.shape):
        return a, b
    raise ValueError('incompatible shapes: {} != {}'.format(*[tuple(int(n) if n.isconstant else n for n in arg.shape) for arg in (a, b)]))


def _inflate_scalar(arg, shape):
    arg = asarray(arg)
    assert arg.ndim == 0
    for idim, length in enumerate(shape):
        arg = insertaxis(arg, idim, length)
    return arg


def _isunique(array):
    return numpy.unique(array).size == array.size


def _make_loop_ids_unique(funcs: typing.Tuple[Evaluable, ...]) -> typing.Tuple[Evaluable, ...]:
    # Replaces all `_LoopId` instances such that every distinct `Loop` has its
    # own loop id.

    loops = util.IDSet()
    for func in funcs:
        loops |= func._loops
    old_ids = {loop.loop_id for loop in loops}
    if len(old_ids) == len(loops):
        # All loops already have unique ids.
        return funcs

    new_ids = filter(lambda id: id not in old_ids, map(_LoopId, itertools.count()))
    root_cache = util.IDDict()

    @util.shallow_replace
    def replace_loop_ids(obj, *loop_caches):
        # For each loop in which `obj` is embeded, `loop_caches` lists the old
        # loop id and the loop cache, ordered from outermost to innermost. The
        # `root_cache` is used for `obj`s that are invariant to all outer loops,
        # if any.
        if not isinstance(obj, Evaluable):
            return
        if loop_caches:
            loop_ids = [arg.loop_id for arg in obj.arguments if isinstance(arg, _LoopIndex)]
            if len(loop_ids) != len(loop_caches):
                # Select a new persistent cache. Remove all outer loops from
                # `loop_caches` for which `obj` is invariant while maintaining the
                # order of the outer loops.
                loop_ids = set(loop_ids)
                loop_caches = [(loop_id, cache) for loop_id, cache in loop_caches if loop_id in loop_ids]
                cache = loop_caches[-1][1] if loop_caches else root_cache
                return replace_loop_ids(obj, *loop_caches, __persistent_cache__=cache)
        if isinstance(obj, Loop):
            assert not any(loop_id == obj.loop_id for loop_id, _ in loop_caches)
            cache = util.IDDict()
            cache[obj.loop_id] = next(new_ids)
            constructor, args = obj.__reduce__()
            new_args = replace_loop_ids(args, *loop_caches, (obj.loop_id, cache), __persistent_cache__=cache)
            return constructor(*new_args)

    return replace_loop_ids(funcs, __persistent_cache__=root_cache)


class _Stats:

    def __init__(self, ncalls: int = 0, time: int = 0) -> None:
        self.ncalls = ncalls
        self.time = time
        self._start = None

    def __repr__(self):
        return '_Stats(ncalls={}, time={})'.format(self.ncalls, self.time)

    def __add__(self, other):
        if not isinstance(other, _Stats):
            return NotImplemented
        return _Stats(self.ncalls+other.ncalls, self.time+other.time)

    def __enter__(self) -> None:
        self._start = time.perf_counter_ns()

    def __exit__(self, *exc_info) -> None:
        self.time += time.perf_counter_ns() - self._start
        self.ncalls += 1

# FUNCTIONS


def isarray(arg):
    return isinstance(arg, Array)


def _containsarray(arg):
    return any(map(_containsarray, arg)) if isinstance(arg, (list, tuple)) else isarray(arg)


def constant(v):
    return Constant(types.arraydata(v))


def iszero(arg):
    return isinstance(arg.simplified, Zeros)


def isunit(arg):
    simple = arg.simplified
    return isinstance(simple, Constant) and numpy.all(simple.value == 1)


def zeros(shape, dtype=float):
    return Zeros(shape, dtype)


def zeros_like(arr):
    return zeros(arr.shape, arr.dtype)


def ones(shape, dtype=float):
    return _inflate_scalar(constant(dtype(1)), shape)


def ones_like(arr):
    return ones(arr.shape, arr.dtype)


def singular_like(arr):
    return appendaxes(Singular(arr.dtype), arr.shape)


def reciprocal(arg):
    arg = asarray(arg)
    if arg.dtype in (bool, int):
        raise ValueError('The boolean or integer reciprocal is not supported.')
    return power(arg, astype(-1, arg.dtype))


def negative(arg):
    arg = asarray(arg)
    if arg.dtype == bool:
        raise ValueError('The boolean negative is not supported.')
    else:
        return multiply(arg, astype(-1, arg.dtype))


def sin(x):
    return Sin(x)


def cos(x):
    return Cos(x)


def tan(x):
    return Tan(x)


def arcsin(x):
    return ArcSin(x)


def arccos(x):
    return ArcCos(x)


def arctan(x):
    return ArcTan(x)


def sinc(x):
    return Sinc(x, n=0)


def exp(x):
    return Exp(x)


def ln(x):
    return Log(x)


def divmod(x, y):
    x, y = _numpy_align(x, y)
    return FloorDivide(x, y), Mod(x, y)


def mod(arg1, arg2):
    return Mod(*_numpy_align(arg1, arg2))


def log2(arg):
    arg = asarray(arg)
    return ln(arg) / astype(numpy.log(2), arg.dtype)


def log10(arg):
    arg = asarray(arg)
    return ln(arg) / astype(numpy.log(10), arg.dtype)


def sqrt(arg):
    arg = asarray(arg)
    if arg.dtype in (bool, int):
        raise ValueError('The boolean or integer square root is not supported.')
    return power(arg, astype(.5, arg.dtype))


def arctan2(arg1, arg2):
    return ArcTan2(*_numpy_align(arg1, arg2))


def abs(arg):
    if arg.dtype == complex:
        return sqrt(arg.real**2. + arg.imag**2.)
    else:
        return arg * sign(arg)


def sinh(arg):
    return SinH(arg)


def cosh(arg):
    return CosH(arg)


def tanh(arg):
    return TanH(arg)


def arctanh(arg):
    return ArcTanH(arg)


def divide(arg1, arg2):
    return multiply(arg1, reciprocal(arg2))


def subtract(arg1, arg2):
    return add(arg1, negative(arg2))


def insertaxis(arg, n, length):
    return Transpose.from_end(InsertAxis(arg, length), n)


def concatenate(args, axis=0):
    if len(args) == 1:
        return args[0]
    lengths = [arg.shape[axis] for arg in args]
    *offsets, totlength = util.cumsum(lengths + [0])
    return Transpose.from_end(util.sum(Inflate(Transpose.to_end(arg, axis), Range(length) + offset, totlength) for arg, length, offset in zip(args, lengths, offsets)), axis)


def stack(args, axis=0):
    return Transpose.from_end(util.sum(Inflate(arg, constant(i), constant(len(args))) for i, arg in enumerate(args)), axis)


def repeat(arg, length, axis):
    arg = asarray(arg)
    assert isunit(arg.shape[axis])
    return insertaxis(get(arg, axis, constant(0)), axis, length)


def get(arg, iax, item):
    if numeric.isint(item):
        if numeric.isint(arg.shape[iax]):
            item = numeric.normdim(arg.shape[iax], item)
        else:
            assert item >= 0
    return Take(Transpose.to_end(arg, iax), item)


def determinant(arg, axes=(-2, -1)):
    arg = asarray(arg)
    if arg.dtype == bool:
        raise ValueError('The boolean determinant is not supported.')
    if arg.dtype == int:
        arg = IntToFloat(arg)
    return Determinant(Transpose.to_end(arg, *axes))


def grammium(arg, axes=(-2, -1)):
    arg = Transpose.to_end(arg, *axes)
    grammium = einsum('Aki,Akj->Aij', arg, arg)
    return Transpose.from_end(grammium, *axes)


def sqrt_abs_det_gram(arg, axes=(-2, -1)):
    arg = Transpose.to_end(arg, *axes)
    if _certainly_equal(arg.shape[-1], arg.shape[-2]):
        return abs(determinant(arg))
    else:
        return sqrt(abs(determinant(grammium(arg))))


def inverse(arg, axes=(-2, -1)):
    arg = asarray(arg)
    if arg.dtype == bool:
        raise ValueError('The boolean inverse is not supported.')
    if arg.dtype == int:
        arg = IntToFloat(arg)
    return Transpose.from_end(Inverse(Transpose.to_end(arg, *axes)), *axes)


def takediag(arg, axis=-2, rmaxis=-1):
    arg = asarray(arg)
    axis = numeric.normdim(arg.ndim, axis)
    rmaxis = numeric.normdim(arg.ndim, rmaxis)
    assert axis != rmaxis
    return Transpose.from_end(_takediag(arg, axis, rmaxis), axis-(axis>=rmaxis))


def _takediag(arg, axis1=-2, axis2=-1):
    return TakeDiag(Transpose.to_end(arg, axis1, axis2))


def derivative(func, var, seen=None):
    'derivative'

    assert isinstance(var, DerivativeTargetBase), 'invalid derivative target {!r}'.format(var)
    if var.dtype in (bool, int) or var not in func.arguments:
        return Zeros(func.shape + var.shape, dtype=func.dtype)
    if seen is None:
        seen = {}
    if func in seen:
        result = seen[func]
    else:
        result = func._derivative(var, seen)
        seen[func] = result
    assert not _any_certainly_different(result.shape, func.shape+var.shape) and result.dtype == func.dtype, 'bug in {}._derivative'.format(type(func).__name__)
    return result


def diagonalize(arg, axis=-1, newaxis=-1):
    arg = asarray(arg)
    axis = numeric.normdim(arg.ndim, axis)
    newaxis = numeric.normdim(arg.ndim+1, newaxis)
    return Transpose.from_end(Diagonalize(Transpose.to_end(arg, axis)), axis + (axis>=newaxis), newaxis)


def sign(arg):
    arg = asarray(arg)
    if arg.dtype == complex:
        raise ValueError('sign is not defined for complex numbers')
    return Sign(arg)


def eig(arg, axes=(-2, -1), symmetric=False):
    eigval, eigvec = Eig(Transpose.to_end(arg, *axes), symmetric)
    return Tuple(tuple(Transpose.from_end(v, *axes) for v in [diagonalize(eigval), eigvec]))


def _takeslice(arg: Array, s: slice, axis: int):
    assert isinstance(arg, Array), f'arg={arg!r}'
    assert isinstance(s, slice), f's={s!r}'
    assert isinstance(axis, int), f'axis={axis!r}'
    n = arg.shape[axis]
    if s.step == None or s.step == 1:
        start = 0 if s.start is None else s.start if s.start >= 0 else s.start + n
        stop = n if s.stop is None else s.stop if s.stop >= 0 else s.stop + n
        if start == 0 and stop == n:
            return arg
        index = Range(asarray(stop-start)) + start
    elif n.isconstant:
        index = constant(numpy.arange(*s.indices(arg.shape[axis])))
    else:
        raise Exception('a non-unit slice requires a constant-length axis')
    return take(arg, index, axis)


def take(arg: Array, index: Array, axis: int):
    assert isinstance(arg, Array), f'arg={arg!r}'
    assert isinstance(index, Array) and index.dtype in (bool, int) and index.ndim == 1, f'index={index!r}'
    assert isinstance(axis, int), f'axis={axis!r}'
    length = arg.shape[axis]
    if index.dtype == bool:
        assert not _certainly_different(index.shape[0], length)
        index = Find(index)
    elif isinstance(index, Constant):
        index_ = index.value
        ineg = numpy.less(index_, 0)
        if not length.isconstant:
            if ineg.any():
                raise IndexError('negative indices only allowed for constant-length axes')
        elif ineg.any():
            if numpy.less(index_, -int(length)).any():
                raise IndexError('indices out of bounds: {} < {}'.format(index_, -int(length)))
            return _take(arg, constant(index_ + ineg * int(length)), axis)
        elif numpy.greater_equal(index_, int(length)).any():
            raise IndexError('indices out of bounds: {} >= {}'.format(index_, int(length)))
    return _take(arg, index, axis)


def _take(arg: Array, index: Array, axis: int):
    assert isinstance(arg, Array), f'arg={arg!r}'
    assert isinstance(index, Array) and index.dtype == int, f'index={index!r}'
    assert isinstance(axis, int), f'axis={axis!r}'
    axis = numeric.normdim(arg.ndim, axis)
    return Transpose.from_end(Take(Transpose.to_end(arg, axis), index), *range(axis, axis+index.ndim))


def _inflate(arg: Array, dofmap: Array, length: Array, axis: int):
    assert isinstance(arg, Array), f'arg={arg!r}'
    assert isinstance(dofmap, Array) and dofmap.dtype == int, f'dofmap={dofmap!r}'
    assert _isindex(length), f'length={length!r}'
    assert isinstance(axis, int), f'axis={axis!r}'
    axis = numeric.normdim(arg.ndim+1-dofmap.ndim, axis)
    assert not _any_certainly_different(dofmap.shape, arg.shape[axis:axis+dofmap.ndim])
    return Transpose.from_end(Inflate(Transpose.to_end(arg, *range(axis, axis+dofmap.ndim)), dofmap, length), axis)


def unravel(func, axis, shape):
    if not shape:
        raise ValueError('cannot unravel to an empty shape')
    func = asarray(func)
    axis = numeric.normdim(func.ndim, axis)
    if len(shape) == 1:
        assert not _certainly_different(func.shape[axis], shape[0])
        return func
    f = Transpose.to_end(func, axis)
    for i in range(len(shape)-1):
        f = Unravel(f, shape[i], util.product(shape[i+1:]))
    return Transpose.from_end(f, *range(axis, axis+len(shape)))


def ravel(func, axis):
    func = asarray(func)
    axis = numeric.normdim(func.ndim-1, axis)
    return Transpose.from_end(Ravel(Transpose.to_end(func, axis, axis+1)), axis)


def _flat(func, ndim=1):
    func = asarray(func)
    if func.ndim == ndim-1:
        return InsertAxis(func, constant(1))
    while func.ndim > ndim:
        func = Ravel(func)
    return func


def prependaxes(func, shape):
    'Prepend axes with specified `shape` to `func`.'

    func = asarray(func)
    for i, n in enumerate(shape):
        func = insertaxis(func, i, n)
    return func


def appendaxes(func, shape):
    'Append axes with specified `shape` to `func`.'

    func = asarray(func)
    for n in shape:
        func = InsertAxis(func, n)
    return func


def loop_index(name, length):
    if not isinstance(name, str):
        return ValueError('`name` must be a `str` but got `{name!r}`')
    return _LoopIndex(_LoopId(name), asarray(length))


def loop_sum(func, index):
    func = asarray(func)
    if not isinstance(index, _LoopIndex):
        raise TypeError(f'expected _LoopIndex, got {index!r}')
    return LoopSum(index.loop_id, index.length, func, func.shape)


def loop_concatenate(func, index):
    func = asarray(func)
    if not isinstance(index, _LoopIndex):
        raise TypeError(f'expected _LoopIndex, got {index!r}')
    chunk_size = func.shape[-1]
    if chunk_size.isconstant:
        chunk_sizes = InsertAxis(chunk_size, index.length)
    else:
        chunk_sizes = loop_concatenate(InsertAxis(func.shape[-1], constant(1)), index)
    offsets = _SizesToOffsets(chunk_sizes)
    start = Take(offsets, index)
    stop = Take(offsets, index+1)
    concat_length = Take(offsets, index.length)
    return LoopConcatenate(index.loop_id, index.length, func, start, stop, concat_length)


@util.shallow_replace
def replace_arguments(value, arguments):
    '''Replace :class:`Argument` objects in ``value``.

    Replace :class:`Argument` objects in ``value`` according to the ``arguments``
    map, taking into account derivatives to the local coordinates.

    Args
    ----
    value : :class:`Array`
        Array to be edited.
    arguments : :class:`collections.abc.Mapping` with :class:`Array`\\s as values
        :class:`Argument`\\s replacements.  The key correspond to the ``name``
        passed to an :class:`Argument` and the value is the replacement.

    Returns
    -------
    :class:`Array`
        The edited ``value``.
    '''
    if isinstance(value, Argument) and value.name in arguments:
        v = asarray(arguments[value.name])
        assert not _any_certainly_different(value.shape, v.shape), (value.shape, v.shape)
        assert value.dtype == v.dtype, (value.dtype, v.dtype)
        return v


def einsum(fmt, *args, **dims):
    '''Multiply and/or contract arrays via format string.

    The format string consists of a comma separated list of axis labels, followed
    by ``->`` and the axis labels of the return value. For example, the following
    swaps the axes of a matrix:

    >>> a45 = ones(tuple(map(constant, [4,5]))) # 4x5 matrix
    >>> einsum('ij->ji', a45)
    nutils.evaluable.Transpose<f:(5),(4)>

    Axis labels that do not occur in the return value are summed. For example,
    the following performs a matrix-vector product:

    >>> a5 = ones(tuple(map(constant, [5]))) # vector with length 5
    >>> einsum('ij,j->i', a45, a5)
    nutils.evaluable.Sum<f:4>

    The following contracts a third order tensor, a matrix, and a vector, and
    transposes the result:

    >>> a234 = ones(tuple(map(constant, [2,3,4]))) # 2x3x4 tensor
    >>> einsum('ijk,kl,l->ji', a234, a45, a5)
    nutils.evaluable.Sum<f:3,2>

    In case the dimension of the input and output arrays may vary, a variable
    length axes group can be denoted by a capital. Its length is automatically
    established based on the dimension of the input arrays. The following example
    performs a tensor product of an array and a vector:

    >>> einsum('A,i->Ai', a234, a5)
    nutils.evaluable.Multiply<f:2,3,4,5>

    The format string may contain multiple variable length axes groups, but their
    lengths must be resolvable from left to right. In case this is not possible,
    lengths may be specified as keyword arguments.

    >>> einsum('AjB,i->AijB', a234, a5, B=1)
    nutils.evaluable.Multiply<f:2,5,3,4>
    '''

    if not all(isinstance(arg, Array) for arg in args):
        raise ValueError('arguments must be Array valued')

    sin, sout = fmt.split('->')
    sin = sin.split(',')

    if len(sin) != len(args):
        raise ValueError('number of arguments does not match format string')

    if any(len(s) != len(set(s)) for s in (*sin, sout)):
        raise ValueError('internal repetitions are not supported')

    if any(n < 0 for n in dims.values()):
        raise ValueError('axis group dimensions cannot be negative')

    for c in 'abcdefghijklmnopqrstuvwxyz':
        dims.setdefault(c, 1)  # lowercase characters default to single dimension

    for s, arg in zip(sin, args):
        missing_dims = arg.ndim - builtins.sum(dims.get(c, 0) for c in s)
        unknown_axes = [c for c in s if c not in dims]
        if len(unknown_axes) == 1 and missing_dims >= 0:
            dims[unknown_axes[0]] = missing_dims
        elif len(unknown_axes) > 1:
            raise ValueError('cannot establish length of variable groups {}'.format(', '.join(unknown_axes)))
        elif missing_dims:
            raise ValueError('argument dimensions are inconsistent with format string')

    # expand characters to match argument dimension
    *sin, sout = [[(c, d) for c in s for d in range(dims[c])] for s in (*sin, sout)]
    sall = sout + sorted({c for s in sin for c in s if c not in sout})

    shapes = {}
    for s, arg in zip(sin, args):
        assert len(s) == arg.ndim
        for c, sh in zip(s, arg.shape):
            if _certainly_different(shapes.setdefault(c, sh), sh):
                raise ValueError('shapes do not match for axis {0[0]}{0[1]}'.format(c))

    ret = None
    for s, arg in zip(sin, args):
        index = {c: i for i, c in enumerate(s)}
        for c in sall:
            if c not in index:
                index[c] = arg.ndim
                arg = InsertAxis(arg, shapes[c])
        v = transpose(arg, tuple(index[c] for c in sall))
        ret = v if ret is None else ret * v
    for i in range(len(sout), len(sall)):
        ret = Sum(ret)
    return ret


def eval_once(func: AsEvaluableArray, *, stats: typing.Optional[str] = None, arguments: typing.Mapping[str, numpy.ndarray] = {}, _simplify: bool = True, _optimize: bool = True) -> typing.Tuple[numpy.ndarray, ...]:
    '''Evaluate one or several Array objects by compiling it for single use.

    Args
    ----
    func : :class:`Evaluable` or (possibly nested) tuples of :class:`Evaluable`\\s
        The function or functions to compile.
    stats : ``'log'`` or ``None``
        If ``'log'`` the compiled function will log the durations of individual
        :class:`Evaluable`\\s referenced by ``func``.
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.

    Returns
    -------
    results : :class:`tuple` of (values, indices, shape) triplets
    '''

    f = compile(func, _simplify=_simplify, _optimize=_optimize, stats=stats, cache_const_intermediates=False)
    return f(**arguments)


@log.withcontext
def compile(func, /, *, stats: typing.Optional[str] = None, cache_const_intermediates: bool = True, _simplify: bool = True, _optimize: bool = True):
    '''Returns a callable that evaluates ``func``.

    The return value of the callable matches the structure of ``func``. If
    ``func`` is a single :class:`Evaluable`, the returned callable returns a
    single array. If ``func`` is a tuple, the returned callable returns a tuple
    as well.

    If ``func`` depends on :class:`Argument`\\s, their values must be passed to
    the callable using keywords arguments. Arguments that are unknown to
    ``func`` are silently ignored.

    Args
    ----
    func : :class:`Evaluable` or (possibly nested) tuples of :class:`Evaluable`\\s
        The function or functions to compile.
    stats : ``'log'`` or ``None``
        If ``'log'`` the compiled function will log the durations of individual
        :class:`Evaluable`\\s referenced by ``func``.
    cache_const_intermediates : :class:`bool`
        If true, the returned callable caches parts of ``func`` that can be
        reused for a second call.

    Returns
    -------
    :any:`callable`
        The callable that evaluates ``func``.

    Examples
    --------

    Compiling a single function with argument ``arg``.

    >>> arg = Argument('arg', (), int)
    >>> f = arg + constant(1)
    >>> compiled = compile(f)
    >>> compiled(arg=1)
    2

    The return value of the compiled function matches the structure of the
    functions passed to :func:`compile`:

    >>> g = arg * constant(3)
    >>> h = arg * arg
    >>> compiled = compile((f, (g, h)))
    >>> compiled(arg=2)
    (3, (6, 4))
    '''

    # Compiles `Evaluable`s to Python code. Every `Loop` is assigned a unique
    # loop id with `_define_loop_block_structure` and based on these loop ids
    # every `Evaluable` defines a block id where the Python code for that
    # evaluable should be positioned.
    #
    # The loop and block ids are multi-indices, tuples of ints, defined such
    # that the loop id of a child loop always starts with the loop id of the
    # parent loop and such that if some loop B depends on some loop A, both
    # having the same parent loop (or no parent loop), the id of A is smaller
    # than the id of B. Finally, two loops in the same parent loop are assigned
    # the same id if they have the same length and do not depend on each other.
    #
    # The block ids are derived from the loop ids. If some evaluable depends on
    # one or more loop indices, the block id of that evaluable starts with the
    # loop id of the innermost loop and has one extra index that does not refer
    # to a loop. The block id of a loop is the loop id with the rightmost index
    # incremented by one. If an `Evaluable` does not depend on loop indices or
    # loops, the block id is `(0,)`.
    #
    # The following two examples illustrate the definition of the loop ids from
    # the original loop names. The first example has two nested loops, because
    # the inner loop depends on both the index for the inner loop `j` and the
    # outer loop `i`. The outer loop, `e0`, gets loop id `(0,)`, the inner
    # loop, `e1`, id `(0,0)`:
    #
    #                            loop_id      block_id
    #     e0: LoopSum            'i'->(0,)    (1,)
    #     └ e1: LoopSum          'j'->(0,0)   (0,1)
    #       └ e2: Add                         (0,0,0)
    #         ├ e3: LoopIndex    'j'->(0,0)   (0,0,0)
    #         │ └ e4: Constant                (0,)
    #         └ e5: LoopIndex    'i'->(0,)    (0,0)
    #           └ e6: Constant                (0,)
    #
    # Note that `e4` and `e6` are the lengths of the loops. In the second
    # example loop `e9` does not depend on loop index `i`, but is a dependency
    # of loop `e7`. Therefor, `e9` gets loop id `(0,)` and `e7` id `(1,)`
    # (adjacent to `e9`):
    #
    #                            loop_id      block_id
    #     e7: LoopSum            'i'->(1,)    (2,)
    #     └ e8: Add                           (1,0)
    #       ├ e9: LoopSum        'j'->(0,)    (1,)
    #       │ └ e10: LoopIndex   'j'->(0,)    (0,0)
    #       │   └ e4: Constant                (0,)
    #       └ e11: LoopIndex     'i'->(1,)    (1,0)
    #         └ e6: Constant                  (0,)
    #
    # Having defined the loop and block ids, the loops (without content) are
    # serialized. For the first example this gives
    #
    #     def compiled(**a):
    #         # contents of blocks[0,]
    #         for v5 in range(c6):
    #             # contents of blocks[0,0]
    #             for v3 in range(c4):
    #                 # contents of blocks[0,0,0]
    #             # contents of blocks[0,1]
    #         # contents of blocks[1,]
    #         return v0
    #
    # The final step is the serialization of the `Evaluable`s. Each `Evaluable`
    # compiles first its dependencies, then itself, placing the statements in
    # the block with the corresponding block id. The serialization of the first
    # example (without inplace additions):
    #
    #     def compiled(**a):
    #         v0 = numpy.zeros((), int) # e0 init
    #         for v5 in range(c6):
    #             v1 = numpy.zeros((), int) # e1 init
    #             for v3 in range(c4):
    #                 v2 = v3 + v5
    #                 v1 += v2 # e1 update
    #             # e1 exit
    #             v0 += v1 # e0 update
    #         # e0 exit
    #         return v0

    if stats is None:
        stats = 'log' if graphviz else False
    elif stats not in ('log', False):
        raise ValueError(f'`stats` must be `None`, `False` or `"log"` but got {stats!r}')

    # Build return value format string `ret` with the same structure as `func`
    # and convert `func` to a flat list.
    stack = [func]
    ret_fmt = []
    funcs = []
    MakeTuple = collections.namedtuple('MakeTuple', ('n'))
    while stack:
        obj = stack.pop()
        if isinstance(obj, MakeTuple):
            m = len(ret_fmt) - obj.n
            ret_fmt[m:] = ['(' + ', '.join(ret_fmt[m:]) + (',)' if obj.n == 1 else ')')]
        elif isinstance(obj, (tuple, list)):
            stack.append(MakeTuple(len(obj)))
            stack.extend(reversed(obj))
        elif isinstance(obj, Array) or isinstance(obj, Evaluable) and not cache_const_intermediates:
            funcs.append(obj)
            ret_fmt.append('{}')
        else:
            raise ValueError(f'expected a `nutils.evaluable.Array`, `tuple` or `list` but got {obj!r}')
    ret_fmt, = ret_fmt

    # Simplify and optimize `funcs`.
    if _simplify:
        funcs = [func.simplified for func in funcs]
    if _optimize:
        funcs = [func._optimized_for_numpy1 for func in funcs]
    funcs = _define_loop_block_structure(tuple(funcs))
    assert not any(isinstance(arg, _LoopIndex) for func in funcs for arg in func.arguments)

    # The globals of the compiled function.
    from . import evaluable
    globals = dict(
        collections=collections,
        first_run=True,
        log_stats=_log_stats,
        multiprocessing=multiprocessing,
        evaluable=evaluable,
        numeric=numeric,
        numpy=numpy,
        parallel=parallel,
        poly=poly,
        ret_tuple=Tuple(funcs),
        Stats=_Stats,
        treelog=log,
        warnings=warnings,
    )
    # Counter for generating unique indices, e.g. for creating variables.
    new_index = itertools.count()
    # Dict of encountered evaluables and their assigned index.
    evaluables = {}
    # Cache of compiled evaluables. Maps `Evaluable` to `_pyast.Expression`,
    # but mostly `_pyast.Variable`.
    cache = {}
    # Dict of evaluable (`Evaluable`) to list of compiled blocks
    # (`_pyast.Block`). Blocks for evaluables compiled using
    # `_compile_with_out` are added to the origin, the evaluable that initiates
    # inplace compilation.
    evaluable_block_map = {}
    # Dict of evaluable (`Evaluable`) to set (`util.IDSet`) of dependencies
    # (`Evaluable`). This is mostly equal to `Evaluable.dependencies`, but in case of
    # inplace compilation the evaluables the dependencies that are compiled
    # inplace are omitted and their dependencies are added to the origin.
    evaluable_deps = {None: util.IDSet()}

    # Count the number of dependents for each `Evaluable` in `funcs`. This is
    # used by the builder to decide if an `Evaluable` can be compiled with
    # `_compile_with_out(..., mode='iadd')`.
    ndependents = collections.defaultdict(lambda: 0)
    def update_ndependents(func):
        for arg in func.dependencies:
            ndependents[arg] += 1
        return func.dependencies
    util.tree_walk(update_ndependents, *funcs)

    # Collect the loop ids and lengths from all loops in `funcs` and assert
    # that all loops with the same id also have the same length. `loop_lengths`
    # maps id to `Evaluable` length. Also populate the `Evaluable` to block id
    # map, `evaluable_block_ids`, with the loops and loop indices. Create
    # blocks for each loop body and trailer.
    blocks = {(0,): _pyast.Block()}
    loop_length_index = {}
    evaluable_block_ids = {}
    for loop in util.IDSet(loop for func in funcs for loop in func._loops):
        loop_id = loop.loop_id.id
        assert isinstance(loop_id, tuple) and loop_id and all(isinstance(n, int) for n in loop_id)
        evaluable_block_ids[loop] = *loop_id[:-1], loop_id[-1] + 1
        if (prev := loop_length_index.get(loop_id)) is not None:
            length, _ = prev
            if length != loop.length:
                raise ValueError(f'multiple loops with the same id ({loop_id}) but different lengths')
            continue
        evaluable_block_ids[loop.index] = *loop_id, 0
        # Reserve a variable for the loop index.
        cache[loop.index] = py_index = _pyast.Variable('i' + '_'.join(map(str, loop_id)))
        loop_length_index[loop_id] = loop.length, py_index
        # Create blocks for the loop body and the loop trailer.
        blocks[(*loop_id, 0)] = _pyast.Block()
        blocks[(*loop_id[:-1], loop_id[-1] + 1)] = _pyast.Block()

    compile_parallel = parallel.maxprocs.current > 1 and any(loop_length_index) and not stats

    builder = _BlockTreeBuilder(blocks, evaluable_block_ids, globals, evaluables, new_index, cache, stats, compile_parallel, ndependents, evaluable_block_map, evaluable_deps)

    # Compile `funcs`.
    py_funcs = builder.compile(funcs)

    # Generate loops and merge loop blocks and trailing blocks into preceding
    # blocks, starting with the inner-most loops which have no succeeding
    # loops, all the way down to block `(0,)`.
    for loop_id, (length, py_loop_index) in sorted(loop_length_index.items(), key=lambda kv: (len(kv[0]), kv[0][-1]), reverse=True):
        py_length = builder.compile(length)
        body = blocks.pop((*loop_id, 0))
        loop_name = _pyast.LiteralStr('loop {}'.format(','.join(map(str, loop_id))))
        py_range = builder.new_var()
        py_range_numpy_int = _pyast.Variable('map').call(_pyast.Variable('numpy').get_attr('int_'), py_range)
        loop_block = _pyast.ForLoop(py_loop_index, py_range_numpy_int, body)
        if len(loop_id) == 1 and compile_parallel:
            iter_context = _pyast.Variable('parallel').get_attr('ctxrange').call(loop_name, py_length)
        else:
            iter_context = _pyast.Variable('treelog').get_attr('iter').get_attr('percentage').call(loop_name, _pyast.Variable('range').call(py_length))
        loop_block = _pyast.With(iter_context, as_=py_range, body=loop_block, omit_if_body_is_empty=True)
        blocks[loop_id].append(loop_block)
        blocks[loop_id].append(blocks.pop((*loop_id[:-1], loop_id[-1] + 1)))
    main = blocks.pop((0,))
    assert not blocks

    if cache_const_intermediates:
        # Collect all evaluables that are to be recomputed on a rerun in
        # `rerun_evaluables` and collect all direct dependencies of the
        # `rerun_evaluables` that are constant in `cache_evaluables`. The
        # `cache_evaluable` are not recursed into.
        rerun_evaluables = util.IDSet()
        cache_evaluables = util.IDSet()
        def collect(evaluable):
            if isinstance(evaluable, Array) and evaluable.isconstant:
                cache_evaluables.add(evaluable)
                return ()
            else:
                rerun_evaluables.add(evaluable)
                return evaluable_deps.get(evaluable, ())
        util.tree_walk(collect, *evaluable_deps[None])
        cache_vars = tuple(cache[evaluable] for evaluable in cache_evaluables)
        # Find all blocks that are to be omitted for a rerun: the blocks of all
        # compiled evaluables that are not part of `rerun_evaluables`.
        rerun_skip_evaluables = util.IDSet(evaluable_block_map) - rerun_evaluables
        rerun_skip_blocks = util.IDSet(itertools.chain.from_iterable(evaluable_block_map[e] for e in rerun_skip_evaluables))
        # Filter the blocks to be skipped from `main` for the rerun.
        main_rerun = main.filter(lambda stmts: _pyast.Block() if stmts in rerun_skip_blocks else None)
        first_run = _pyast.Variable('first_run')
        # Make all cached results immutable.
        for v in cache_vars:
            main.append(_pyast.Exec(v.get_attr('setflags').call(write=_pyast.LiteralBool(False))))
        # Combine `main` (for the first run) and `main_rerun` into `main`.
        main.append(_pyast.Assign(first_run, _pyast.LiteralBool(False)))
        main = _pyast.Block([
            _pyast.Global((first_run,) + cache_vars),
            _pyast.If(first_run, main, main_rerun),
        ])

    if compile_parallel:
        main = _pyast.Block([
            _pyast.Assign(_pyast.Variable('lock'), _pyast.Variable('multiprocessing').get_attr('Lock').call()),
            main,
        ])

    if stats == 'log':
        main = _pyast.Block([
            _pyast.Assign(_pyast.Variable('stats'), _pyast.Variable('collections').get_attr('defaultdict').call(_pyast.Variable('Stats'))),
            main,
            _pyast.Exec(_pyast.Variable('log_stats').call(_pyast.Variable('ret_tuple'), _pyast.Variable('stats'))),
        ])

    script = StringIO()
    print('def compiled(**a):', file=script)
    for line in main.lines:
        print('    ' + line, file=script)
    print('    return ' + ret_fmt.format(*[v.py_expr for v in py_funcs]), file=script)
    script = script.getvalue()
    script_hash = hashlib.sha1(script.encode('utf-8')).digest()

    name = f'compiled_{script_hash.hex()}'

    if debug_flags.compile:
        print(script)

    # Make sure we can see `script` in tracebacks and `pdb`.
    # From: https://stackoverflow.com/a/39625821
    linecache.cache[name] = (len(script), None, [line+'\n' for line in script.splitlines()], name)

    # Compile.
    eval(builtins.compile(script, name, 'exec'), globals)
    compiled = globals['compiled']
    compiled.__nutils_hash__ = script_hash
    return compiled


def _define_loop_block_structure(targets: typing.Tuple[Evaluable, ...]) -> typing.Tuple[Evaluable, ...]:
    # To aid the serialization of the `targets`, this function replaces the
    # existing loop ids of `Loop` subclasses with unique ids, such that
    # every sibling `Evaluable` of the `targets` can be related to exactly
    # one loop (possibly nested in other loops).
    #
    # The new ids are multi-indices, tuples of ints, defined such that the
    # id of a child loop always starts with the id of the parent loop and
    # such that if some loop B depends on some loop A, both having the same
    # parent loop (or no parent loop), the id of A is smaller than the id
    # of B. Finally, two loops in the same parent loop are assigned the
    # same id if they have the same length and do not depend on each other.

    unique_targets = _make_loop_ids_unique(targets)

    queue = util.IDSet()
    for target in unique_targets:
        queue.update(target._loops)
    nloops = len(queue)

    def collect(indices):
        # Find all adjacent loops and form groups of loops that have the same
        # length and can be evaluated simultaneously. For each group the nested
        # loops are collected. After reversal, `groups` lists the groups and
        # the nested groups in the order at which they must be evaluated.
        groups = []
        while True:
            group = []
            for loop in queue:
                if indices and indices.isdisjoint(loop.arguments) or group and loop.length != group[0].length:
                    continue
                if not any(loop in other._loops for other in queue if other is not loop):
                    group.append(loop)
            if not group:
                break
            queue.difference_update(group)
            nested = collect(frozenset(loop.index for loop in group))
            groups.append((group, nested))
        groups.reverse()
        return groups

    groups = collect(frozenset())
    assert not queue

    # Assign new ids.
    def build_id_map(groups, parent_id):
        for i, (loops, nested) in enumerate(groups):
            id = *parent_id, i
            for loop in loops:
                id_map[loop.loop_id] = _LoopId(id)
            build_id_map(nested, id)

    id_map = util.IDDict()
    build_id_map(groups, ())
    return tuple(util.shallow_replace(id_map.get, target) for target in unique_targets)


def _log_stats(func, stats):
    node = func._node({}, None, stats, True)
    maxtime = builtins.max(n.metadata[1].time for n in node.walk(set()))
    tottime = builtins.sum(n.metadata[1].time for n in node.walk(set()))
    aggstats = tuple((key, builtins.sum(v.time for v in values), builtins.sum(v.ncalls for v in values)) for key, values in util.gather(n.metadata for n in node.walk(set())))
    fill_color = (lambda node: '0,{:.2f},1'.format(node.metadata[1].time/maxtime) if node.metadata[1].ncalls else None) if maxtime else None
    if graphviz:
        node.export_graphviz(fill_color=fill_color, dot_path=graphviz)
    log.info('total time: {:.0f}ms\n'.format(tottime/1e6) + '\n'.join('{:4.0f} {} ({} calls, avg {:.3f} per call)'.format(t / 1e6, k, n, t / (1e6*n))
                                                                      for k, t, n in sorted(aggstats, reverse=True, key=lambda item: item[1]) if n))


class _BlockTreeBuilder:

    def __init__(
        self,
        blocks: typing.Dict[_BlockId, _pyast.Block],
        evaluable_block_ids: typing.Dict[Evaluable, _BlockId],
        globals: typing.Dict[str, typing.Any],
        evaluables: typing.Dict[Evaluable, int],
        new_index: typing.Iterator[int],
        compiled_cache: dict,
        stats: bool,
        parallel: bool,
        ndependents: typing.Mapping[Evaluable, int],
        evaluable_block_map: typing.Mapping[Evaluable, typing.List[_pyast.Block]],
        evaluable_deps: typing.Mapping[typing.Optional[Evaluable], typing.Set[Evaluable]],
        origin: typing.Optional[Evaluable] = None,
    ):
        self._blocks = blocks
        self._evaluable_block_ids = evaluable_block_ids
        self._globals = globals
        self._evaluables = evaluables
        self._new_index = new_index
        self._compiled_cache = compiled_cache
        self._stats = stats
        self._parallel = parallel
        self.ndependents = ndependents
        self._evaluable_block_map = evaluable_block_map
        self._evaluable_deps = evaluable_deps
        self._origin = origin
        self._shared_arrays = set()
        self._new_eid = map('e{}'.format, new_index).__next__
        self.new_var = lambda: _pyast.Variable('v{}'.format(next(new_index)))

    def compile(self, evaluable: Evaluable) -> _pyast.Expression:
        # Compiles the tree defined by `evaluable` and returns the result of `evaluable`.
        if isinstance(evaluable, tuple):
            return _pyast.Tuple(tuple(map(self.compile, evaluable)))
        self._evaluable_deps.setdefault(self._origin, util.IDSet()).add(evaluable)
        if (out := self._compiled_cache.get(evaluable)) is None:
            evaluable_builder = _BlockTreeBuilder(
                self._blocks,
                self._evaluable_block_ids,
                self._globals,
                self._evaluables,
                self._new_index,
                self._compiled_cache,
                self._stats,
                self._parallel,
                self.ndependents,
                self._evaluable_block_map,
                self._evaluable_deps,
                evaluable)
            self._compiled_cache[evaluable] = out = evaluable._compile(evaluable_builder)
            if debug_flags.evalf and isinstance(evaluable, Array):
                block = self.get_block(self.get_block_id(evaluable))
                block.assert_equal(out.get_attr('dtype').get_attr('kind'), _pyast.LiteralStr(_array_dtype_to_kind[evaluable.dtype]))
                block.assert_equal(out.get_attr('ndim'), _pyast.LiteralInt(evaluable.ndim))
                for i, n in enumerate(evaluable.shape):
                    if isinstance(n, Constant):
                        block.assert_equal(out.get_attr('shape').get_item(_pyast.LiteralInt(i)), _pyast.LiteralInt(n.__index__()))
        return out

    def compile_with_out(self, evaluable: Evaluable, out: _pyast.Expression, out_block_id: _BlockId, mode: str) -> None:
        self._get_evaluable_index(evaluable)
        evaluable_block_id = self.get_block_id(evaluable)
        if self.ndependents[evaluable] > 1 or evaluable_block_id < out_block_id or evaluable._compile_with_out(self, out, out_block_id, mode) is NotImplemented:
            value = self.compile(evaluable)
            block = self.get_block_for_evaluable(evaluable, block_id=builtins.max(evaluable_block_id, out_block_id))
            if mode == 'assign':
                block.array_copy(out, value)
            elif mode == 'iadd':
                block.array_iadd(out, value)
            else:
                raise ValueError(f'invalid mode: {mode}')

    def new_empty_array_for_evaluable(self, array: Array) -> _pyast.Variable:
        # Creates a new empty array and assigns the array to variable `v{id}`
        # where `id` is the unique index of `array`.
        shape = self.compile(array.shape)
        out = self.get_variable_for_evaluable(array)
        # Allocation of `out` should happen as early as possible (that is: as
        # soon as the shape is known). `out`, however, must not be used outside
        # the loop where `array` resides, if any. This is to prevent compiling
        # a dependency of `array` that lives outside that loop using
        # `_compile_with_out`.
        alloc_block_id = builtins.max(map(self.get_block_id, array.shape)) if array.ndim else (0,)
        out_block_id = builtins.max((*self.get_block_id(array)[:-1], 0), alloc_block_id)
        if self._parallel and len(out_block_id) == 1:
            # Because `self._parallel` is true, the outer-most loops will be
            # parallelized using fork. If `array` is evaluated inside a loop,
            # `out` must not be a shared array. However, if `array` originates
            # outside an outer loop, which is the case in this if-branch, then
            # `array` might initiate an inplace add inside a loop, in which
            # case `out` must be a shared memory array or we'll only see
            # updates to `out` from the parent process.
            self._shared_arrays.add(out)
            py_alloc = _pyast.Variable('parallel').get_attr('shempty')
        else:
            py_alloc = _pyast.Variable('numpy').get_attr('empty')
        alloc_block = self.get_block_for_evaluable(array, block_id=alloc_block_id, comment='alloc')
        alloc_block.assign_to(out, py_alloc.call(shape, dtype=_pyast.Variable(array.dtype.__name__)))
        return out, out_block_id

    def add_constant(self, value) -> _pyast.Variable:
        # Assigns `value` to constant `cx{nutils_hash(value)}`.
        const = _pyast.Variable('c{}'.format(types.nutils_hash(value).hex()))
        old_value = self._globals.setdefault(const.py_expr, value)
        assert old_value is value or type(old_value) is type(value) and old_value == value
        return const

    def get_argument(self, name: str) -> _pyast.Variable:
        # Returns the argument with the given `name`.
        return _pyast.Variable('a').get_item(_pyast.LiteralStr(name))

    def get_block(self, block_id: _BlockId) -> '_BlockBuilder':
        # Returns a block builder for the given block id.
        block = _pyast.Block()
        self._blocks[block_id].append(block)
        if self._origin:
            self._evaluable_block_map.setdefault(self._origin, []).append(block)
        return _BlockBuilder(self, block)

    def get_block_for_evaluable(self, evaluable: Evaluable, *, block_id: typing.Optional[_BlockId] = None, comment: str = '') -> '_BlockBuilder':
        # Appends a comment identifying `evaluable` to the block with the given
        # id, or `self.get_block_id(evaluable)` if absent, optionally suffixed
        # with `comment`, and returns the block.
        eid = 'e{}'.format(self._get_evaluable_index(evaluable))
        if block_id is None:
            block_id = self.get_block_id(evaluable)
        block = _pyast.Block()
        block_builder = _BlockBuilder(self, block)

        description = f'{type(evaluable).__name__} {eid}'
        if self._origin and evaluable is not self._origin:
            description = f'{description}, origin: {type(self._origin).__name__} e{self._get_evaluable_index(self._origin)}'
        if comment:
            description  = f'{description}; {comment}'
        block = _pyast.CommentBlock(description, block)

        if self._stats:
            block = _pyast.With(_pyast.Variable('stats').get_item(_pyast.Variable(eid)), block, omit_if_body_is_empty=True)

        self._blocks[block_id].append(block)
        if self._origin:
            self._evaluable_block_map.setdefault(self._origin, []).append(block)
        return block_builder

    def get_variable_for_evaluable(self, evaluable: Evaluable) -> _pyast.Variable:
        # Returns the variable `v{id}` where `id` is the unique index of `evaluable`.
        #
        # To aid introspection of the generated code, every evaluable has
        # exactly one variable at its disposal that shares the index with the
        # index of the evaluable. Care should be taken that this variable is
        # not reused (within the same scope). The cache in
        # `_BlockTreeBuilder.compile` already prevents repeated calls to
        # `Evaluable._compile` on the same `evaluable`, so calling this method
        # once in `Evaluable._compile` is safe as long as it is not called
        # somewhere else.
        return _pyast.Variable('v{}'.format(self._get_evaluable_index(evaluable)))

    def get_evaluable_expr(self, evaluable: Evaluable) -> _pyast.Variable:
        # Returns the variable `e{id}` referring to `evaluable` where `id` is
        # the unique index of `evaluable`.
        return _pyast.Variable('e{}'.format(self._get_evaluable_index(evaluable)))

    def _get_evaluable_index(self, evaluable: Evaluable) -> int:
        # Assigns a unique index to `evaluable` if not already assigned and
        # returns the index.
        if (index := self._evaluables.get(evaluable)) is None:
            self._evaluables[evaluable] = index = next(self._new_index)
            self._globals[f'e{index}'] = evaluable
        return index

    def get_block_id(self, evaluable: Evaluable) -> _BlockId:
        # The id of the block where this `Evaluable` must be evaluated.
        if (block_id := self._evaluable_block_ids.get(evaluable)) is None:
            assert not isinstance(evaluable, _LoopIndex)
            if evaluable.dependencies:
                arg_block_ids = tuple(map(self.get_block_id, evaluable.dependencies))
                # Select the first block where all arguments are within scope as
                # the block where `evaluable` must be evaluated.
                block_id = builtins.max(arg_block_ids)
                # If an evaluable is evaluated in some (possibly nested) loop,
                # the value goes out of scope as soon as the loop exits. All
                # arguments must therefor be defined in the same loop as
                # `evaluable`, or a parent loop, but not an adjacent loop. All
                # but the last item of the block id identify loops, hence we
                # need to assert that for all arguments all but the last items
                # of the block id matches the start of the chosen block id of
                # `evaluable`.
                assert all(len(arg_block_id) > 0 and arg_block_id[:-1] == block_id[:len(arg_block_id)-1] for arg_block_id in arg_block_ids)
            else:
                block_id = 0,
            self._evaluable_block_ids[evaluable] = block_id
        return block_id


class _BlockBuilder:

    def __init__(self, parent, block):
        self._parent = parent
        self._block = block
        self.new_var = parent.new_var

    def _needs_lock(self, /, *args, **kwargs):
        # Returns true if any of the arguments references variables that
        # require a lock.
        variables = frozenset().union(*(arg.variables for arg in args), *(arg.variables for arg in kwargs.values()))
        return not self._parent._shared_arrays.isdisjoint(variables)

    def _block_for(self, /, *args, **kwargs):
        # If any of the arguments references variables that require a lock,
        # return a new block that is enclosed in a `with lock` block. Otherwise
        # simply return our block.
        if self._needs_lock(*args, **kwargs):
            block = _pyast.Block()
            self._block.append(_pyast.With(_pyast.Variable('lock'), block))
            return block
        else:
            return self._block

    def exec(self, expression: _pyast.Expression):
        # Appends the statement `{expression}`. Returns nothing.
        self._block_for(expression).append(_pyast.Exec(expression))

    def assign_to(self, lhs: _pyast.Expression, rhs: _pyast.Expression) -> _pyast.Variable:
        # Appends the statement `{lhs} = {rhs}` and returns `lhs`.
        self._block_for(lhs, rhs).append(_pyast.Assign(lhs, rhs))
        return lhs

    def eval(self, expression: _pyast.Expression) -> _pyast.Variable:
        # Evaluates `expression` and assigns the value to a new variable.
        return self.assign_to(self.new_var(), expression)

    def assert_true(self, condition: _pyast.Expression):
        # Appends the statement `assert {condition}`.
        self._block_for(condition).append(_pyast.Assert(condition))

    def assert_equal(self, a: _pyast.Expression, b: _pyast.Expression):
        # Appends the statement `assert {a} == {b}`.
        self.assert_true(_pyast.BinOp(a, '==', b))

    def if_(self, condition: _pyast.Expression) -> '_BlockBuilder':
        # Appends the statement `if {condition}` and returns the if-body.
        if self._needs_lock(condition):
            # We don't want to hold the lock longer than necessary and we must
            # prevent acquiring the lock twice,
            #
            # with lock:
            #    if condition: # needs lock
            #        with lock:
            #            # needs lock
            #
            # Instead we evaluate the condition before the if statement and
            # proceed with the if statement with the lock released.
            condition = self.eval(_pyast.Variable('bool').call(condition))
        body = _pyast.Block()
        self._block.append(_pyast.If(condition, body))
        return _BlockBuilder(self._parent, body)

    def raise_(self, exception: _pyast.Expression):
        # Appends the statement `raise {exception}`.
        self._block_for(exception).append(_pyast.Raise(exception))

    def array_copy(self, dst: _pyast.Expression, src: _pyast.Expression) -> _pyast.Variable:
        # Appends statements for copying array `src` to array `dst`.
        self.exec(_pyast.Variable('numpy').get_attr('copyto').call(dst, src))

    def array_iadd(self, acc: _pyast.Expression, inc: _pyast.Expression):
        # Appends statements for incrementing array `acc` with array `inc`.
        self.exec(_pyast.Variable('numpy').get_attr('add').call(acc, inc, out=acc))

    def array_imul(self, product: _pyast.Expression, factor: _pyast.Expression):
        # Appends statements for inplace multiplication of `product` with `factor`.
        self.exec(_pyast.Variable('numpy').get_attr('multiply').call(product, factor, out=product))

    def array_add_at(self, out: _pyast.Expression, indices: _pyast.Expression, values: _pyast.Expression):
        # Appends statements for incrementing array `out` with array `values` at `indices`.
        self.exec(_pyast.Variable('numpy').get_attr('add').get_attr('at').call(out, indices, values))

    def array_fill_zeros(self, array: _pyast.Expression):
        # Appends statements for filling `array` with zeros.
        self.exec(array.get_attr('fill').call(_pyast.LiteralInt(0)))


if __name__ == '__main__':
    # Diagnostics for the development for simplify operations.
    simplify_priority = (
        Transpose, Ravel,  # reinterpretation
        InsertAxis, Inflate, Diagonalize,  # size increasing
        Multiply, Add, LoopSum, Sign, Power, Inverse, Unravel,  # size preserving
        Product, Determinant, Sum, Take, TakeDiag)  # size decreasing
    # The simplify priority defines the preferred order in which operations are
    # performed: shape decreasing operations such as Sum and Take should be done
    # as soon as possible, and shape increasing operations such as Inflate and
    # Diagonalize as late as possible. In shuffling the order of operations the
    # two classes might annihilate each other, for example when a Sum passes
    # through a Diagonalize. Any shape increasing operations that remain should
    # end up at the surface, exposing sparsity by means of the _assparse method.
    attrs = ['_'+cls.__name__.lower() for cls in simplify_priority]
    # The simplify operations responsible for swapping (a.o.) are methods named
    # '_add', '_multiply', etc. In order to avoid recursions the operations
    # should only be defined in the direction defined by operator priority. The
    # following code warns gainst violations of this rule and lists permissible
    # simplifications that have not yet been implemented.
    for i, cls in enumerate(simplify_priority):
        warn = [attr for attr in attrs[:i] if getattr(cls, attr) is not getattr(Array, attr)]
        if warn:
            print('[!] {} should not define {}'.format(cls.__name__, ', '.join(warn)))
        missing = [attr for attr in attrs[i+1:] if not getattr(cls, attr) is not getattr(Array, attr)]
        if missing:
            print('[ ] {} could define {}'.format(cls.__name__, ', '.join(missing)))

# vim:sw=4:sts=4:et
