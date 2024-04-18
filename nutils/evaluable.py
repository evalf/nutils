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

from . import debug_flags, _util as util, types, numeric, cache, warnings, parallel, sparse
from functools import cached_property
from ._graph import Node, RegularNode, DuplicatedLeafNode, InvisibleNode, Subgraph, TupleNode
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

graphviz = os.environ.get('NUTILS_GRAPHVIZ')

isevaluable = lambda arg: isinstance(arg, Evaluable)


def simplified(value: 'Evaluable'):
    assert isinstance(value, Evaluable), f'value={value!r}'
    return value.simplified


_array_dtypes = bool, int, float, complex
asdtype = lambda arg: arg if any(arg is dtype for dtype in _array_dtypes) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]
Dtype = typing.Union[_array_dtypes]


def asarray(arg):
    if hasattr(type(arg), 'as_evaluable_array'):
        return arg.as_evaluable_array
    if _containsarray(arg):
        return stack(arg, axis=0)
    else:
        return constant(arg)


def _isindex(arg):
    return isinstance(arg, Array) and arg.ndim == 0 and arg.dtype == int and arg._intbounds[0] >= 0


def _equals_scalar_constant(arg: 'Array', value: Dtype):
    assert isinstance(arg, Array) and arg.ndim == 0, f'arg={arg!r}'
    assert arg.dtype == type(value), f'arg.dtype={arg.dtype}, type(value)={type(value)}'
    return arg.isconstant and arg.eval() == value


def _equals_simplified(arg1: 'Array', arg2: 'Array'):
    assert isinstance(arg1, Array), f'arg1={arg1!r}'
    assert isinstance(arg2, Array), f'arg2={arg2!r}'
    if arg1 is arg2:
        return True
    assert equalshape(arg1.shape, arg2.shape) and arg1.dtype == arg2.dtype, f'arg1={arg1!r}, arg2={arg2!r}'
    arg1 = arg1.simplified
    arg2 = arg2.simplified
    if arg1 is arg2:
        return True
    if arg1.arguments != arg2.arguments:
        return False
    if arg1.isconstant: # implies arg2.isconstant
        return numpy.all(arg1.eval() == arg2.eval())


def equalshape(N: typing.Tuple['Array', ...], M: typing.Tuple['Array', ...]):
    '''Compare two array shapes.

    Returns `True` if all indices are certainly equal, `False` if any indices are
    certainly not equal, or `None` if equality cannot be determined at compile
    time.
    '''

    assert isinstance(N, tuple) and all(_isindex(n) for n in N), f'N={N!r}'
    assert isinstance(M, tuple) and all(_isindex(n) for n in M), f'M={M!r}'
    if N == M:
        return True
    if len(N) != len(M):
        return False
    retval = True
    for eq in map(_equals_simplified, N, M):
        if eq == False:
            return False
        if eq == None:
            retval = None
    return retval


class ExpensiveEvaluationWarning(warnings.NutilsInefficiencyWarning):
    pass


class Evaluable(types.Singleton):
    'Base class'

    def __init__(self, args: typing.Tuple['Evaluable', ...]):
        assert isinstance(args, tuple) and all(isinstance(arg, Evaluable) for arg in args), f'args={args!r}'
        super().__init__()
        self.__args = args

    @staticmethod
    def evalf(*args):
        raise NotImplementedError('Evaluable derivatives should implement the evalf method')

    def evalf_withtimes(self, times, *args):
        with times[self]:
            return self.evalf(*args)

    @cached_property
    def dependencies(self):
        '''collection of all function arguments'''
        deps = {}
        for func in self.__args:
            funcdeps = func.dependencies
            deps.update(funcdeps)
            deps[func] = len(funcdeps)
        return types.frozendict(deps)

    @cached_property
    def arguments(self):
        'a frozenset of all arguments of this evaluable'
        return frozenset().union(*(child.arguments for child in self.__args))

    @property
    def isconstant(self):
        return EVALARGS not in self.dependencies and not self.arguments

    @cached_property
    def ordereddeps(self):
        '''collection of all function arguments such that the arguments to
        dependencies[i] can be found in dependencies[:i]'''
        deps = self.dependencies.copy()
        deps.pop(EVALARGS, None)
        return tuple([EVALARGS] + sorted(deps, key=deps.__getitem__))

    @cached_property
    def dependencytree(self):
        '''lookup table of function arguments into ordereddeps, such that
        ordereddeps[i].__args[j] == ordereddeps[dependencytree[i][j]], and
        self.__args[j] == ordereddeps[dependencytree[-1][j]]'''
        args = self.ordereddeps
        return tuple(tuple(map(args.index, func.__args)) for func in args+(self,))

    @property
    def serialized(self):
        return zip(self.ordereddeps[1:]+(self,), self.dependencytree[1:])

    # This property is a derivation of `ordereddeps[1:]` where the `Evaluable`
    # instances are mapped to the `evalf` methods of the instances. Asserting
    # that functions are immutable is difficult and currently
    # `types._isimmutable` marks all functions as mutable. Since the
    # `types.CacheMeta` machinery asserts immutability of the property, we have
    # to resort to a regular `functools.cached_property`. Nevertheless, this
    # property should be treated as if it is immutable.
    @cached_property
    def _serialized_evalf_head(self):
        return tuple(op.evalf for op in self.ordereddeps[1:])

    @property
    def _serialized_evalf(self):
        return zip(itertools.chain(self._serialized_evalf_head, (self.evalf,)), self.dependencytree[1:])

    def _node(self, cache, subgraph, times):
        if self in cache:
            return cache[self]
        args = tuple(arg._node(cache, subgraph, times) for arg in self.__args)
        label = '\n'.join(filter(None, (type(self).__name__, self._node_details)))
        cache[self] = node = RegularNode(label, args, {}, (type(self).__name__, times[self]), subgraph)
        return node

    @property
    def _node_details(self):
        return ''

    def asciitree(self, richoutput=False):
        'string representation'

        return self._node({}, None, collections.defaultdict(_Stats)).generate_asciitree(richoutput)

    def __str__(self):
        return self.__class__.__name__

    def eval(self, **evalargs):
        '''Evaluate function on a specified element, point set.'''

        values = [evalargs]
        try:
            values.extend(op_evalf(*[values[i] for i in indices]) for op_evalf, indices in self._serialized_evalf)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(self._format_stack(values, e))
            raise
        else:
            return values[-1]

    def eval_withtimes(self, times, **evalargs):
        '''Evaluate function on a specified element, point set while measure time of each step.'''

        values = [evalargs]
        try:
            values.extend(op.evalf_withtimes(times, *[values[i] for i in indices]) for op, indices in self.serialized)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(self._format_stack(values, e))
            raise
        else:
            return values[-1]

    @contextlib.contextmanager
    def session(self, graphviz):
        if graphviz is None:
            yield self.eval
            return
        stats = collections.defaultdict(_Stats)

        def eval(**args):
            return self.eval_withtimes(stats, **args)
        with log.context('eval'):
            yield eval
            node = self._node({}, None, stats)
            maxtime = builtins.max(n.metadata[1].time for n in node.walk(set()))
            tottime = builtins.sum(n.metadata[1].time for n in node.walk(set()))
            aggstats = tuple((key, builtins.sum(v.time for v in values), builtins.sum(v.ncalls for v in values)) for key, values in util.gather(n.metadata for n in node.walk(set())))
            fill_color = (lambda node: '0,{:.2f},1'.format(node.metadata[1].time/maxtime)) if maxtime else None
            node.export_graphviz(fill_color=fill_color, dot_path=graphviz)
            log.info('total time: {:.0f}ms\n'.format(tottime/1e6) + '\n'.join('{:4.0f} {} ({} calls, avg {:.3f} per call)'.format(t / 1e6, k, n, t / (1e6*n))
                                                                              for k, t, n in sorted(aggstats, reverse=True, key=lambda item: item[1]) if n))

    def _iter_stack(self):
        yield '%0 = EVALARGS'
        for i, (op, indices) in enumerate(self.serialized, start=1):
            s = [f'%{i} = {op}']
            if indices:
                args = [f'%{i}' for i in indices]
                try:
                    sig = inspect.signature(op.evalf)
                except ValueError:
                    pass
                else:
                    for i, param in enumerate(sig.parameters.values()):
                        if i < len(args) and param.kind == param.POSITIONAL_OR_KEYWORD:
                            args[i] = param.name + '=' + args[i]
                s.extend(args)
            yield ' '.join(s)

    def _format_stack(self, values, e):
        lines = [f'evaluation failed in step {len(values)}/{len(self.dependencies)+1}']
        stack = self._iter_stack()
        for v, op in zip(values, stack): # NOTE values must come first to avoid popping next item from stack
            s = f'{type(v).__name__}'
            if numeric.isarray(v):
                s += f'<{v.dtype.kind}:{",".join(str(n) for n in v.shape)}>'
            lines.append(f'{op} --> {s}')
        lines.append(f'{next(stack)} --> {e}')
        return '\n  '.join(lines)

    @util.deep_replace_property
    def simplified(obj):
        retval = obj._simplified()
        if retval is None:
            return obj
        if isinstance(obj, Array):
            assert isinstance(retval, Array) and equalshape(retval.shape, obj.shape) and retval.dtype == obj.dtype, '{} --simplify--> {}'.format(obj, retval)
        return retval

    def _simplified(self):
        return

    @cached_property
    def optimized_for_numpy(self):
        return self.simplified \
                   ._optimized_for_numpy1 \
                   ._deep_flatten_constants() \
                   ._combine_loops()

    @util.deep_replace_property
    def _optimized_for_numpy1(obj):
        retval = obj._optimized_for_numpy()
        if retval is None:
            return obj
        if isinstance(obj, Array):
            assert isinstance(retval, Array) and equalshape(retval.shape, obj.shape), '{0}._optimized_for_numpy or {0}._simplified resulted in shape change'.format(type(obj).__name__)
        return retval

    def _optimized_for_numpy(self):
        return

    @util.shallow_replace
    def _deep_flatten_constants(self):
        if isinstance(self, Array):
            return self._flatten_constant()

    @cached_property
    def _loop_deps(self):
        deps = util.IDSet()
        for arg in self.__args:
            deps |= arg._loop_deps
        return deps.view()

    def _combine_loops(self, candidates=None):
        if candidates is None:
            candidates = self._loop_deps
        candidates = candidates.copy()

        while candidates:
            # Select the loops from candidates that can be combined. Given an
            # index, we select all loops that have that index and are not a
            # dependency of another loop in `candidates`. The latter ensures
            # that the `candidates` do not disappear from `self` other than the
            # deliberate combining performed here. To illustrate this, consider
            # the following abstract loop structure
            #
            #     A: Add
            #       B: LoopSum, loop_index=i
            #         C: Add
            #           D: LoopSum, loop_index=j
            #             ..., does not depend on i
            #           E: LoopSum, loop_index=j
            #             ..., does not depend on i
            #       F: LoopSum, loop_index=i
            #         ...
            #
            # Loops D and E are invariant loops of B. The default candidates of
            # A are B, D, E and F. By combing D and E first, B would be replaced by B',
            #
            #     A: Add
            #       B': LoopSum, loop_index=i
            #         C': Add
            #           D': ArrayFromTuple, index=0
            #             DE': LoopTuple(D, E), loop_index=j
            #           E': ArrayFromTuple, index=0
            #             DE'
            #       F: LoopSum, loop_index=i
            #         ...
            #
            # which is not in the set of candidates and we'd miss the
            # opportunity to combine B' with F. Or worse: we combine B with F,
            # ignore the former (`ArrayFromTuple`) *and* keep B'.
            loops = []
            for loop in candidates:
                if loops and loop.index != loops[0].index:
                    continue # take one index at a time
                if all(loop not in other._loop_deps for other in candidates if other is not loop):
                    loops.append(loop)
            candidates.difference_update(loops)
            index = loops[0].index

            replacements = util.IDDict()
            if len(loops) >= 2:
                combined = _LoopTuple(tuple(loops), index.name, index.length)
                combined = combined._combine_loops(combined.body_arg._loop_deps - combined._loop_deps)
                for i, loop in enumerate(loops):
                    replacements[loop] = ArrayFromTuple(combined, i, loop.shape, loop.dtype)
            else:
                loop, = loops
                combined = loop._combine_loops(loop.body_arg._loop_deps - loop._loop_deps)
                if combined != loop:
                    replacements[loop] = combined
            if replacements:
                self = util.shallow_replace(replacements.get, self)

        return self


class EVALARGS(Evaluable):
    def __init__(self):
        super().__init__(args=())

    def _node(self, cache, subgraph, times):
        return InvisibleNode((type(self).__name__, _Stats()))


EVALARGS = EVALARGS()


class Tuple(Evaluable):

    def __init__(self, items):
        self.items = items
        super().__init__(items)

    @staticmethod
    def evalf(*items):
        return items

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

    def _node(self, cache, subgraph, times):
        if (cached := cache.get(self)) is not None:
            return cached
        cache[self] = node = TupleNode(tuple(item._node(cache, subgraph, times) for item in self.items), (type(self).__name__, times[self]), subgraph=subgraph)
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
        arg = Transpose.inv(arg, where)
    assert equalshape(arg.shape, shape), f'arg.shape={arg.shape!r}, shape={shape!r}'
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
    if naxes is None:
        if any(arg.ndim != args[0].ndim for arg in args[1:]):
            raise ValueError('varying dimensions in unalign')
        naxes = args[0].ndim
    elif any(arg.ndim < naxes for arg in args):
        raise ValueError('one or more arguments have fewer axes than expected')
    uninserted = []
    for arg in args:
        where = list(range(naxes))
        while True:
            for arg, axis, length in arg._as(insertaxis, lambda f, i, n: i < len(where)):
                del where[axis]
                break
            else:
                break
        uninserted.append((arg, tuple(where)))
    if len(args) == 1:
        return uninserted[0]
    commonwhere = tuple(sorted(functools.reduce(operator.or_, [set(axes) for func, axes in uninserted])))
    if len(commonwhere) == naxes:
        return (*args, tuple(range(naxes)))
    ret = []
    for arg, (unaligned, where) in zip(args, uninserted):
        # both where and commonwhere are sorted
        if where != commonwhere:
            missing = [i for i, n in enumerate(commonwhere) if n not in where]
            shape = [arg.shape[commonwhere[i]] for i in missing]
            unaligned = Transpose.from_end(appendaxes(unaligned, shape), *missing)
        ret.append(unaligned)
    return (*ret, commonwhere)

# ARRAYS


_ArrayMeta = type(Evaluable)

if debug_flags.sparse:
    def _chunked_assparse_checker(orig):
        assert isinstance(orig, cached_property)

        @cached_property
        def _assparse(self):
            chunks = orig.func(self)
            assert isinstance(chunks, tuple)
            assert all(isinstance(chunk, tuple) for chunk in chunks)
            assert all(all(isinstance(item, Array) for item in chunk) for chunk in chunks)
            if self.ndim:
                for *indices, values in chunks:
                    assert len(indices) == self.ndim
                    assert all(idx.dtype == int for idx in indices)
                    assert all(equalshape(idx.shape, values.shape) for idx in indices)
            elif chunks:
                assert len(chunks) == 1
                chunk, = chunks
                assert len(chunk) == 1
                values, = chunk
                assert values.shape == ()
            return chunks
        return _assparse

    class _ArrayMeta(_ArrayMeta):
        def __new__(mcls, name, bases, namespace):
            if '_assparse' in namespace:
                namespace['_assparse'] = _chunked_assparse_checker(namespace['_assparse'])
            return super().__new__(mcls, name, bases, namespace)

if debug_flags.evalf:
    class _evalf_checker:
        def __init__(self, orig):
            self.orig = orig

        def __set_name__(self, owner, name):
            if hasattr(self.orig, '__set_name__'):
                self.orig.__set_name__(owner, name)

        def __get__(self, instance, owner):
            evalf = self.orig.__get__(instance, owner)

            @functools.wraps(evalf)
            def evalf_with_check(*args, **kwargs):
                res = evalf(*args, **kwargs)
                assert not hasattr(instance, 'dtype') or asdtype(res.dtype) == instance.dtype, ((instance.dtype, res.dtype), instance, res)
                assert not hasattr(instance, 'ndim') or res.ndim == instance.ndim
                assert not hasattr(instance, 'shape') or all(m == n for m, n in zip(res.shape, instance.shape) if isinstance(n, int)), 'shape mismatch'
                return res
            return evalf_with_check

    class _ArrayMeta(_ArrayMeta):
        def __new__(mcls, name, bases, namespace):
            if 'evalf' in namespace:
                namespace['evalf'] = _evalf_checker(namespace['evalf'])
            return super().__new__(mcls, name, bases, namespace)


class AsEvaluableArray(Protocol):
    'Protocol for conversion into an :class:`Array`.'

    @property
    def as_evaluable_array(self) -> 'Array':
        'Lower this object to a :class:`nutils.evaluable.Array`.'


class Array(Evaluable, metaclass=_ArrayMeta):
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

    def __init__(self, args, shape: typing.Tuple['Array', ...], dtype: Dtype):
        assert isinstance(shape, tuple) and all(_isindex(n) for n in shape), f'shape={shape!r}'
        assert isinstance(dtype, type) and dtype in _array_dtypes, f'dtype={dtype!r}'
        self.shape = shape
        self.dtype = dtype
        super().__init__(args=args)

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
            for op, args in self.representations:
                if op is insertaxis:
                    f, axis, length = args
                    shape[axis] = '(' + shape[axis] + ')'
                elif op is _inflate:
                    f, dofmap, length, axis = args
                    shape[axis] = '~' + shape[axis]
                elif op is diagonalize:
                    f, axis, newaxis = args
                    shape[axis+(newaxis<=axis)] += '/'
                    shape[newaxis] += '/'
        except:
            suffix = '(e)'
        return prefix + ','.join(shape) + suffix

    sum = sum
    prod = product
    dot = dot
    swapaxes = swapaxes
    transpose = transpose
    choose = lambda self, choices: Choose(self, *choices)
    conjugate = conjugate

    @property
    def real(self):
        return real(self)

    @property
    def imag(self):
        return imag(self)

    @cached_property
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

    def _node(self, cache, subgraph, times):
        if self in cache:
            return cache[self]
        args = tuple(arg._node(cache, subgraph, times) for arg in self._Evaluable__args)
        bounds = '[{},{}]'.format(*self._intbounds) if self.dtype == int else None
        label = '\n'.join(filter(None, (type(self).__name__, self._node_details, self._shape_str(form=repr), bounds)))
        cache[self] = node = RegularNode(label, args, {}, (type(self).__name__, times[self]), subgraph)
        return node

    # simplifications
    _eig = lambda self, symmetric: None

    representations = ()

    def _as(self, op, condition=None):
        return (args for op_, args in self.representations if op_ is op and (condition is None or condition(*args)))

    def _as_any(self, *ops):
        for op, args in self.representations:
            if op in ops:
                return op(*args)

    def _iter_into(self, binary_op):
        queue = [(self, (), set())]
        for f, rem, ops in queue:
            yield f, rem, ops
            ops = ops.copy()
            for op, args in f.representations:
                if op is binary_op:
                    a, b = args
                    queue.append((a, (b, *rem), ops))
                    queue.append((b, (a, *rem), ops))
                    break
                ops.add(op)

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
        if self.ndim == 0 and self.dtype == int and self.isconstant:
            value = self.__index__()
            return value, value
        else:
            lower, upper = self._intbounds_impl()
            assert isinstance(lower, int) or lower == float('-inf')
            assert isinstance(upper, int) or upper == float('inf')
            assert lower <= upper
            return lower, upper

    def _intbounds_impl(self):
        return float('-inf'), float('inf')

    @property
    def _const_uniform(self):
        if self.dtype == int:
            lower, upper = self._intbounds
            return lower if lower == upper else None

    def _flatten_constant(self):
        if self.isconstant:
            return constant(self.eval())


class Orthonormal(Array):
    'make a vector orthonormal to a subspace'

    def __init__(self, basis: Array, vector: Array):
        assert isinstance(basis, Array) and basis.ndim >= 2 and basis.dtype not in (bool, complex), f'basis={basis!r}'
        assert isinstance(vector, Array) and vector.ndim >= 1 and vector.dtype not in (bool, complex), f'vector={vector!r}'
        assert equalshape(basis.shape[:-1], vector.shape)
        self._basis = basis
        self._vector = vector
        super().__init__(args=(basis, vector), shape=vector.shape, dtype=float)

    def _simplified(self):
        if _equals_scalar_constant(self.shape[-1], 1):
            return Sign(self._vector)
        basis, vector, where = unalign(self._basis, self._vector, naxes=self.ndim - 1)
        if len(where) < self.ndim - 1:
            return align(Orthonormal(basis, vector), (*where, self.ndim - 1), self.shape)

    @staticmethod
    def evalf(G, n):
        GG = numpy.einsum('...ki,...kj->...ij', G, G)
        v1 = numpy.einsum('...ij,...i->...j', G, n)
        v2 = numpy.linalg.solve(GG, v1)
        v3 = numpy.einsum('...ij,...j->...i', G, v2)
        return numeric.normalize(n - v3)

    def _derivative(self, var, seen):
        if _equals_scalar_constant(self.shape[-1], 1):
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

        G = self._basis
        invGG = inverse(einsum('Aki,Akj->Aij', G, G))

        Q = -einsum('Aim,Amn,AjnB->AijB', G, invGG, derivative(G, var, seen))
        QN = einsum('Ai,AjiB->AjB', self, Q)

        if _equals_simplified(G.shape[-1], G.shape[-2] - 1): # dim(kern(G)) = 1
            # In this situation, since N is a basis for the kernel of G, we
            # have the identity P == N N^T which cancels the entire second term
            # of N' along with any reference to v', reducing it to N' = Q N.
            return QN

        v = self._vector
        P = Diagonalize(ones(self.shape)) - einsum('Aim,Amn,Ajn->Aij', G, invGG, G)
        Z = P - einsum('Ai,Aj->Aij', self, self) # P - N N^T

        return QN + einsum('A,AiB->AiB',
            power(einsum('Ai,Aij,Aj->A', v, P, v), -.5),
            einsum('Aij,AjB->AiB', Z, einsum('Ai,AijB->AjB', v, Q) + derivative(v, var, seen)))


class Constant(Array):

    def __init__(self, value: types.arraydata):
        assert isinstance(value, types.arraydata), f'value={value!r}'
        self.value = numpy.asarray(value)
        super().__init__(args=(), shape=tuple(constant(n) for n in value.shape), dtype=value.dtype)

    @property
    def representations(self):
        return (constant, (self.value,)),

    def _simplified(self):
        if not self.value.any():
            return zeros_like(self)
        if self.ndim == 1 and self.dtype == int and numpy.all(self.value == numpy.arange(self.value.shape[0])):
            return Range(self.shape[0])
        for i, sh in enumerate(self.shape):
            # Find and replace invariant axes with InsertAxis. Since `self.value.any()`
            # is False for arrays with a zero-length axis, we can arrive here only if all
            # axes have at least length one, hence the following statement should work.
            first, *others = numpy.rollaxis(self.value, i)
            if all(numpy.equal(first, other).all() for other in others):
                return insertaxis(constant(first), i, sh)

    def evalf(self):
        return self.value

    def _node(self, cache, subgraph, times):
        if self.ndim:
            return super()._node(cache, subgraph, times)
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

    def _eig(self, symmetric):
        eigval, eigvec = (numpy.linalg.eigh if symmetric else numpy.linalg.eig)(self.value)
        if not symmetric:
            eigval = eigval.astype(complex, copy=False)
            eigvec = eigvec.astype(complex, copy=False)
        return Tuple((constant(eigval), constant(eigvec)))

    def _intbounds_impl(self):
        if self.dtype == int and self.value.size:
            return int(self.value.min()), int(self.value.max())
        else:
            return super()._intbounds_impl()

    @property
    def _const_uniform(self):
        if self.ndim == 0:
            return self.dtype(self.value[()])

    def _flatten_constant(self):
        pass


class InsertAxis(Array):

    def __init__(self, func: Array, length: Array):
        assert isinstance(func, Array), f'func={func!r}'
        assert _isindex(length), f'length={length!r}'
        self.func = func
        self.length = length
        super().__init__(args=(func, length), shape=(*func.shape, length), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield insertaxis, (self.func, self.ndim-1, self.length)
        for op, args in self.func.representations:
            if op is insertaxis:
                f, axis, length = args
                yield op, (InsertAxis(f, self.length), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                yield op, (InsertAxis(f, self.length), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                yield op, (InsertAxis(f, self.length), dofmap, length, axis)
            elif op in (add, multiply):
                a, b = args
                yield op, (InsertAxis(a, self.length), InsertAxis(b, self.length))
            elif op is ravel:
                f, axis = args
                yield op, (InsertAxis(f, self.length), axis)

    @property
    def _diagonals(self):
        return self.func._diagonals

    @cached_property
    def _inflations(self):
        return tuple((axis, types.frozendict((dofmap, InsertAxis(func, self.length)) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if _equals_scalar_constant(self.length, 0):
            return zeros_like(self)
        if simple := self._as_any(ravel):
            return simple

    @staticmethod
    def evalf(func, length):
        if length == 1:
            return func[..., numpy.newaxis]
        try:
            return numpy.ndarray(buffer=func, dtype=func.dtype, shape=(*func.shape, length), strides=(*func.strides, 0))
        except ValueError:  # non-contiguous data
            return numpy.repeat(func[..., numpy.newaxis], length, -1)

    def _derivative(self, var, seen):
        return insertaxis(derivative(self.func, var, seen), self.ndim-1, self.length)

    @cached_property
    def _assparse(self):
        return tuple((*(InsertAxis(idx, self.length) for idx in indices), prependaxes(Range(self.length), values.shape), InsertAxis(values, self.length)) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds

    @property
    def _const_uniform(self):
        return self.func._const_uniform

    def _flatten_constant(self):
        pass


class Transpose(Array):

    @classmethod
    def _mk_axes(cls, ndim, axes):
        axes = [numeric.normdim(ndim, axis) for axis in axes]
        if all(a == b for a, b in enumerate(axes, start=ndim-len(axes))):
            return
        trans = [i for i in range(ndim) if i not in axes]
        trans.extend(axes)
        if len(trans) != ndim:
            raise Exception('duplicate axes')
        return tuple(trans)

    @classmethod
    def from_end(cls, array, *axes):
        trans = cls._mk_axes(array.ndim, axes)
        return cls.inv(array, trans) if trans else array

    @classmethod
    def to_end(cls, array, *axes):
        trans = cls._mk_axes(array.ndim, axes)
        return cls(array, trans) if trans else array

    @classmethod
    def inv(cls, func, axes):
        return cls(func, tuple(n.__index__() for n in numpy.argsort(axes)))

    def __init__(self, func: Array, axes: typing.Tuple[int, ...]):
        assert isinstance(func, Array), f'func={func!r}'
        assert isinstance(axes, tuple) and all(isinstance(axis, int) for axis in axes), f'axes={axes!r}'
        assert sorted(axes) == list(range(func.ndim))
        assert axes != tuple(range(func.ndim))
        self.func = func
        self.axes = axes
        super().__init__(args=(func,), shape=tuple(func.shape[n] for n in axes), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield transpose, (self.func, self.axes)
        for op, args in self.func.representations:
            if op is transpose:
                f, axes = args
                yield op, (f, tuple(axes[i] for i in self.axes))
            elif op in (insertaxis, legendre):
                f, axis, arg = args
                yield op, (transpose(f, tuple(i-(i>axis) for i in self.axes if i != axis)), self.axes.index(axis), arg)
            elif op is diagonalize:
                f, axis, newaxis = args
                axes = tuple(i-(i>newaxis) for i in self.axes if i != newaxis)
                yield op, (transpose(f, axes), axes.index(axis), self.axes.index(newaxis))
            elif op is _inflate:
                f, dofmap, length, axis = args
                axes = [i if i <= axis else i + (dofmap.ndim-1) for i in self.axes]
                newaxis = self.axes.index(axis)
                axes[newaxis:newaxis+1] = range(axis, axis+dofmap.ndim)
                yield op, (transpose(f, tuple(axes)), dofmap, length, newaxis)
            elif op is _take:
                f, indices, axis = args
                if indices.ndim == 0:
                    yield op, (transpose(f, (*self.axes, f.ndim-1)), indices, f.ndim-1)
                elif numpy.ptp(takeaxes := self.axes[axis:axis+indices.ndim]) == indices.ndim-1:
                    axes = [i if i <= axis else i - (indices.ndim-1) for i in self.axes if not axis < i < axis + indices.ndim]
                    iaxes = [i - min(takeaxes) for i in takeaxes]
                    yield op, (transpose(f, axes), transpose(indices, iaxes), axes.index(axis))
            elif op is polyval:
                f, coeffs, axis = args
                if coeffs.ndim == 1:
                    yield op, (transpose(f, (*self.axes, f.ndim-1)), coeffs, f.ndim-1)
                elif numpy.ptp(takeaxes := self.axes[axis:axis+coeffs.ndim-1]) == coeffs.ndim-2:
                    axes = [i if i <= axis else i - (coeffs.ndim-2) for i in self.axes if not axis < i < axis + coeffs.ndim - 1]
                    caxes = [i - min(takeaxes) for i in takeaxes] + [coeffs.ndim-1]
                    yield op, (transpose(f, axes), transpose(coeffs, caxes), axes.index(axis))
            elif op is polymul:
                c1, c2, v, axis = args
                yield op, (Transpose(c1, self.axes), Transpose(c2, self.axes), v, self.axes.index(axis))
            elif op is polygrad:
                c, nvars, caxis, deraxis = args
                axes = tuple(i-(i>deraxis) for i in self.axes if i != deraxis)
                yield op, (transpose(c, axes), nvars, axes.index(caxis), self.axes.index(deraxis))
            elif op is takediag:
                f, axis, rmaxis = args
                yield op, (Transpose(f, (*self.axes, self.ndim)), self.axes.index(axis), self.ndim)
            elif op in (add, multiply, power, Choose, sign, *_POINTWISE):
                yield op, tuple(Transpose(arg, self.axes) for arg in args)
            elif op is ravel:
                f, axis = args
                axes = [i + (i>axis) for i in self.axes]
                newaxis = self.axes.index(axis)
                axes.insert(newaxis+1, axis+1)
                yield op, (transpose(f, tuple(axes)), newaxis)
            elif op is unravel:
                f, axis, shape = args
                if self.axes[axis+1] == self.axes[axis]+1:
                    axes = [i if i <= axis else i - 1 for i in self.axes if i != axis + 1]
                    yield op, (transpose(f, tuple(axes)), axes.index(axis), shape)
            elif op is constant:
                v, = args
                yield op, (v.transpose(self.axes),)
            elif op in (loop_sum, SearchSorted):
                f, *_args = args
                yield op, (Transpose(f, self.axes), *_args)
            elif op is loop_concatenate:
                f, index, axis = args
                yield op, (Transpose(f, self.axes), index, self.axes.index(axis))
            elif op is inverse:
                f, invaxes = args
                yield op, (Transpose(f, self.axes), tuple(map(self.axes.index, invaxes)))
            elif op is determinant:
                f, axes = args
                yield op, (Transpose(f, (*self.axes, self.ndim, self.ndim+1)), (self.ndim, self.ndim+1))
            elif op in (sum, product):
                f, axis = args
                yield op, (Transpose(f, (*self.axes, self.ndim)), self.ndim)
            elif op is ravelindex:
                ia, ib, na, nb, where = args
                align = numpy.argsort(self.axes)
                axesa = numpy.argsort(align[~where])
                axesb = numpy.argsort(align[where])
                yield op, (transpose(ia, axesa), transpose(ib, axesb), na, nb, where[list(self.axes)])
            elif op is Choose:
                yield op, tuple(Transpose(arg, self.axes) for arg in args)

    @cached_property
    def _diagonals(self):
        return tuple(frozenset(self._invaxes[i] for i in axes) for axes in self.func._diagonals)

    @cached_property
    def _inflations(self):
        return tuple((self._invaxes[axis], types.frozendict((dofmap, transpose(func, self._axes_for(dofmap.ndim, self._invaxes[axis]))) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

    @cached_property
    def _invaxes(self):
        return tuple(n.__index__() for n in numpy.argsort(self.axes))

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        for f, axes in self.func._as(transpose):
            return transpose(f, tuple(axes[i] for i in self.axes))

    def _optimized_for_numpy(self):
        if isinstance(self.func, Einsum):
            return Einsum(self.func.args, self.func.args_idx, tuple(self.func.out_idx[i] for i in self.axes))

    def evalf(self, arr):
        return arr.transpose(self.axes)

    @property
    def _node_details(self):
        return ','.join(map(str, self.axes))

    def _derivative(self, var, seen):
        return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

    def _axes_for(self, ndim, axis):
        funcaxis = self.axes[axis]
        axes = [ax+(ax > funcaxis)*(ndim-1) for ax in self.axes if ax != funcaxis]
        axes[axis:axis] = range(funcaxis, funcaxis + ndim)
        return tuple(axes)

    @cached_property
    def _assparse(self):
        return tuple((*(indices[i] for i in self.axes), values) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds

    @property
    def _const_uniform(self):
        return self.func._const_uniform

    def _flatten_constant(self):
        pass


class Product(Array):

    def __init__(self, func: Array):
        assert isinstance(func, Array), f'func={func!r}'
        self.func = func
        self.evalf = functools.partial(numpy.all if func.dtype == bool else numpy.prod, axis=-1)
        super().__init__(args=(func,), shape=func.shape[:-1], dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield product, (self.func, self.ndim)
        for op, args in self.func.representations:
            if op is constant:
                v, = args
                yield op, (self.evalf(v),)
            elif op is insertaxis:
                f, axis, length = args
                if axis < f.ndim:
                    yield op, (Product(f), axis, length)
            elif op is multiply:
                a, b = args
                yield op, (Product(a), Product(b))
            elif op in (IntToFloat, FloatToComplex):
                f, = args
                yield op, (Product(f),)

    def _simplified(self):
        if _equals_scalar_constant(self.func.shape[-1], 1):
            return get(self.func, self.ndim, constant(0))
        if simple := self._as_any(constant, insertaxis, multiply, IntToFloat, FloatToComplex):
            return simple
        for f, axis, newaxis in self.func._as(diagonalize):
            if newaxis == f.ndim or axis == f.ndim-1:
                # TODO what if func.shape[-1] is not constant 1 but _may_ be one
                return zeros_like(self)
        for f, axis in self.func._as(ravel):
            if axis == f.ndim-2:
                return Product(Product(f))
        for f, axis, length in self.func._as(insertaxis):
            if axis == self.ndim:
                return f if self.dtype == bool else f**astype(length, self.dtype)

    def _derivative(self, var, seen):
        grad = derivative(self.func, var, seen)
        funcs = Product(insertaxis(self.func, -2, self.func.shape[-1]) + Diagonalize(astype(1, self.func.dtype) - self.func))  # replace diagonal entries by 1
        return einsum('Ai,AiB->AB', funcs, grad)


class Inverse(Array):
    '''
    Matrix inverse of ``func`` over the last two axes.  All other axes are
    treated element-wise.
    '''

    def __init__(self, func: Array):
        assert isinstance(func, Array) and func.dtype in (float, complex) and func.ndim >= 2 and _equals_simplified(func.shape[-1], func.shape[-2]), f'func={func!r}'
        self.func = func
        super().__init__(args=(func,), shape=func.shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield inverse, (self.func, (self.ndim-2, self.ndim-1))
        for op, args in self.func.representations:
            if op is insertaxis:
                f, axis, length = args
                if axis < f.ndim-2:
                    yield op, (Inverse(f), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-2 and newaxis <= f.ndim-2:
                    yield op, (Inverse(f), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim-2:
                    yield op, (Inverse(f), dofmap, length, axis)
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-2:
                    yield op, (Inverse(f), axis)
            elif op is constant:
                v, = args
                yield op, (numpy.linalg.inv(v),)

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return singular_like(self)
        if _equals_scalar_constant(self.func.shape[-1], 1):
            return reciprocal(self.func)
        if _equals_scalar_constant(self.func.shape[-1], 0):
            return singular_like(self)
        for f, axis, newaxis in self.func._as(diagonalize, lambda f, i, j: i == f.ndim-1 and j >= f.ndim-1):
            return Diagonalize(reciprocal(f))
        # try to separate scalar multiplications
        for factor, rem, ops in self.func._iter_into(multiply):
            for f, axis, length in factor._as(insertaxis, lambda f, i, n: i >= self.ndim-2):
                if not rem:
                    # TODO what if func.shape[-1] is not constant 1 but _may_ be one
                    return singular_like(self)
                for f, axis, length in f._as(insertaxis, lambda f, i, n: i >= self.ndim-2):
                    return reciprocal(factor) * Inverse(multiply(*rem))

    evalf = staticmethod(numeric.inv)

    def _derivative(self, var, seen):
        return -einsum('Aij,AjkB,Akl->AilB', self, derivative(self.func, var, seen), self)

    def _eig(self, symmetric):
        eigval, eigvec = Eig(self.func, symmetric)
        return Tuple((reciprocal(eigval), eigvec))


class Determinant(Array):

    def __init__(self, func: Array):
        assert isarray(func) and func.dtype in (float, complex) and func.ndim >= 2 and _equals_simplified(func.shape[-1], func.shape[-2])
        self.func = func
        super().__init__(args=(func,), shape=func.shape[:-2], dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield determinant, (self.func, (self.func.ndim-2, self.func.ndim-1))
        for op, args in self.func.representations:
            if op is insertaxis:
                f, axis, length = args
                if axis < f.ndim-2:
                    yield op, (Determinant(f), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-2 and newaxis <= f.ndim-2:
                    yield op, (Determinant(f), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim-2:
                    yield op, (Determinant(f), dofmap, length, axis)
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-2:
                    yield op, (Determinant(f), axis)
            elif op is constant:
                v, = args
                yield op, (numpy.linalg.det(v),)

    def _simplified(self):
        if _equals_scalar_constant(self.func.shape[-1], 0):
            return ones_like(self)
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if _equals_scalar_constant(self.func.shape[-1], 1):
            return Take(Take(self.func, zeros((), int)), zeros((), int))
        for f, *axes in self.func._as(inverse):
            if min(axes) == self.func.ndim-2 and max(axes) == self.func.ndim-1:
                return reciprocal(Determinant(f))
        for f, axis, newaxis in self.func._as(diagonalize, lambda f, i, j: i == f.ndim-1 and j >= f.ndim-1):
            return Product(f)
        if simple := self._as_any(constant, insertaxis, diagonalize):
            return simple
        for factor, rem, ops in self.func._iter_into(multiply):
            for f, axis, length in factor._as(insertaxis, lambda f, i, n: i >= self.ndim):
                if not rem:
                    # TODO what if func.shape[-1] is not constant 1 but _may_ be one
                    return zeros_like(self)
                for f, axis, length in f._as(insertaxis, lambda f, i, n: i >= self.ndim):
                    return f**astype(self.func.shape[-1], self.dtype) * Determinant(multiply(*rem))

    evalf = staticmethod(numpy.linalg.det)

    def _derivative(self, var, seen):
        return einsum('A,Aji,AijB->AB', self, inverse(self.func), derivative(self.func, var, seen))


class Multiply(Array):

    def __init__(self, funcs: types.frozenmultiset):
        assert isinstance(funcs, types.frozenmultiset), f'funcs={funcs!r}'
        self.funcs = funcs
        func1, func2 = funcs
        assert equalshape(func1.shape, func2.shape) and func1.dtype == func2.dtype, 'Multiply({}, {})'.format(func1, func2)
        super().__init__(args=tuple(self.funcs), shape=func1.shape, dtype=func1.dtype)

    @property
    def _factors(self):
        for func in self.funcs:
            if isinstance(func, Multiply):
                yield from func._factors
            else:
                yield func

    @util.reentrant_iter.property
    def representations(self):
        yield multiply, tuple(self.funcs)
        func1, func2 = self.funcs
        for (op1, args1), (op2, args2) in util.iter_product(func1.representations, func2.representations):
            if op1 is op2 and op1 in (transpose, insertaxis, diagonalize, ravel, constant, BoolToInt, IntToFloat, FloatToComplex, transpose, product) and args1[1:] == args2[1:]:
                # excluding _inflate because of potential repeated indices
                yield op1, (args1[0] * args2[0], *args1[1:])

    def _simplified(self):
        func1, func2 = self.funcs
        if isinstance(func1, Zeros) or isinstance(func2, Zeros):
            return zeros_like(self)
        if func1._const_uniform == self.dtype(1):
            return func2
        if func2._const_uniform == self.dtype(1):
            return func1
        combine_ops = transpose, BoolToInt, IntToFloat, FloatToComplex, constant, ravel, diagonalize, insertaxis # highest priority last
        for (f1, rem1, ops1), (f2, rem2, ops2) in util.iter_product(func1._iter_into(multiply), func2._iter_into(multiply)):
            # note: in first iteration, f1 + f2 == self
            if self.dtype == bool and f1 == f2:
                return multiply(f1, *rem1, *rem2)
            severed_ops = ops1 | ops2
            for op, args in (f1 * f2).representations:
                if op in combine_ops and severed_ops.isdisjoint(combine_ops[combine_ops.index(op):]):
                    return multiply(op(*args), *rem1, *rem2)
        for a, b in (func1, func2), (func2, func1):
            stack = [a]
            terms = []
            while stack:
                term = stack.pop()
                for op, args in term.representations:
                    if op is diagonalize:
                        f, axis, newaxis = args
                        terms.append(op(f * takediag(b, axis+(axis>=newaxis), newaxis), axis, newaxis))
                        break
                    elif op is _inflate:
                        f, dofmap, length, axis = args
                        terms.append(op(f * _take(b, dofmap, axis), dofmap, length, axis))
                        break
                    elif op is ravel:
                        f, axis = args
                        terms.append(op(f * unravel(b, axis, f.shape[axis:axis+2]), axis))
                        break
                    elif op is loop_sum:
                        f, index = args
                        if index not in b.arguments:
                            terms.append(op(f * b, index))
                            break
                    elif op is add:
                        stack.extend(args)
                        break
                else:
                    break
            else:
                return add(*terms)

    def _optimized_for_numpy(self):
        if self.dtype == bool:
            return
        func1, func2 = self.funcs
        if func1._const_uniform == -1:
            return Negative(func2)
        if func2._const_uniform == -1:
            return Negative(func1)
        if isinstance(func1, Sign) and func1.func == func2:
            return Absolute(func2)
        if isinstance(func2, Sign) and func2.func == func1:
            return Absolute(func1)
        if self.ndim:
            r = tuple(range(self.ndim))
            return Einsum((func1, func2), (r, r), r)

    evalf = staticmethod(numpy.multiply)

    def _derivative(self, var, seen):
        func1, func2 = self.funcs
        return einsum('A,AB->AB', func1, derivative(func2, var, seen)) \
            + einsum('A,AB->AB', func2, derivative(func1, var, seen))

    @cached_property
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


class Add(Array):

    def __init__(self, funcs: types.frozenmultiset):
        assert isinstance(funcs, types.frozenmultiset) and len(funcs) == 2, f'funcs={funcs!r}'
        self.funcs = funcs
        func1, func2 = funcs
        assert equalshape(func1.shape, func2.shape) and func1.dtype == func2.dtype, 'Add({}, {})'.format(func1, func2)
        super().__init__(args=tuple(self.funcs), shape=func1.shape, dtype=func1.dtype)

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
            dofmaps = set(parts1) | set(parts2)
            if (len(parts1) < len(dofmaps) and len(parts2) < len(dofmaps)  # neither set is a subset of the other; total may be dense
                    and self.shape[axis].isconstant and all(dofmap.isconstant for dofmap in dofmaps)):
                mask = numpy.zeros(int(self.shape[axis]), dtype=bool)
                for dofmap in dofmaps:
                    mask[dofmap.eval()] = True
                if mask.all():  # axis adds up to dense
                    continue
            inflations.append((axis, types.frozendict((dofmap, util.sum(parts[dofmap] for parts in (parts1, parts2) if dofmap in parts)) for dofmap in dofmaps)))
        return tuple(inflations)

    @property
    def _terms(self):
        for func in self.funcs:
            if isinstance(func, Add):
                yield from func._terms
            else:
                yield func

    @util.reentrant_iter.property
    def representations(self):
        yield add, tuple(self.funcs)
        func1, func2 = self.funcs
        for (op1, args1), (op2, args2) in util.iter_product(func1.representations, func2.representations):
            if op1 is op2 and op1 in (transpose, insertaxis, _inflate, diagonalize, ravel, constant, loop_sum, IntToFloat, FloatToComplex) and args1[1:] == args2[1:]:
                yield op1, (args1[0] + args2[0], *args1[1:])

    def _simplified(self):
        func1, func2 = self.funcs
        if isinstance(func1, Zeros):
            return func2
        if isinstance(func2, Zeros):
            return func1
        combine_ops = transpose, BoolToInt, IntToFloat, FloatToComplex, loop_sum, constant, ravel, diagonalize, _inflate, insertaxis # highest priority last
        for (f1, rem1, ops1), (f2, rem2, ops2) in util.iter_product(func1._iter_into(add), func2._iter_into(add)):
            # note: in first iteration, f1 + f2 == self
            if f1 == f2:
                return add(f1 * self.dtype(2), *rem1, *rem2)
            severed_ops = ops1 | ops2
            for op, args in (f1 + f2).representations:
                if op in combine_ops and severed_ops.isdisjoint(combine_ops[combine_ops.index(op):]):
                    return add(op(*args), *rem1, *rem2)
        # try to factor out a * b + a * c = a * (b + c)
        for (f1, rem1, ops2), (f2, rem2, ops2) in util.iter_product(func1._iter_into(multiply), func2._iter_into(multiply)):
            if f1 == f2:
                if builtins.abs(len(rem1) - len(rem2)) == 1 and len(s := tuple(types.frozenmultiset(rem1)^types.frozenmultiset(rem2))) == 1 and s[0]._const_uniform == -1:
                    return zeros_like(self)
                elif rem1 and rem2:
                    return f1 * (multiply(*rem1) + multiply(*rem2))

    evalf = staticmethod(numpy.add)

    def _derivative(self, var, seen):
        return add(*[derivative(f, var, seen) for f in self._terms])

    @cached_property
    def _assparse(self):
        if self.dtype == bool:
            return super()._assparse
        else:
            return _gathersparsechunks(itertools.chain(*[f._assparse for f in self._terms]))

    def _intbounds_impl(self):
        lowers, uppers = zip(*[f._intbounds for f in self._terms])
        return builtins.sum(lowers), builtins.sum(uppers)


class Einsum(Array):

    def __init__(self, args: typing.Tuple[Array, ...], args_idx: typing.Tuple[typing.Tuple[int, ...], ...], out_idx: typing.Tuple[int, ...]):
        assert isinstance(args, tuple) and all(isinstance(arg, Array) for arg in args), f'args={args!r}'
        assert isinstance(args_idx, tuple) and all(isinstance(arg_idx, tuple) and all(isinstance(n, int) for n in arg_idx) for arg_idx in args_idx), f'args_idx={args_idx!r}'
        assert isinstance(out_idx, tuple) and all(isinstance(n, int) for n in out_idx) and len(out_idx) == len(set(out_idx)), f'out_idx={out_idx!r}'
        assert len(args_idx) == len(args) and all(len(idx) == arg.ndim for idx, arg in zip(args_idx, args)), f'len(args_idx)={len(args_idx)}, len(args)={len(args)}'
        dtype = args[0].dtype
        if dtype == bool or any(arg.dtype != dtype for arg in args[1:]):
            raise ValueError('Inconsistent or invalid dtypes.')
        lengths = {}
        for idx, arg in zip(args_idx, args):
            for i, length in zip(idx, arg.shape):
                if i not in lengths:
                    lengths[i] = length
                elif not _equals_simplified(lengths[i], length):
                    raise ValueError('Axes with index {} have different lengths.'.format(i))
        try:
            shape = tuple(lengths[i] for i in out_idx)
        except KeyError:
            raise ValueError('Output axis {} is not listed in any of the arguments.'.format(', '.join(str(i) for i in out_idx if i not in lengths)))
        self.args = args
        self.args_idx = args_idx
        self.out_idx = out_idx
        self._einsumfmt = ','.join(''.join(chr(97+i) for i in idx) for idx in args_idx) + '->' + ''.join(chr(97+i) for i in out_idx)
        self._has_summed_axes = len(lengths) > len(out_idx)
        super().__init__(args=self.args, shape=shape, dtype=dtype)

    def evalf(self, *args):
        if self._has_summed_axes:
            args = tuple(numpy.asarray(arg, order='F') for arg in args)
        return numpy.core.multiarray.c_einsum(self._einsumfmt, *args)

    @property
    def _node_details(self):
        return self._einsumfmt

    def _optimized_for_numpy(self):
        for i, arg in enumerate(self.args):
            if isinstance(arg, Einsum) and not arg._has_summed_axes:
                new_arg = arg.args
                new_idx = tuple(tuple(self.args_idx[i][arg.out_idx.index(j)] for j in arg_idx) for arg_idx in arg.args_idx)
            elif isinstance(arg, Transpose):
                new_arg = arg.func,
                new_idx = tuple(self.args_idx[i][j] for j in numpy.argsort(arg.axes)),
            elif isinstance(arg, InsertAxis):
                new_arg = arg.func,
                new_idx = self.args_idx[i][:-1],
            else:
                continue
            return Einsum(self.args[:i] + new_arg + self.args[i+1:], self.args_idx[:i] + new_idx + self.args_idx[i+1:], self.out_idx)

class Sum(Array):

    def __init__(self, func: Array):
        assert isinstance(func, Array), f'func={func!r}'
        self.func = func
        self.evalf = functools.partial(numpy.any if func.dtype == bool else numpy.sum, axis=-1)
        super().__init__(args=(func,), shape=func.shape[:-1], dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield sum, (self.func, self.ndim)
        for op, args in self.func.representations:
            if op is sum:
                f, axis = args
                if axis < f.ndim-1:
                    yield op, (Sum(f), axis)
                else:
                    yield op, (sum(f, -2), self.ndim)
            elif op is insertaxis:
                f, axis, length = args
                if axis < f.ndim:
                    yield op, (Sum(f), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-1 and newaxis < f.ndim:
                    yield op, (Sum(f), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim:
                    yield op, (Sum(f), dofmap, length, axis)
            elif op is add:
                a, b = args
                yield op, (Sum(a), Sum(b))
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-1:
                    yield op, (Sum(f), axis)
            elif op is constant:
                v, = args
                yield op, (v.sum(-1) if self.dtype != bool else v.any(-1),)
            elif op in (IntToFloat, FloatToComplex):
                f, = args
                yield op, (Sum(f),)
            elif op is loop_sum:
                f, index = args
                yield op, (Sum(f), index)

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if _equals_scalar_constant(self.func.shape[-1], 1):
            return Take(self.func, constant(0))
        if simple := self._as_any(insertaxis, diagonalize, _inflate, add, ravel, constant, IntToFloat, FloatToComplex, loop_sum):
            return simple
        for op, args in self.func.representations:
            if op is _inflate:
                f, dofmap, length, axis = args
                if axis == self.ndim:
                    # NOTE is this correct if dofmap has repeated indices?
                    for i in range(dofmap.ndim):
                        f = Sum(f)
                    return f
            elif op is insertaxis:
                f, axis, length = args
                if axis == f.ndim:
                    return f * astype(length, f.dtype) if f.dtype != bool else f
            elif op is diagonalize:
                f, axis, newaxis = args
                if newaxis == self.func.ndim-1:
                    return f
                if axis == f.ndim-1:
                    return Transpose.from_end(f, newaxis)
            elif op is ravel:
                f, axis = args
                if axis == f.ndim-2:
                    return Sum(Sum(f))
        for factor, rem, ops in self.func._iter_into(multiply):
            for f, axis, length in factor._as(insertaxis, lambda f, i, n: i == self.ndim):
                if rem:
                    return f * Sum(multiply(*rem))
                elif self.dtype == bool:
                    return f
                else:
                    return f * astype(length, self.dtype)

    def _optimized_for_numpy(self):
        if isinstance(self.func, Einsum):
            return Einsum(self.func.args, self.func.args_idx, self.func.out_idx[:-1])
        if isinstance(self.func, Multiply) and self.dtype != bool:
            return Einsum(tuple(self.func.funcs), (tuple(range(self.func.ndim)),)*2, tuple(range(self.ndim)))

    def _derivative(self, var, seen):
        return sum(derivative(self.func, var, seen), self.ndim)

    @cached_property
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


class TakeDiag(Array):

    def __init__(self, func: Array):
        assert isinstance(func, Array) and func.ndim >= 2 and _equals_simplified(*func.shape[-2:]), f'func={func!r}'
        self.func = func
        super().__init__(args=(func,), shape=func.shape[:-1], dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield takediag, (self.func, self.func.ndim-2, self.func.ndim-1)
        for op, args in self.func.representations:
            if op is takediag:
                f, axis, rmaxis = args
                if axis < f.ndim-2 and rmaxis < f.ndim-2:
                    yield op, (TakeDiag(f), axis, rmaxis)
                # TODO expand
            elif op is _take:
                f, indices, axis = args
                if axis < f.ndim-2:
                    yield op, (TakeDiag(f), indices, axis)
                elif indices.ndim == 0:
                    yield op, (takediag(f, -3, -2 if axis == f.ndim-1 else -1), indices, f.ndim-2)
            elif op is insertaxis:
                f, axis, length = args
                if axis < f.ndim-1:
                    yield op, (TakeDiag(f), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-3 and newaxis < f.ndim-2:
                    yield op, (TakeDiag(f), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis < self.func.ndim-2:
                    yield op, (TakeDiag(f), dofmap, length, axis)
                else:
                    taken = _take(f, dofmap, -1 if axis == self.func.ndim-2 else self.func.ndim-2)
                    for i in range(dofmap.ndim):
                        taken = takediag(taken, self.func.ndim-2+i, self.func.ndim-2+dofmap.ndim)
                    yield op, (taken, dofmap, length, self.func.ndim-2)
            elif op in (add, multiply, power, Choose, sign, *_POINTWISE):
                yield op, tuple(TakeDiag(arg) for arg in args)
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-2:
                    yield op, (TakeDiag(f), axis)
                else:
                    unraveled = unravel(f, -1 if axis == self.func.ndim-2 else -3, f.shape[axis:axis+2])
                    yield op, (TakeDiag(takediag(unraveled, -4, -2)), self.ndim-1)
            elif op is unravel:
                f, axis, shape = args
                if axis < f.ndim-2:
                    yield op, (TakeDiag(f), axis, shape)
            elif op in (sum, product):
                f, axis = args
                if axis >= f.ndim-2:
                    yield op, (takediag(f, -3, -2 if axis == f.ndim-1 else -1), f.ndim-2)
                else:
                    yield op, (TakeDiag(f), axis)
            elif op is loop_concatenate:
                f, index, axis = args
                if axis < f.ndim-2:
                    yield op, (TakeDiag(f), index, axis)
            elif op is constant:
                v, = args
                yield op, (numpy.einsum('...kk->...k', v),)
            elif op in (inverse, determinant):
                f, (axis1, axis2) = args
                if axis1 < f.ndim-2 and axis2 < f.ndim-2:
                    yield op, (TakeDiag(f), (axis1, axis2))
            elif op is constant:
                v, = args
                yield op, (TakeDiag.evalf(v),)
            elif op in (loop_sum, SearchSorted):
                f, *args = args
                yield op, (TakeDiag(f), *args)
            elif op is polyval:
                points, coeffs, axis = args
                if axis < points.ndim-2:
                    yield op, (TakeDiag(points), coeffs, axis)
                elif coeffs.ndim == 1:
                    untested
                    yield op, (takediag(points, -3, -1 if axis == points.ndim-2 else -2), coeffs, points.ndim-2)
            elif op is polymul:
                c1, c2, v, axis = args
                if axis < self.func.ndim-2:
                    yield op, (TakeDiag(c1), TakeDiag(c2), v, axis)
            elif op is polygrad:
                c, nvars, caxis, deraxis = args
                if caxis < self.func.ndim-2 and deraxis < self.func.ndim-2:
                    yield op, (TakeDiag(c), nvars, caxis, deraxis)
            elif op is legendre:
                f, axis, degree = args
                if axis < self.func.ndim-2:
                    yield op, (TakeDiag(f), axis, degree)
            elif op is ravelindex:
                ia, ib, na, nb, where = args
                if where[-2] == where[-1]:
                    if where[-1]:
                        ib = TakeDiag(ib, self.indices)
                    else:
                        ia = TakeDiag(ia, self.indices)
                    yield op, (ia, ib, na, nb, numpy.concatenate([where[:-2], numpy.full(self.indices.ndim, where[-1])]))

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        for f, axis, length in self.func._as(insertaxis, lambda f, a, l: a >= f.ndim-1):
            return f
        for f, axis, newaxis in self.func._as(diagonalize):
            if axis == f.ndim-1 and newaxis >= f.ndim-1:
                return f
        if simple := self._as_any(ravel, unravel, insertaxis, _inflate, diagonalize, add, multiply, constant, product, inverse, determinant, sum, loop_sum, loop_concatenate, Choose, ravelindex, polyval, polymul, polygrad, legendre, SearchSorted, *_POINTWISE, _take, power, sign):
            return simple

    @staticmethod
    def evalf(arr):
        return numpy.einsum('...kk->...k', arr, optimize=False)

    def _derivative(self, var, seen):
        return takediag(derivative(self.func, var, seen), self.ndim-1, self.ndim)

    def _intbounds_impl(self):
        return self.func._intbounds


class Take(Array):

    def __init__(self, func: Array, indices: Array):
        assert isinstance(func, Array) and func.ndim > 0, f'func={func!r}'
        assert isinstance(indices, Array) and indices.dtype == int, f'indices={indices!r}'
        self.func = func
        self.indices = indices
        super().__init__(args=(func, indices), shape=func.shape[:-1]+indices.shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield _take, (self.func, self.indices, self.func.ndim-1)
        for op, args in self.func.representations:
            if op is _take:
                f, indices, axis = args
                if axis < self.func.ndim-1:
                    yield op, (Take(f, self.indices), indices, axis)
            elif op is insertaxis:
                f, axis, length = args
                if axis < f.ndim:
                    yield op, (Take(f, self.indices), axis, length)
                else: # axis == f.ndim
                    for i in range(self.indices.ndim):
                        args = appendaxes(f, self.indices.shape[:i] + self.indices.shape[i+1:]), f.ndim+i, length
                        yield op, (appendaxes(f, self.indices.shape[:i] + self.indices.shape[i+1:]), f.ndim+i, self.indices.shape[i])
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-1 and newaxis < f.ndim:
                    yield op, (Take(f, self.indices), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim:
                    yield op, (Take(f, self.indices), dofmap, length, axis)
            elif op in (add, multiply, power, Choose, sign, *_POINTWISE):
                yield op, tuple(Take(arg, self.indices) for arg in args)
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-1:
                    yield op, (Take(f, self.indices), axis)
            elif op is unravel:
                f, axis, shape = args
                if axis < f.ndim-1:
                    yield op, (Take(f, self.indices), axis, shape)
            elif op is constant:
                v, = args
                if self.indices.isconstant:
                    yield op, (v.take(self.indices.eval(), -1),)
            elif op in (inverse, determinant):
                f, (axis1, axis2) = args
                if axis1 < f.ndim-1 and axis2 < f.ndim-1:
                    yield op, (Take(f, self.indices), (axis1, axis2))
            elif op in (loop_sum, SearchSorted):
                f, *args = args
                yield op, (Take(f, self.indices), *args)
            elif op in (sum, product):
                f, axis = args
                if axis < f.ndim-1:
                    yield op, (Take(f, self.indices), axis)
                else:
                    yield op, (_take(f, self.indices, -2), self.ndim)
            elif op is loop_concatenate:
                f, index, axis = args
                if axis < f.ndim-1:
                    yield op, (Take(f, self.indices), index, axis)
            elif op is takediag:
                f, axis, rmaxis = args
                if axis < f.ndim-1 and rmaxis < f.ndim-1:
                    yield op, (Take(f, self.indices), axis, rmaxis)
                elif axis < f.ndim-2 and rmaxis == f.ndim-1:
                    yield op, (_take(f, self.indices, -2), axis, self.ndim)
            elif op is ravelindex:
                ia, ib, na, nb, where = args
                if where[-1]:
                    ib = Take(ib, self.indices)
                else:
                    ia = Take(ia, self.indices)
                yield op, (ia, ib, na, nb, numpy.concatenate([where[:-1], numpy.full(self.indices.ndim, where[-1])]))
            elif op is polyval:
                points, coeffs, axis = args
                if axis < points.ndim-1:
                    yield op, (Take(points, self.indices), coeffs, axis+self.indices.ndim-1)
                elif coeffs.ndim == 1:
                    yield op, (_take(points, self.indices, -2), coeffs, points.ndim+self.indices.ndim-2)
                else:
                    yield op, (points, _take(coeffs, self.indices, -2), axis)
            elif op is polymul:
                c1, c2, v, axis = args
                if axis < c1.ndim-1:
                    yield op, (Take(c1, self.indices), Take(c2, self.indices), v, axis)
            elif op is polygrad:
                c, nvars, caxis, deraxis = args
                if caxis < c.ndim-1 and deraxis < c.ndim:
                    yield op, (Take(c, self.indices), nvars, caxis, deraxis)
            elif op is legendre:
                f, axis, degree = args
                if axis < f.ndim:
                    yield op, (Take(f, self.indices), axis, degree)

    def _simplified(self):
        if any(isinstance(n, Zeros) for n in self.indices.shape):
            return zeros_like(self)
        for f, axis, newaxis in self.func._as(diagonalize, lambda f, a, na: a == f.ndim-1 or na == f.ndim):
            if newaxis != f.ndim:
                f = Transpose.from_end(f, newaxis)
                axis = newaxis
            taken = _take(f, self.indices, axis)
            for i in range(self.indices.ndim):
                taken = diagonalize(taken, axis+i)
            return _inflate(taken, self.indices, f.shape[axis], axis)
        for f, dofmap, length, axis in self.func._as(_inflate, lambda f, d, l, a: a + d.ndim == f.ndim):
            newindex, newdofmap = SwapInflateTake(dofmap, self.indices)
            if dofmap.ndim:
                for i in range(dofmap.ndim-1):
                    f = Ravel(f)
                intersection = Take(f, newindex)
            else: # kronecker; newindex is all zeros (but of varying length)
                intersection = InsertAxis(f, newindex.shape[0])
            if self.indices.ndim:
                swapped = Inflate(intersection, newdofmap, util.product(self.indices.shape))
                for i in range(self.indices.ndim-1):
                    swapped = Unravel(swapped, self.indices.shape[i], util.product(self.indices.shape[i+1:]))
            else: # get; newdofmap is all zeros (but of varying length)
                swapped = Sum(intersection)
            return swapped
        for length, in self.func._as(Range):
            return InRange(self.indices, length)
        if self.indices.ndim == 0:
            for f, axis, length in self.func._as(insertaxis, lambda f, a, l: a == f.ndim):
                return f # TODO check indices < length or add runtime check?
        for indices, axis in self.indices._as(ravel):
            return ravel(Take(self.func, indices), self.func.ndim-1+axis)
        for ia, ib, na, nb, where in self.indices._as(ravelindex):
            if _equals_simplified(self.func.shape[-1], na * nb):
                wb, = where.nonzero()
                return Transpose.to_end(Take(_take(Unravel(self.func, na, nb), ia, -2), ib), *wb+self.func.ndim-1)
        for length, in self.indices._as(Range):
            if self.func.shape[-1] == length:
                return self.func
        if simple := self._as_any(ravel, insertaxis, _inflate, diagonalize, add, multiply, constant, product, inverse, determinant, sum, loop_sum, loop_concatenate, Choose, ravelindex, polyval, polymul, polygrad, legendre, SearchSorted, *_POINTWISE):
            return simple

    @staticmethod
    def evalf(arr, indices):
        return arr[..., indices]

    def _derivative(self, var, seen):
        return _take(derivative(self.func, var, seen), self.indices, self.func.ndim-1)

    def _intbounds_impl(self):
        return self.func._intbounds


class Power(Array):

    def __init__(self, func: Array, power: Array):
        assert isinstance(func, Array) and func.dtype != bool, f'func={func!r}'
        assert isinstance(power, Array) and (power.dtype in (float, complex) or power.dtype == int and power._intbounds[0] >= 0), f'power={power!r}'
        assert equalshape(func.shape, power.shape) and func.dtype == power.dtype, 'Power({}, {})'.format(func, power)
        self.func = func
        self.power = power
        super().__init__(args=(func, power), shape=func.shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield power, (self.func, self.power)
        for (op, args1), (op_, args2) in util.iter_product(self.func.representations, self.power.representations):
            if op is op_:
                if op is constant:
                    v1, = args1
                    v2, = args2
                    yield op, (numpy.power(v1, v2),)
                elif op is insertaxis:
                    f1, axis1, length1 = args1
                    f2, axis2, length2 = args2
                    if axis1 == axis2:
                        assert _equals_simplified(length1, length2)
                        yield op, (Power(f1, f2), axis1, length1)
                elif op is ravel:
                    f1, axis1 = args1
                    f2, axis2 = args2
                    if axis1 == axis2:
                        yield op, (Power(f1, f2), axis1)

    def _simplified(self):
        if isinstance(self.power, Zeros):
            return ones_like(self)
        if simple := self._as_any(constant, insertaxis, ravel):
            return simple
        p = self.power._const_uniform
        if p == 1:
            return self.func
        elif p == 2:
            return self.func * self.func
        if self.dtype != complex:
            for f, q in self.func._as(power):
                if p and p % 2 != 0 and (q_ := q._const_uniform) and q_ % 2 == 0: # power was even, becomes odd
                    f = abs(f)
                return Power(f, self.power * q)

    def _optimized_for_numpy(self):
        p = self.power._const_uniform
        if p == -1:
            return Reciprocal(self.func)
        elif p == -2:
            return Reciprocal(self.func * self.func)

    evalf = staticmethod(numpy.power)

    def _derivative(self, var, seen):
        if self.power.isconstant:
            p = self.power.eval()
            return einsum('A,A,AB->AB', constant(p), power(self.func, p - (p != 0)), derivative(self.func, var, seen))
        if self.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        # self = func**power
        # ln self = power * ln func
        # self` / self = power` * ln func + power * func` / func
        # self` = power` * ln func * self + power * func` * func**(power-1)
        return einsum('A,A,AB->AB', self.power, power(self.func, self.power - astype(1, self.power.dtype)), derivative(self.func, var, seen)) \
            + einsum('A,A,AB->AB', ln(self.func), self, derivative(self.power, var, seen))


class Pointwise(Array):
    '''
    Abstract base class for pointwise array functions.
    '''

    deriv = None
    return_type = None
    linear = () # indices of linear arguments
    zero_preserving = () # indices of nonlinear arguments that map zero to zero

    def __init__(self, *args: Array, **params):
        assert all(isinstance(arg, Array) for arg in args), f'args={args!r}'
        dtype = self.__class__.return_type(*[arg.dtype for arg in args], **params)
        shape0 = args[0].shape
        assert all(equalshape(arg.shape, shape0) for arg in args[1:]), 'pointwise arguments have inconsistent shapes'
        self.args = args
        self.params = params
        if params:
            self.evalf = functools.partial(self.evalf, **params)
        super().__init__(args=args, shape=shape0, dtype=dtype)

    @util.reentrant_iter.property
    def representations(self):
        if self.params:
            return
        yield self.__class__, self.args
        for opargs in util.iter_many(arg.representations for arg in self.args):
            for i, (op, args) in enumerate(opargs):
                if op is ravel:
                    f, axis = args
                    yield op, (self.__class__(*[f if i == j else unravel(arg, axis, f.shape[axis:axis+2]) for j, arg in enumerate(self.args)]), axis)
                elif op in (transpose, insertaxis):
                    if i == 0 and all(op_ is op and args_[1:] == args[1:] for op_, args_ in opargs[1:]):
                        yield op, (self.__class__(*[f_ for op_, (f_, *_) in opargs]), *args[1:])
                elif op is constant:
                    if i == 0 and all(op_ is op for op_, args_ in opargs[1:]):
                        yield op, (self.evalf(*[v_ for op_, (v_,) in opargs]),)
                elif op is diagonalize:
                    if i in self.linear or i in self.zero_preserving:
                        f, axis, newaxis = args
                        yield op, (self.__class__(*[f if i == j else takediag(arg, axis, newaxis) for j, arg in enumerate(self.args)]), axis, newaxis)
                elif op is _inflate:
                    if i in self.linear:
                        f, dofmap, length, axis = args
                        yield op, (self.__class__(*[f if i == j else _take(arg, axis) for j, arg in enumerate(self.args)]), dofmap, length, axis)

    def _newargs(self, *args):
        '''
        Reinstantiate self with different arguments. Parameters are preserved,
        as these are considered part of the type.
        '''

        return self.__class__(*args, **self.params)

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
        for i in self.linear + self.zero_preserving:
            if iszero(self.args[i]):
                return zeros_like(self)
            for axis1, axis2, *other in map(sorted, self.args[i]._diagonals):
                return diagonalize(self._newargs(*[takediag(arg, axis1, axis2) for arg in self.args]), axis1, axis2)
        for i in self.linear:
            for axis, parts in self.args[i]._inflations:
                return util.sum(_inflate(self._newargs(*self.args[:i], func, *self.args[i+1:]), dofmap, self.shape[axis], axis) for dofmap, func in parts.items())
        if isinstance(self.args[0], Transpose) and all(isinstance(arg, Transpose) and arg.axes == self.args[0].axes for arg in self.args[1:]):
            return Transpose(self._newargs(*[arg.func for arg in self.args]), self.args[0].axes)
        *uninserted, where = unalign(*self.args)
        if len(where) != self.ndim:
            return align(self._newargs(*uninserted), where, self.shape)

    def _derivative(self, var, seen):
        if self.dtype == complex or var.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        elif self.deriv is not None:
            return util.sum(einsum('A,AB->AB', deriv(*self.args, **self.params), derivative(arg, var, seen)) for arg, deriv in zip(self.args, self.deriv))
        else:
            return super()._derivative(var, seen)


class Holomorphic(Pointwise):
    '''
    Abstract base class for holomorphic array functions.
    '''

    @staticmethod
    def return_type(*dtypes, **params):
        return_type = dtypes[-1]
        if not all(dtype == return_type for dtype in dtypes[:-1]):
            raise ValueError('All arguments must have the same dtype but got {} and {}.'.format(', '.join(map(str, dtypes[:-1])), return_type))
        if return_type not in (float, complex):
            raise ValueError(f'{self.__class__.__name__} is not defined for arguments of dtype {return_type}')
        return return_type

    def _derivative(self, var, seen):
        if self.deriv is not None:
            return util.sum(einsum('A,AB->AB', deriv(*self.args, **self.params), derivative(arg, var, seen)) for arg, deriv in zip(self.args, self.deriv))
        else:
            return super()._derivative(var, seen)


class Reciprocal(Holomorphic):
    evalf = staticmethod(numpy.reciprocal)


class Negative(Pointwise):
    linear = 0,
    evalf = staticmethod(numpy.negative)
    def return_type(T):
        if T == bool:
            raise ValueError('boolean values cannot be negated')
        return T

    def _intbounds_impl(self):
        lower, upper = self.args[0]._intbounds
        return -upper, -lower


class FloorDivide(Pointwise):
    zero_preserving = 0,
    evalf = staticmethod(numpy.floor_divide)
    def return_type(dividend, divisor):
        if dividend != divisor:
            raise ValueError(f'All arguments must have the same dtype but got {dividend} and {divisor}.')
        if dividend == bool:
            raise ValueError(f'The boolean floor division is not supported.')
        return dividend

    def _intbounds_impl(self):
        lower, upper = self.args[0]._intbounds
        divisor_lower, divisor_upper = self.args[1]._intbounds
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
    zero_preserving = 0,
    evalf = staticmethod(numpy.absolute)

    def return_type(T):
        if T == bool:
            raise ValueError('The boolean absolute value is not implemented.')
        return float if T == complex else T

    def _intbounds_impl(self):
        lower, upper = self.args[0]._intbounds
        extrema = builtins.abs(lower), builtins.abs(upper)
        if lower <= 0 and upper >= 0:
            return 0, max(extrema)
        else:
            return min(extrema), max(extrema)


class Cos(Holomorphic):
    'Cosine, element-wise.'
    evalf = staticmethod(numpy.cos)
    deriv = lambda x: -Sin(x),


class Sin(Holomorphic):
    'Sine, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.sin)
    deriv = Cos,


class Tan(Holomorphic):
    'Tangent, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.tan)
    deriv = lambda x: Cos(x)**astype(-2, x.dtype),


class ArcSin(Holomorphic):
    'Inverse sine, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.arcsin)
    deriv = lambda x: reciprocal(sqrt(astype(1, x.dtype)-x**astype(2, x.dtype))),


class ArcCos(Holomorphic):
    'Inverse cosine, element-wise.'
    evalf = staticmethod(numpy.arccos)
    deriv = lambda x: -reciprocal(sqrt(astype(1, x.dtype)-x**astype(2, x.dtype))),


class ArcTan(Holomorphic):
    'Inverse tangent, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.arctan)
    deriv = lambda x: reciprocal(astype(1, x.dtype)+x**astype(2, x.dtype)),


class Sinc(Holomorphic):
    evalf = staticmethod(numeric.sinc)
    deriv = lambda x, n: Sinc(x, n=n+1),


class CosH(Holomorphic):
    'Hyperbolic cosine, element-wise.'
    evalf = staticmethod(numpy.cosh)
    deriv = lambda x: SinH(x),


class SinH(Holomorphic):
    'Hyperbolic sine, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.sinh)
    deriv = CosH,


class TanH(Holomorphic):
    'Hyperbolic tangent, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.tanh)
    deriv = lambda x: astype(1, x.dtype) - TanH(x)**astype(2, x.dtype),


class ArcTanH(Holomorphic):
    'Inverse hyperbolic tangent, element-wise.'
    zero_preserving = 0,
    evalf = staticmethod(numpy.arctanh)
    deriv = lambda x: reciprocal(astype(1, x.dtype)-x**astype(2, x.dtype)),


class Exp(Holomorphic):
    evalf = staticmethod(numpy.exp)
    deriv = lambda x: Exp(x),


class Log(Holomorphic):
    evalf = staticmethod(numpy.log)
    deriv = lambda x: reciprocal(x),


class Mod(Pointwise):
    zero_preserving = 0,
    evalf = staticmethod(numpy.mod)

    def return_type(dividend, divisor):
        if dividend != divisor:
            raise ValueError(f'All arguments must have the same dtype but got {dividend} and {divisor}.')
        if dividend == bool:
            raise ValueError(f'The boolean floor division is not supported.')
        if dividend == complex:
            raise ValueError(f'The complex floor division is not supported.')
        return dividend

    def _intbounds_impl(self):
        dividend, divisor = self.args
        lower_divisor, upper_divisor = divisor._intbounds
        if lower_divisor > 0:
            lower_dividend, upper_dividend = dividend._intbounds
            if 0 <= lower_dividend and upper_dividend < lower_divisor:
                return lower_dividend, upper_dividend
            else:
                return 0, upper_divisor - 1
        else:
            return super()._intbounds_impl()

    def _simplified(self):
        dividend, divisor = self.args
        lower_divisor, upper_divisor = divisor._intbounds
        if lower_divisor > 0:
            lower_dividend, upper_dividend = dividend._intbounds
            if 0 <= lower_dividend and upper_dividend < lower_divisor:
                return dividend
        return super()._simplified()


class ArcTan2(Pointwise):
    evalf = staticmethod(numpy.arctan2)
    deriv = lambda x, y: y / (x**astype(2, x.dtype) + y**astype(2, x.dtype)), lambda x, y: -x / (x**astype(2, x.dtype) + y**astype(2, x.dtype))
    def return_type(T1, T2):
        if T1 == complex or T2 == complex:
            raise ValueError('arctan2 is not defined for complex numbers')
        return float


class Greater(Pointwise):
    evalf = staticmethod(numpy.greater)
    def return_type(T1, T2):
        if T1 != T2:
            raise ValueError('Cannot compare different dtypes.')
        elif T1 == complex:
            raise ValueError('Complex numbers have no total order.')
        elif T1 == bool:
            raise ValueError('Use logical operators to compare booleans.')
        return bool


class Equal(Pointwise):
    evalf = staticmethod(numpy.equal)
    def return_type(T1, T2):
        if T1 != T2:
            raise ValueError('Cannot compare different dtypes.')
        return bool

    def _simplified(self):
        a1, a2 = self.args
        if a1 == a2:
            return ones(self.shape, bool)
        if self.ndim == 2:
            u1, w1 = unalign(a1)
            u2, w2 = unalign(a2)
            if u1.ndim == u2.ndim == 1 and u1 == u2 and w1 != w2 and isinstance(u1, Range):
                # NOTE: Once we introduce isunique we can relax the Range bound
                return Diagonalize(ones(u1.shape, bool))
        return super()._simplified()


class Less(Pointwise):
    evalf = staticmethod(numpy.less)
    def return_type(T1, T2):
        if T1 != T2:
            raise ValueError('Cannot compare different dtypes.')
        elif T1 == complex:
            raise ValueError('Complex numbers have no total order.')
        elif T1 == bool:
            raise ValueError('Use logical operators to compare booleans.')
        return bool


class LogicalNot(Pointwise):
    evalf = staticmethod(numpy.logical_not)
    def return_type(T):
        if T != bool:
            raise ValueError(f'Expected a boolean but got {T}.')
        return bool

    def _simplified(self):
        arg, = self.args
        if isinstance(arg, LogicalNot):
            return arg.args[0]
        return super()._simplified()


class Minimum(Pointwise):
    evalf = staticmethod(numpy.minimum)
    deriv = lambda x, y: .5 - .5 * Sign(x - y), lambda x, y: .5 + .5 * Sign(x - y)
    def return_type(T1, T2):
        if T1 == complex or T2 == complex:
            raise ValueError('Complex numbers have no total order.')
        return float if float in (T1, T2) else int if int in (T1, T2) else bool

    def _simplified(self):
        if self.dtype == int:
            lower1, upper1 = self.args[0]._intbounds
            lower2, upper2 = self.args[1]._intbounds
            if upper1 <= lower2:
                return self.args[0]
            elif upper2 <= lower1:
                return self.args[1]
        return super()._simplified()

    def _intbounds_impl(self):
        lower1, upper1 = self.args[0]._intbounds
        lower2, upper2 = self.args[1]._intbounds
        return min(lower1, lower2), min(upper1, upper2)


class Maximum(Pointwise):
    evalf = staticmethod(numpy.maximum)
    deriv = lambda x, y: .5 + .5 * Sign(x - y), lambda x, y: .5 - .5 * Sign(x - y)
    def return_type(T1, T2):
        if T1 == complex or T2 == complex:
            raise ValueError('Complex numbers have no total order.')
        return float if float in (T1, T2) else int if int in (T1, T2) else bool

    def _simplified(self):
        if self.dtype == int:
            lower1, upper1 = self.args[0]._intbounds
            lower2, upper2 = self.args[1]._intbounds
            if upper2 <= lower1:
                return self.args[0]
            elif upper1 <= lower2:
                return self.args[1]
        return super()._simplified()

    def _intbounds_impl(self):
        lower1, upper1 = self.args[0]._intbounds
        lower2, upper2 = self.args[1]._intbounds
        return max(lower1, lower2), max(upper1, upper2)


class Conjugate(Pointwise):
    linear = 0,
    evalf = staticmethod(numpy.conjugate)
    def return_type(T):
        if T != complex:
            raise ValueError(f'Conjugate is not defined for arguments of type {T}')
        return complex

    def _simplified(self):
        for f, in self.args[0]._as(FloatToComplex):
            return FloatToComplex(f)
        return super()._simplified()


class Real(Pointwise):
    linear = 0,
    evalf = staticmethod(numpy.real)
    def return_type(T):
        if T != complex:
            raise ValueError(f'Real is not defined for arguments of type {T}')
        return float

    def _simplified(self):
        for f, in self.args[0]._as(FloatToComplex):
            return f
        return super()._simplified()


class Imag(Pointwise):
    linear = 0,
    evalf = staticmethod(numpy.imag)
    def return_type(T):
        if T != complex:
            raise ValueError(f'Real is not defined for arguments of type {T}')
        return float

    def _simplified(self):
        for f, in self.args[0]._as(FloatToComplex):
            return zeros_like(self)
        return super()._simplified()


class Cast(Pointwise):

    def evalf(self, arg):
        return numpy.array(arg, dtype=self.dtype)

    def _intbounds_impl(self):
        if self.args[0].dtype == bool:
            return 0, 1
        else:
            return self.args[0]._intbounds

    @property
    def _const_uniform(self):
        value = self.args[0]._const_uniform
        if value is not None:
            return self.dtype(value)


class BoolToInt(Cast):
    zero_preserving = 0,
    def return_type(T):
        if T != bool:
            raise TypeError(f'Expected an array with dtype bool but got {T.__name__}.')
        return int


class IntToFloat(Cast):
    linear = 0,
    def return_type(T):
        if T != int:
            raise TypeError(f'Expected an array with dtype int but got {T.__name__}.')
        return float

    def _derivative(self, var, seen):
        return Zeros(self.shape + var.shape, dtype=self.dtype)


class FloatToComplex(Cast):
    linear = 0,
    def return_type(T):
        if T != float:
            raise TypeError(f'Expected an array with dtype float but got {T.__name__}.')
        return complex

    def _derivative(self, var, seen):
        if var.dtype == complex:
            raise ValueError('The complex derivative does not exist.')
        arg, = self.args
        return FloatToComplex(derivative(arg, var, seen))


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

    def __init__(self, func: Array):
        assert isinstance(func, Array) and func.dtype != complex, f'func={func!r}'
        self.func = func
        super().__init__(args=(func,), shape=func.shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield sign, (self.func,)
        for op, args in self.func.representations:
            if op is constant:
                v, = args
                yield op, (numpy.sign(v),)
            elif op in (insertaxis, diagonalize, ravel):
                f, *args = args
                yield op, (Sign(f), *args)
            elif op in (multiply, IntToFloat):
                yield op, tuple(Sign(arg) for arg in args)

    def _simplified(self):
        if simple := self._as_any(constant, insertaxis, multiply, IntToFloat, diagonalize, ravel):
            return simple
        for f, in self.func._as(sign):
            return self.func

    evalf = staticmethod(numpy.sign)

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

    def __init__(self, points: Array, target: Array, interpolation: str):
        assert isinstance(points, Array) and points.ndim == 2, f'points={points!r}'
        assert isinstance(target, Array) and target.ndim == 2, f'target={target!r}'
        assert points.shape[1] == target.shape[1]
        if interpolation == 'none':
            self.evalf = self.evalf_none
        elif interpolation == 'nearest':
            self.evalf = self.evalf_nearest
        else:
            raise ValueError(f'invalid interpolation {interpolation!r}; valid values are "none" and "nearest"')
        super().__init__(args=(points, target), shape=(points.shape[0], target.shape[0]), dtype=float)

    @staticmethod
    def evalf_none(points, target):
        if points.shape != target.shape or not numpy.equal(points, target).all():
            raise ValueError('points do not correspond to the target sample; consider using "nearest" interpolation if this is desired')
        return numpy.eye(len(points))

    @staticmethod
    def evalf_nearest(points, target):
        nearest = numpy.linalg.norm(points[:,numpy.newaxis,:] - target[numpy.newaxis,:,:], axis=2).argmin(axis=1)
        return numpy.eye(len(target))[nearest]


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
    cumprod = [shape[i] for i in var_axes]
    for i in reversed(range(len(cumprod)-1)):
        cumprod[i] *= cumprod[i+1]  # work backwards so that the shape check matches in Unravel
    offsets = _SizesToOffsets(asarray([d.shape[-1] for d in raveled]))
    elemwise = Take(concat, Range(cumprod[0]) + Take(offsets, index))
    for i in range(len(var_axes)-1):
        elemwise = Unravel(elemwise, shape[var_axes[i]], cumprod[i+1])
    return Transpose.from_end(elemwise, *var_axes)


class Eig(Evaluable):

    def __init__(self, func: Array, symmetric: bool = False):
        assert isinstance(func, Array) and func.ndim >= 2 and _equals_simplified(func.shape[-1], func.shape[-2]), f'func={func!r}'
        assert isinstance(symmetric, bool), f'symmetric={symmetric!r}'
        self.symmetric = symmetric
        self.func = func
        self._w_dtype = float if symmetric else complex
        self._vt_dtype = float if symmetric and func.dtype != complex else complex
        super().__init__(args=(func,))

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

    def __init__(self, arrays: Evaluable, index: int, shape: typing.Tuple[Array, ...], dtype: Dtype):
        assert isinstance(arrays, Evaluable), f'arrays={arrays!r}'
        assert isinstance(index, int), f'index=f{index}'
        self.arrays = arrays
        self.index = index
        super().__init__(args=(arrays,), shape=shape, dtype=dtype)

    def _simplified(self):
        if isinstance(self.arrays, Tuple):
            # This allows the self.arrays evaluable to simplify itself into a
            # Tuple and its components be exposed to the function tree.
            return self.arrays[self.index]

    def evalf(self, arrays):
        assert isinstance(arrays, tuple)
        return arrays[self.index]

    def _node(self, cache, subgraph, times):
        if self in cache:
            return cache[self]
        node = self.arrays._node(cache, subgraph, times)
        if isinstance(node, TupleNode):
            node = node[self.index]
        cache[self] = node
        return node

    def _intbounds_impl(self):
        intbounds = getattr(self.arrays, '_intbounds_tuple', None)
        return intbounds[self.index] if intbounds else (float('-inf'), float('inf'))


class Singular(Array):

    def __init__(self, dtype):
        assert dtype in (float, complex), 'Singular dtype must be float or complex'
        super().__init__(args=(), shape=(), dtype=dtype)

    def evalf(self):
        warnings.warn('singular matrix', RuntimeWarning)
        return numpy.array(numpy.nan, self.dtype)


class Zeros(Array):
    'zero'

    def __init__(self, shape, dtype):
        super().__init__(args=shape, shape=shape, dtype=dtype)

    @util.reentrant_iter.property
    def representations(self):
        for axis in range(self.ndim):
            yield insertaxis, (Zeros(self.shape[:axis] + self.shape[axis+1:], self.dtype), axis, self.shape[axis])

    def evalf(self, *shape):
        return numpy.zeros(shape, dtype=self.dtype)

    def _node(self, cache, subgraph, times):
        if self.ndim:
            return super()._node(cache, subgraph, times)
        elif self in cache:
            return cache[self]
        else:
            cache[self] = node = DuplicatedLeafNode('0', (type(self).__name__, times[self]))
            return node

    @cached_property
    def _assparse(self):
        return ()

    def _intbounds_impl(self):
        return 0, 0


class Inflate(Array):

    def __init__(self, func: Array, dofmap: Array, length: Array):
        assert isinstance(func, Array), f'func={func!r}'
        assert isinstance(dofmap, Array), f'dofmap={dofmap!r}'
        assert _isindex(length), f'length={length!r}'
        assert equalshape(func.shape[func.ndim-dofmap.ndim:], dofmap.shape), f'func.shape={func.shape!r}, dofmap.shape={dofmap.shape!r}'
        self.func = func
        self.dofmap = dofmap
        self.length = length
        self.warn = not dofmap.isconstant
        super().__init__(args=(func, dofmap, length), shape=(*func.shape[:func.ndim-dofmap.ndim], length), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield _inflate, (self.func, self.dofmap, self.length, self.ndim-1)
        for op, args in self.func.representations:
            if op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim <= f.ndim - self.dofmap.ndim:
                    yield op, (Inflate(f, self.dofmap, self.length), dofmap, length, axis)
            elif op is insertaxis:
                f, axis, length = args
                if axis < self.ndim-1:
                    yield op, (Inflate(f, self.dofmap, self.length), axis, length)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < self.ndim-2 and newaxis <= self.ndim-2:
                    yield op, (Inflate(f, self.dofmap, self.length), axis, newaxis)
            elif op is add:
                a, b = args
                yield op, (Inflate(a, self.dofmap, self.length), Inflate(b, self.dofmap, self.length))
            elif op is ravel:
                f, axis = args
                if axis < self.ndim-1:
                    yield op, (Inflate(f, self.dofmap, self.length), axis)

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
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        for axis in range(self.dofmap.ndim):
            if _equals_scalar_constant(self.dofmap.shape[axis], 1):
                return Inflate(_take(self.func, constant(0), self.func.ndim-self.dofmap.ndim+axis), _take(self.dofmap, constant(0), axis), self.length)
        if self.dofmap.ndim == 0 and _equals_scalar_constant(self.dofmap, 0) and _equals_scalar_constant(self.length, 1):
            return InsertAxis(self.func, constant(1))
        for f, axis in self.func._as(ravel):
            if axis >= self.ndim-1:
                return Inflate(f, unravel(self.dofmap, axis-(self.ndim-1), f.shape[axis:axis+2]), self.length)
        for f, dofmap, length, axis in self.func._as(_inflate):
            i = axis - (self.ndim-1)
            if i >= 0:
                return Inflate(f, _take(self.dofmap, dofmap, i), self.length)
        for ia, ib, na, nb, where in self.dofmap._as(ravelindex):
            if _equals_simplified(self.length, na * nb) and where[ia.ndim:].all():
                return Ravel(Inflate(_inflate(self.func, ia, na, self.func.ndim - self.dofmap.ndim), ib, nb))
        for length, in self.dofmap._as(Range):
            if length == self.length:
                return self.func
        if simple := self._as_any(ravel, insertaxis):
            return simple

    def evalf(self, array, indices, length):
        assert indices.ndim == self.dofmap.ndim
        assert length.ndim == 0
        if self.warn and int(length) > indices.size:
            warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
        inflated = numpy.zeros(array.shape[:array.ndim-indices.ndim] + (length,), dtype=self.dtype)
        numpy.add.at(inflated, (slice(None),)*(self.ndim-1)+(indices,), array)
        return inflated

    def _derivative(self, var, seen):
        return _inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.ndim-1)

    @cached_property
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


class SwapInflateTake(Evaluable):

    def __init__(self, inflateidx, takeidx):
        self.inflateidx = inflateidx
        self.takeidx = takeidx
        super().__init__(args=(inflateidx, takeidx))

    def _simplified(self):
        if self.isconstant:
            return Tuple(tuple(map(constant, self.eval())))

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


class Diagonalize(Array):

    def __init__(self, func: Array):
        assert isinstance(func, Array) and func.ndim > 0, f'func={func!r}'
        self.func = func
        super().__init__(args=(func,), shape=(*func.shape, func.shape[-1]), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield diagonalize, (self.func, self.ndim-2, self.ndim-1)
        for op, args in self.func.representations:
            if op is diagonalize:
                f, axis, newaxis = args
                assert f.ndim == self.ndim-2
                if newaxis == self.ndim-2:
                    yield op, (self.func, axis, self.ndim-1)
                    yield op, (self.func, axis, self.ndim-2)
                else:
                    yield op, (Diagonalize(f), axis, newaxis)
            elif op is insertaxis:
                f, axis, length = args
                if axis < self.ndim-2:
                    yield op, (Diagonalize(f), axis, length)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim:
                    yield op, (Diagonalize(f), dofmap, length, axis)
            elif op in (add, multiply):
                a, b = args
                yield op, (Diagonalize(a), Diagonalize(b))
            elif op is ravel:
                f, axis = args
                if axis < self.func.ndim-1:
                    yield op, (Diagonalize(f), axis)
                #else:
                #    assert axis == self.func.ndim-1
                #    unraveled = Diagonalize(diagonalize(f, -2, -2))
                #    yield op, (ravel(unraveled, axis), axis+1)
                #    yield op, (ravel(unraveled, axis+2), axis)

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
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if self.shape[-1] == 1:
            return InsertAxis(self.func, 1)
        if simple := self._as_any(ravel, insertaxis, _inflate):
            return simple

    @staticmethod
    def evalf(arr):
        result = numpy.zeros(arr.shape+(arr.shape[-1],), dtype=arr.dtype, order='F')
        diag = numpy.core.multiarray.c_einsum('...ii->...i', result)
        diag[:] = arr
        return result

    def _derivative(self, var, seen):
        return diagonalize(derivative(self.func, var, seen), self.ndim-2, self.ndim-1)

    @cached_property
    def _assparse(self):
        return tuple((*indices, indices[-1], values) for *indices, values in self.func._assparse)


class Guard(Array):
    'bar all simplifications'

    def __init__(self, fun: Array):
        assert isinstance(fun, Array), f'fun={fun!r}'
        self.fun = fun
        super().__init__(args=(fun,), shape=fun.shape, dtype=fun.dtype)

    @property
    def isconstant(self):
        return False  # avoid simplifications based on fun being constant

    @staticmethod
    def evalf(dat):
        return dat

    def _derivative(self, var, seen):
        return Guard(derivative(self.fun, var, seen))


class Find(Array):
    'indices of boolean index vector'

    def __init__(self, where: Array):
        assert isarray(where) and where.ndim == 1 and where.dtype == bool
        self.where = where
        super().__init__(args=(where,), shape=(Sum(BoolToInt(where)),), dtype=int)

    @staticmethod
    def evalf(where):
        return where.nonzero()[0]

    def _simplified(self):
        if self.isconstant:
            return constant(self.eval())


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

    def __init__(self, func: Array, var: DerivativeTargetBase, derivative: Array) -> None:
        self._func = func
        self._var = var
        self._deriv = derivative
        super().__init__(args=(func,), shape=func.shape, dtype=func.dtype)

    @property
    def arguments(self):
        return self._func.arguments | {self._var}

    @staticmethod
    def evalf(func: numpy.ndarray) -> numpy.ndarray:
        return func

    def _derivative(self, var: DerivativeTargetBase, seen) -> Array:
        if var == self._var:
            return self._deriv
        else:
            return derivative(self._func, var, seen)

    def _simplified(self) -> Array:
        return self._func


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

    def __init__(self, name: str, shape: typing.Tuple[Array, ...], dtype: Dtype = float):
        assert isinstance(name, str), f'name={name!r}'
        assert isinstance(shape, tuple) and all(_isindex(n) for n in shape), f'shape={shape!r}'
        self._name = name
        super().__init__(args=(EVALARGS, *shape), shape=shape, dtype=dtype)

    def evalf(self, evalargs, *shape):
        try:
            value = evalargs[self._name]
        except KeyError:
            raise ValueError(f'argument {self._name!r} missing')
        value = numpy.asarray(value)
        if value.shape != shape:
            raise ValueError(f'argument {self._name!r} has the wrong shape: expected {shape}, got {value.shape}')
        return value.astype(self.dtype, casting='safe', copy=False)

    def _derivative(self, var, seen):
        if isinstance(var, Argument) and var._name == self._name and self.dtype in (float, complex):
            result = ones(self.shape, self.dtype)
            for i, sh in enumerate(self.shape):
                result = diagonalize(result, i, i+self.ndim)
            return result
        else:
            return zeros(self.shape+var.shape)

    def __str__(self):
        return '{} {!r} <{}>'.format(self.__class__.__name__, self._name, self._shape_str(form=str))

    def _node(self, cache, subgraph, times):
        if self in cache:
            return cache[self]
        else:
            label = '\n'.join(filter(None, (type(self).__name__, self._name, self._shape_str(form=repr))))
            cache[self] = node = DuplicatedLeafNode(label, (type(self).__name__, times[self]))
            return node

    @property
    def arguments(self):
        return frozenset({self})


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

    def __init__(self, identifier, shape):
        self.identifier = identifier
        super().__init__(args=(), shape=shape, dtype=float)

    def evalf(self):
        raise Exception('{} cannot be evaluabled'.format(type(self).__name__))


class Ravel(Array):

    def __init__(self, func: Array):
        assert isinstance(func, Array) and func.ndim >= 2, f'func={func!r}'
        self.func = func
        super().__init__(args=(func,), shape=(*func.shape[:-2], func.shape[-2] * func.shape[-1]), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield ravel, (self.func, self.ndim-1)
        for op, args in self.func.representations:
            if op is ravel:
                f, axis = args
                if axis+1 < f.ndim-2:
                    yield op, (Ravel(f), axis)
            elif op is insertaxis:
                f, axis, length = args
                if axis < f.ndim-1:
                    yield op, (Ravel(f), axis, length)
                #else:
                #    for f2, axis2, length2 in f._as(op):
                #        if axis2 == f2.ndim:
                #            yield op, (f2, axis2, self.shape[-1])
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-2 and newaxis <= f.ndim-2:
                    yield op, (Ravel(f), axis, newaxis)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis + dofmap.ndim < f.ndim-1:
                    yield op, (Ravel(f), dofmap, length, axis)
                else:
                    if axis + dofmap.ndim == f.ndim-1:
                        indices = dofmap, Range(f.shape[-1])
                    else:
                        indices = Range(f.shape[axis-1]), dofmap
                    yield op, (f, RavelIndex(*indices, *self.func.shape[-2:]), self.shape[-1], f.ndim - (dofmap.ndim+1))
            elif op in (add, multiply, power, Choose):
                yield op, tuple(Ravel(arg) for arg in args)

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
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if _equals_scalar_constant(self.func.shape[-2], 1):
            return get(self.func, -2, constant(0))
        if _equals_scalar_constant(self.func.shape[-1], 1):
            return get(self.func, -1, constant(0))

    @staticmethod
    def evalf(f):
        return f.reshape(f.shape[:-2] + (f.shape[-2]*f.shape[-1],))

    def _derivative(self, var, seen):
        return ravel(derivative(self.func, var, seen), axis=self.ndim-1)

    @cached_property
    def _assparse(self):
        return tuple((*indices[:-2], indices[-2]*self.func.shape[-1]+indices[-1], values) for *indices, values in self.func._assparse)

    def _intbounds_impl(self):
        return self.func._intbounds_impl()


class Unravel(Array):

    def __init__(self, func: Array, sh1: Array, sh2: Array):
        assert isinstance(func, Array) and func.ndim > 0, f'func={func!r}'
        assert _isindex(sh1), f'sh1={sh1!r}'
        assert _isindex(sh2), f'sh2={sh2!r}'
        assert _equals_simplified(func.shape[-1], sh1 * sh2)
        self.func = func
        super().__init__(args=(func, sh1, sh2), shape=(*func.shape[:-1], sh1, sh2), dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield unravel, (self.func, self.ndim-2, self.shape[-2:])
        for op, args in self.func.representations:
            if op is constant:
                if self.shape[-2].isconstant and self.shape[-1].isconstant:
                    v, = args
                    return op, (v.reshape(*v.shape[:-1], self.shape[-2].eval(), self.shape[-1].eval()),)
            elif op is insertaxis:
                f, axis, length = args
                if axis != f.ndim:
                    yield op, (Unravel(f, *self.shape[-2:]), axis, length)
                else:
                    yield op, (InsertAxis(f, self.shape[-2]), self.ndim-1, self.shape[-1])
                    yield op, (InsertAxis(f, self.shape[-1]), self.ndim-2, self.shape[-2])
            elif op is inverse:
                f, invaxes = args
                if max(invaxes) < self.func.ndim-1:
                    yield op, (Unravel(f, *self.shape[-2:]), invaxes)
            elif op in (multiply, add, power, Choose, sign, *_POINTWISE):
                yield op, tuple(Unravel(arg, *self.shape[-2:]) for arg in args)
            elif op is _inflate:
                f, dofmap, length, axis = args
                if axis+dofmap.ndim < self.func.ndim-1:
                    yield op, (Unravel(f, *self.shape[-2:]), dofmap, length, axis)
            elif op is diagonalize:
                f, axis, newaxis = args
                if axis < f.ndim-1 and newaxis < f.ndim:
                    yield op, (Unravel(f, *self.shape[-2:]), axis, newaxis)
            elif op is ravel:
                f, axis = args
                if axis+1 < f.ndim-1:
                    yield op, (Unravel(f, *self.shape[-2:]), axis)
            elif op is ravelindex:
                ia, ib, na, nb, where = args
                if where[-1]:
                    ib = Unravel(ib, *self.shape[-2:])
                else:
                    ia = Unravel(ia, *self.shape[-2:])
                yield op, (ia, ib, na, nb, numpy.concatenate([where, where[-1:]]))
            elif op is polyval:
                f, coeffs, axis = args
                if axis < f.ndim-1:
                    yield op, (Unravel(f, *self.shape[-2:]), coeffs, axis)
                elif coeffs.ndim == 0:
                    yield op, (unravel(f, -2, self.shape[-2:]), coeffs, axis+1)
            elif op is polymul:
                c1, c2, v, axis = args
                if axis < c1.ndim-1:
                    yield op, (Unravel(c1, *self.shape[-2:]), Unravel(c2, *self.shape[-2:]), v, axis)
            elif op is polygrad:
                c, nvars, caxis, deraxis = args
                if deraxis != c.ndim and caxis != c.ndim-1:
                    yield op, (Unravel(c, *self.shape[-2:]), nvars, caxis, deraxis)
            elif op is legendre:
                f, axis, degree = args
                if axis != f.ndim:
                    yield op, (Unravel(f, *self.shape[-2:]), axis, degree)
            elif op is loop_concatenate:
                f, index, axis = args
                if axis < f.ndim-1:
                    yield op, (Unravel(f, *self.shape[-2:]), index, axis)
            elif op in (loop_sum, SearchSorted):
                f, *args = args
                yield op, (Unravel(f, *self.shape[-2:]), *args)

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if _equals_scalar_constant(self.shape[-2], 1):
            return insertaxis(self.func, self.ndim-2, constant(1))
        if _equals_scalar_constant(self.shape[-1], 1):
            return insertaxis(self.func, self.ndim-1, constant(1))
        if simple := self._as_any(_inflate, diagonalize, ravel, constant, insertaxis, inverse, multiply, add, power, Choose, sign, *_POINTWISE, ravelindex, polyval, polymul, polygrad, legendre, loop_sum, loop_concatenate, SearchSorted):
            return simple
        for f, axis, newaxis in self.func._as(diagonalize):
            if newaxis == f.ndim:
                f = unravel(f, axis, self.shape[-2:])
                f = diagonalize(f, axis, newaxis+1)
                f = diagonalize(f, axis+1, newaxis+2)
                return ravel(f, axis)
            elif axis == f.ndim-1:
                f = unravel(f, axis, self.shape[-2:])
                f = diagonalize(f, axis, newaxis)
                f = diagonalize(f, axis+2, newaxis+1)
                return ravel(f, newaxis)
        for f, axis in self.func._as(ravel):
            if axis == f.ndim-2 and equalshape(f.shape[-2:], self.shape[-2:]):
                return f
        for length, in self.func._as(Range):
            return RavelIndex(Range(self.shape[0]), Range(self.shape[1]), self.shape[0], self.shape[1])

    def _derivative(self, var, seen):
        return unravel(derivative(self.func, var, seen), axis=self.ndim-2, shape=self.shape[-2:])

    @staticmethod
    def evalf(f, sh1, sh2):
        return f.reshape(f.shape[:-1] + (sh1, sh2))

    @cached_property
    def _assparse(self):
        return tuple((*indices[:-1], *divmod(indices[-1], appendaxes(self.shape[-1], values.shape)), values) for *indices, values in self.func._assparse)


class RavelIndex(Array):

    def __init__(self, ia: Array, ib: Array, na: Array, nb: Array):
        assert isinstance(ia, Array), f'ia={ia!r}'
        assert isinstance(ib, Array), f'ib={ib!r}'
        assert _isindex(na), f'na={na!r}'
        assert _isindex(nb), f'nb={nb!r}'
        self._ia = ia
        self._ib = ib
        self._na = na
        self._nb = nb
        self._length = na * nb
        super().__init__(args=(ia, ib, nb), shape=ia.shape + ib.shape, dtype=int)

    @util.reentrant_iter.property
    def representations(self):
        yield ravelindex, (self._ia, self._ib, self._na, self._nb, numpy.concatenate([numpy.zeros(self._ia.ndim, bool), numpy.ones(self._ib.ndim, bool)]))
        for op, args in self._ia.representations:
            if op is insertaxis:
                f, axis, length = args
                yield op, (RavelIndex(f, self._ib, self._na, self._nb), axis, length)
        for op, args in self._ib.representations:
            if op is insertaxis:
                f, axis, length = args
                yield op, (RavelIndex(self._ia, f, self._na, self._nb), self._ia.ndim+axis, length)

    @staticmethod
    def evalf(ia, ib, nb):
        return ia[(...,)+(numpy.newaxis,)*ib.ndim] * nb + ib

    def _intbounds_impl(self):
        nbmin, nbmax = self._nb._intbounds
        iamin, iamax = self._ia._intbounds
        ibmin, ibmax = self._ib._intbounds
        return iamin * nbmin + ibmin, (iamax and nbmax and iamax * nbmax) + ibmax


class Range(Array):

    def __init__(self, length: Array):
        assert _isindex(length), f'length={length!r}'
        self.length = length
        super().__init__(args=(length,), shape=(length,), dtype=int)

    @property
    def representations(self):
        return (Range, (self.length,)),

    evalf = staticmethod(numpy.arange)

    def _intbounds_impl(self):
        lower, upper = self.length._intbounds
        assert lower >= 0
        return 0, max(0, upper - 1)


class InRange(Array):

    def __init__(self, index: Array, length: Array):
        assert isinstance(index, Array) and index.dtype == int, f'index={index!r}'
        assert isinstance(length, Array) and length.dtype == int and length.ndim == 0, f'length={length!r}'
        self.index = index
        self.length = length
        super().__init__(args=(index, length), shape=index.shape, dtype=int)

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

    def __init__(self, coeffs: Array, points: Array):
        assert isinstance(coeffs, Array) and coeffs.dtype == float and coeffs.ndim >= 1, f'coeffs={coeffs!r}'
        assert isinstance(points, Array) and points.dtype == float and points.ndim >= 1 and points.shape[-1].isconstant, f'points={points!r}'
        self.points_ndim = int(points.shape[-1])
        self.coeffs = coeffs
        self.points = points
        super().__init__(args=(coeffs, points), shape=points.shape[:-1]+coeffs.shape[:-1], dtype=float)

    @util.reentrant_iter.property
    def representations(self):
        yield polyval, (self.points, self.coeffs, self.points.ndim-1)
        for op, args in self.points.representations:
            if op is insertaxis:
                f, axis, length = args
                if axis != f.ndim:
                    yield op, (Polyval(self.coeffs, f), axis, length)

    evalf = staticmethod(poly.eval_outer)

    def _derivative(self, var, seen):
        if self.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        dpoints = einsum('ABi,AiD->ABD', Polyval(PolyGrad(self.coeffs, self.points_ndim), self.points), derivative(self.points, var, seen), A=self.points.ndim-1)
        dcoeffs = Transpose.from_end(Polyval(Transpose.to_end(derivative(self.coeffs, var, seen), *range(self.coeffs.ndim)), self.points), *range(self.points.ndim-1, self.ndim))
        return dpoints + dcoeffs

    def _simplified(self):
        ncoeffs_lower, ncoeffs_upper = self.coeffs.shape[-1]._intbounds
        if iszero(self.coeffs):
            return zeros_like(self)
        elif _equals_scalar_constant(self.coeffs.shape[-1], 1):
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

    def __init__(self, ncoeffs: Array, nvars: int) -> None:
        assert isinstance(ncoeffs, Array) and ncoeffs.ndim == 0 and ncoeffs.dtype == int, 'ncoeffs={ncoeffs!r}'
        assert isinstance(nvars, int) and nvars >= 0, 'nvars={nvars!r}'
        self.ncoeffs = ncoeffs
        self.nvars = nvars
        super().__init__(args=(ncoeffs,), shape=(), dtype=int)

    def evalf(self, ncoeffs):
        return numpy.array(poly.degree(self.nvars, ncoeffs.__index__()))

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

    def __init__(self, nvars: int, degree: Array) -> None:
        assert isinstance(degree, Array) and degree.ndim == 0 and degree.dtype == int, f'degree={degree!r}'
        assert isinstance(nvars, int) and nvars >= 0, 'nvars={nvars!r}'
        self.nvars = nvars
        self.degree = degree
        super().__init__(args=(degree,), shape=(), dtype=int)

    def evalf(self, degree):
        return numpy.array(poly.ncoeffs(self.nvars, degree.__index__()))

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

    def __init__(self, coeffs_left: Array, coeffs_right: Array, vars: typing.Tuple[poly.MulVar, ...]):
        assert isinstance(coeffs_left, Array) and coeffs_left.ndim >= 1 and coeffs_left.dtype == float, f'coeffs_left={coeffs_left!r}'
        assert isinstance(coeffs_right, Array) and coeffs_right.ndim >= 1 and coeffs_right.dtype == float, f'coeffs_right={coeffs_right!r}'
        assert equalshape(coeffs_left.shape[:-1], coeffs_right.shape[:-1]), 'PolyMul({}, {})'.format(coeffs_left, coeffs_right)
        self.coeffs_left = coeffs_left
        self.coeffs_right = coeffs_right
        self.vars = vars
        self.degree_left = PolyDegree(coeffs_left.shape[-1], builtins.sum(var != poly.MulVar.Right for var in vars))
        self.degree_right = PolyDegree(coeffs_right.shape[-1], builtins.sum(var != poly.MulVar.Left for var in vars))
        ncoeffs = PolyNCoeffs(len(vars), self.degree_left + self.degree_right)
        super().__init__(args=(coeffs_left, coeffs_right), shape=(*coeffs_left.shape[:-1], ncoeffs), dtype=float)

    @util.reentrant_iter.property
    def representations(self):
        yield polymul, (self.coeffs_left, self.coeffs_right, self.vars, self.ndim-1)
        for (op1, args1), (op2, args2) in util.iter_product(self.coeffs_left.representations, self.coeffs_right.representations):
            if op1 is op2 and op1 is insertaxis and args1[1:] == args2[1:]:
                f1, axis, length = args1
                f2, _axis, _length = args2
                if axis < f1.ndim:
                    yield insertaxis, (PolyMul(f1, f2, self.vars), axis, length)

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

    def __init__(self, coeffs: Array, nvars: int):
        assert isinstance(coeffs, Array) and coeffs.dtype == float and coeffs.ndim >= 1, f'coeffs={coeffs!r}'
        assert isinstance(nvars, int) and nvars >= 0, f'nvars={nvars!r}'
        self.coeffs = coeffs
        self.nvars = nvars
        self.degree = PolyDegree(coeffs.shape[-1], nvars)
        ncoeffs = PolyNCoeffs(nvars, Maximum(constant(0), self.degree - constant(1)))
        shape = *coeffs.shape[:-1], constant(nvars), ncoeffs
        super().__init__(args=(coeffs,), shape=shape, dtype=float)

    @util.reentrant_iter.property
    def representations(self):
        yield polygrad, (self.coeffs, self.nvars, self.ndim-2, self.ndim-1)
        for op, args in self.coeffs.representations:
            if op is insertaxis:
                f, axis, length = args
                if axis < f.ndim-1:
                    yield op, (PolyGrad(f, self.nvars), axis, length)

    @cached_property
    def evalf(self):
        try:
            degree = self.degree.__index__()
        except TypeError as e:
            return functools.partial(poly.grad, nvars=self.nvars)
        else:
            return poly.GradPlan(self.nvars, degree)

    def _simplified(self):
        if iszero(self.coeffs) or _equals_scalar_constant(self.degree, 0):
            return zeros_like(self)
        elif _equals_scalar_constant(self.degree, 1):
            return InsertAxis(Take(self.coeffs, constant(self.nvars - 1) - Range(constant(self.nvars))), constant(1))


class Legendre(Array):
    '''Series of Legendre polynomial up to and including the given degree.

    Parameters
    ---------
    x : :class:`Array`
        The coordinates to evaluate the series at.
    degree : :class:`int`
        The degree of the last polynomial of the series.
    '''

    def __init__(self, x: Array, degree: int) -> None:
        assert isinstance(x, Array) and x.dtype == float, f'x={x!r}'
        assert isinstance(degree, int) and degree >= 0, f'degree={degree!r}'
        self._x = x
        self._degree = degree
        super().__init__(args=(x,), shape=(*x.shape, constant(degree+1)), dtype=float)

    @util.reentrant_iter.property
    def representations(self):
        yield legendre, (self._x, self.ndim-1, self._degree)
        for op, args in self._x.representations:
            if op is insertaxis:
                f, axis, length = args
                yield op, (Legendre(f, self._degree), axis, length)

    def evalf(self, x: numpy.ndarray) -> numpy.ndarray:
        P = numpy.empty((*x.shape, self._degree+1), dtype=float)
        P[..., 0] = 1
        if self._degree:
            P[..., 1] = x
        for i in range(2, self._degree+1):
            P[..., i] = (2-1/i)*P[..., 1]*P[..., i-1] - (1-1/i)*P[..., i-2]
        return P

    def _derivative(self, var, seen):
        if self.dtype == complex:
            raise NotImplementedError('The complex derivative is not implemented.')
        d = numpy.zeros((self._degree+1,)*2, dtype=int)
        for i in range(self._degree+1):
            d[i, i+1::2] = 2*i+1
        dself = einsum('Ai,ij->Aj', self, astype(d, self.dtype))
        return einsum('Ai,AB->AiB', dself, derivative(self._x, var, seen))

    def _simplified(self):
        unaligned, where = unalign(self._x)
        if where != tuple(range(self._x.ndim)):
            return align(Legendre(unaligned, self._degree), (*where, self.ndim-1), self.shape)


class Choose(Array):
    '''Function equivalent of :func:`numpy.choose`.'''

    def __init__(self, index: Array, *choices: Array):
        assert isinstance(index, Array) and index.dtype == int, f'index={index!r}'
        assert isinstance(choices, tuple) and all(isinstance(choice, Array) for choice in choices), f'choices={choices!r}'
        dtype = choices[0].dtype
        assert all(choice.dtype == dtype for choice in choices[1:])
        shape = index.shape
        assert all(equalshape(choice.shape, shape) for choice in choices)
        self.index = index
        self.choices = choices
        super().__init__(args=(index,)+choices, shape=shape, dtype=dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield Choose, (self.index, *self.choices)

    @staticmethod
    def evalf(index, *choices):
        return numpy.choose(index, choices)

    def _derivative(self, var, seen):
        return Choose(appendaxes(self.index, var.shape), *(derivative(choice, var, seen) for choice in self.choices))

    def _simplified(self):
        if all(choice == self.choices[0] for choice in self.choices[1:]):
            return self.choices[0]
        index, *choices, where = unalign(self.index, *self.choices)
        if len(where) < self.ndim:
            return align(Choose(index, *choices), where, self.shape)

    def _get(self, i, item):
        return Choose(get(self.index, i, item), *(get(choice, i, item) for choice in self.choices))


class NormDim(Array):

    def __init__(self, length: Array, index: Array):
        assert isinstance(length, Array) and length.dtype == int, f'length={length!r}'
        assert isinstance(index, Array) and index.dtype == int, f'index={index!r}'
        assert equalshape(length.shape, index.shape)
        # The following corner cases makes the assertion fail, hence we can only
        # assert the bounds if the arrays are guaranteed to be unempty:
        #
        #     Take(func, NormDim(func.shape[-1], Range(0) + func.shape[-1]))
        if all(n._intbounds[0] > 0 for n in index.shape):
            assert -length._intbounds[1] <= index._intbounds[0] and index._intbounds[1] <= length._intbounds[1] - 1
        self.length = length
        self.index = index
        super().__init__(args=(length, index), shape=index.shape, dtype=index.dtype)

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
        if self.length.isconstant and self.index.isconstant:
            return constant(self.eval())

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

    def __init__(self, target, source, index: Array, coords: Array):
        if index.dtype != int or index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        if coords.dtype != float:
            raise ValueError('argument `coords` must be a real-valued array with at least one axis')
        self._target = target
        self._source = source
        self._index = index
        self._coords = coords
        target_dim = source.todims if target is None else target.fromdims
        super().__init__(args=(index, coords), shape=(*coords.shape[:-1], constant(target_dim)), dtype=float)

    def evalf(self, index, coords):
        chain = self._source[index.__index__()]
        if self._target is not None:
            _, chain = self._target.index_with_tail(chain)
        return functools.reduce(lambda c, t: t.apply(c), reversed(chain), coords)

    def _derivative(self, var, seen):
        linear = TransformLinear(self._target, self._source, self._index)
        dcoords = derivative(self._coords, var, seen)
        return einsum('ij,AjB->AiB', linear, dcoords, A=self._coords.ndim - 1, B=var.ndim)

    def _simplified(self):
        if self._target == self._source:
            return self._coords
        cax = self.ndim - 1
        coords, where = unalign(self._coords, naxes=cax)
        if len(where) < cax:
            return align(TransformCoords(self._target, self._source, self._index, coords), (*where, cax), self.shape)


class TransformIndex(Array):
    '''Transform coordinates from one coordinate system to another (index part)

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

    def __init__(self, target, source, index: Array):
        if index.dtype != int or index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        self._target = target
        self._source = source
        self._index = index
        super().__init__(args=(index,), shape=(), dtype=int)

    def evalf(self, index):
        if self._target is not None:
            index, _ = self._target.index_with_tail(self._source[index.__index__()])
        else:
            index = 0
        return numpy.array(index)

    def _intbounds_impl(self):
        len_target = 1 if self._target is None else len(self._target)
        return 0, len_target - 1

    def _simplified(self):
        if self._target is None:
            return ones((1,), dtype=int)
        elif self._target == self._source:
            return self._index


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

    def __init__(self, target, source, index: Array):
        if index.dtype != int or index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        self._target = target
        self._source = source
        target_dim = source.todims if target is None else target.fromdims
        super().__init__(args=(index,), shape=(constant(target_dim), constant(source.fromdims)), dtype=float)

    def evalf(self, index):
        chain = self._source[index.__index__()]
        if self._target is not None:
            _, chain = self._target.index_with_tail(chain)
        if chain:
            return functools.reduce(lambda r, i: i @ r, (item.linear for item in reversed(chain)))
        else:
            return numpy.eye(self._source.fromdims)

    def _simplified(self):
        if self._target == self._source:
            return diagonalize(ones((constant(self._source.fromdims),), dtype=float))


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

    def __init__(self, source, index: Array):
        if index.dtype != int or index.ndim != 0:
            raise ValueError('argument `index` must be a scalar, integer `nutils.evaluable.Array`')
        self._source = source
        super().__init__(args=(index,), shape=(constant(source.todims), constant(source.todims)), dtype=float)

    def evalf(self, index):
        chain = self._source[index.__index__()]
        linear = numpy.eye(self._source.fromdims)
        for item in reversed(chain):
            linear = item.linear @ linear
            assert item.fromdims <= item.todims <= item.fromdims + 1
            if item.todims == item.fromdims + 1:
                linear = numpy.concatenate([linear, item.ext[:, numpy.newaxis]], axis=1)
        assert linear.shape == (self._source.todims, self._source.todims)
        return linear

    def _simplified(self):
        if self._source.todims == self._source.fromdims:
            # Since we only guarantee that the basis spans the space of source
            # coordinates mapped to the root and the map is a bijection (every
            # `Transform` is assumed to be injective), we can return the unit
            # vectors here.
            return diagonalize(ones((self._source.fromdims,), dtype=float))


class _LoopIndex(Argument):

    def __init__(self, name: str, length: Array):
        assert isinstance(name, str), f'name={name!r}'
        assert _isindex(length), f'length={length!r}'
        self.name = name
        self.length = length
        super().__init__(name, (), int)

    def __str__(self):
        try:
            length = self.length.__index__()
        except:
            length = '?'
        return 'LoopIndex({}, length={})'.format(self._name, length)

    def _node(self, cache, subgraph, times):
        if self in cache:
            return cache[self]
        cache[self] = node = RegularNode('LoopIndex', (), dict(length=self.length._node(cache, subgraph, times)), (type(self).__name__, _Stats()), subgraph)
        return node

    def _intbounds_impl(self):
        lower_length, upper_length = self.length._intbounds
        return 0, max(0, upper_length - 1)

    def _simplified(self):
        if _equals_scalar_constant(self.length, 1):
            return Zeros((), int)


class Loop(Evaluable):
    '''Base class for evaluable loops.

    Subclasses must implement

    *   method ``evalf_loop_init(init_arg)`` and
    *   method ``evalf_loop_body(output, body_arg)``.
    '''

    def __init__(self, index_name: str, length: Array, init_arg: Evaluable, body_arg: Evaluable, *args, **kwargs):
        assert isinstance(index_name, str), f'index_name={index_name!r}'
        assert isinstance(length, Array), f'length={length!r}'
        assert isinstance(init_arg, Evaluable), f'init_arg={init_arg!r}'
        assert isinstance(body_arg, Evaluable), f'body_arg={init_arg!r}'
        self.index_name = index_name
        self.length = length
        self.index = _LoopIndex(index_name, length)
        self.init_arg = init_arg
        self.body_arg = body_arg
        if self.index in init_arg.arguments:
            raise ValueError('the loop initialization arguments must not depend on the index')
        self._invariants, self._dependencies = _dependencies_sans_invariants(body_arg, self.index)
        super().__init__(args=(length, init_arg, *self._invariants), *args, **kwargs)

    @cached_property
    def _serialized_loop(self):
        indices = {d: i for i, d in enumerate(itertools.chain([self.index], self._invariants, self._dependencies))}
        return tuple((dep, tuple(map(indices.__getitem__, dep._Evaluable__args))) for dep in self._dependencies)

    @cached_property
    def _serialized_loop_evalf(self):
        return tuple((dep.evalf, indices) for dep, indices in self._serialized_loop)

    def evalf(self, length, init_arg, *invariants):
        serialized_evalf = self._serialized_loop_evalf
        output = self.evalf_loop_init(init_arg)
        length = length.__index__()
        values = [None] + list(invariants) + [None] * len(serialized_evalf)
        with log.context(f'loop {self.index.name}'.replace('{', '{{').replace('}', '}}') + ' {:3.0f}%', 0) as log_ctx:
            fork = parallel.fork(length)
            if fork:
                raw_index = multiprocessing.RawValue('i', 0)
                lock = multiprocessing.Lock()
                with fork as pid:
                    with lock:
                        index = raw_index.value
                        raw_index.value = index + 1
                    while index < length:
                        if not pid:
                            log_ctx(100*index/length)
                        values[0] = numpy.array(index)
                        for o, (op_evalf, indices) in enumerate(serialized_evalf, len(invariants) + 1):
                            values[o] = op_evalf(*[values[i] for i in indices])
                        with lock:
                            self.evalf_loop_body(output, values[-1])
                            index = raw_index.value
                            raw_index.value = index + 1
            else:
                for index in range(length):
                    values[0] = numpy.array(index)
                    for o, (op_evalf, indices) in enumerate(serialized_evalf, len(invariants) + 1):
                        values[o] = op_evalf(*[values[i] for i in indices])
                    self.evalf_loop_body(output, values[-1])
                    log_ctx(100*(index+1)/length)
            return output

    def evalf_withtimes(self, times, length, init_arg, *invariants):
        serialized = self._serialized_loop
        subtimes = times.setdefault(self, collections.defaultdict(_Stats))
        output = self.evalf_loop_init(init_arg)
        values = [None] + list(invariants) + [None] * len(serialized)
        for index in range(length):
            values[0] = numpy.array(index)
            for o, (op, indices) in enumerate(serialized, len(invariants) + 1):
                values[o] = op.evalf_withtimes(subtimes, *[values[i] for i in indices])
            self.evalf_loop_body_withtimes(subtimes, output, values[-1])
        return output

    def evalf_loop_body_withtimes(self, times, output, body_arg):
        with times[self]:
            self.evalf_loop_body(output, body_arg)

    def _node(self, cache, subgraph, times):
        if (cached := cache.get(self)) is not None:
            return cached
        for arg in itertools.chain(self._invariants, (self.init_arg,)):
            arg._node(cache, subgraph, times)
        loopcache = cache.copy()
        loopcache.pop(self.index, None)
        loopgraph = Subgraph('Loop', subgraph)
        looptimes = times.get(self, collections.defaultdict(_Stats))
        cache[self] = node = self._node_loop_body(loopcache, loopgraph, looptimes)
        return node

    @property
    def _loop_deps(self):
        deps = util.IDSet([self])
        deps |= self.init_arg._loop_deps
        for arg in self._invariants:
            deps |= arg._loop_deps
        return deps.view()


class _LoopTuple(Loop):

    def __init__(self, loops: typing.Tuple[Loop], index_name: str, length: Array):
        assert isinstance(loops, tuple) and all(isinstance(loop, Loop) and loop.index_name == index_name and loop.length == length for loop in loops), f'loops={loops}'
        self.loops = loops
        super().__init__(
            index_name=index_name,
            length=length,
            init_arg=Tuple(tuple(loop.init_arg for loop in loops)),
            body_arg=Tuple(tuple(loop.body_arg for loop in loops)),
        )

    def evalf_loop_init(self, args):
        return tuple(loop.evalf_loop_init(arg) for loop, arg in zip(self.loops, args))

    def evalf_loop_body(self, outputs, args):
        for loop, output, arg in zip(self.loops, outputs, args):
            loop.evalf_loop_body(output, arg)

    def evalf_loop_body_withtimes(self, times, outputs, args):
        for loop, output, arg in zip(self.loops, outputs, args):
            loop.evalf_loop_body_withtimes(times, output, arg)

    def _node_loop_body(self, cache, subgraph, times):
        if (cached := cache.get(self)) is not None:
            return cached
        cache[self] = node = TupleNode(tuple(item._node_loop_body(cache, subgraph, times) for item in self.loops), metadata=(type(self).__name__, times[self]), subgraph=subgraph)
        return node

    @property
    def _intbounds_tuple(self):
        return tuple(loop._intbounds for loop in self.loops)


class LoopSum(Loop, Array):

    def __init__(self, func: Array, shape: typing.Tuple[Array, ...], index_name: str, length: Array):
        assert isinstance(func, Array) and func.dtype != bool, f'func={func!r}'
        assert func.ndim == len(shape)
        self.func = func
        super().__init__(init_arg=Tuple(shape), body_arg=func, index_name=index_name, length=length, shape=shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield loop_sum, (self.func, self.index)
        for op, args in self.func.representations:
            if op in (insertaxis, diagonalize, ravel) \
            or op is _inflate and self.index not in args[1].arguments:
                yield op, (loop_sum(args[0], self.index), *args[1:])

    def evalf_loop_init(self, shape):
        return parallel.shzeros(tuple(n.__index__() for n in shape), dtype=self.dtype)

    @staticmethod
    def evalf_loop_body(output, func):
        output += func

    def _derivative(self, var, seen):
        return loop_sum(derivative(self.func, var, seen), self.index)

    def _node_loop_body(self, cache, subgraph, times):
        if (cached := cache.get(self)) is not None:
            return cached
        kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times) for i, n in enumerate(self.shape)}
        kwargs['func'] = self.func._node(cache, subgraph, times)
        cache[self] = node = RegularNode('LoopSum', (), kwargs, (type(self).__name__, times[self]), subgraph)
        return node

    def _simplified(self):
        if isinstance(self.func, Zeros):
            return zeros_like(self)
        if self.index not in self.func.arguments:
            return self.func * astype(self.index.length, self.func.dtype)
        if simple := self._as_any(insertaxis, diagonalize, ravel):
            return simple

    @cached_property
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


class _SizesToOffsets(Array):

    def __init__(self, sizes):
        assert sizes.ndim == 1
        assert sizes.dtype == int
        assert sizes._intbounds[0] >= 0
        self._sizes = sizes
        super().__init__(args=(sizes,), shape=(sizes.shape[0]+1,), dtype=int)

    @staticmethod
    def evalf(sizes):
        return numpy.cumsum([0, *sizes])

    def _simplified(self):
        unaligned, where = unalign(self._sizes)
        if not where:
            return Range(self.shape[0]) * appendaxes(unaligned, self.shape[:1])

    def _intbounds_impl(self):
        n = self._sizes.shape[0]._intbounds[1]
        m = self._sizes._intbounds[1]
        return 0, (0 if n == 0 or m == 0 else n * m)


class LoopConcatenate(Loop, Array):

    def __init__(self, func: Array, start: Array, stop: Array, concat_length: Array, index_name: str, length: Array):
        assert isinstance(func, Array), f'func={func}'
        assert _isindex(start), f'start={start}'
        assert _isindex(stop), f'stop={stop}'
        assert _isindex(concat_length), f'concat_length={concat_length}'
        self.func = func
        self.start = start
        self.stop = stop
        if not self.func.ndim:
            raise ValueError('expected an array with at least one axis')
        shape = *func.shape[:-1], concat_length
        super().__init__(init_arg=Tuple(shape), body_arg=Tuple((func, start, stop)), index_name=index_name, length=length, shape=shape, dtype=func.dtype)

    @util.reentrant_iter.property
    def representations(self):
        yield loop_concatenate, (self.func, self.index, self.ndim-1)

    def evalf_loop_init(self, shape):
        return parallel.shempty(tuple(n.__index__() for n in shape), dtype=self.dtype)

    @staticmethod
    def evalf_loop_body(output, arg):
        func, start, stop = arg
        output[..., start:stop] = func

    def _derivative(self, var, seen):
        return Transpose.from_end(loop_concatenate(Transpose.to_end(derivative(self.func, var, seen), self.ndim-1), self.index), self.ndim-1)

    def _node_loop_body(self, cache, subgraph, times):
        if (cached := cache.get(self)) is not None:
            return cached
        kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times) for i, n in enumerate(self.shape)}
        kwargs['start'] = self.start._node(cache, subgraph, times)
        kwargs['stop'] = self.stop._node(cache, subgraph, times)
        kwargs['func'] = self.func._node(cache, subgraph, times)
        cache[self] = node = RegularNode('LoopConcatenate', (), kwargs, (type(self).__name__, times[self]), subgraph)
        return node

    def _simplified(self):
        if iszero(self.func):
            return zeros_like(self)
        elif self.index not in self.func.arguments:
            return Ravel(Transpose.from_end(InsertAxis(self.func, self.index.length), -2))
        unaligned, where = unalign(self.func)
        if self.ndim-1 not in where:
            # reinsert concatenation axis, at unit length if possible so we can
            # insert the remainder outside of the loop
            unaligned = InsertAxis(unaligned, self.func.shape[-1] if self.index in self.func.shape[-1].arguments else constant(1))
            where += self.ndim-1,
        elif where[-1] != self.ndim-1:
            # bring concatenation axis to the end
            unaligned = Transpose.inv(unaligned, where)
            where = tuple(sorted(where))
        f = loop_concatenate(unaligned, self.index)
        if not _equals_simplified(self.shape[-1], f.shape[-1]):
            # last axis was reinserted at unit length AND it was not unit length
            # originally - if it was unit length originally then we proceed only if
            # there are other insertions to promote, otherwise we'd get a recursion.
            f = Ravel(InsertAxis(f, self.func.shape[-1]))
        elif len(where) == self.ndim:
            return
        return align(f, where, self.shape)

    @cached_property
    def _assparse(self):
        chunks = []
        for *indices, last_index, values in self.func._assparse:
            last_index = last_index + prependaxes(self.start, last_index.shape)
            chunks.append(tuple(loop_concatenate(_flat(arr), self.index) for arr in (*indices, last_index, values)))
        return tuple(chunks)

    def _intbounds_impl(self):
        return self.func._intbounds


class SearchSorted(Array):
    '''Find index of evaluable array into sorted numpy array.'''

    # NOTE: SearchSorted is essentially pointwise in its only evaluable
    # argument, but the Pointwise class currently does not allow for
    # additional, static arguments. The following constructor makes the static
    # arguments keyword-only in anticipation of potential future support.

    def __init__(self, arg: Array, array: types.arraydata, side: str, sorter: typing.Optional[types.arraydata]):
        assert isinstance(arg, Array), f'arg={arg!r}'
        assert isinstance(array, types.arraydata) and array.ndim == 1, f'array={array!r}'
        assert side in ('left', 'right'), f'side={side!r}'
        assert sorter is None or isinstance(sorter, types.arraydata) and sorter.dtype == int and sorter.shape == array.shape, f'sorter={sorter!r}'
        self._arg = arg
        self._array = array
        self._side = side
        self._sorter = sorter
        super().__init__(args=(arg,), shape=arg.shape, dtype=int)

    @util.reentrant_iter.property
    def representations(self):
        yield SearchSorted, (self._arg, self._array, self._side, self._sorter)

    def evalf(self, values):
        index = numpy.searchsorted(self._array, values, side=self._side, sorter=self._sorter)
        # on some platforms (windows) searchsorted does not return indices as
        # numpy.dtype(int), so we type cast it for consistency
        return index.astype(int, copy=False)

    def _intbounds_impl(self):
        return 0, self._array.shape[0]


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
    if equalshape(a.shape, b.shape):
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


_AddDependency = collections.namedtuple('_AddDependency', ['dependency'])

def _dependencies_sans_invariants(func, arg):
    invariants = []
    dependencies = []
    cache = {arg}
    stack = [func]
    while stack:
        func_ = stack.pop()
        if isinstance(func_, _AddDependency):
            dependencies.append(func_.dependency)
        elif func_ not in cache:
            cache.add(func_)
            if arg in func_.arguments:
                stack.append(_AddDependency(func_))
                stack.extend(func_._Evaluable__args)
            else:
                invariants.append(func_)
    assert (dependencies or invariants or [arg])[-1] == func
    return tuple(invariants), tuple(dependencies)


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
    lengths = [arg.shape[axis] for arg in args]
    *offsets, totlength = util.cumsum(lengths + [0])
    return Transpose.from_end(util.sum(Inflate(Transpose.to_end(arg, axis), Range(length) + offset, totlength) for arg, length, offset in zip(args, lengths, offsets)), axis)


def stack(args, axis=0):
    return Transpose.from_end(util.sum(Inflate(arg, constant(i), constant(len(args))) for i, arg in enumerate(args)), axis)


def repeat(arg, length, axis):
    arg = asarray(arg)
    assert _equals_scalar_constant(arg.shape[axis], 1)
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
    if _equals_simplified(arg.shape[-1], arg.shape[-2]):
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
    return Transpose.from_end(TakeDiag(Transpose.to_end(arg, axis, rmaxis)), axis-(axis>=rmaxis))


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
    assert equalshape(result.shape, func.shape+var.shape) and result.dtype == func.dtype, 'bug in {}._derivative'.format(type(func).__name__)
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
        assert _equals_simplified(index.shape[0], length)
        index = Find(index)
    elif index.isconstant:
        index_ = index.eval()
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
        elif numpy.greater(numpy.diff(index_), 0).all():
            return mask(arg, numeric.asboolean(index_, int(length)), axis)
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
    assert equalshape(dofmap.shape, arg.shape[axis:axis+dofmap.ndim])
    return Transpose.from_end(Inflate(Transpose.to_end(arg, *range(axis, axis+dofmap.ndim)), dofmap, length), axis)


def mask(arg, mask: Array, axis: int = 0):
    assert isinstance(arg, Array), f'arg={arg!r}'
    assert isinstance(mask, numpy.ndarray) and mask.dtype == bool and mask.ndim == 1 and _equals_scalar_constant(arg.shape[axis], len(mask)), f'mask={mask!r}'
    index, = mask.nonzero()
    return _take(arg, constant(index), axis)


def unravel(func, axis, shape):
    func = asarray(func)
    axis = numeric.normdim(func.ndim, axis)
    assert len(shape) == 2
    return Transpose.from_end(Unravel(Transpose.to_end(func, axis), *shape), axis, axis+1)


def ravel(func, axis):
    func = asarray(func)
    axis = numeric.normdim(func.ndim-1, axis)
    return Transpose.from_end(Ravel(Transpose.to_end(func, axis, axis+1)), axis)


def ravelindex(ia, ib, na, nb, where):
    assert where.dtype == bool and len(where) == ia.ndim + ib.ndim
    wb, = where.nonzero()
    assert len(wb) == ib.ndim
    return Transpose.from_end(RavelIndex(ia, ib, na, nb), *wb)


def polymul(c1, c2, v, axis):
    assert c1.ndim == c2.ndim
    axis = numeric.normdim(c1.ndim, axis)
    assert equalshape(c1.shape[:axis] + c1.shape[axis+1:], c2.shape[:axis] + c2.shape[axis+1:]), f'incompatible shapes: {c1}, {c2}'
    return Transpose.from_end(PolyMul(Transpose.to_end(c1, axis), Transpose.to_end(c2, axis), v), axis)


def polyval(points, coeffs, axis):
    axis = numeric.normdim(points.ndim, axis)
    return Transpose.from_end(Polyval(coeffs, Transpose.to_end(points, axis)), *numpy.arange(axis, axis+coeffs.ndim-1))


def polygrad(coeffs, nvars, caxis, deraxis):
    caxis = numeric.normdim(coeffs.ndim, caxis)
    deraxis = numeric.normdim(coeffs.ndim+1, deraxis)
    return Transpose.from_end(PolyGrad(Transpose.to_end(coeffs, caxis), nvars), caxis + (caxis>=deraxis), deraxis)


def legendre(x, axis, degree):
    axis = numeric.normdim(x.ndim+1, axis)
    return Transpose.from_end(Legendre(x, degree), axis)


def _flat(func):
    func = asarray(func)
    if func.ndim == 0:
        return InsertAxis(func, constant(1))
    while func.ndim > 1:
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
    return _LoopIndex(name, asarray(length))


def loop_sum(func, index):
    func = asarray(func)
    if not isinstance(index, _LoopIndex):
        raise TypeError(f'expected _LoopIndex, got {index!r}')
    return LoopSum(func, func.shape, index.name, index.length)


def loop_concatenate(func, index, axis=-1):
    axis = numeric.normdim(func.ndim, axis)
    func = Transpose.to_end(func, axis)
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
    return Transpose.from_end(LoopConcatenate(func, start, stop, concat_length, index.name, index.length), axis)


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
    if isinstance(value, Argument) and value._name in arguments:
        v = asarray(arguments[value._name])
        assert equalshape(value.shape, v.shape), (value.shape, v.shape)
        assert value.dtype == v.dtype, (value.dtype, v.dtype)
        return v


def einsum(fmt, *args, **dims):
    '''Multiply and/or contract arrays via format string.

    The format string consists of a comma separated list of axis labels, followed
    by ``->`` and the axis labels of the return value. For example, the following
    swaps the axes of a matrix:

    >>> a45 = constant(numpy.ones([4,5])) # 4x5 matrix
    >>> einsum('ij->ji', a45)
    nutils.evaluable.Transpose<f:5,4>

    Axis labels that do not occur in the return value are summed. For example,
    the following performs a matrix-vector product:

    >>> a5 = constant(numpy.ones([5])) # vector with length 5
    >>> einsum('ij,j->i', a45, a5)
    nutils.evaluable.Sum<f:4>

    The following contracts a third order tensor, a matrix, and a vector, and
    transposes the result:

    >>> a234 = constant(numpy.ones([2,3,4])) # 2x3x4 tensor
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
            if not _equals_simplified(shapes.setdefault(c, sh), sh):
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


@util.single_or_multiple
def eval_sparse(funcs: AsEvaluableArray, **arguments: typing.Mapping[str, numpy.ndarray]) -> typing.Tuple[numpy.ndarray, ...]:
    '''Evaluate one or several Array objects as sparse data.

    Args
    ----
    funcs : :class:`tuple` of Array objects
        Arrays to be evaluated.
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.

    Returns
    -------
    results : :class:`tuple` of sparse data arrays
    '''

    funcs = [func.as_evaluable_array for func in funcs]
    shape_chunks = Tuple(tuple(Tuple(builtins.sum(func.simplified._assparse, func.shape)) for func in funcs))
    with shape_chunks.optimized_for_numpy.session(graphviz=graphviz) as eval:
        for func, args in zip(funcs, eval(**arguments)):
            shape = tuple(map(int, args[:func.ndim]))
            chunks = [args[i:i+func.ndim+1] for i in range(func.ndim, len(args), func.ndim+1)]
            length = builtins.sum(values.size for *indices, values in chunks)
            data = numpy.empty((length,), dtype=sparse.dtype(shape, func.dtype))
            start = 0
            for *indices, values in chunks:
                stop = start + values.size
                d = data[start:stop].reshape(values.shape)
                d['value'] = values
                for idim, ii in enumerate(indices):
                    d['index']['i'+str(idim)] = ii
                start = stop
            yield data


_POINTWISE = [Pointwise]
for _c in _POINTWISE:
    _POINTWISE.extend(_c.__subclasses__())


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
