'''Expression parser version 1 and namespace.

The syntax of an expression is as follows:

*   **Integers** or **decimal numbers** are denoted in the usual way.
    Examples: ``1``, ``1.2``, ``.2``.  A number may not start with a zero,
    except when followed by a dot: ``0.1`` is valid, but ``01`` is not.

*   **Variables** are denoted with a string of alphanumeric characters.  The
    first character may not be a numeral.  Unlike Python variables,
    underscores are not allowed, as they have a special meaning.  If the
    variable is an array with one or more axes, all those axes should be
    labeled with a latin character, the index, and appended to the variable
    with an underscore.  For example an array ``a`` with two axes can be
    denoted with ``a_ij``.  Optionally, a single numeral may be used to
    select an item at the concerning axis.  Example: in ``a_i0`` the first
    axis of ``a`` is labeled ``i`` and the first element of the second axis
    is selected.  If the same index occurs twice, the trace is taken along
    the concerning axes.  Example: the trace of the first and third axes of
    ``b`` is denoted by ``b_iji``.  It is invalid to specify an index more
    than twice.  The following names cannot be used as variables: ``n``,
    ``δ``, ``$``.  The variable named ``x``, or the value of argument
    ``default_geometry_name``, has a special meaning, detailed below.

*   A term, the **product** of two or more arrays or scalars, is denoted by
    space-separated variables, constants or compound expressions.  Example:
    ``a b c`` denotes the product of the scalars ``a``, ``b`` and ``c``.  A
    term may start with a number, but a number is not allowed in other parts
    of the term.  Example: ``2 a`` denotes two times ``a``; ``2 2 a`` and ``2
    a 2``` are invalid.  When two arrays in a term have the same index, this
    index is summed.  Example: ``a_i b_i`` denotes the inner product of ``a``
    and ``b`` and ``A_ij b_j``` a matrix vector product.  It is not allowed
    to use an index more than twice in a term.

*   The operator ``/`` denotes a **fraction**.  Example: in ``a b / c d`` ``a
    b`` is the numerator and ``c d`` the denominator.  Both the numerator and
    the denominator may start with a number.  Example: ``2 a / 3 b``.  The
    denominator must be a scalar.  Example: ``2 / a_i b_i`` is valid, but ``2
    a_i / b_i`` is not.

    .. warning::

        This syntax is different from the Python syntax.  In Python ``a*b /
        c*d`` is mathematically equivalent to ``a*b*d/c``.

*   The operators ``+`` and ``-`` denote **add** and **subtract**.  Both
    operators should be surrounded by whitespace, e.g. ``a + b``.  Both
    operands should have the same shape.  Example: ``a_ij + b_i c_j`` is a
    valid, provided that the lengths of the axes with the same indices match,
    but ``a_ij + b_i`` is invalid.  At the beginning of an expression or a
    compound ``-`` may be used to negate the following term.  Example: in
    ``-a b + c`` the term ``a b`` is negated before adding ``c``.  It is not
    allowed to negate other terms: ``a + -b`` is invalid, so is ``a -b``.

*   An expression surrounded by parentheses is a **compound expression** and
    can be used as single entity in a term.  Example: ``(a_i + b_i) c_i``
    denotes the inner product of ``a_i + b_i`` with ``c_i``.

*   **Exponentiation** is denoted by a ``^``, where the left and right
    operands should be a number, variable or compound expression and the
    right operand should be a scalar.  Example: ``a^2`` denotes the square of
    ``a``, ``a^-2`` denotes ``a`` to the power ``-2`` and ``a^(1 / 2)`` the
    square root of ``a``.

*   An **argument** is denoted by a name — following the same rules as a
    variable name — prefixed with a question mark.  An argument is a scalar
    or array with a yet unknown value.  Example: ``basis_i ?coeffs_i``
    denotes the inner product of a basis with unknown coefficient vector
    ``?coeffs``.  If possible the shape of the argument is deduced from the
    expression.  In the previous example the shape of ``?coeffs`` is equal to
    the shape of ``basis``.  If the shape cannot be deduced from the
    expression the shape should be defined manually (see :func:`parse`).
    Arguments and variables live in separate namespaces: ``?x`` and ``x`` are
    different entities.

*   An argument may be **substituted** by appending without whitespace
    ``(arg = value)`` to a variable of compound expression, where ``arg`` is
    an argument and ``value`` the substitution.  The substitution applies to
    the variable of compound expression only.  The value may be an
    expression.  Example: ``2 ?x(x = 3 + y)`` is equivalent to ``2 (3 + y)``
    and ``2 ?x(x=y) + 3`` is equivalent to ``2 (y) + 3``.  It is possible to
    apply multiple substitutions.  Example: ``(?x + ?y)(x = 1, y = )2`` is
    equivalent to ``1 + 2``.

*   The **gradient** of a variable to the default geometry — the default
    geometry is variable ``x`` unless overriden by the argument
    ``default_geometry_name`` — is denoted by an underscore, a comma and an
    index.  If the variable is an array with more than one axis, the
    underscore is omitted.  Example: ``a_,i`` denotes the gradient of the
    scalar ``a`` to the geometry and ``b_i,j`` the gradient of vector ``b``.
    The gradient of a compound expression is denoted by an underscore, a
    comma and an index.  Example: ``(a_i + b_j)_,k`` denotes the gradient of
    ``a_i + b_j``.  The usual summation rules apply and it is allowed to use
    a numeral as index.  The **surface gradient** is denoted with a semicolon
    instead of a comma, but follows the same rules as the gradient otherwise.
    Example: ``a_i;j`` is the sufrace gradient of ``a_i`` to the geometry.

*   The **normal** of the default geometry is denoted by ``n_i``, where the
    index ``i`` may be replaced with an index of choice.

*   A **dirac** is denoted by ``δ`` or ``$`` and takes two indices.  The
    shape of the dirac is deduced from the expression.  Example: let ``A`` be
    a square matrix with three rows and columns, then ``δ_ij`` in ``(A_ij - λ
    δ_ij) x_j`` has three rows and columns as well.

*   An expression surrounded by square brackets or curly braces denotes the
    **jump** or **mean**, respectively, of the enclosed expression.  Example:
    ``[ a_i ]`` denotes the jump of ``a_i`` and ``{ a_i + b_i }`` denotes the
    mean of ``a_i + b_i``.

*   A **function call** is denoted by a name — following the same rules as
    for a variable name — optionally followed by ``_`` and indices,
    optionally followed by ``:`` and indices, directly followed by the left
    parenthesis ``(``, without a space.  The arguments to the function are
    separated by a comma and at least one space.  The function is applied
    pointwise to the arguments and summation convection is applied to the
    result. Example: assume ``mul(...)`` returns the product of its
    arguments, then ``mul(x_i, y_j)`` is equivalent to ``x_i y_j`` and
    ``mul(x_i, y_i)`` to ``x_i y_i``. Functions and variables share a
    namespace: defining a variable with the same name as a function renders
    the function inaccessible. Functions of the form ``f_i(...)`` and
    ``f_ij(...)`` etc. generate one and two axes, respectively. Functions
    of the form ``f:i(...)`` and ``f:ij(...)`` etc. consume the axes labelled
    ``i`` and ``i`` and ``j`` respectively. If all axes are consumed, it is
    allowed to omit the axes: ``sum(u)`` is equivalent to ``sum:i(u_i)``.

*   A **stack** of two or more arrays along an axis is denoted by a ``<``
    followed by comma and space separated arrays followed by ``>`` and an
    index.  All arguments must have the same shape and must not have an axis labelled with the stack axis.
    Example: ``<1, 2>_i`` creates an array with components ``1`` and ``2``.

.. _`Einstein Summation Convection`: https://en.wikipedia.org/wiki/Einstein_notation
'''

import re
import collections
import functools
import itertools
import numpy
import operator
import types as builtin_types
from typing import Any, Callable, Dict, List, Mapping, Optional, overload, Tuple, Union
from . import function, types, warnings


# Convenience function to create a constant in ExpressionAST (details in
# docstring of `parse` below).
_ = lambda arg: (None, arg)


def _sp(count, singular, plural):
    '''format ``count``+ ``singular` or ``plural`` depending on ``count``'''
    return '{} {}'.format(count, singular if count == 1 else plural)


class ExpressionSyntaxError(Exception):
    pass


class AmbiguousAlignmentError(Exception):
    pass


class _IntermediateError(Exception):
    '''Intermediate exception, to be catched and converted into an ``ExpressionSyntaxError``.'''

    def __init__(self, msg, at=None, count=None):
        self.msg = msg
        self.at = at
        self.count = count
        super().__init__(msg)


_Token = collections.namedtuple('_Token', ['type', 'data', 'pos'])
_Token.__doc__ = 'An indivisible part of an expression string.'
_Token.type.__doc__ = 'The type of this token.'
_Token.data.__doc__ = 'Substring of the expression string that belongs to this token.'
_Token.pos.__doc__ = 'The start position of the token in the expression string.'


_Length = collections.namedtuple('_Length', ['pos'])
_Length.__doc__ = 'Yet unknown length, introduced at ``pos`` in the expression string.'
_Length.pos.__doc__ = 'The position where this :class:`_Length` is introduced.'


class _Array:
    '''ExpressionAST with shape, indices.

    The :class:`_Array` class combines an ExpressionAST with shape and indices
    and maintains a list of summed indices in the expression string resulting in
    this :class:`_Array`.

    Attributes
    ----------
    ast : :class:`tuple`
        The ExpressionAST (see :func:`parse`).
    indices : :class:`str`
        The indices of the array represented by the :attr:`ast`.
    shape : :class:`tuple` of :class:`int`\\s or :class:`_Length`\\s
        The shape of the array represented by the :attr:`ast`.
    summed : :class:`frozenset` of indices (:class:`str`)
        A set of indices that are summed in the expression string resulting in
        this :class:`_Array`.  The indices are not allowed in expressions
        involving this :class:`_Array`.  For example, index `i` in expression
        string ``'a_ij b_i'`` is summed and cannot be used in an expression like
        ``('a_ij b_i) c_i``.
    linked_lengths : :class:`frozenset` of :class:`frozensets` of :class:`_Length`\\s and :class:`int`\\s
        A set of sets of :class:`_Length`\\s and :class:`int`\\s.  A
        :class:`_Length` is introduced if an axis of an :class:`_Array` has an
        unknown length, e.g. a dirac has two axes of equal, but unknown length.
        All class:`_Length`\\s in a set have the same length.  If a set contains
        an :class:`int` the class:`_Length`\\s are resolved.
    ndim : :class:`int`
        The number of dimensions of this :class:`_Array`.

    Args
    ----
    ast : :class:`tuple`
        See :attr:`_Array.ast`.
    indices : :class:`str`
        See :attr:`_Array.indices`.
    shape : :class:`tuple` of :class:`int`\\s or :class:`_Length`\\s
        See :attr:`_Array.shape`.
    summed : :class:`frozenset` of indices (:class:`str`)
        See :attr:`_Array.summed`.
    linked_lengths : :class:`frozenset` of :class:`frozensets` of :class:`_Length`\\s and :class:`int`\\s
        See :attr:`_Array.linked_lengths`.
    '''

    @classmethod
    def wrap(cls, ast, indices, shape, linked_lengths=None):
        '''Create an :class:`_Array` by wrapping ``ast``.

        The ``ast`` should be a constant or variable.  Duplicate indices are summed
        and numeric indices are replaced by a getitem.
        '''

        if len(indices) != len(shape):
            raise _IntermediateError('Expected {}, got {}.'.format(_sp(len(shape), 'index', 'indices'), len(indices)))
        return cls._apply_indices(ast, 0, indices, shape, frozenset(), linked_lengths or frozenset())

    @classmethod
    def _apply_indices(cls, ast, offset, indices, shape, summed, linked_lengths):
        '''Wrap ``ast`` in an :class:`_Array`, thereby summing indices occuring twice and applying numeric indices.

        When wrapping a variable or gradient the indices of may appear twice,
        indicating summation, or numeric, indicating a getitem.  This method wraps
        ``ast`` and applies summation and getitem if needed.

        Args
        ----
        ast : :class:`tuple`
            See :attr:`_Array.ast`.
        offset : :class:`int`
            Start at index ``offset`` when looking for indices occuring twice (in
            the entire list of ``indices``, not only those in ``indices[offset:]``)
            or numeric indices.  The list ``indices[offset:]`` is assumed to be
            already processed.
        indices : :class:`str`
            See :attr:`_Array.indices`.
        shape : :class:`tuple` of :class:`int`\\s or :class:`_Length`\\s
            See :attr:`_Array.shape`.
            ``indices``.
        summed : :class:`frozenset` of indices (:class:`str`)
            See :attr:`_Array.summed`.
        linked_lengths : :class:`frozenset` of :class:`frozensets` of :class:`_Length`\\s and :class:`int`\\s
            See :attr:`_Array.linked_lengths`.

        Returns
        -------
        wrapped_ast : :class:`_Array foo : bar`
        '''

        summed = set(summed)
        linked_lengths = set(linked_lengths)
        i = offset
        dims = tuple(range(len(indices)))
        while i < len(indices):
            index = indices[i]
            j = indices.index(index)
            if '0' <= index <= '9':
                index = int(index)
                if isinstance(shape[i], int) and index >= shape[i]:
                    raise _IntermediateError('Index of dimension {} with length {} out of range.'.format(dims[i], shape[i]))
                ast = 'getitem', ast, _(i), _(index)
                indices = indices[:i] + indices[i+1:]
                shape = shape[:i] + shape[i+1:]
                dims = dims[:i] + dims[i+1:]
            elif index in summed:
                raise _IntermediateError('Index {!r} occurs more than twice.'.format(index))
            elif j < i:
                linked_lengths = set(cls._update_lengths(linked_lengths, index, shape[j], shape[i]))
                ast = 'trace', ast, _(j), _(i)
                indices = indices[:j] + indices[j+1:i] + indices[i+1:]
                shape = shape[:j] + shape[j+1:i] + shape[i+1:]
                dims = dims[:j] + dims[j+1:i] + dims[i+1:]
                summed.add(index)
                i -= 1
            else:
                if isinstance(shape[i], _Length) and not any(shape[i] in g for g in linked_lengths):
                    linked_lengths.add(frozenset([shape[i]]))
                i += 1

        return cls(ast, indices, shape, summed, linked_lengths)

    @classmethod
    def stack(cls, arrays, index):
        '''Stack ``arrays`` along axis ``index``.

        The arrays are stacked in given order.  All arrays should have matching
        shapes, except for the axis labeled ``index``.  If an array does not have
        the supplied ``index``, the array is expanded with an axis of length one
        before stacking.  For example, stacking a scalar and an array with shape
        ``{i: 2}`` along ``i`` gives an array with shape ``{i: 3}``.

        Args
        ----
        arrays : a :class:`~collections.abc.Sequence` of :class:`_Array` objects
            The arrays to stack.
        index : :class:`str`
            The index along which to stack the ``arrays``.

        Returns
        -------
        array : :class:`_Array`
            The stacked array.
        '''

        # TODO: assert is_valid_lhs_indices(index)
        if len(arrays) == 0:
            raise _IntermediateError('Cannot stack 0 arrays.')
        if len(set(frozenset(array.indices) - {index} for array in arrays)) != 1:
            raise _IntermediateError(
                'Cannot stack arrays with unmatched indices (excluding the stack index {!r}): {}.'
                .format(index, ', '.join(array.indices for array in arrays)))
        if any(index in array.indices for array in arrays):
            warnings.deprecation('Concatenating arrays with `<a_i, b_i>_i` syntax is deprecated.')
        indices = index + ''.join(i for i in arrays[0].indices if i != index)
        arrays = [(array.append_axis(index, 1) if index not in array.indices else array).transpose(indices) for array in arrays]

        if len(arrays) == 1:
            return arrays[0]

        helper = arrays[0].replace(indices=arrays[0].indices[1:], shape=arrays[0].shape[1:])
        for other in arrays[1:]:
            other = other.replace(indices=other.indices[1:], shape=other.shape[1:])
            shape, linked_lengths = helper._join_shapes(other)
            helper = helper.replace(shape=shape, linked_lengths=linked_lengths, summed=helper.summed | other.summed)

        # Apply `helper.linked_lengths` to all `arrays`.  If the lengths at
        # `index` is not known at this point, we won't be able to resolve this
        # ever, so raise an exception here.
        length = 0
        for array in arrays:
            shape = array._simplify_shape(helper.linked_lengths)
            if isinstance(shape[0], _Length):
                raise _IntermediateError('Cannot determine the length of the stack axis, because the length at {} is unknown.'.format(shape[0].pos), at=shape[0].pos)
            length += shape[0]

        ast = ('concatenate',) + tuple(array.ast for array in arrays)
        return helper.replace(ast=ast, indices=indices, shape=(length,)+helper.shape)

    @staticmethod
    def align(*arrays):
        '''Align ``arrays`` to the first array.

        Args
        ----
        arrays : :class:`_Array`
            The arrays to align.

        Returns
        -------
        aligned_arrays : :class:`tuple` of :class:`_Array` objects
            The aligned arrays.
        '''

        assert len(arrays) > 0
        if len(set(frozenset(array.indices) for array in arrays)) != 1:
            raise _IntermediateError(
                'Cannot align arrays with unmatched indices: {}.'
                .format(', '.join(array.indices for array in arrays)))
        arrays = [array.transpose(arrays[0].indices) for array in arrays]

        helper = arrays[0]
        for other in arrays[1:]:
            shape, linked_lengths = helper._join_shapes(other)
            helper = helper.replace(shape=shape, linked_lengths=linked_lengths, summed=helper.summed | other.summed)

        return tuple(array.replace(shape=helper.shape, linked_lengths=helper.linked_lengths, summed=helper.summed) for array in arrays)

    def __init__(self, ast, indices, shape, summed, linked_lengths):
        assert isinstance(indices, str)

        self.ast = tuple(ast)
        self.indices = indices
        self.shape = tuple(shape)
        self.summed = frozenset(summed)
        self.linked_lengths = frozenset(linked_lengths)

        self.ndim = len(self.indices)

    def _join_shapes(self, other):
        '''Verify ``self + other`` is valid and return the resulting shape and linked lengths.

        Args
        ----
        other : :class:`_Array`
            Should have the same (order of) indices as this array.

        Returns
        -------
        shape : :class:`tuple`
            The simplified shape of ``self + other``.
        linked_lengths : :class:`frozenset` of :class:`frozensets` of :class:`_Length`\\s and :class:`int`\\s
            See :attr:`_Array.linked_lengths`.  Updated with links resulting from
            applying ``self + other``.
        '''

        assert self.indices == other.indices, 'unaligned'
        groups = set(self.linked_lengths | other.linked_lengths)
        for index, a, b in zip(self.indices, self.shape, other.shape):
            if a == b:
                continue
            if not isinstance(a, _Length) and not isinstance(b, _Length):
                raise _IntermediateError('Shapes at index {!r} differ: {}, {}.'.format(index, a, b))
            groups.add(frozenset({a, b}))
        linked_lengths = self._join_lengths(other, groups)
        return self._simplify_shape(linked_lengths), linked_lengths

    def _simplify_shape(self, linked_lengths):
        '''Return simplified shape by replacing :class:`_Length`\\s with :class:`int`\\s according to the ``linked_lengths``.'''

        shape = []
        cache = {k: v for v in linked_lengths for k in v}
        for length in self.shape:
            if isinstance(length, _Length):
                for l in cache[length]:
                    if not isinstance(l, _Length):
                        length = l
                        break
            shape.append(length)
        return shape

    def _join_lengths(*args):
        '''Return updated linked lengths resulting from ``self + other``.'''

        groups = set()
        for arg in args:
            groups |= arg.linked_lengths if isinstance(arg, _Array) else arg
        cache = {}
        for g in groups:
            # g = frozenset(itertools.chain.from_iterable(map(linked_lenghts.get, g)))
            new_g = set()
            for k in g:
                new_g |= cache.get(k, frozenset([k]))
            new_g = frozenset(new_g)
            cache.update((k, new_g) for k in new_g)
        linked_lengths = frozenset(cache.values())
        # Verify.
        for g in linked_lengths:
            known = tuple(sorted(set(k for k in g if not isinstance(k, _Length))))
            if len(known) > 1:
                raise _IntermediateError('Axes have different lengths: {}.'.format(', '.join(map(str, known))))
        return linked_lengths

    @staticmethod
    def _update_lengths(linked_lengths, index, a, b):
        '''Add link ``a``, ``b`` to ``linked_lengths``.'''

        cache = {l: g for g in linked_lengths for l in g}
        if a != b:
            if not isinstance(a, _Length) and not isinstance(b, _Length):
                raise _IntermediateError('Shapes at index {!r} differ: {}, {}.'.format(index, a, b))
            g = cache.get(a, frozenset([a])) | cache.get(b, frozenset([b]))
            cache.update((k, g) for k in g)
            # Verify.
            known = tuple(sorted(set(k for k in g if not isinstance(k, _Length))))
            if len(known) > 1:
                raise _IntermediateError('Shapes at index {!r} differ: {}.'.format(index, ', '.join(map(str, known))))
        elif isinstance(a, _Length):
            cache.setdefault(a, frozenset([a]))
        return frozenset(cache.values())

    def __neg__(self):
        '''Return -self.'''

        return self.replace(ast=('neg', self.ast))

    def _add_sub(self, other, op, name):
        '''Return op(self, other).'''

        if frozenset(self.indices) != frozenset(other.indices):
            raise _IntermediateError('Cannot {} arrays with unmatched indices: {!r}, {!r}.'.format(name, self.indices, other.indices))
        other = other.transpose(self.indices)
        shape, linked_lengths = self._join_shapes(other)
        return _Array((op, self.ast, other.ast), self.indices, shape, self.summed, linked_lengths)

    def __add__(self, other):
        '''Return self+other.'''

        return self._add_sub(other, 'add', 'add')

    def __sub__(self, other):
        '''Return self-other.'''

        return self._add_sub(other, 'sub', 'subtract')

    def __mul__(self, other):
        '''Return self*other.'''

        for a, b in ((self, other), (other, self)):
            for index in sorted(frozenset(a.indices) | a.summed):
                if index in b.summed:
                    raise _IntermediateError('Index {!r} occurs more than twice.'.format(index))
        common = []
        for index, length in zip(self.indices, self.shape):
            if index in other.indices:
                common.append(index)
            else:
                other = other.append_axis(index, length)
        for index, length in zip(other.indices, other.shape):
            if index not in self.indices:
                self = self.append_axis(index, length)
        indices = self.indices
        other = other.transpose(indices)
        shape, linked_lengths = self._join_shapes(other)
        ast = 'mul', self.ast, other.ast
        for index in reversed(common):
            i = self.indices.index(index)
            ast = 'sum', ast, _(i)
            indices = indices[:i] + indices[i+1:]
            shape = shape[:i] + shape[i+1:]
        return _Array(ast, indices, shape, self.summed | other.summed | frozenset(common), linked_lengths)

    def __truediv__(self, other):
        '''Return self/value.'''

        if other.ndim > 0:
            raise _IntermediateError('A denominator must have dimension 0.')
        for index in sorted((self.summed | set(self.indices)) & other.summed):
            raise _IntermediateError('Index {!r} occurs more than twice.'.format(index))
        return _Array(('truediv', self.ast, other.ast), self.indices, self.shape, self.summed | other.summed, self._join_lengths(other))

    def __pow__(self, other):
        '''Return self**value.'''

        if other.ndim > 0:
            raise _IntermediateError('An exponent must have dimension 0.')
        for index in sorted((self.summed | set(self.indices)) & other.summed):
            raise _IntermediateError('Index {!r} occurs more than twice.'.format(index))
        return _Array(('pow', self.ast, other.ast), self.indices, self.shape, self.summed | other.summed, self._join_lengths(other))

    def grad(self, index, geom, type):
        '''Return the gradient w.r.t. ``geom``.'''

        assert geom.ndim == 1
        assert not isinstance(geom.shape[0], _Length)
        assert type in ('grad', 'surfgrad')
        ast = type, self.ast, _(geom)
        return _Array._apply_indices(ast, self.ndim, self.indices+index, self.shape+geom.shape, self.summed, self.linked_lengths)

    def derivative(self, arg):
        'Return the derivative to ``arg``.'

        return _Array._apply_indices(('derivative', self.ast, arg.ast), self.ndim, self.indices+arg.indices, self.shape+arg.shape, self.summed, self.linked_lengths)

    def append_axis(self, index, length):
        '''Return an :class:`_Array` with one additional axis.'''

        if index in self.indices or index in self.summed:
            raise _IntermediateError('Duplicate index: {!r}.'.format(index))
        linked_lengths = self.linked_lengths
        if isinstance(length, _Length):
            for group in linked_lengths:
                if length in group:
                    break
            else:
                linked_lengths |= frozenset({frozenset({length})})
        return _Array(('append_axis', self.ast, _(length)), self.indices+index, self.shape+(length,), self.summed, linked_lengths)

    def transpose(self, indices):
        '''Return an :class:`_Array` transposed according to ``indices``.'''

        if len(indices) != len(set(indices)):
            raise _IntermediateError('Cannot transpose from {!r} to {!r}: duplicate indices.'.format(self.indices, indices))
        elif set(self.indices) != set(indices):
            raise _IntermediateError('Cannot transpose from {!r} to {!r}: indices differ.'.format(self.indices, indices))
        if self.indices == indices:
            return self
        else:
            transpose = tuple(map(self.indices.index, indices))
            shape = tuple(map(self.shape.__getitem__, transpose))
            return _Array(('transpose', self.ast, _(transpose)), indices, shape, self.summed, self.linked_lengths)

    def replace(self, **updates):
        '''Return a copy of this :class:`_Array` with attributes replaced by ``updates``.'''

        kwargs = dict(ast=self.ast, indices=self.indices, shape=self.shape, summed=self.summed, linked_lengths=self.linked_lengths)
        kwargs.update(updates)
        return _Array(**kwargs)


class _ArrayOmittedIndices:

    def __init__(self, ast, shape):
        self.ast = tuple(ast)
        self.shape = tuple(shape)

        self.ndim = len(self.shape)

    def __add__(self, other):
        if self.shape != other.shape:
            raise _IntermediateError('Cannot add arrays with omitted indices because the shapes differ: {}, {}.'.format(self.shape, other.shape))
        return _ArrayOmittedIndices(('add', self.ast, other.ast), self.shape)

    def __sub__(self, other):
        if self.shape != other.shape:
            raise _IntermediateError('Cannot subtract arrays with omitted indices because the shapes differ: {}, {}.'.format(self.shape, other.shape))
        return _ArrayOmittedIndices(('sub', self.ast, other.ast), self.shape)

    def __mul__(self, other):
        if self.ndim != 0:
            raise _IntermediateError('Arrays with omitted indices cannot be multiplied.')
        self_ast = self.ast
        for n in other.shape:
            self_ast = ('append_axis', self_ast, _(n))
        return _ArrayOmittedIndices(('mul', self_ast, other.ast), other.shape)

    def __neg__(self):
        return _ArrayOmittedIndices(('neg', self.ast), self.shape)

    def __truediv__(self, other):
        if other.ndim > 0:
            raise _IntermediateError('A denominator must have dimension 0.')
        return _ArrayOmittedIndices(('truediv', self.ast, other.ast), self.shape)

    def __pow__(self, other):
        if other.ndim > 0:
            raise _IntermediateError('An exponent must have dimension 0.')
        return _ArrayOmittedIndices(('pow', self.ast, other.ast), self.shape)

    def replace(self, ast=None):
        return _ArrayOmittedIndices(self.ast if ast is None else ast, self.shape)


class _ExpressionParser:
    '''Expression parser

    Args
    ----
    expression : :class:`str`
        See argument ``expression`` of :func:`parse`.
    variables : :class:`dict` of :class:`str` and :class:`nutils.function.Array` pairs
        See argument ``variables`` of :func:`parse`.
    arg_shapes : :class:`dict` of :class:`str` and :class:`tuple` or :class:`int`\\s pairs
        See argument ``arg_shapes`` of :func:`parse`.
    default_geometry_name : class:`str`
        See argument ``default_geometry_name`` of :func:`parse`.
    fixed_lengths : :class:`dict` of :class:`str` and :class:`int`
        See argument ``fixed_lengths`` of :func:`parse`.
    '''

    eye_symbols = '$', 'δ'
    normal_symbols = 'n',

    def __init__(self, expression, variables, arg_shapes, default_geometry_name, fixed_lengths):
        self.expression = expression
        self.variables = variables
        self.arg_shapes = dict(arg_shapes)
        self.default_geometry_name = default_geometry_name
        self.fixed_lengths = fixed_lengths

    def highlight(f):
        'wrap ``f`` in a function that converts ``_IntermediateError`` objects'

        def wrapper(self, *args, **kwargs):
            if hasattr(self, '_tokens'):
                pos = self._next.pos
            else:
                pos = 0
            try:
                return f(self, *args, **kwargs)
            except _IntermediateError as e:
                if e.at is None:
                    at = pos
                    count = self._next.pos - pos if self._next.pos > pos else len(self._next.data)
                else:
                    at = e.at
                    count = 1 if e.count is None else e.count
                raise ExpressionSyntaxError(e.msg + '\n' + self.expression + '\n' + ' '*at + '^'*count) from e
        return wrapper

    def _consume(self):
        'advance to next token'

        self._index += 1
        if self._index >= len(self._tokens):
            raise _IntermediateError('Unexpected end of expression.', at=len(self.expression))
        return self._current

    def _consume_if_whitespace(self):
        'advance to next token if it is a whitespace'

        if self._next.type == 'whitespace':
            self._consume()

    @highlight
    def _consume_assert_whitespace(self):
        'assert the next token is whitespace, skip it, and advance to next token'

        if self._consume().type != 'whitespace':
            raise _IntermediateError('Missing whitespace.', at=self._current.pos)

    @highlight
    def _consume_assert_equal(self, value, msg=None):
        'assert the next token is equal to ``value``'

        token = self._consume()
        if token.type != value:
            if msg is None:
                msg = 'Expected {!r}.'.format(value)
            raise _IntermediateError(msg, at=token.pos)
        return token

    @property
    def _current(self):
        'the current token'

        return self._tokens[self._index]

    @property
    def _next(self):
        'the next token'

        return self._tokens[min(len(self._tokens)-1, self._index+1)]

    @property
    def _next_non_whitespace(self):
        'the next non-whitespace token'

        return self._tokens[self._index+2] if self._next.type == 'whitespace' else self._next

    def _asarray(self, ast, indices_token, shape, omitted_indices):
        indices = indices_token.data if indices_token else ''
        if omitted_indices:
            assert not indices
            return _ArrayOmittedIndices(ast, shape)
        if len(indices) != len(shape):
            raise _IntermediateError('Expected {}, got {}.'.format(_sp(len(shape), 'index', 'indices'), len(indices)))
        linked_lengths = set()
        for iaxis, (length, index) in enumerate(zip(shape, indices)):
            fixed = self.fixed_lengths.get(index)
            if fixed is None:
                pass
            elif isinstance(length, _Length):
                linked_lengths.add(frozenset([length, fixed]))
            elif fixed != length:
                raise _IntermediateError('Length of index {} is fixed at {} but the expression has length {}.'.format(index, fixed, length), at=indices_token.pos+iaxis, count=1)
        return _Array.wrap(ast, indices, shape, linked_lengths)

    def _get_variable(self, name):
        'get variable by ``name`` or raise an error'

        value = self.variables.get(name, None)
        if value is None:
            raise _IntermediateError('Unknown variable: {!r}.'.format(name))
        return value

    def _get_geometry(self, name):
        'get geometry by ``name`` or raise an error'

        geom = self._get_variable(name)
        if geom.ndim != 1:
            raise _IntermediateError('Invalid geometry: expected 1 dimension, but {!r} has {}.'.format(name, geom.ndim))
        return geom

    def _get_arg(self, name, indices_token):
        'get arg by ``name`` or raise an error'

        indices = indices_token.data if indices_token else ''
        if name in self.arg_shapes:
            shape = self.arg_shapes[name]
            if len(shape) != len(indices):
                raise _IntermediateError('Argument {!r} previously defined with {} instead of {}.'.format(name, _sp(len(shape), 'axis', 'axes'), len(indices)))
        else:
            shape = tuple(_Length(indices_token.pos+i) for i, j in enumerate(indices))
            self.arg_shapes[name] = shape
        return self._asarray(('arg', _(name)) + tuple(map(_, shape)), indices_token, shape, False)

    @highlight
    def parse_lhs_arg(self, seen_lhs):
        'parse lhs arg, e.g. the "x_ij" in "x_kk(x_ij=a_ij)"'

        token = self._consume()
        if token.type != 'variable':
            raise _IntermediateError("Expected an argument, e.g. 'argname'.")
        if token.data.startswith('?'):
            raise _IntermediateError("The argument name at the left hand side of a substitution must not be prefixed by a '?'.")
        name = token.data
        if name in seen_lhs:
            raise _IntermediateError("Argument {!r} occurs more than once.".format(name))
        seen_lhs[name] = token
        indices = self._consume() if self._next.type == 'indices' else ''
        for i, index in enumerate(indices and indices.data):
            if index in indices.data[i+1:]:
                raise _IntermediateError('Repeated indices are not allowed on the left hand side.')
            elif '0' <= index <= '9':
                raise _IntermediateError('Numeric indices are not allowed on the left hand side.')
        return self._get_arg(name, indices)

    @highlight
    def parse_var(self, omitted_indices):
        'parse a component of a term, e.g. "1", "a_i", "(2 a_i)", "a_i^2", "abs(x)"'

        if self._next.type == '(':
            self._consume()
            value = self.parse_subexpression_cast(omitted_indices)
            self._consume_assert_equal(')')
            value = value.replace(ast=('group', value.ast))
        elif self._next.type == '[':
            self._consume()
            value = self.parse_subexpression_cast(omitted_indices)
            self._consume_assert_equal(']')
            value = value.replace(ast=('jump', value.ast))
            if self._next.type == 'geometry':
                geometry_name = self._consume().data
            else:
                geometry_name = self.default_geometry_name
            if not omitted_indices and self._next.type == 'indices':
                geom = self._get_geometry(geometry_name)
                warnings.deprecation('`[f]_i` and `[f]_x_i` are deprecated; use `[f] n({x}_i)` instead'.format(x=geometry_name))
                value *= self._asarray(('normal', _(geom)), self._consume(), geom.shape, False)
        elif self._next.type == '{':
            self._consume()
            value = self.parse_subexpression_cast(omitted_indices)
            self._consume_assert_equal('}')
            value = value.replace(ast=('mean', value.ast))
        elif not omitted_indices and self._next.type == '<':
            self._consume()
            args = self.parse_comma_separated(end='>', parse_item=functools.partial(self.parse_subexpression, omitted_indices=False))
            indices = self._consume()
            if indices.type != 'indices':
                raise _IntermediateError('Expected 1 index.', at=indices.pos, count=len(indices.data))
            if len(indices.data) != 1:
                raise _IntermediateError('Expected 1 index, got {}.'.format(len(indices.data)), at=indices.pos, count=len(indices.data))
            if '0' <= indices.data <= '9':
                raise _IntermediateError('Expected a non-numeric index, got {!r}.'.format(indices.data), at=indices.pos, count=len(indices.data))
            value = _Array.stack(args, indices.data)
        elif not omitted_indices and self._next.type == 'jacobian':
            nbounds = len(self._consume().data)-1
            geometry_name = self._consume_assert_equal('geometry').data
            geom = self._get_geometry(geometry_name)
            value = self._asarray(('jacobian', _(geom), _(len(geom)-nbounds)), None, (), False)
        elif not omitted_indices and self._next.type == 'old-jacobian':
            self._consume()
            geometry_name = self._consume_assert_equal('geometry').data
            geom = self._get_geometry(geometry_name)
            value = self._asarray(('jacobian', _(geom), _(None)), None, (), False)
        elif not omitted_indices and self._next.type == 'derivative':
            self._consume()
            target = self._consume()
            assert target.type in ('geometry', 'argument')
            indices = self._consume() if self._next.type == 'indices' else None
            if target.type == 'geometry':
                warnings.deprecation('the gradient syntax `dx_i:u` is deprecated; use `d(u, x_i)` instead')
                geom = self._get_geometry(target.data)
            elif target.type == 'argument':
                assert target.data.startswith('?')
                warnings.deprecation('the derivative syntax `d?a:u` is deprecated; use `d(u, ?a)` instead')
                arg = self._get_arg(target.data[1:], indices)
            func = self.parse_var(False)
            if target.type == 'geometry':
                return func.grad(indices.data if indices else '', geom, 'grad')
            else:
                return func.derivative(arg)
        elif not omitted_indices and self._next.type == 'eye':
            self._consume()
            indices = self._consume() if self._next.type == 'indices' else None
            length = _Length(self._current.pos)
            value = self._asarray(('eye', _(length)), indices, (length, length), False)
        elif self._next.type == 'variable':
            token = self._consume()
            name = token.data
            if not omitted_indices and name.startswith('?'):
                indices = self._consume() if self._next.type == 'indices' else None
                value = self._get_arg(name[1:], indices)
            elif name not in self.variables and next(t for t in self._tokens[self._index+1:] if t.type not in ('indices', 'consumes')).type == '(':  # assume function
                if not omitted_indices and self._next.type == 'indices':
                    generates_token = self._consume()
                    generates = generates_token.data
                    generates_shape = tuple(_Length(pos) for pos, index in enumerate(generates, generates_token.pos) if not '0' <= index <= '9')
                else:
                    generates = ''
                    generates_shape = ()
                consumes = self._consume().data if not omitted_indices and self._next.type == 'consumes' else ''
                self._consume_assert_equal('(')
                args_omitted_indices = None
                if omitted_indices:
                    args_omitted_indices = self.parse_comma_separated(end=')', parse_item=functools.partial(self.parse_subexpression, omitted_indices=True))
                elif not consumes:
                    try:
                        index = self._index
                        args_omitted_indices = self.parse_comma_separated(end=')', parse_item=functools.partial(self.parse_subexpression, omitted_indices=True))
                    except ExpressionSyntaxError:
                        self._index = index
                if args_omitted_indices:
                    if not all(arg.shape == args_omitted_indices[0].shape for arg in args_omitted_indices):
                        raise _IntermediateError('All arguments should have the same shape.')
                    ast = ('call', _(name), _(len(generates)), _(args_omitted_indices[0].ndim), *(arg.ast for arg in args_omitted_indices))
                    if omitted_indices:
                        value = _ArrayOmittedIndices(ast, generates_shape)
                    else:
                        value = _Array._apply_indices(ast,
                                                      offset=0,
                                                      indices=generates,
                                                      shape=generates_shape,
                                                      summed=frozenset(),
                                                      linked_lengths=frozenset())
                else:
                    args = self.parse_comma_separated(end=')', parse_item=functools.partial(self.parse_subexpression, omitted_indices=False))
                    if consumes:
                        if not all(set(consumes) <= set(arg.indices) for arg in args):
                            raise _IntermediateError('All axes to be consumed ({}) must be present in all arguments.'.format(consumes))
                        args = tuple(arg.transpose(''.join(i for i in arg.indices if i not in consumes)+consumes) for arg in args)
                    value = _Array._apply_indices(ast=('call', _(name), _(len(generates)), _(len(consumes)), *(arg.ast for arg in args)),
                                                  offset=0,
                                                  indices=''.join(arg.indices[:arg.ndim-len(consumes)] for arg in args)+generates,
                                                  shape=sum((arg.shape[:arg.ndim-len(consumes)] for arg in args), ())+generates_shape,
                                                  summed=functools.reduce(operator.or_, (arg.summed for arg in args), frozenset(consumes)),
                                                  linked_lengths=functools.reduce(operator.or_, (arg.linked_lengths for arg in args), frozenset()))
            elif name in self.normal_symbols:
                if self._next.type == 'geometry':
                    warnings.deprecation('the normal syntax with explicitly geometry `n:x_i` is deprecated; use `n(x_i)` instead')
                    geometry_name = self._consume().data
                else:
                    geometry_name = self.default_geometry_name
                geom = self._get_geometry(geometry_name)
                if omitted_indices:
                    value = _ArrayOmittedIndices(('normal', _(geom)), geom.shape)
                else:
                    indices = self._consume() if self._next.type == 'indices' else None
                    value = self._asarray(('normal', _(geom)), indices, geom.shape, False)
            else:
                raw = self._get_variable(name)
                if omitted_indices:
                    value = _ArrayOmittedIndices(_(raw), raw.shape)
                else:
                    indices = self._consume() if self._next.type == 'indices' else None
                    value = self._asarray(_(raw), indices, raw.shape, False)
        else:
            raise _IntermediateError('Expected a variable, group or function call.')

        if not omitted_indices and self._next.type == 'gradient':
            gradient = self._consume()
            target = self._consume()
            assert target.type in ('geometry', 'argument')
            indices = self._consume() if self._next.type == 'indices' else None
            if target.type == 'geometry':
                assert indices
                gradtype = {',': 'grad', ';': 'surfgrad'}[gradient.data]
                if target.data:
                    if gradient.data == ',':
                        warnings.deprecation('the gradient syntax with explicit geometry `u_,x_i` is deprecated; use `d(u, x_i)` instead')
                    else:
                        warnings.deprecation('the surface gradient syntax with explicit geometry `u_;x_i` is deprecated; use `surfgrad(u, x_i)` instead')
                geom = self._get_geometry(target.data or self.default_geometry_name)
                for i, index in enumerate(indices.data):
                    value = value.grad(index, geom, gradtype)
            elif target.type == 'argument':
                assert gradient.data == ','
                assert target.data.startswith('?')
                warnings.deprecation('the derivative to argument syntax `u_,?a` is deprecated; use `d(u, ?a)` instead')
                arg = self._get_arg(target.data[1:], indices)
                value = value.derivative(arg)
        elif not omitted_indices and self._next.type == 'indices':
            raise _IntermediateError("Indices can only be specified for variables, e.g. 'a_ij', not for groups, e.g. '(a+b)_ij'.", at=self._next.pos, count=len(self._next.data))

        if not omitted_indices and self._next.type == '(':
            self._consume()
            subs = self.parse_comma_separated(end=')', parse_item=functools.partial(self.parse_substitution, seen_lhs={}))
            if not subs:
                raise _IntermediateError("Zero substitutions are not allowed.")
            ast = ['substitute', value.ast]
            links = []  # type: List[LinkedLengths]
            for lhs, rhs in subs:
                ast += [lhs.ast, rhs.ast]
                links += [rhs.linked_lengths, frozenset(frozenset({l, r}) for l, r in zip(lhs.shape, rhs.shape))]
            value = value.replace(ast=tuple(ast), linked_lengths=value._join_lengths(*links))

        if self._next.type == '^':
            token = self._consume()
            if self._next.type == '(':
                self._consume()
                exponent = self.parse_subexpression(omitted_indices)
                self._consume_assert_equal(')')
            else:
                if self._next.type == '-':
                    self._consume()
                    negate = True
                else:
                    negate = False
                exponent = self.parse_const_scalar(omitted_indices)
                if negate:
                    exponent = -exponent
            value = value**exponent

        return value

    @highlight
    def parse_const_scalar(self, omitted_indices):
        'parse a constant scalar, e.g. "1", "1.0", "0.1", "1e3", ".1e0", "1.2e03"'

        token = self._consume()
        if token.type == 'int':
            value = self._asarray(_(int(token.data)), '', [], omitted_indices)
        elif token.type == 'float':
            value = self._asarray(_(float(token.data)), '', [], omitted_indices)
        else:
            raise _IntermediateError('Expected a number.')
        if not omitted_indices and self._next.type == 'gradient':
            self._consume()
            self._consume()
            if self._next.type == 'indices':
                self._consume()
            raise _IntermediateError('Taking a derivative of a constant is not allowed.')

        return value

    @highlight
    def parse_const(self, omitted_indices):
        'parse a const, possibly with indices, e.g. "1_j"'

        value = self.parse_const_scalar(omitted_indices)
        if not omitted_indices and self._next.type == 'indices':
            token = self._consume()
            indices = token.data
            for i, index in enumerate(indices):
                if '0' <= index <= '9':
                    raise _IntermediateError('Numeric indices are not allowed on constant values.')
                if index in indices[1+i:]:
                    raise _IntermediateError('Indices of a constant value may not be repeated.')
                value = value.append_axis(index, _Length(pos=token.pos+i))
        if not omitted_indices and self._next.type == 'gradient':
            self._consume()
            self._consume()
            if self._next.type == 'indices':
                self._consume()
            raise _IntermediateError('Taking a derivative of a constant is not allowed.')
        if self._next.type == '^':
            token = self._consume()
            if self._next.type == '(':
                self._consume()
                exponent = self.parse_subexpression_cast(omitted_indices)
                self._consume_assert_equal(')')
            else:
                if self._next.type == '-':
                    self._consume()
                    negate = True
                else:
                    negate = False
                exponent = self.parse_const_scalar(omitted_indices)
                if negate:
                    exponent = -exponent
            value = value**exponent
        return value

    @highlight
    def parse_numerator(self, omitted_indices):
        'parse the numerator part of a fraction'

        if self._next.type in ('int', 'float'):
            value = self.parse_const(omitted_indices)
        else:
            value = self.parse_var(omitted_indices)

        while True:
            stop = self._next.pos
            if self._next_non_whitespace.type in (')', ']', '}', '>', 'EOF', '+', '-', '/', '|', ','):
                break
            self._consume_assert_whitespace()
            value *= self.parse_var(omitted_indices)

        return value

    @highlight
    def parse_denominator(self, omitted_indices):
        'parse the denominator part of a fraction'

        value = self.parse_numerator(omitted_indices)
        if value.ndim > 0:
            raise _IntermediateError('A denominator must have dimension 0.')
        return value

    def parse_comma_separated(self, end, parse_item):
        'parse comma separated values until end token, e.g. "1, 2 (a_ij b_j + 3))" with end token ")"'

        items = []
        self._consume_if_whitespace()
        if self._next.type != end:
            while True:
                items.append(parse_item())
                self._consume_if_whitespace()
                if self._next.type != ',':
                    break
                self._consume_assert_equal(',')
                self._consume_assert_whitespace()
        self._consume_assert_equal(end)
        return items

    @highlight
    def parse_substitution(self, seen_lhs):
        'parse a substitution, e.g. "x_ij=a_ij" in "?x_kk(x_ij=a_ij)"'

        lhs = self.parse_lhs_arg(seen_lhs)
        self._consume_if_whitespace()
        self._consume_assert_equal('=')
        self._consume_if_whitespace()
        rhs = self.parse_subexpression(False)
        if set(lhs.indices) != set(rhs.indices):
            raise _IntermediateError('Left and right hand side should have the same indices, got {!r} and {!r}.'.format(lhs.indices, rhs.indices))
        rhs = rhs.transpose(lhs.indices)
        return lhs, rhs

    @highlight
    def parse_term(self, omitted_indices):
        'parse a term, e.g. "a b_i (2 c_i + 1)"'

        value = self.parse_numerator(omitted_indices)
        if self._next_non_whitespace.type == '/':
            self._consume_assert_whitespace()
            token = self._consume()
            assert token.type == '/'
            self._consume_assert_whitespace()
            denominator = self.parse_denominator(omitted_indices)
            value /= denominator
        return value

    @highlight
    def parse_subexpression(self, omitted_indices):
        'parse a scope: the entire expression or a subexpression between parentheses'

        self._consume_if_whitespace()
        negate = self._next.type == '-'
        if negate:
            self._consume()
        self._consume_if_whitespace()
        value = self.parse_term(omitted_indices)
        if negate:
            value = -value

        while self._next_non_whitespace.type not in ('|', 'EOF', '_', ')', ']', '}', '>', ','):
            self._consume_assert_whitespace()
            op_token = self._consume()
            if op_token.type not in '+-':
                raise _IntermediateError('Expected {!r} or {!r}.'.format('+', '-'), at=op_token.pos, count=len(op_token.data))
            self._consume_assert_whitespace()
            r_value = self.parse_term(omitted_indices)
            value = {'+': value.__add__, '-': value.__sub__}[op_token.type](r_value)

        self._consume_if_whitespace()
        return value

    @highlight
    def parse_subexpression_cast(self, omitted_indices):
        if omitted_indices:
            return self.parse_subexpression(omitted_indices)
        try:
            index = self._index
            value = self.parse_subexpression(True)
            if value.ndim == 0:
                return _Array(value.ast, '', (), frozenset(), frozenset())
        except ExpressionSyntaxError:
            pass
        self._index = index
        return self.parse_subexpression(False)

    @highlight
    def tokenize(self):
        'subdivide :attr:`expression` in indivisible tokens'

        pos = 0
        tokens = [_Token('BOF', '', pos)]
        while pos < len(self.expression):
            m = re.match(r'\s+', self.expression[pos:])
            if m:
                tokens.append(_Token('whitespace', m.group(0), pos))
                pos += m.end()
                continue
            if self.expression[pos] in '+-^/|=[]{}()<>,':
                tokens.append(_Token(self.expression[pos], self.expression[pos], pos))
                pos += 1
                continue
            m = re.match(r'(J\^*):([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)', self.expression[pos:])
            if m:
                tokens.append(_Token('jacobian', m.group(1), pos))
                tokens.append(_Token('geometry', m.group(2), 1+len(m.group(1))))
                pos += m.end()
                continue
            m = re.match(r'd:[a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*', self.expression[pos:])
            if m:
                tokens.append(_Token('old-jacobian', m.group(0)[:1], pos))
                tokens.append(_Token('geometry', m.group(0)[2:], pos+2))
                pos += m.end()
                continue
            m = re.match(r'd(\??[a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-zA-Z0-9]+)?:', self.expression[pos:])
            if m:
                tokens.append(_Token('derivative', 'd', pos))
                tokens.append(_Token('argument' if m.group(1).startswith('?') else 'geometry', m.group(1), pos+m.start(1)))
                if m.group(2):
                    tokens.append(_Token('indices', m.group(2)[1:], pos+m.start(2)+1))
                pos += m.end()
                continue
            m = re.match(r'({}):([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)_([a-zA-Z0-9])'.format('|'.join(map(re.escape, self.normal_symbols))), self.expression[pos:])
            if m:
                tokens.append(_Token('variable', m.group(1), pos))
                tokens.append(_Token('geometry', m.group(2), pos+m.start(2)))
                tokens.append(_Token('indices', m.group(3), pos+m.start(3)))
                pos += m.end()
                continue
            m_variable = re.match(r'[?]?[a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*', self.expression[pos:])
            m_variable = m_variable.group(0) if m_variable else ''
            m_eye = _string_startswith(self.expression, self.eye_symbols, start=pos)
            # Insert eye or normal symbols only if we can't match a longer variable name.
            if m_eye and len(m_variable) <= len(m_eye):
                tokens.append(_Token('eye', m_eye, pos))
                pos += len(m_eye)
                continue
            m_normal = _string_startswith(self.expression, self.normal_symbols, start=pos)
            if m_variable:
                tokens.append(_Token('variable', m_variable, pos))
                pos += len(m_variable)
                continue
            m = re.match(r'[0-9]+e-?[0-9]+|([0-9]+[.][0-9]*|[.][0-9]+)(e-?[0-9]+)?', self.expression[pos:])
            if m:
                if m.group(0).startswith('0') and not (m.group(0).startswith('0.') or m.group(0).startswith('0e')):
                    raise _IntermediateError('Leading zeros are forbidden.', at=pos, count=len(m.group(0)))
                tokens.append(_Token('float', m.group(0), pos))
                pos += m.end()
                continue
            m = re.match(r'[0-9]+', self.expression[pos:])
            if m:
                if m.group(0).startswith('0') and not m.group(0) == '0':
                    raise _IntermediateError('Leading zeros are forbidden.', at=pos, count=len(m.group(0)))
                tokens.append(_Token('int', m.group(0), pos))
                pos += m.end()
                continue
            if self.expression[pos] == '_':
                pos += 1
                parts = 0
                m = re.match(r'[a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*_', self.expression[pos:])
                if m:
                    withgeom = m.group(0)[:-1]
                    tokens.append(_Token('geometry', m.group(0)[:-1], pos))
                    pos += m.end()
                else:
                    withgeom = None
                m = re.match(r'[a-zA-Z0-9]+', self.expression[pos:])
                if m:
                    tokens.append(_Token('indices', m.group(0), pos))
                    pos += m.end()
                    parts += 1
                m_arg = re.match(r'(,)([?][a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-zA-Z0-9]+)?', self.expression[pos:])
                m_geom = re.match(r'([,;])(([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)_)?([a-zA-Z0-9]+)', self.expression[pos:])
                if m_arg:
                    tokens.append(_Token('gradient', m_arg.group(1), pos))
                    tokens.append(_Token('argument', m_arg.group(2), pos+m_arg.start(2)))
                    if m_arg.group(3):
                        tokens.append(_Token('indices', m_arg.group(3)[1:], pos+m_arg.start(3)+1))
                    pos += m_arg.end()
                    parts += 1
                elif m_geom:
                    if withgeom is not None and not m_geom.group(2):
                        variant_geom = m_geom.group(1) + withgeom + '_' + m_geom.group(4)
                        variant_default = m_geom.group(1) + self.default_geometry_name + '_' + m_geom.group(4)
                        raise _IntermediateError('Missing geometry, e.g. {!r} or {!r}.'.format(variant_geom, variant_default), at=pos)
                    tokens.append(_Token('gradient', m_geom.group(1), pos))
                    tokens.append(_Token('geometry', m_geom.group(3), pos+m_geom.start(3)))
                    tokens.append(_Token('indices', m_geom.group(4), pos+m_geom.start(4)))
                    pos += m_geom.end()
                    parts += 1
                if parts == 0:
                    raise _IntermediateError('Missing indices.', at=pos)
                continue
            m = re.match(r':([a-zA-Z]+)', self.expression[pos:])
            if m:
                tokens.append(_Token('consumes', m.group(1), pos+1))
                pos += m.end()
                continue
            raise _IntermediateError('Unknown symbol: {!r}.'.format(self.expression[pos]), at=pos)
        tokens.append(_Token('EOF', '', pos))
        self._tokens = tokens
        self._index = 0


def _string_startswith(string, prefixes, start=0):
    assert not isinstance(prefixes, str)
    for prefix in prefixes:
        if string.startswith(prefix, start):
            return prefix


def _replace_lengths(ast, lengths):
    'replace all :class:`_Length` objects in ``ast`` with the lengths in ``lengths``'

    if ast[0] is not None:
        return (ast[0],) + tuple(_replace_lengths(arg, lengths) for arg in ast[1:])
    elif isinstance(ast[1], _Length):
        return _(lengths[ast[1]])
    else:
        return ast


def parse(expression, variables, indices, arg_shapes={}, default_geometry_name='x', fixed_lengths=None, fallback_length=None, functions=None):
    '''Parse ``expression`` and return AST.

    This function parses a tensor expression according to the syntax described in
    module :mod:`nutils.expression_v1` and returns an Abstract Syntax Tree (AST).

    Args
    ----
    expression : :class:`str`
        The expression to parse.  See :mod:`~nutils.expression_v1` for the
        expression syntax.
    variables : :class:`dict` of :class:`str` and :class:`nutils.function.Array` pairs
        A :class:`dict` of variable names and array pairs.  All variables used in
        the ``expression`` should exist in ``variables``.
    indices : :class:`str`
        The indices used for aligning the resulting array.  For example, let
        ``expression`` be ``'a_ij'``.  If ``indices`` is ``'ij'``, then the
        returned array is simply ``variables['a']``, but if ``indices`` is
        ``'ji'`` the transpose of ``variables['a']`` is returned.  All indices of
        the ``expression`` should be listed precisely once.
    arg_shapes : :class:`dict` of :class:`str` and :class:`tuple` or :class:`int`\\s pairs
        A :class:`dict` of argument names and shapes.  If ``expression`` contains
        an argument not present in ``arg_shapes`` the shape will be decuded from
        the expression and added to a copy of ``arg_shapes``.
    default_geometry_name : :class:`str`
        The name of the default geometry variable.  When computing a gradient or
        the normal, e.g. ``'f_,i'`` or ``'n_i'``, this variable is used as the
        geometry, unless the geometry is explicitly mentioned in the expression.
        Default: ``'x'``.
    fixed_lengths : :class:`dict` of :class:`str` and :class:`int` pairs, optional
        A :class:`dict` of indices and lengths.  All axes in the expression
        marked with an index of fixed length are asserted to have the fixed
        length.
    fallback_length : :class:`int`, optional
        The fallback length of an axis if the length cannot be determined from
        the expression.

    Returns
    -------
    ast : :class:`tuple`
        The parsed ``expression`` as an abstract syntax tree (AST).  The AST is a
        :class:`tuple` of an opcode and arguments.  The special opcode ``None``
        indicates that the single argument is used verbatim.  All other opcodes
        have AST as arguments.  The following opcodes exist::

            (None, const)
            ('group', group)
            ('arg', name, *shape)
            ('substitute', array, arg, value)
            ('call', func, arg)
            ('eye', length)
            ('normal', geom)
            ('getitem', array, dim, index)
            ('trace', array, n1, n2)
            ('sum', array, axis)
            ('concatenate', *args)
            ('grad', array, geom)
            ('surfgrad', array, geom)
            ('derivative', func, target)
            ('append_axis', array, length)
            ('transpose', array, trans)
            ('jump', array)
            ('mean', array)
            ('neg', array)
            ('add', left, right)
            ('sub', left, right)
            ('mul', left, right)
            ('truediv', left, right)
            ('pow', left, right)
    arg_shapes : :class:`dict` of :class:`str` and :class:`tuple` of :class:`int`\\s pairs
        A copy of ``arg_shapes`` updated with shapes of arguments present in this
        ``expression``.
    '''

    if functions is not None:
        warnings.deprecation('argument `functions` is deprecated; the existence and number of arguments is not checked during parsing')
    parser = _ExpressionParser(expression, variables, arg_shapes, default_geometry_name, fixed_lengths or {})
    parser.tokenize()
    if indices is None:
        try:
            value = parser.parse_subexpression(True)
            return value.ast, arg_shapes
        except ExpressionSyntaxError:
            pass
        parser._index = 0
    value = parser.parse_subexpression(False)
    parser._consume_assert_equal('EOF', msg='Unexpected symbol at end of expression.')
    if indices is None:
        if value.ndim > 1:
            raise AmbiguousAlignmentError(
                'Cannot unambiguously align the array because the array has more than one dimension.\n'
                + expression + '\n'
                + '^'*len(expression))
        ast = value.ast
    else:
        try:
            ast = value.transpose(indices).ast
        except _IntermediateError as e:
            raise ExpressionSyntaxError(e.msg + '\n' + expression + '\n' + '^'*len(expression)) from e
    lengths = {}
    undetermined = set()
    for group in value.linked_lengths:
        ints = tuple(i for i in group if not isinstance(i, _Length))
        assert len(ints) <= 1, 'multiple integers in linked lengths group'
        val = ints[0] if ints else fallback_length
        if val is None:
            undetermined.update(i.pos for i in group)
        else:
            lengths.update((length, val) for length in group)
    for pos in sorted(undetermined):
        raise ExpressionSyntaxError('Length of axis cannot be determined from the expression.' + '\n' + expression + '\n' + ' '*pos + '^')
    arg_shapes = dict(arg_shapes)
    for arg, shape in parser.arg_shapes.items():
        arg_shapes[arg] = tuple(lengths.get(i, i) for i in shape)
    return _replace_lengths(ast, lengths), arg_shapes


def _eval_ast(ast, functions):
    '''evaluate ``ast`` generated by :func:`parse`'''

    op, *args = ast
    if op is None:
        value, = args
        return value

    args = [_eval_ast(arg, functions) for arg in args]
    if op == 'group':
        array, = args
        return array
    elif op == 'arg':
        name, *shape = args
        return function.Argument(name, shape)
    elif op == 'substitute':
        array, *arg_value_pairs = args
        subs = {}
        assert len(arg_value_pairs) % 2 == 0
        for arg, value in zip(arg_value_pairs[0::2], arg_value_pairs[1::2]):
            assert arg.name not in subs
            subs[arg.name] = value
        return function.replace_arguments(array, subs)
    elif op == 'call':
        func, generates, consumes, *args = args
        args = tuple(map(function.Array.cast, args))
        kwargs = {}
        if generates:
            kwargs['generates'] = generates
        if consumes:
            kwargs['consumes'] = consumes
        result = functions[func](*args, **kwargs)
        shape = sum((arg.shape[:arg.ndim-consumes] for arg in args), ())
        if result.ndim != len(shape) + generates or result.shape[:len(shape)] != shape:
            raise ValueError('expected an array with shape {} and {} additional axes when calling {} but got {}'.format(shape, generates, func, result.shape))
        return result
    elif op == 'jacobian':
        geom, ndims = args
        return function.J(geom, ndims)
    elif op == 'eye':
        length, = args
        return function.eye(length)
    elif op == 'normal':
        geom, = args
        return function.normal(geom)
    elif op == 'getitem':
        array, dim, index = args
        return function.get(array, dim, index)
    elif op == 'trace':
        array, n1, n2 = args
        return numpy.trace(array, axis1=n1, axis2=n2)
    elif op == 'sum':
        array, axis = args
        return numpy.sum(array, axis)
    elif op == 'concatenate':
        return numpy.concatenate(args, axis=0)
    elif op == 'grad':
        array, geom = args
        return function.grad(array, geom)
    elif op == 'surfgrad':
        array, geom = args
        return function.grad(array, geom, len(geom)-1)
    elif op == 'derivative':
        func, target = args
        return function.derivative(func, target)
    elif op == 'append_axis':
        array, length = args
        return function.insertaxis(array, -1, length)
    elif op == 'transpose':
        array, trans = args
        return numpy.transpose(array, trans)
    elif op == 'jump':
        array, = args
        return function.jump(array)
    elif op == 'mean':
        array, = args
        return function.mean(array)
    elif op == 'neg':
        array, = args
        return -function.Array.cast(array)
    elif op in ('add', 'sub', 'mul', 'truediv', 'pow'):
        left, right = args
        return getattr(operator, '__{}__'.format(op))(function.Array.cast(left), function.Array.cast(right))
    else:
        raise ValueError('unknown opcode: {!r}'.format(op))


def _sum_expr(arg: function.Array, *, consumes: int = 0) -> function.Array:
    if consumes == 0:
        raise ValueError('sum must consume at least one axis but got zero')
    return numpy.sum(arg, range(arg.ndim-consumes, arg.ndim))


def _norm2_expr(arg: function.Array, *, consumes: int = 0) -> function.Array:
    if consumes == 0:
        raise ValueError('sum must consume at least one axis but got zero')
    return numpy.sqrt(numpy.sum(arg * numpy.conjugate(arg), range(arg.ndim-consumes, arg.ndim)))


def _J_expr(geom: function.Array, *, consumes: int = 0) -> function.Array:
    if geom.ndim == 0:
        return function.J(function.insertaxis(geom, 0, 1))
    if consumes > 1:
        raise ValueError('J consumes at most one axis but got {}'.format(consumes))
    if geom.ndim > consumes:
        raise NotImplementedError('currently J cannot be vectorized')
    return function.J(geom)


def _arctan2_expr(_a: function.Array, _b: function.Array) -> function.Array:
    a = function.Array.cast(_a)
    b = function.Array.cast(_b)
    return numpy.arctan2(function._append_axes(a, b.shape), function._prepend_axes(b, a.shape))


class Namespace:
    '''Namespace for :class:`~nutils.function.Array` objects supporting assignments with tensor expressions.

    The :class:`Namespace` object is used to store
    :class:`~nutils.function.Array` objects.

    >>> from nutils import expression_v1, function
    >>> ns = expression_v1.Namespace()
    >>> ns.A = function.zeros([3, 3])
    >>> ns.x = function.zeros([3])
    >>> ns.c = 2

    In addition to the assignment of :class:`~nutils.function.Array` objects, it
    is also possible to specify an array using a tensor expression string — see
    :mod:`nutils.expression_v1` for the syntax.  All attributes defined in this
    namespace are available as variables in the expression.  If the array defined
    by the expression has one or more dimensions the indices of the axes should
    be appended to the attribute name.  Examples:

    >>> ns.cAx_i = 'c A_ij x_j'
    >>> ns.xAx = 'x_i A_ij x_j'

    It is also possible to simply evaluate an expression without storing its
    value in the namespace by passing the expression to the method ``eval_``
    suffixed with appropriate indices:

    >>> ns.eval_('2 c')
    Array<>
    >>> ns.eval_i('c A_ij x_j')
    Array<3>
    >>> ns.eval_ij('A_ij + A_ji')
    Array<3,3>

    For zero and one dimensional expressions the following shorthand can be used:

    >>> '2 c' @ ns
    Array<>
    >>> 'A_ij x_j' @ ns
    Array<3>

    Sometimes the dimension of an expression cannot be determined, e.g. when
    evaluating the identity array:

    >>> ns.eval_ij('δ_ij')
    Traceback (most recent call last):
    ...
    nutils.expression_v1.ExpressionSyntaxError: Length of axis cannot be determined from the expression.
    δ_ij
      ^

    There are two ways to inform the namespace of the correct lengths.  The first is to
    assign fixed lengths to certain indices via keyword argument ``length_<indices>``:

    >>> ns_fixed = expression_v1.Namespace(length_ij=2)
    >>> ns_fixed.eval_ij('δ_ij')
    Array<2,2>

    Note that evaluating an expression with an incompatible length raises an
    exception:

    >>> import numpy
    >>> ns = expression_v1.Namespace(length_i=2)
    >>> ns.a = numpy.array([1,2,3])
    >>> 'a_i' @ ns
    Traceback (most recent call last):
    ...
    nutils.expression_v1.ExpressionSyntaxError: Length of index i is fixed at 2 but the expression has length 3.
    a_i
      ^

    The second is to define a fallback length via the ``fallback_length`` argument:

    >>> ns_fallback = expression_v1.Namespace(fallback_length=2)
    >>> ns_fallback.eval_ij('δ_ij')
    Array<2,2>

    When evaluating an expression through this namespace the following functions
    are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
    ``tanh``, ``arcsin``, ``arccos``, ``arctan2``, ``arctanh``, ``exp``, ``abs``,
    ``ln``, ``log``, ``log2``, ``log10``, ``sqrt`` and ``sign``.

    Additional pointwise functions can be passed to argument ``functions``. All
    functions should take :class:`~nutils.function.Array` objects as arguments
    and must return an :class:`~nutils.function.Array` with as shape the sum of
    all shapes of the arguments.

    >>> def sqr(a):
    ...   return a**2
    >>> def mul(a, b):
    ...   return a[(...,)+(None,)*b.ndim] * b[(None,)*a.ndim]
    >>> ns_funcs = expression_v1.Namespace(functions=dict(sqr=sqr, mul=mul))
    >>> ns_funcs.a = numpy.array([1,2,3])
    >>> ns_funcs.b = numpy.array([4,5])
    >>> 'sqr(a_i)' @ ns_funcs # same as 'a_i^2'
    Array<3>
    >>> ns_funcs.eval_ij('mul(a_i, b_j)') # same as 'a_i b_j'
    Array<3,2>
    >>> 'mul(a_i, a_i)' @ ns_funcs # same as 'a_i a_i'
    Array<>

    Args
    ----
    default_geometry_name : :class:`str`
        The name of the default geometry.  This argument is passed to
        :func:`nutils.expression_v1.parse`.  Default: ``'x'``.
    fallback_length : :class:`int`, optional
        The fallback length of an axis if the length cannot be determined from
        the expression.
    length_<indices> : :class:`int`
        The fixed length of ``<indices>``.  All axes in the expression marked
        with one of the ``<indices>`` are asserted to have the specified length.
    functions : :class:`dict`, optional
        Pointwise functions that should be available in the namespace,
        supplementing the default functions listed above. All functions should
        return arrays with as shape the sum of all shapes of the arguments.

    Attributes
    ----------
    arg_shapes : :class:`dict`
        A readonly map of argument names and shapes.
    default_geometry_name : :class:`str`
        The name of the default geometry.  See argument with the same name.
    '''

    __slots__ = '_attributes', '_arg_shapes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'

    _re_assign = re.compile('^([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-z]+)?$')

    _default_functions = dict(
        opposite=function.opposite, sin=numpy.sin, cos=numpy.cos,
        tan=numpy.tan, sinh=numpy.sinh, cosh=numpy.cosh,
        tanh=numpy.tanh, arcsin=numpy.arcsin, arccos=numpy.arccos,
        arctan=numpy.arctan, arctan2=_arctan2_expr,
        arctanh=numpy.arctanh, exp=numpy.exp, abs=numpy.abs,
        ln=numpy.log, log=numpy.log, log2=numpy.log2, log10=numpy.log10,
        sqrt=numpy.sqrt, sign=numpy.sign, d=function.d,
        surfgrad=function.surfgrad, n=function.normal, sum=_sum_expr,
        norm2=_norm2_expr, J=_J_expr,
    )

    def __init__(self, *, default_geometry_name: str = 'x', fallback_length: Optional[int] = None, functions: Optional[Mapping[str, Callable]] = None, **kwargs: Any) -> None:
        if not isinstance(default_geometry_name, str):
            raise ValueError('default_geometry_name: Expected a str, got {!r}.'.format(default_geometry_name))
        if '_' in default_geometry_name or not self._re_assign.match(default_geometry_name):
            raise ValueError('default_geometry_name: Invalid variable name: {!r}.'.format(default_geometry_name))
        fixed_lengths = {}
        for name, value in kwargs.items():
            if not name.startswith('length_'):
                raise TypeError('__init__() got an unexpected keyword argument {!r}'.format(name))
            for index in name[7:]:
                if index in fixed_lengths:
                    raise ValueError('length of index {} specified more than once'.format(index))
                fixed_lengths[index] = value
        super().__setattr__('_attributes', {})
        super().__setattr__('_arg_shapes', {})
        super().__setattr__('_fixed_lengths', types.frozendict({i: l for indices, l in fixed_lengths.items() for i in indices} if fixed_lengths else {}))
        super().__setattr__('_fallback_length', fallback_length)
        super().__setattr__('default_geometry_name', default_geometry_name)
        super().__setattr__('_functions', dict(itertools.chain(self._default_functions.items(), () if functions is None else functions.items())))
        super().__init__()

    def __getstate__(self) -> Dict[str, Any]:
        'Pickle instructions'
        attrs = '_arg_shapes', '_attributes', 'default_geometry_name', '_fixed_lengths', '_fallback_length', '_functions'
        return {k: getattr(self, k) for k in attrs}

    def __setstate__(self, d: Mapping[str, Any]) -> None:
        'Unpickle instructions'
        for k, v in d.items():
            super().__setattr__(k, v)

    @property
    def arg_shapes(self) -> Mapping[str, function.Shape]:
        return builtin_types.MappingProxyType(self._arg_shapes)

    @property
    def default_geometry(self) -> str:
        ''':class:`nutils.function.Array`: The default geometry, shorthand for ``getattr(ns, ns.default_geometry_name)``.'''
        return getattr(self, self.default_geometry_name)

    def __call__(*args, **subs: function.IntoArray) -> 'Namespace':
        '''Return a copy with arguments replaced by ``subs``.

        Return a copy of this namespace with :class:`~nutils.function.Argument`
        objects replaced according to ``subs``.

        Args
        ----
        **subs : :class:`dict` of :class:`str` and :class:`nutils.function.Array` objects
            Replacements of the :class:`~nutils.function.Argument` objects,
            identified by their names.

        Returns
        -------
        ns : :class:`Namespace`
            The copy of this namespace with replaced :class:`~nutils.function.Argument` objects.
        '''

        if len(args) != 1:
            raise TypeError('{} instance takes 1 positional argument but {} were given'.format(type(args[0]).__name__, len(args)))
        self, = args
        ns = Namespace(default_geometry_name=self.default_geometry_name)
        for k, v in self._attributes.items():
            setattr(ns, k, function.replace_arguments(v, subs))
        return ns

    def copy_(self, *, default_geometry_name: Optional[str] = None) -> 'Namespace':
        '''Return a copy of this namespace.'''

        if default_geometry_name is None:
            default_geometry_name = self.default_geometry_name
        ns = Namespace(default_geometry_name=default_geometry_name, fallback_length=self._fallback_length, functions=self._functions, **{'length_{i}': l for i, l in self._fixed_lengths.items()})
        for k, v in self._attributes.items():
            setattr(ns, k, v)
        return ns

    def __getattr__(self, name: str) -> Any:
        '''Get attribute ``name``.'''

        if name.startswith('eval_'):
            return lambda expr: _eval_ast(parse(expr, variables=self._attributes, indices=name[5:], arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0], self._functions)
        try:
            return self._attributes[name]
        except KeyError:
            pass
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> Any:
        '''Set attribute ``name`` to ``value``.'''

        if name in self.__slots__:
            raise AttributeError('readonly')
        m = self._re_assign.match(name)
        if not m or m.group(2) and len(set(m.group(2))) != len(m.group(2)):
            raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))
        else:
            name, indices = m.groups()
            indices = indices[1:] if indices else None
            if isinstance(value, str):
                ast, arg_shapes = parse(value, variables=self._attributes, indices=indices, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)
                value = _eval_ast(ast, self._functions)
                self._arg_shapes.update(arg_shapes)
            else:
                assert not indices
            self._attributes[name] = function.Array.cast(value)

    def __delattr__(self, name: str) -> None:
        '''Delete attribute ``name``.'''

        if name in self.__slots__:
            raise AttributeError('readonly')
        elif name in self._attributes:
            del self._attributes[name]
        else:
            raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))

    @overload
    def __rmatmul__(self, expr: str) -> function.Array: ...
    @overload
    def __rmatmul__(self, expr: Union[Tuple[str, ...], List[str]]) -> Tuple[function.Array, ...]: ...

    def __rmatmul__(self, expr: Union[str, Tuple[str, ...], List[str]]) -> Union[function.Array, Tuple[function.Array, ...]]:
        '''Evaluate zero or one dimensional ``expr`` or a list of expressions.'''

        if isinstance(expr, (tuple, list)):
            return tuple(map(self.__rmatmul__, expr))
        if not isinstance(expr, str):
            return NotImplemented
        try:
            ast = parse(expr, variables=self._attributes, indices=None, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0]
        except AmbiguousAlignmentError:
            raise ValueError('`expression @ Namespace` cannot be used because the expression has more than one dimension.  Use `Namespace.eval_...(expression)` instead')
        return _eval_ast(ast, self._functions)

# vim:sw=4:sts=4:et
