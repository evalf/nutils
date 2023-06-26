'''Expression parser version 2 and namespace.

The syntax of an expression is as follows:

*   **Integers** or **decimal numbers** are denoted in the usual way.
    Examples: ``1``, ``1.2``, ``.2``.

*   **Variables** are denoted with a string of characters. The first character
    must not be a digit. Unlike Python variables, underscores are not allowed,
    as they have a special meaning.  If the variable is an array with one or
    more axes, all those axes should be labeled with a latin character, the
    index, and appended to the variable with an underscore.  For example an
    array ``a`` with two axes can be denoted with ``a_ij``.  Optionally, a
    single numeral may be used to select an item at the concerning axis.
    Example: in ``a_i0`` the first axis of ``a`` is labeled ``i`` and the first
    element of the second axis is selected.  If the same index occurs twice,
    the trace is taken along the concerning axes.  Example: the trace of the
    first and third axes of ``b`` is denoted by ``b_iji``.  It is invalid to
    specify an index more than twice.

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
    square root of ``a``. Note that the power has precedence over a unary
    minus: ``-2^2`` is interpreted as ``-(2^2)``.

*   An expression surrounded by square brackets or curly braces denotes the
    **jump** or **mean**, respectively, of the enclosed expression.  Example:
    ``[a_i]`` denotes the jump of ``a_i`` and ``{a_i + b_i}`` denotes the
    mean of ``a_i + b_i``.

*   A **function call** is denoted by a name — following the same rules as
    for a variable name — optionally followed by ``_`` and indices for **generated axes**, directly
    followed by the left parenthesis ``(``, without a space.  A function takes
    a single argument with any shape and returns an array with the same shape
    plus an axis per index listed after the underscore. The function is applied
    pointwise to the argument. If an index for a generated axis is also present
    in the argument, the trace is taken along the concerning axes after the
    function call.

.. _`Einstein Summation Convection`: https://en.wikipedia.org/wiki/Einstein_notation
'''

import typing
if typing.TYPE_CHECKING:  # pragma: nocover
    from typing_extensions import Protocol
else:
    class _Protocol(type):
        def __getitem__(cls, item):
            return cls

    class Protocol(metaclass=_Protocol):
        pass

from typing import Callable, FrozenSet, Generic, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union
import functools
import numpy
from . import function

T = TypeVar('T')


class _Substring:

    def __init__(self, base: str, start: Optional[int] = None, stop: Optional[int] = None) -> None:
        self.base = base
        self.start = 0 if start is None else start
        self.stop = len(base) if stop is None else stop
        assert 0 <= self.start <= self.stop <= len(self.base)

    def __len__(self) -> int:
        return self.stop - self.start

    def __str__(self) -> str:
        return self.base[self.start:self.stop]

    def __iter__(self) -> Iterator['_Substring']:
        for i in range(self.start, self.stop):
            yield _Substring(self.base, i, i + 1)

    def __getitem__(self, item: Union[int, slice]) -> '_Substring':
        # Since this is for internal use, we use asserts instead of proper
        # exceptions.
        assert isinstance(item, (int, slice))
        if isinstance(item, int):
            assert 0 <= item < len(self)
            return _Substring(self.base, self.start + item, self.start + item + 1)
        else:
            start, stop, stride = item.indices(len(self))
            assert stride == 1
            return _Substring(self.base, self.start + start, self.start + stop)

    def __contains__(self, item: str) -> bool:
        return self._find(_match(item))[0] >= 0

    def trim(self) -> '_Substring':
        return self.trim_end().trim_start()

    def trim_start(self) -> '_Substring':
        start = self.start
        while start < self.stop and self.base[start] == ' ':
            start += 1
        return _Substring(self.base, start, self.stop)

    def trim_end(self) -> '_Substring':
        stop = self.stop
        while stop > self.start and self.base[stop - 1] == ' ':
            stop -= 1
        return _Substring(self.base, self.start, stop)

    def starts_with(self, prefix: str) -> bool:
        return str(self).startswith(prefix)

    def ends_with(self, suffix: str) -> bool:
        return str(self).endswith(suffix)

    def strip_prefix(self, prefix: str) -> Optional['_Substring']:
        return self[len(prefix):] if self.starts_with(prefix) else None

    def strip_suffix(self, suffix: str) -> Optional['_Substring']:
        return self[:len(self)-len(suffix)] if self.ends_with(suffix) else None

    def _find(self, *matchers: Callable[[str], int]) -> Tuple[int, int, int]:
        # Returns the index of the first successful matcher, the position of the
        # match and the length of the match, or `-1`, the length of the substring
        # and `0` if nothing matches.
        level = 0
        for offset, ch in enumerate(self.base[self.start:self.stop]):
            if ch in (')', ']', '}', '>'):
                level -= 1
            if level == 0:
                tail = self.base[self.start+offset:self.stop]
                for imatcher, matcher in enumerate(matchers):
                    length = matcher(tail)
                    if length:
                        return imatcher, offset, length
            if ch in ('(', '[', '{', '<'):
                level += 1
        return -1, len(self), 0

    def split(self, *matchers: Callable[[str], int]) -> Iterator['_Substring']:
        # Split the substring at every non-overlapping match.
        n = 1
        while n:
            _, i, n = self._find(*matchers)
            yield self[:i]
            self = self[i+n:]

    def isplit(self, *matchers: Callable[[str], int], first: int) -> Iterator[Tuple[int, '_Substring']]:
        # Split the substring at every non-overlapping match and yield both the
        # index of the successful matcher and the (subsequently splitted) substring
        # to the *right* of the match. The item to the left of the first match, or
        # the entire substring if nothing matches, gets `first` as matcher index.
        imatcher = first
        n = 1
        while n:
            imatcher_next, i, n = self._find(*matchers)
            yield imatcher, self[:i]
            self = self[i+n:]
            imatcher = imatcher_next

    def partition(self, *matchers: Callable[[str], int]) -> Tuple['_Substring', '_Substring', '_Substring']:
        _, i, n = self._find(*matchers)
        return self[:i], self[i:i+n], self[i+n:]

    def partition_scope(self) -> Tuple['_Substring', '_Substring', '_Substring', '_Substring', '_Substring']:
        _, i, n = self._find(lambda tail: tail[0] in ('(', '[', '{', '<'))
        _, j, n = self[i:]._find(lambda tail: tail[0] in (')', ']', '}', '>'))
        j += i
        return self[:i], self[i:i+1], self[i+1:j], self[j:j+1], self[j+1:]


class ExpressionSyntaxError(ValueError):

    def __init__(self, message: str, caret: Optional['_Substring'] = None, tilde: Optional['_Substring'] = None) -> None:
        expression, = {s.base for s in (caret, tilde) if s is not None}
        markers = ' '*len(expression)
        for marker, s in ('^', caret), ('~', tilde):
            if s is not None:
                n = max(1, len(s))
                markers = markers[:s.start] + marker * n + markers[s.start+n:]
        markers = markers.rstrip()
        super().__init__('\n'.join((message, expression, markers)))


class _InvalidDimension:

    def __init__(self, __actual_ndim: int) -> None:
        self.actual_ndim = __actual_ndim


_Shape = Tuple[int, ...]


class _ArrayOps(Protocol[T]):

    def from_int(self, __value: int) -> T: ...
    def from_float(self, __value: float) -> T: ...
    def get_variable(self, __name: str, __ndim: int) -> Optional[Union[Tuple[T, _Shape], _InvalidDimension]]: ...
    def call(self, __name: str, __ngenerates: int, arg: T) -> Optional[Union[Tuple[T, _Shape], _InvalidDimension]]: ...
    def get_element(self, __array: T, __axis: int, __index: int) -> T: ...
    def transpose(self, __array: T, __axes: Tuple[int, ...]) -> T: ...
    def trace(self, __array: T, __axis1: int, __axis2: int) -> T: ...
    def scope(self, __array: T) -> T: ...
    def mean(self, __array: T) -> T: ...
    def jump(self, __array: T) -> T: ...
    def add(self, *args: Tuple[bool, T]) -> T: ...
    def multiply(self, *args: T) -> T: ...
    def divide(self, __numerator: T, __denominator: T) -> T: ...
    def power(self, __base: T, __exponent: T) -> T: ...


_ORDINALS = 'zeroth', 'first', 'second', 'third', 'fourth', 'fifth'


def _nth(n: int) -> str:
    return _ORDINALS[n] if 0 <= n < len(_ORDINALS) else '{}th'.format(n)


def _sp(n: int, s: str, p: str) -> str:
    return '{} {}'.format(n, s if n == 1 else p)


def _match(s: str) -> Callable[[str], int]:
    def matcher(tail: str) -> int:
        return len(s) if tail.startswith(s) else 0
    return matcher


def _match_spaces(tail: str) -> int:
    return len(tail) - len(tail.lstrip(' '))


class _Parser(Generic[T]):

    def __init__(self, __array_ops: _ArrayOps[T]) -> None:
        self.array = __array_ops

    def parse_expression(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        s_tail = s
        # Parse optional leading minus. The leading minus applies to the entire
        # term, not to the first number, if any, e.g. `-2^2` is interpreted as
        # `-(2^2)`. See also
        # https://en.wikipedia.org/wiki/Order_of_operations#Unary_minus_sign
        negate = False
        s_try_strip = s_tail.trim_start().strip_prefix('-')
        if s_try_strip:
            s_tail = s_try_strip
            negate = True
        # Parse terms separated by ` + ` or ` - `. If the expression is empty,
        # `split` yields once and `self.parse_fraction` will raise an exception.
        unaligned = tuple((imatcher == 1, s_term, self.parse_fraction(s_term)) for imatcher, s_term in s_tail.isplit(_match(' + '), _match(' - '), first=1 if negate else 0))
        # Check that all terms have the same indices and transpose all but the
        # first array such that all terms have the same order of indices.
        negate, s_first, (term, shape, indices, summed_indices) = unaligned[0]
        if not negate and len(unaligned) == 1:
            # There is only one term without unary minus. Return the term as is.
            return term, shape, indices, summed_indices
        aligned = [(negate, term)]
        for iterm, (negate, s_term, (term, term_shape, term_indices, term_summed_indices)) in enumerate(unaligned[1:], 2):
            if term_indices != indices:
                # The indices of the current term don't match the indices of the first
                # term. Check if there are no missing indices and transpose.
                for index in sorted(set(indices) - set(term_indices)):
                    raise ExpressionSyntaxError('Index {} of the first term [^] is missing in the {} term [~].'.format(index, _nth(iterm)), caret=s_first.trim(), tilde=s_term.trim())
                for index in sorted(set(term_indices) - set(indices)):
                    raise ExpressionSyntaxError('Index {} of the {} term [~] is missing in the first term [^].'.format(index, _nth(iterm)), caret=s_first.trim(), tilde=s_term.trim())
                axes = tuple(map(term_indices.index, indices))
                term = self.array.transpose(term, axes)
                term_shape = tuple(map(term_shape.__getitem__, axes))
            # Verify the shape of the current (transposed) term with the first
            # term.
            for n, m, index in zip(shape, term_shape, indices):
                if n != m:
                    raise ExpressionSyntaxError('Index {} has length {} in the first term [^] but length {} in the {} term [~].'.format(index, n, m, _nth(iterm)), caret=s_first.trim(), tilde=s_term.trim())
            aligned.append((negate, term))
            summed_indices |= term_summed_indices
        result = self.array.add(*aligned)
        return result, shape, indices, summed_indices

    def parse_fraction(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        s_parts = tuple(s.split(_match(' / ')))
        if len(s_parts) > 2:
            raise ExpressionSyntaxError('Repeated fractions are not allowed. Use parentheses if necessary.', s.trim())
        # Parse the numerator.
        numerator, shape, indices, numerator_summed_indices = self.parse_term(s_parts[0])
        if len(s_parts) == 1:
            # There is no denominator. Return the numerator as is.
            return numerator, shape, indices, numerator_summed_indices
        # Parse the denominator.
        denominator, denominator_shape, denominator_indices, denominator_summed_indices = self.parse_term(s_parts[1])
        # Verify and merge indices. The denominator must have dimension zero.
        # Summed indices of the numerator and denominator are treated as if the
        # numerator and denominator are multiplied.
        if denominator_indices:
            raise ExpressionSyntaxError('The denominator must have dimension zero.', s_parts[1].trim())
        summed_indices = self._merge_summed_indices_same_term(s.trim(), numerator_summed_indices, denominator_summed_indices)
        self._verify_indices_summed(s.trim(), indices, summed_indices)
        return self.array.divide(numerator, denominator), shape, indices, summed_indices

    def parse_term(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        s_trimmed = s.trim()
        if not s_trimmed:
            # If the string is empty, let `parse_power` raise an exception. We
            # don't trim the string, because we want to highlight the entire part of
            # the expression that we are currently parsing.
            return self.parse_power(s, allow_number=True)
        # Split the substring at spaces and parse the items using `parse_power`.
        # The first items is allowed to be a number, the remainder is not.
        parts = tuple(self.parse_power(s_part, allow_number=i == 0) for i, s_part in enumerate(s_trimmed.split(_match_spaces)))
        if len(parts) == 1:
            # There is only one item in the term. Return this item as is.
            return parts[0]
        items, shapes, indices, summed_indices = zip(*parts)
        shape = tuple(n for shape in shapes for n in shape)
        # Sum duplicate indices, e.g. index `i` in `a_ij b_ik`.
        return self._trace(s_trimmed, self.array.multiply(*items), shape, ''.join(indices), *summed_indices)

    def parse_power(self, s: _Substring, allow_number: bool) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        s_parts = tuple(s.trim().split(_match('^')))
        if len(s_parts) > 2:
            raise ExpressionSyntaxError('Repeated powers are not allowed. Use parentheses if necessary.', s.trim())
        if len(s_parts) == 2:
            if s_parts[0].ends_with(' '):
                raise ExpressionSyntaxError('Unexpected whitespace before `^`.', s_parts[0][-1:])
            if s_parts[1].starts_with(' '):
                raise ExpressionSyntaxError('Unexpected whitespace after `^`.', s_parts[1][:1])
        # Parse the base.
        base, shape, indices, base_summed_indices = self.parse_item(s_parts[0], allow_number=allow_number)
        if len(s_parts) == 1:
            # There's no exponent. Return the base as is.
            return base, shape, indices, base_summed_indices
        # Parse the exponent. This should either be a scoped expression, or a signed int.
        s_head, s_open, s_scope, s_close, s_tail = s_parts[1].partition_scope()
        if not s_head and not s_tail and str(s_open) == '(' and str(s_close) == ')':
            exponent, exponent_shape, exponent_indices, exponent_summed_indices = self.parse_expression(s_scope)
        elif s_parts[1] and ('0' <= str(s_parts[1][0]) <= '9' or str(s_parts[1][0]) == '-'):
            exponent, exponent_shape, exponent_indices, exponent_summed_indices = self.parse_signed_int(s_parts[1])
        else:
            raise ExpressionSyntaxError('Expected an int or scoped expression.', s_parts[1])
        # Verify and merge indices. The exponent must have dimension zero. Summed
        # indices of the base and exponent are treated as if base and exponent are
        # multiplied.
        if exponent_indices:
            raise ExpressionSyntaxError('The exponent must have dimension zero.', s_parts[1])
        summed_indices = self._merge_summed_indices_same_term(s.trim(), base_summed_indices, exponent_summed_indices)
        self._verify_indices_summed(s.trim(), indices, summed_indices)
        return self.array.power(base, exponent), shape, indices, summed_indices

    def parse_item(self, s: _Substring, allow_number: bool) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        s_trimmed = s.trim()
        if allow_number:
            msg = 'Expected a number, variable, scope, mean, jump or function call.'
        else:
            msg = 'Expected a variable, scope, mean, jump or function call.'
        if any(op in s_trimmed for op in ('+', '-', '/')):
            msg += ' Hint: the operators `+`, `-` and `/` must be surrounded by spaces.'
        error = ExpressionSyntaxError(msg, s_trimmed or s)
        if not s_trimmed:
            raise error
        # If the expression starts with a digit or a dot, we assume this is a
        # number. We try to parse the expression as an int or a float, in that
        # order. Otherwise we raise `error`.
        if '0' <= str(s_trimmed[0]) <= '9' or str(s_trimmed[0]) == '.':
            if not allow_number:
                raise ExpressionSyntaxError('Numbers are only allowed at the start of a term.', s_trimmed)
            for parse in self.parse_unsigned_int, self.parse_unsigned_float:
                try:
                    return parse(s_trimmed)
                except ExpressionSyntaxError:
                    pass
            raise error
        # If the expression contains a scope, partition it and verify that opening
        # and closing parentheses match. If there is no scope, `head` will be the
        # entire expression and `scope` will be empty.
        s_head, s_open, s_scope, s_close, s_tail = s_trimmed.partition_scope()
        parentheses = {'(': ')', '[': ']', '{': '}', '<': '>'}
        if s_open:
            if not s_close:
                raise ExpressionSyntaxError("Unclosed `{}`.".format(s_open), caret=s_open, tilde=s_close)
            if parentheses[str(s_open)] != str(s_close):
                raise ExpressionSyntaxError("Parenthesis `{}` closed by `{}`.".format(s_open, s_close), caret=s_open, tilde=s_close)
        # Under no circumstances we allow anything after a scope.
        if s_tail:
            raise ExpressionSyntaxError('Unexpected symbols after scope.', s_tail)
        # If there are symbols (before an optional scope), assume this is a variable
        # (no scope) or a function (with scope).
        if s_head:
            s_name, s_underscore, s_generated_indices = s_head.partition(_match('_'))
            if not s_open:
                # There is no scope. Parse as a variable.
                indices = ''
                summed_indices = frozenset()
                result = self.array.get_variable(str(s_name), len(s_generated_indices))
                if result is None:
                    raise ExpressionSyntaxError('No such variable: `{}`.'.format(s_name), s_name)
                elif isinstance(result, _InvalidDimension):
                    raise ExpressionSyntaxError('Expected {} for variable `{}` but got {}.'.format(_sp(result.actual_ndim, 'index', 'indices'), s_name, len(s_generated_indices)), s_trimmed)
                array, shape = result
                assert len(shape) == len(s_generated_indices), 'array backend returned an array with incorrect dimension'
            elif str(s_open) == '(':
                # Parse the argument and call the function.
                arg, shape, indices, summed_indices = self.parse_expression(s_scope)
                result = self.array.call(str(s_name), len(s_generated_indices), arg)
                if result is None:
                    raise ExpressionSyntaxError('No such function: `{}`.'.format(s_name), s_name)
                elif isinstance(result, _InvalidDimension):
                    raise ExpressionSyntaxError('Expected {} for axes generated by function `{}` but got {}.'.format(_sp(result.actual_ndim, 'index', 'indices'), s_name, len(s_generated_indices)), s_trimmed)
                array, generated_shape = result
                assert len(generated_shape) == len(s_generated_indices), 'array backend returned an array with incorrect dimension'
                shape = (*shape, *generated_shape)
            else:
                raise error
            # Process generated indices. If an index is numeric, get the element at
            # the index, otherwise add the index to result indices.
            for s_index in s_generated_indices:
                index = str(s_index)
                if '0' <= index <= '9':
                    index = int(index)
                    axis = len(indices)
                    if index >= shape[axis]:
                        raise ExpressionSyntaxError('Index of axis with length {} out of range.'.format(shape[axis]), s_index)
                    array = self.array.get_element(array, axis, index)
                    shape = shape[:axis] + shape[axis+1:]
                elif 'a' <= index <= 'z':
                    indices += str(s_index)
                else:
                    raise ExpressionSyntaxError('Symbol `{}` is not allowed as index.'.format(s_index), s_index)
            # Verify indices and sum indices that occur twice.
            return self._trace(s_trimmed, array, shape, indices, summed_indices)
        elif str(s_open) in ('(', '[', '{'):
            array, shape, indices, summed_indices = self.parse_expression(s_scope)
            array = {'(': self.array.scope, '{': self.array.mean, '[': self.array.jump}[str(s_open)](array)
            return array, shape, indices, summed_indices
        else:
            raise error

    def parse_signed_int(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        try:
            value = int(str(s.trim()))
        except ValueError:
            raise ExpressionSyntaxError('Expected an int.', s.trim() or s) from None
        return self.array.from_int(value), (), '', frozenset(())

    def parse_unsigned_int(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        try:
            value = int(str(s.trim()))
        except ValueError:
            raise ExpressionSyntaxError('Expected an int.', s.trim() or s) from None
        if value < 0:
            raise ExpressionSyntaxError('Expected an int.', s.trim() or s)
        return self.array.from_int(value), (), '', frozenset(())

    def parse_unsigned_float(self, s: _Substring) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        try:
            value = float(str(s.trim()))
        except ValueError:
            raise ExpressionSyntaxError('Expected a float.', s.trim() or s) from None
        if value < 0:
            raise ExpressionSyntaxError('Expected a float.', s.trim() or s)
        return self.array.from_float(value), (), '', frozenset(())

    def _verify_indices_summed(self, s: _Substring, indices: str, summed: FrozenSet[str]) -> None:
        # Check that none of `indices` occur in `summed`. Note that all `indices`
        # are assumed to be unique. If this is not the case, duplicates will
        # silently be ignored.
        for index in indices:
            if index in summed:
                raise ExpressionSyntaxError('Index {} occurs more than twice.'.format(index), s)

    def _merge_summed_indices_same_term(self, s: _Substring, *parts: FrozenSet[str]) -> FrozenSet[str]:
        # Merge `items` into a single set of indices and check that we don't have
        # duplicates.
        merged = set()  # type: Set[str]
        for part in parts:
            for index in sorted(merged & part):
                raise ExpressionSyntaxError('Index {} occurs more than twice.'.format(index), s)
            merged |= part
        return frozenset(merged)

    def _trace(self, s: _Substring, array: T, shape: _Shape, indices: str, *summed_indices_parts: FrozenSet[str]) -> Tuple[T, _Shape, str, FrozenSet[str]]:
        # Sum duplicate indices.
        summed_indices = set(self._merge_summed_indices_same_term(s, *summed_indices_parts))
        j = 0
        while j < len(indices):
            index = indices[j]
            i = indices.index(index)
            if index in summed_indices:
                raise ExpressionSyntaxError('Index {} occurs more than twice.'.format(index), s)
            elif i < j:
                if shape[i] != shape[j]:
                    raise ExpressionSyntaxError('Index {} is assigned to axes with different lengths: {} and {}.'.format(index, shape[i], shape[j]), s)
                array = self.array.trace(array, i, j)
                shape = shape[:i] + shape[i+1:j] + shape[j+1:]
                indices = indices[:i] + indices[i+1:j] + indices[j+1:]
                summed_indices.add(index)
                j -= 1
            else:
                j += 1
        return array, shape, indices, frozenset(summed_indices)


def _grad(geom: function.Array, func: function.Array) -> function.Array:
    return function.grad(func, geom)


def _curl(geom: function.Array, func: function.Array) -> function.Array:
    return numpy.sum(function.levicivita(3) * function.grad(func, geom)[..., numpy.newaxis, :, numpy.newaxis], axis=-2)


class Namespace:
    '''Namespace for :class:`~nutils.function.Array` objects supporting assignments with tensor expressions.

    The :class:`Namespace` object is used to store :class:`~nutils.function.Array` objects.

    >>> from nutils import function
    >>> ns = Namespace()
    >>> ns.A = function.zeros([2, 3])
    >>> ns.x = function.zeros([3])
    >>> ns.c = 2

    In addition to the assignment of :class:`~nutils.function.Array` objects, it is also possible
    to specify an array using a tensor expression string — see
    :mod:`nutils.expression_v2` for the syntax.  All attributes defined in this
    namespace are available as variables in the expression.  If the array defined
    by the expression has one or more dimensions the indices of the axes should
    be appended to the attribute name.  Example:

    >>> ns.cAx_i = 'c A_ij x_j'

    It is also possible to simply evaluate an expression without storing its
    value in the namespace using ``expression @ ns``:

    >>> '2 c' @ ns
    Array<>
    >>> 'c A_ij x_j' @ ns
    Array<2>
    >>> 'A_ij' @ ns # indices are ordered alphabetically
    Array<2,3>

    Note that evaluating an expression with an incompatible length raises an
    exception:

    >>> 'A_ij + A_ji' @ ns
    Traceback (most recent call last):
    ...
    nutils.expression_v2.ExpressionSyntaxError: Index i has length 2 in the first term [^] but length 3 in the second term [~].
    A_ij + A_ji
    ^^^^   ~~~~

    When evaluating an expression through this namespace the following functions
    are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
    ``tanh``, ``arcsin``, ``arccos``, ``arctanh``, ``exp``, ``abs``, ``ln``,
    ``log``, ``log2``, ``log10``, ``sqrt``, ``sign``, ``conj``, ``real`` and
    ``imag``.

    Additional pointwise functions can be assigned to the namespace similar to variables:

    >>> ns.sqr = lambda u: u**2
    >>> 'sqr(x_i)' @ ns # same as 'x_i^2'
    Array<3>
    '''

    def __init__(self) -> None:
        self.opposite = function.opposite
        self.sin = numpy.sin
        self.cos = numpy.cos
        self.tan = numpy.tan
        self.sinh = numpy.sinh
        self.cosh = numpy.cosh
        self.tanh = numpy.tanh
        self.arcsin = numpy.arcsin
        self.arccos = numpy.arccos
        self.arctan = numpy.arctan
        self.arctanh = numpy.arctanh
        self.exp = numpy.exp
        self.abs = numpy.abs
        self.ln = numpy.log
        self.log = numpy.log
        self.log2 = numpy.log2
        self.log10 = numpy.log10
        self.sqrt = numpy.sqrt
        self.sign = numpy.sign
        self.conj = numpy.conj
        self.real = numpy.real
        self.imag = numpy.imag

    def __setattr__(self, attr: str, value: Union[function.Array, str]) -> None:
        name, underscore, indices = attr.partition('_')
        if isinstance(value, (int, float, complex, numpy.ndarray)):
            value = function.Array.cast(value)
        if hasattr(value, '__array_ufunc__') and hasattr(value, '__array_function__'):
            if underscore:
                raise AttributeError('Cannot assign an array to an attribute with an underscore.')
            super().__setattr__(name, value)
        elif isinstance(value, str):
            if not all('a' <= index <= 'z' for index in indices):
                raise AttributeError('Only lower case latin characters are allowed as indices.')
            if len(set(indices)) != len(indices):
                raise AttributeError('All indices must be unique.')
            ops = _FunctionArrayOps(self)
            array, shape, expression_indices, summed = _Parser(ops).parse_expression(_Substring(value))
            assert numpy.shape(array) == shape
            if expression_indices != indices:
                for index in sorted(set(indices) - set(expression_indices)):
                    raise AttributeError('Index {} of the namespace attribute is missing in the expression.'.format(index))
                for index in sorted(set(expression_indices) - set(indices)):
                    raise AttributeError('Index {} of the expression is missing in the namespace attribute.'.format(index))
                array = ops.align(array, expression_indices, indices)
            super().__setattr__(name, array)
        elif callable(value):
            if underscore:
                raise AttributeError('Cannot assign a function to an attribute with an underscore.')
            super().__setattr__(name, value)
        else:
            raise AttributeError('Cannot assign an object of type {} to the namespace.'.format(type(value)))

    def __rmatmul__(self, expression):
        ops = _FunctionArrayOps(self)
        parser = _Parser(ops)
        if isinstance(expression, str):
            array, shape, indices, summed = parser.parse_expression(_Substring(expression))
            assert numpy.shape(array) == shape
            array = ops.align(array, indices, ''.join(sorted(indices)))
            return array
        elif isinstance(expression, tuple):
            return tuple(item @ self for item in expression)
        elif isinstance(expression, list):
            return list(item @ self for item in expression)
        else:
            return NotImplemented

    def define_for(self, __name: str, *, gradient: Optional[str] = None, curl: Optional[str] = None, normal: Optional[str] = None, jacobians: Sequence[str] = ()) -> None:
        '''Define gradient, normal or jacobian for the given geometry.

        Parameters
        ----------
        name : :class:`str`
            Define the gradient, normal or jacobian for the geometry with the given
            name in this namespace.
        gradient : :class:`str`, optional
            Define the gradient function with the given name. The function
            generates axes with the same shape as the given geometry.
        curl : :class:`str`, optional
            Define the curl function with the given name. The function generates
            two axes of length 3 where the last axis should be traced with an axis
            of the argument, e.g. `curl_ij(u_j)`.
        normal : :class:`str`, optional
            Define the normal with the given name. The normal has the same shape as
            the geometry.
        jacobians : sequence of :class:`str`, optional
            Define the jacobians for decreasing dimensions, starting at the
            dimensions of the geometry. The jacobians are always scalars.

        Example
        -------

        >>> from nutils import function, mesh
        >>> ns = Namespace()
        >>> topo, ns.x = mesh.rectilinear([2, 2])
        >>> ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
        >>> ns.basis = topo.basis('spline', degree=1)
        >>> ns.u = function.dotarg('u', ns.basis)
        >>> ns.v = function.dotarg('v', ns.basis)
        >>> res = topo.integral('-∇_i(v) ∇_i(u) dV' @ ns, degree=2)
        >>> res += topo.boundary.integral('∇_i(v) u n_i dS' @ ns, degree=2)
        '''

        geom = getattr(self, __name)
        if gradient:
            setattr(self, gradient, functools.partial(_grad, geom))
        if curl:
            if numpy.shape(geom) != (3,):
                raise ValueError('The curl can only be defined for a geometry with shape (3,) but got {}.'.format(numpy.shape(geom)))
            # Definition: `curl_ki(u_...)` := `ε_kji ∇_j(u_...)`. Should be used as
            # `curl_ki(u_i)`, which is equivalent to `ε_kji ∇_j(u_i)`.
            setattr(self, curl, functools.partial(_curl, geom))
        if normal:
            setattr(self, normal, function.normal(geom))
        for i, jacobian in enumerate(jacobians):
            if i > numpy.size(geom):
                raise ValueError('Cannot define the jacobian {!r}: dimension is negative.'.format(jacobian))
            setattr(self, jacobian, function.jacobian(geom, numpy.size(geom) - i))

    def add_field(self, __names: Union[str, Sequence[str]], *__bases, shape: Tuple[int, ...] = (), dtype: function.DType = float):
        '''Add field(s) of the form ns.u = function.dotarg('u', ...)

        Parameters
        ----------
        names : :class:`str` or iterable thereof
            Name of both the generated field and the function argument.
        bases : :class:`nutils.function.Array` or something that can be :meth:`nutils.function.Array.cast` into one
            The arrays to take inner products with.
        shape : :class:`tuple` of :class:`int`, optional
            The shape to be appended to the argument.
        dtype : :class:`bool`, :class:`int`, :class:`float` or :class:`complex`
            The dtype of the argument.
        '''

        for name in (__names,) if isinstance(__names, str) else __names:
            setattr(self, name, function.dotarg(name, *__bases, shape=shape, dtype=dtype))

    def copy_(self, **replacements: Mapping[str, function.Array]) -> 'Namespace':
        '''Return a copy of this namespace.

        Parameters
        ----------
        **replacements : :class:`nutils.function.Array`
            Argument replacements to apply to the copy of this namespace.

        Returns
        -------
        :class:`Namespace`
            A copy of this namespace.
        '''

        ns = Namespace()
        for attr, value in vars(self).items():
            if replacements and hasattr(value, '__array_ufunc__') and hasattr(value, '__array_function__'):
                value = function.replace_arguments(value, replacements)
            object.__setattr__(ns, attr, value)
        return ns


class _FunctionArrayOps:

    def __init__(self, namespace: Namespace) -> None:
        self.namespace = namespace

    def align(self, array: function.Array, in_indices: str, out_indices: str) -> function.Array:
        assert set(in_indices) == set(out_indices) and len(in_indices) == len(out_indices) == len(set(in_indices))
        return self.transpose(array, tuple(map(in_indices.index, out_indices)))

    def from_int(self, value: int) -> function.Array:
        return function.Array.cast(value)

    def from_float(self, value: float) -> function.Array:
        return function.Array.cast(value)

    def get_variable(self, name: str, ndim: int) -> Optional[Union[Tuple[function.Array, _Shape], _InvalidDimension]]:
        try:
            array = getattr(self.namespace, name)
        except AttributeError:
            return None
        if callable(array):
            return None
        elif numpy.ndim(array) == ndim:
            return array, numpy.shape(array)
        else:
            return _InvalidDimension(numpy.ndim(array))

    def call(self, name: str, ngenerates: int, arg: function.Array) -> Optional[Union[Tuple[function.Array, _Shape], _InvalidDimension]]:
        try:
            func = getattr(self.namespace, name)
        except AttributeError:
            return None
        array = func(arg)
        assert numpy.shape(array)[:numpy.ndim(arg)] == numpy.shape(arg)
        if numpy.ndim(array) == numpy.ndim(arg) + ngenerates:
            return array, numpy.shape(array)[numpy.ndim(arg):]
        else:
            return _InvalidDimension(numpy.ndim(array) - numpy.ndim(arg))

    def get_element(self, array: function.Array, axis: int, index: int) -> function.Array:
        assert 0 <= axis < numpy.ndim(array) and 0 <= index < numpy.shape(array)[axis]
        return numpy.take(array, index, axis)

    def transpose(self, array: function.Array, axes: Tuple[int, ...]) -> function.Array:
        assert numpy.ndim(array) == len(axes)
        return numpy.transpose(array, axes)

    def trace(self, array: function.Array, axis1: int, axis2: int) -> function.Array:
        return numpy.trace(array, axis1=axis1, axis2=axis2)

    def scope(self, array: function.Array) -> function.Array:
        return array

    def mean(self, array: function.Array) -> function.Array:
        return function.mean(array)

    def jump(self, array: function.Array) -> function.Array:
        return function.jump(array)

    def add(self, *args: Tuple[bool, function.Array]) -> function.Array:
        assert all(numpy.shape(arg) == numpy.shape(args[0][1]) for neg, arg in args[1:])
        negated = (-arg if neg else arg for neg, arg in args)
        return functools.reduce(numpy.add, negated)

    def append_axes(self, array, shape):
        shuffle = numpy.concatenate([len(shape) + numpy.arange(numpy.ndim(array)), numpy.arange(len(shape))])
        return numpy.transpose(numpy.broadcast_to(array, shape + numpy.shape(array)), shuffle)

    def multiply(self, *args: function.Array) -> function.Array:
        result = args[0]
        for arg in args[1:]:
            result = numpy.multiply(self.append_axes(result, numpy.shape(arg)), arg)
        return result

    def divide(self, numerator: function.Array, denominator: function.Array) -> function.Array:
        assert numpy.ndim(denominator) == 0
        return numpy.true_divide(numerator, denominator)

    def power(self, base: function.Array, exponent: function.Array) -> function.Array:
        assert numpy.ndim(exponent) == 0
        return numpy.power(base, exponent)
