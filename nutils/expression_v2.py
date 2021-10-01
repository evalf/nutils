# Copyright (c) 2021 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
if typing.TYPE_CHECKING: # pragma: nocover
  from typing_extensions import Protocol
else:
  class _Protocol(type):
    def __getitem__(cls, item):
      return cls
  class Protocol(metaclass=_Protocol): pass

from typing import Callable, FrozenSet, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar, Union

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
    parts = tuple(self.parse_power(s_part, allow_number=i==0) for i, s_part in enumerate(s_trimmed.split(_match_spaces)))
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
    merged = set() # type: Set[str]
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
