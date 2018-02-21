# Copyright (c) 2014 Evalf
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

"""
The util module provides a collection of general purpose methods. Most
importantly it provides the :func:`run` method which is the preferred entry
point of a nutils application, taking care of command line parsing, output dir
creation and initiation of a log file.
"""

from . import numeric, config
import sys, os, numpy, collections.abc, inspect, functools, operator, numbers, pathlib

supports_outdirfd = os.open in os.supports_dir_fd and os.listdir in os.supports_fd

sum = functools.partial(functools.reduce, operator.add)
product = functools.partial(functools.reduce, operator.mul)

def cumsum(seq):
  offset = 0
  for i in seq:
    yield offset
    offset += i

def gather(items):
  gathered = []
  d = {}
  for key, value in items:
    try:
      values = d[key]
    except KeyError:
      d[key] = values = []
      gathered.append((key, values))
    values.append(value)
  return gathered

def allequal(seq1, seq2):
  seq1 = iter(seq1)
  seq2 = iter(seq2)
  for item1, item2 in zip(seq1, seq2):
    if item1 != item2:
      return False
  if list(seq1) or list(seq2):
    return False
  return True

class NanVec(numpy.ndarray):
  'nan-initialized vector'

  def __new__(cls, length):
    vec = numpy.empty(length, dtype=float).view(cls)
    vec[:] = numpy.nan
    return vec

  @property
  def where(self):
    return ~numpy.isnan(self.view(numpy.ndarray))

  def __iand__(self, other):
    if self.dtype != float:
      return self.view(numpy.ndarray).__iand__(other)
    where = self.where
    if numpy.isscalar(other):
      self[where] = other
    else:
      assert numeric.isarray(other) and other.shape == self.shape
      self[where] = other[where]
    return self

  def __and__(self, other):
    if self.dtype != float:
      return self.view(numpy.ndarray).__and__(other)
    return self.copy().__iand__(other)

  def __ior__(self, other):
    if self.dtype != float:
      return self.view(numpy.ndarray).__ior__(other)
    wherenot = ~self.where
    self[wherenot] = other if numpy.isscalar(other) else other[wherenot]
    return self

  def __or__(self, other):
    if self.dtype != float:
      return self.view(numpy.ndarray).__or__(other)
    return self.copy().__ior__(other)

  def __invert__(self):
    if self.dtype != float:
      return self.view(numpy.ndarray).__invert__()
    nanvec = NanVec(len(self))
    nanvec[numpy.isnan(self)] = 0
    return nanvec

def regularize(bbox, spacing, xy=numpy.empty((0,2))):
  xy = numpy.asarray(xy)
  index0 = numeric.floor(bbox[:,0] / (2*spacing)) * 2 - 1
  shape = numeric.ceil(bbox[:,1] / (2*spacing)) * 2 + 2 - index0
  index = numeric.round(xy / spacing) - index0
  keep = numpy.logical_and(numpy.greater_equal(index, 0), numpy.less(index, shape)).all(axis=1)
  mask = numpy.zeros(shape, dtype=bool)
  for i, ind in enumerate(index):
    if keep[i]:
      if not mask[tuple(ind)]:
        mask[tuple(ind)] = True
      else:
        keep[i] = False
  coursex = mask[0:-2:2] | mask[1:-1:2] | mask[2::2]
  coarsexy = coursex[:,0:-2:2] | coursex[:,1:-1:2] | coursex[:,2::2]
  vacant, = (~coarsexy).ravel().nonzero()
  newindex = numpy.array(numpy.unravel_index(vacant, coarsexy.shape)).T * 2 + index0 + 1
  return numpy.concatenate([newindex * spacing, xy[keep]], axis=0)

class hashlessdict(collections.abc.MutableMapping):
  __slots__ = '__keys', '__values'

  def __new__(cls, *args, **kwargs):
    self = object.__new__(cls)
    self.__keys = []
    self.__values = []
    return self

  def __init__(self, init=()):
    for key, value in init.items() if isinstance(init, collections.abc.Mapping) else init:
      self.__keys.append(key)
      self.__values.append(value)

  def __getitem__(self, key):
    try:
      index = self.__keys.index(key)
    except ValueError as e:
      raise KeyError(key) from e
    else:
      return self.__values[index]

  def __setitem__(self, key, value):
    try:
      index = self.__keys.index(key)
    except ValueError:
      self.__keys.append(key)
      self.__values.append(value)
    else:
      self.__values[index] = value

  def __delitem__(self, key):
    try:
      index = self.__keys.index(key)
    except ValueError as e:
      raise KeyError(key) from e
    else:
      del self.__keys[index]
      del self.__values[index]

  def __iter__(self):
    return iter(self.__keys)

  def __len__(self):
    return len(self.__keys)

  def __bool__(self):
    return len(self.__keys) > 0

  def __contains__(self, key):
    return key in self.__keys

  def __eq__(self, other):
    return isinstance(other, hashlessdict) and self.__keys == other.__keys and self.__values == other.__values

  def get(self, key, value=None):
    try:
      index = self.__keys.index(key)
    except ValueError:
      return value
    else:
      return self.__values[index]

  def keys(self):
    return tuple(self.__keys)

  def values(self):
    return tuple(self.__values)

  def items(self):
    return zip(self.__keys, self.__values)

  def copy(self):
    return hashlessdict(self)

class frozenmultiset(collections.abc.Container):
  __slots__ = '__items', '__key'

  def __new__(cls, items):
    if isinstance(items, frozenmultiset):
      return items
    self = object.__new__(cls)
    self.__items = tuple(items)
    self.__key = frozenset((item, self.__items.count(item)) for item in self.__items)
    return self

  def __and__(self, other):
    items = list(self.__items)
    isect = []
    for item in other:
      try:
        items.remove(item)
      except ValueError:
        pass
      else:
        isect.append(item)
    return frozenmultiset(isect)

  def __sub__(self, other):
    items = list(self.__items)
    for item in other:
      items.remove(item)
    return frozenmultiset(items)

  __reduce__ = lambda self: (frozenmultiset, (self.__items,))
  __hash__ = lambda self: hash(self.__key)
  __eq__ = lambda self, other: isinstance(other, frozenmultiset) and self.__key == other.__key
  __contains__ = lambda self, item: item in self.__items
  __iter__ = lambda self: iter(self.__items)
  __len__ = lambda self: len(self.__items)
  __bool__ = lambda self: bool(self.__items)
  __add__ = lambda self, other: frozenmultiset(self.__items + tuple(other))

  isdisjoint = lambda self, other: not any(item in self.__items for item in other)

def obj2str(obj):
  '''compact, lossy string representation of arbitrary object'''
  return '['+','.join(obj2str(item) for item in obj)+']' if isinstance(obj, collections.abc.Iterable) \
    else str(obj).strip('0').rstrip('.') or '0' if isinstance(obj, numbers.Real) \
    else str(obj)

def single_or_multiple(f):
  """
  Method wrapper, converts first positional argument to tuple: tuples/lists
  are passed on as tuples, other objects are turned into tuple singleton.
  Return values should match the length of the argument list, and are unpacked
  if the original argument was not a tuple/list.

  >>> class Test:
  ...   @single_or_multiple
  ...   def square(self, args):
  ...     return [v**2 for v in args]
  ...
  >>> T = Test()
  >>> T.square(2)
  4
  >>> T.square([2,3])
  (4, 9)

  Args
  ----
  f: method
      Method that expects a tuple as first positional argument, and that
      returns a list/tuple of the same length.

  Returns
  -------
  Wrapped method.
  """

  @functools.wraps(f)
  def wrapped(self, arg0, *args, **kwargs):
    ismultiple = isinstance(arg0, (list,tuple))
    arg0mod = tuple(arg0) if ismultiple else (arg0,)
    retvals = tuple(f(self, arg0mod, *args, **kwargs))
    if not ismultiple:
      retvals, = retvals
    return retvals
  return wrapped

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
