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

from . import debug_flags, util, types, numeric, cache, transform, expression, warnings, parallel, sparse
from ._graph import Node, RegularNode, DuplicatedLeafNode, InvisibleNode, Subgraph
import numpy, sys, itertools, functools, operator, inspect, numbers, builtins, re, types as builtin_types, abc, collections.abc, math, treelog as log, weakref, time, contextlib, subprocess
_ = numpy.newaxis

isevaluable = lambda arg: isinstance(arg, Evaluable)

def strictevaluable(value):
  if not isinstance(value, Evaluable):
    raise ValueError('expected an object of type {!r} but got {!r} with type {!r}'.format(Evaluable.__qualname__, value, type(value).__qualname__))
  return value

def simplified(value):
  return strictevaluable(value).simplified

asdtype = lambda arg: arg if any(arg is dtype for dtype in (bool, int, float, complex)) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]

def asarray(arg):
  if hasattr(arg, 'as_evaluable_array'):
    return arg.as_evaluable_array()
  elif _containsarray(arg):
    return stack(arg, axis=0)
  else:
    return Constant(arg)

asarrays = types.tuple[asarray]

def asindex(arg):
  arg = asarray(arg)
  if arg.ndim or arg.dtype not in (int, bool): # NOTE: bool to be removed after introduction of Cast
    raise ValueError('argument is not an index: {}'.format(arg))
  return arg

@types.apply_annotations
def equalindex(n:asindex, m:asindex):
  '''Compare two array indices.

  Returns `True` if the two indices are certainly equal, `False` if they are
  certainly not equal, or `None` if equality cannot be determined at compile
  time.
  '''

  if n is m:
    return True
  n = n.simplified
  m = m.simplified
  if n is m:
    return True
  if n.arguments != m.arguments:
    return False
  if n.isconstant: # implies m.isconstant
    return int(n) == int(m)

asshape = types.tuple[asindex]

@types.apply_annotations
def equalshape(N:asshape, M:asshape):
  '''Compare two array shapes.

  Returns `True` if all indices are certainly equal, `False` if any indices are
  certainly not equal, or `None` if equality cannot be determined at compile
  time.
  '''

  assert len(N) == len(M)
  retval = True
  for eq in map(equalindex, N, M):
    if eq == False:
      return False
    if eq == None:
      retval = None
  return retval

class ExpensiveEvaluationWarning(warnings.NutilsInefficiencyWarning): pass

def replace(func=None, depthfirst=False, recursive=False, lru=4):
  '''decorator for deep object replacement

  Generates a deep replacement method for general objects based on a callable
  that is applied (recursively) on individual constructor arguments.

  Args
  ----
  func
      Callable which maps an object onto a new object, or `None` if no
      replacement is made. It must have one positional argument for the object,
      and may have any number of additional positional and/or keyword
      arguments.
  depthfirst : :class:`bool`
      If `True`, decompose each object as far a possible, then apply `func` to
      all arguments as the objects are reconstructed. Otherwise apply `func`
      directly on each new object that is encountered in the decomposition,
      proceding only if the return value is `None`.
  recursive : :class:`bool`
      If `True`, repeat replacement for any object returned by `func` until it
      returns `None`. Otherwise perform a single, non-recursive sweep.
  lru : :class:`int`
      Maximum size of the least-recently-used cache. A persistent weak-key
      dictionary is maintained for every unique set of function arguments. When
      the size of `lru` is reached, the least recently used cache is dropped.

  Returns
  -------
  :any:`callable`
      The method that searches the object to perform the replacements.
  '''

  if func is None:
    return functools.partial(replace, depthfirst=depthfirst, recursive=recursive, lru=lru)

  signature = inspect.signature(func)
  arguments = [] # list of past function arguments, least recently used last
  caches = [] # list of weak-key dictionaries matching arguments (above)

  remember = object() # token to signal that rstack[-1] can be cached as the replacement of fstack[-1]
  recreate = object() # token to signal that all arguments for object recreation are ready on rstack
  pending = object() # token to hold the place of a cachable object pending creation
  identity = object() # token to hold the place of the cache value in case it matches key, to avoid circular references

  @functools.wraps(func)
  def wrapped(target, *funcargs, **funckwargs):

    # retrieve or create a weak-key dictionary
    bound = signature.bind(None, *funcargs, **funckwargs)
    bound.apply_defaults()
    try:
      index = arguments.index(bound.arguments) # by using index, arguments need not be hashable
    except ValueError:
      index = -1
      cache = weakref.WeakKeyDictionary()
    else:
      cache = caches[index]
    if index != 0: # function arguments are not the most recent (possibly new)
      if index > 0 or len(arguments) >= lru:
        caches.pop(index) # pop matching (or oldest) item
        arguments.pop(index)
      caches.insert(0, cache) # insert popped (or new) item to front
      arguments.insert(0, bound.arguments)

    fstack = [target] # stack of unprocessed objects and command tokens
    rstack = [] # stack of processed objects
    _stack = fstack if recursive else rstack

    try:
      while fstack:
        obj = fstack.pop()

        if obj is recreate:
          args = [rstack.pop() for obj in range(fstack.pop())]
          f = fstack.pop()
          r = f(*args)
          if depthfirst:
            newr = func(r, *funcargs, **funckwargs)
            if newr is not None:
              _stack.append(newr)
              continue
          rstack.append(r)
          continue

        if obj is remember:
          obj = fstack.pop()
          cache[obj] = rstack[-1] if rstack[-1] is not obj else identity
          continue

        if isinstance(obj, (tuple, list, dict, set, frozenset)):
          if not obj:
            rstack.append(obj) # shortcut to avoid recreation of empty container
          else:
            fstack.append(lambda *x, T=type(obj): T(x))
            fstack.append(len(obj))
            fstack.append(recreate)
            fstack.extend(obj if not isinstance(obj, dict) else obj.items())
          continue

        try:
          r = cache[obj]
        except KeyError: # object can be weakly cached, but isn't
          cache[obj] = pending
          fstack.append(obj)
          fstack.append(remember)
        except TypeError: # object cannot be referenced or is not hashable
          pass
        else: # object is in cache
          if r is pending:
            pending_objs = [k for k, v in cache.items() if v is pending]
            index = pending_objs.index(obj)
            raise Exception('{}@replace caught in a circular dependence\n'.format(func.__name__) + Tuple(pending_objs[index:]).asciitree().split('\n', 1)[1])
          rstack.append(r if r is not identity else obj)
          continue

        if not depthfirst:
          newr = func(obj, *funcargs, **funckwargs)
          if newr is not None:
            _stack.append(newr)
            continue

        try:
          f, args = obj.__reduce__()
        except: # obj cannot be reduced into a constructor and its arguments
          rstack.append(obj)
        else:
          fstack.append(f)
          fstack.append(len(args))
          fstack.append(recreate)
          fstack.extend(args)

      assert len(rstack) == 1

    finally:
      while fstack:
        if fstack.pop() is remember:
          assert cache.pop(fstack.pop()) is pending

    return rstack[0]

  return wrapped

class Evaluable(types.Singleton):
  'Base class'

  __slots__ = '__args'
  __cache__ = 'dependencies', 'arguments', 'ordereddeps', 'dependencytree', 'optimized_for_numpy', '_loop_concatenate_deps'

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable]):
    super().__init__()
    self.__args = args

  def evalf(self, *args):
    raise NotImplementedError('Evaluable derivatives should implement the evalf method')

  def evalf_withtimes(self, times, *args):
    with times[self]:
      return self.evalf(*args)

  @property
  def dependencies(self):
    '''collection of all function arguments'''
    deps = {}
    for func in self.__args:
      funcdeps = func.dependencies
      deps.update(funcdeps)
      deps[func] = len(funcdeps)
    return types.frozendict(deps)

  @property
  def arguments(self):
    'a frozenset of all arguments of this evaluable'
    return frozenset().union(*(child.arguments for child in self.__args))

  @property
  def isconstant(self):
    return EVALARGS not in self.dependencies

  @property
  def ordereddeps(self):
    '''collection of all function arguments such that the arguments to
    dependencies[i] can be found in dependencies[:i]'''
    deps = self.dependencies.copy()
    deps.pop(EVALARGS, None)
    return tuple([EVALARGS] + sorted(deps, key=deps.__getitem__))

  @property
  def dependencytree(self):
    '''lookup table of function arguments into ordereddeps, such that
    ordereddeps[i].__args[j] == ordereddeps[dependencytree[i][j]], and
    self.__args[j] == ordereddeps[dependencytree[-1][j]]'''
    args = self.ordereddeps
    return tuple(tuple(map(args.index, func.__args)) for func in args+(self,))

  @property
  def serialized(self):
    return zip(self.ordereddeps[1:]+(self,), self.dependencytree[1:])

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
      values.extend(op.evalf(*[values[i] for i in indices]) for op, indices in self.serialized)
    except KeyboardInterrupt:
      raise
    except Exception as e:
      raise EvaluationError(self, values) from e
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
      raise EvaluationError(self, values) from e
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

  def _stack(self, values):
    lines = ['  %0 = EVALARGS']
    for (op, indices), v in zip(self.serialized, values):
      lines[-1] += ' --> ' + type(v).__name__
      if numeric.isarray(v):
        lines[-1] += '({})'.format(','.join(map(str, v.shape)))
      try:
        code = op.evalf.__code__
        offset = 1 if getattr(op.evalf, '__self__', None) is not None else 0
        names = code.co_varnames[offset:code.co_argcount]
        names += tuple('{}[{}]'.format(code.co_varnames[code.co_argcount], n) for n in range(len(indices) - len(names)))
        args = map(' {}=%{}'.format, names, indices)
      except:
        args = map(' %{}'.format, indices)
      lines.append('  %{} = {}:{}'.format(len(lines), op, ','.join(args)))
    return lines

  @property
  @replace(depthfirst=True, recursive=True)
  def simplified(obj):
    if isinstance(obj, Evaluable):
      retval = obj._simplified()
      if retval is not None and isinstance(obj, Array):
        assert isinstance(retval, Array) and equalshape(retval.shape, obj.shape), '{}._simplified resulted in shape change'.format(type(obj).__name__)
      return retval

  def _simplified(self):
    return

  @property
  def optimized_for_numpy(self):
    retval = self._optimized_for_numpy1() or self
    return retval._combine_loop_concatenates(frozenset())

  @types.apply_annotations
  @replace(depthfirst=True, recursive=True)
  def _optimized_for_numpy1(obj: simplified.fget):
    if isinstance(obj, Evaluable):
      retval = obj._simplified() or obj._optimized_for_numpy()
      if retval is not None and isinstance(obj, Array):
        assert isinstance(retval, Array) and equalshape(retval.shape, obj.shape), '{0}._optimized_for_numpy or {0}._simplified resulted in shape change'.format(type(obj).__name__)
      return retval

  def _optimized_for_numpy(self):
    return

  @property
  def _loop_concatenate_deps(self):
    deps = []
    for arg in self.__args:
      deps += [dep for dep in arg._loop_concatenate_deps if dep not in deps]
    return tuple(deps)

  def _combine_loop_concatenates(self, outer_exclude):
    while True:
      exclude = set(outer_exclude)
      combine = {}
      # Collect all top-level `LoopConcatenate` instances in `combine` and all
      # their dependent `LoopConcatenate` instances in `exclude`.
      for lc in self._loop_concatenate_deps:
        lcs = combine.setdefault((lc.index, lc.length), [])
        if lc not in lcs:
          lcs.append(lc)
          exclude.update(set(lc._loop_concatenate_deps) - {lc})
      # Combine top-level `LoopConcatenate` instances excluding those in
      # `exclude`.
      replacements = {}
      for (index, length), lcs in combine.items():
        lcs = [lc for lc in lcs if lc not in exclude]
        if not lcs:
          continue
        # We're extracting data from `LoopConcatenate` in favor of using
        # `loop_concatenate_combined(lcs, ...)` because the later requires
        # reapplying simplifications that are already applied in the former.
        # For example, in `loop_concatenate_combined` the offsets (used by
        # start, stop and the concatenation length) are formed by
        # `loop_concatenate`-ing `func.shape[-1]`. If the shape is constant,
        # this can be simplified to a `Range`.
        data = Tuple((Tuple(lc.funcdata) for lc in lcs))
        # Combine `LoopConcatenate` instances in `data` excluding
        # `outer_exclude` and those that will be processed in a subsequent loop
        # (the remainder of `exclude`). The latter consists of loops that are
        # invariant w.r.t. the current loop `index`.
        data = data._combine_loop_concatenates(exclude)
        combined = LoopConcatenateCombined(data, index, length)
        for i, lc in enumerate(lcs):
          replacements[lc] = ArrayFromTuple(combined, i, lc.shape, lc.dtype)
      if replacements:
        self = replace(lambda key: replacements.get(key) if isinstance(key, LoopConcatenate) else None, recursive=False, depthfirst=False)(self)
      else:
        return self

class EvaluationError(Exception):
  def __init__(self, f, values):
    super().__init__('evaluation failed in step {}/{}\n'.format(len(values), len(f.dependencies)) + '\n'.join(f._stack(values)))

class EVALARGS(Evaluable):
  def __init__(self):
    super().__init__(args=())
  def _node(self, cache, subgraph, times):
    return InvisibleNode((type(self).__name__, _Stats()))

EVALARGS = EVALARGS()

class Tuple(Evaluable):

  __slots__ = 'items', 'indices'

  @types.apply_annotations
  def __init__(self, items:tuple): # FIXME: shouldn't all items be Evaluable?
    self.items = items
    args = []
    indices = []
    for i, item in enumerate(self.items):
      if isevaluable(item):
        args.append(item)
        indices.append(i)
    self.indices = tuple(indices)
    super().__init__(args)

  def evalf(self, *items):
    'evaluate'

    T = list(self.items)
    for index, item in zip(self.indices, items):
      T[index] = item
    return tuple(T)

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

class SparseArray(Evaluable):
  'sparse array'

  @types.apply_annotations
  def __init__(self, chunks:types.tuple[asarrays], shape:asarrays, dtype:asdtype):
    self._shape = shape
    self._dtype = dtype
    super().__init__(args=[Tuple(shape), *map(Tuple, chunks)])

  def evalf(self, shape, *chunks):
    length = builtins.sum(values.size for *indices, values in chunks)
    data = numpy.empty((length,), dtype=sparse.dtype(tuple(map(int, shape)), self._dtype))
    start = 0
    for *indices, values in chunks:
      stop = start + values.size
      d = data[start:stop].reshape(values.shape)
      d['value'] = values
      for idim, ii in enumerate(indices):
        d['index']['i'+str(idim)] = ii
      start = stop
    return data

# TRANSFORMCHAIN

class TransformChain(Evaluable):
  '''Chain of affine transformations.

  Evaluates to a tuple of :class:`nutils.transform.TransformItem` objects.
  '''

  __slots__ = 'todims',

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], todims:types.strictint=None):
    self.todims = todims
    super().__init__(args)

class SelectChain(TransformChain):

  __slots__ = 'n'

  @types.apply_annotations
  def __init__(self, n:types.strictint=0):
    self.n = n
    super().__init__(args=[EVALARGS])

  def evalf(self, evalargs):
    trans = evalargs['_transforms'][self.n]
    assert isinstance(trans, tuple)
    return trans

  @property
  def _node_details(self):
    return 'index={}'.format(self.n)

class TransformChainFromSequence(TransformChain):

  @types.apply_annotations
  def __init__(self, seq, index):
    self._seq = seq
    super().__init__(args=[index])

  def evalf(self, index):
    return self._seq[index.__index__()]

class PopHead(TransformChain):

  __slots__ = 'trans',

  @types.apply_annotations
  def __init__(self, todims:types.strictint, trans):
    self.trans = trans
    super().__init__(args=[self.trans], todims=todims)

  def evalf(self, trans):
    assert trans[0].fromdims == self.todims
    return trans[1:]

class SelectBifurcation(TransformChain):

  __slots__ = 'trans', 'first'

  @types.apply_annotations
  def __init__(self, trans:strictevaluable, first:bool, todims:types.strictint=None):
    self.trans = trans
    self.first = first
    super().__init__(args=[trans], todims=todims)

  def evalf(self, trans):
    assert isinstance(trans, tuple)
    bf = trans[0]
    assert isinstance(bf, transform.Bifurcate)
    selected = bf.trans1 if self.first else bf.trans2
    return selected + trans[1:]

class TransformChainFromTuple(TransformChain):

  __slots__ = 'index',

  def __init__(self, values:strictevaluable, index:types.strictint, todims:types.strictint=None):
    assert 0 <= index < len(values)
    self.index = index
    super().__init__(args=[values], todims=todims)

  def evalf(self, values):
    return values[self.index]

class TransformsIndexWithTail(Evaluable):

  __slots__ = '_transforms'

  @types.apply_annotations
  def __init__(self, transforms, trans:types.strict[TransformChain]):
    self._transforms = transforms
    super().__init__(args=[trans])

  def evalf(self, trans):
    index, tail = self._transforms.index_with_tail(trans)
    return numpy.array(index), tail

  def __len__(self):
    return 2

  @property
  def index(self):
    return ArrayFromTuple(self, index=0, shape=(), dtype=int)

  @property
  def tail(self):
    return TransformChainFromTuple(self, index=1, todims=self._transforms.fromdims)

  def __iter__(self):
    yield self.index
    yield self.tail

# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

def add(a, b):
  a, b = _numpy_align(a, b)
  return Add([a, b])

def multiply(a, b):
  a, b = _numpy_align(a, b)
  return Multiply([a, b])

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

  return multiply(a, b).sum(axes)

def transpose(arg, trans=None):
  arg = asarray(arg)
  if trans is None:
    normtrans = range(arg.ndim-1, -1, -1)
  else:
    normtrans = _normdims(arg.ndim, trans)
    assert sorted(normtrans) == list(range(arg.ndim))
  return Transpose(arg, normtrans)

def swapaxes(arg, axis1, axis2):
  arg = asarray(arg)
  trans = numpy.arange(arg.ndim)
  trans[axis1], trans[axis2] = trans[axis2], trans[axis1]
  return transpose(arg, trans)

# AXIS PROPERTIES

class Axis(types.Immutable):
  __slots__ = 'length'
  @types.apply_annotations
  def __init__(self, length:asindex):
    self.length = length
  def __str__(self):
    return str(int(self.length)) if self.length.isconstant else '?'
  def __repr__(self):
    return '{}{}'.format(type(self).__name__[0].lower(), self)

class Inserted(Axis):
  __slots__ = ()

class Raveled(Axis):
  __slots__ = 'shape'
  @types.apply_annotations
  def __init__(self, shape:asshape):
    assert len(shape) == 2
    self.shape = shape
    super().__init__(shape[0] * shape[1])

class Diagonal(Axis):
  __slots__ = ()
  @types.apply_annotations
  def __init__(self, length:asindex, marker):
    super().__init__(length)

class Sparse(Axis):
  __slots__ = 'mask'
  @types.apply_annotations
  def __init__(self, length:asindex, mask:types.arraydata=False):
    assert mask.dtype == bool
    self.mask = numpy.asarray(mask) # True for indices that are certain to be filled, used to detect dense addition
    super().__init__(length)

def as_axis_property(value):
  return value if isinstance(value, Axis) else Axis(value)

# ARRAYS

_ArrayMeta = type(Evaluable)

if debug_flags.sparse:
  def _chunked_assparse_checker(orig):
    assert isinstance(orig, property)
    @property
    def _assparse(self):
      chunks = orig.fget(self)
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
      self.evalf_obj = getattr(orig, '__get__', lambda *args: orig)
    def __get__(self, instance, owner):
      evalf = self.evalf_obj(instance, owner)
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

  __slots__ = '_axes', 'dtype', '__index'
  __cache__ = 'blocks', 'assparse', '_assparse'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], shape:types.tuple[as_axis_property], dtype:asdtype):
    self._axes = shape
    self.dtype = dtype
    super().__init__(args=args)

  @property
  def shape(self):
    return tuple(axis.length for axis in self._axes)

  @property
  def ndim(self):
    return len(self._axes)

  def __getitem__(self, item):
    if not isinstance(item, tuple):
      item = item,
    iell = None
    nx = self.ndim - len(item)
    for i, it in enumerate(item):
      if it is ...:
        assert iell is None, 'at most one ellipsis allowed'
        iell = i
      elif it is _:
        nx += 1
    array = self
    axis = 0
    for it in item + (slice(None),)*nx if iell is None else item[:iell] + (slice(None),)*(nx+1) + item[iell+1:]:
      if numeric.isint(it):
        array = get(array, axis, item=it)
      else:
        array = insertaxis(array, axis, 1) if it is _ \
           else _takeslice(array, it, axis) if isinstance(it, slice) \
           else take(array, it, axis)
        axis += 1
    assert axis == array.ndim
    return array

  def __bool__(self):
    return True

  def __len__(self):
    if self.ndim == 0:
      raise TypeError('len() of unsized object')
    return self.shape[0]

  def __iter__(self):
    if not self.shape:
      raise TypeError('iteration over a 0-d array')
    return (self[i,...] for i in range(self.shape[0]))

  def __index__(self):
    try:
      index = self.__index
    except AttributeError:
      if self.ndim or self.dtype not in (int, bool) or not self.isconstant:
        raise TypeError('cannot convert {!r} to int'.format(self))
      index = self.__index = int(self.simplified.eval())
    return index

  size = property(lambda self: util.product(self.shape) if self.ndim else 1)
  T = property(lambda self: transpose(self))

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
  __mod__  = lambda self, other: mod(self, other)
  __int__ = __index__
  __str__ = __repr__ = lambda self: '{}.{}<{}>'.format(type(self).__module__, type(self).__name__, self._shape_str(form=str))
  _shape_str = lambda self, form: '{}:{}'.format(self.dtype.__name__[0] if hasattr(self, 'dtype') else '?', ','.join(map(form, self._axes)) if hasattr(self, '_axes') else '?')

  sum = sum
  prod = product
  dot = dot
  swapaxes = swapaxes
  transpose = transpose
  choose = lambda self, choices: Choose(self, _numpy_align(*choices))

  @property
  def blocks(self):
    blocks = []
    pool = [(tuple(Range(sh) for sh in self.shape), self.simplified)]
    while pool:
      indices, f = pool.pop()
      i = 0
      for n, j in ((n, j) for n, index in enumerate(indices) for j in range(index.ndim)):
        if isinstance(f._axes[i], Sparse):
          pool.extend((indices[:n]+(_take(indices[n], ind, j).simplified,)+indices[n+1:], f.simplified) for ind, f in f._desparsify(i))
          break
        i += 1
      else:
        assert i == f.ndim
        blocks.append((indices, f))
    return _gatherblocks(blocks)

  @property
  def assparse(self):
    'Convert to a :class:`SparseArray`.'

    return SparseArray(self.simplified._assparse, self.shape, self.dtype)

  @property
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

    # Fallback using `blocks`.
    if self.ndim:
      chunks = []
      for indices, values in self.blocks:
        *offsets, ndim = (0, *itertools.accumulate(idx.ndim for idx in indices))
        assert ndim == values.ndim
        indices = (prependaxes(appendaxes(idx, values.shape[offset+idx.ndim:]), values.shape[:offset]) for idx, offset in zip(indices, offsets))
        chunks.append((*indices, values))
      return tuple(chunks)
    else:
      return (util.sum((values for indices, values in self.blocks), zeros((), self.dtype)),),

  def _node(self, cache, subgraph, times):
    if self in cache:
      return cache[self]
    args = tuple(arg._node(cache, subgraph, times) for arg in self._Evaluable__args)
    label = '\n'.join(filter(None, (type(self).__name__, self._node_details, self._shape_str(form=repr))))
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
  _determinant = lambda self, axis1, axis2: None
  _inverse = lambda self, axis1, axis2: None
  _takediag = lambda self, axis1, axis2: None
  _diagonalize = lambda self, axis: None
  _product = lambda self: None
  _sign = lambda self: None
  _eig = lambda self, symmetric: None
  _inflate = lambda self, dofmap, length, axis: None
  _unravel = lambda self, axis, shape: None
  _ravel = lambda self, axis: None
  _loopsum = lambda self, index, length: None

  def _uninsert(self, axis):
    assert isinstance(self._axes[axis], Inserted)
    item = Array(args=[EVALARGS], shape=(), dtype=int)
    uninserted = _take(self, item, axis).simplified
    assert item not in uninserted.dependencies, 'failed to uninsert axis'
    return uninserted

  def _desparsify(self, axis):
    if not isinstance(self._axes[axis], Sparse):
      raise Exception('attempting to desparsify a non-sparse axis')
    raise NotImplementedError('_desparsify implementation missing for {}'.format(type(self).__name__))

  def _resparsify(self, axis):
    if not isinstance(self._axes[axis], Sparse):
      raise Exception('attempting to resparsify a non-sparse axis')
    items = [_inflate(f, index, self.shape[axis], axis) for index, f in self._desparsify(axis)]
    return util.sum(items) if items else zeros_like(self)

  def _derivative(self, var, seen):
    if self.dtype in (bool, int) or var not in self.dependencies:
      return Zeros(self.shape + var.shape, dtype=self.dtype)
    raise NotImplementedError('derivative not defined for {}'.format(self.__class__.__name__))

  def as_evaluable_array(self):
    'return self'

    return self

class NPoints(Array):
  'The length of the points axis.'

  __slots__ = ()

  def __init__(self):
    super().__init__(args=[EVALARGS], shape=(), dtype=int)

  def evalf(self, evalargs):
    points = evalargs['_points'].coords
    return types.frozenarray(points.shape[0])

class Points(Array):

  __slots__ = ()

  def __init__(self, npoints, ndim):
    super().__init__(args=[EVALARGS], shape=(npoints, ndim), dtype=float)

  def evalf(self, evalargs):
    return evalargs['_points'].coords

class Weights(Array):

  __slots__ = ()

  def __init__(self, npoints):
    super().__init__(args=[EVALARGS], shape=(npoints,), dtype=float)

  def evalf(self, evalargs):
    weights = evalargs['_points'].weights
    assert numeric.isarray(weights) and weights.ndim == 1
    return weights

class Normal(Array):
  'normal'

  __slots__ = 'lgrad',

  @types.apply_annotations
  def __init__(self, lgrad:asarray):
    assert lgrad.ndim >= 2 and equalindex(lgrad.shape[-2], lgrad.shape[-1])
    self.lgrad = lgrad
    super().__init__(args=[lgrad], shape=lgrad.shape[:-1], dtype=float)

  def _simplified(self):
    if equalindex(self.shape[-1], 1):
      return Sign(Take(self.lgrad, 0))

  def evalf(self, lgrad):
    n = lgrad[...,-1]
    # orthonormalize n to G
    G = lgrad[...,:-1]
    GG = numeric.contract(G[...,:,_,:], G[...,:,:,_], axis=-3)
    v1 = numeric.contract(G, n[...,:,_], axis=-2)
    v2 = numpy.linalg.solve(GG, v1)
    v3 = numeric.contract(G, v2[...,_,:], axis=-1)
    return numeric.normalize(n - v3)

  def _derivative(self, var, seen):
    if equalindex(self.shape[-1], 1):
      return zeros(self.shape + var.shape)
    G = self.lgrad[...,:-1]
    m, n = G.shape[-2:]
    GG = dot(insertaxis(G, -1, n), insertaxis(G, -2, n), -3)
    GinvGG = dot(insertaxis(G, -1, n), insertaxis(inverse(GG), -3, m), -2)
    Gder = derivative(G, var, seen)
    nGder = dot(appendaxes(self, (n, *var.shape)), Gder, self.ndim-1)
    return -dot(appendaxes(GinvGG, var.shape), insertaxis(nGder, self.ndim-1, m), self.ndim)

class Constant(Array):

  __slots__ = 'value',
  __cache__ = '_isunit'

  @types.apply_annotations
  def __init__(self, value:types.arraydata):
    self.value = numpy.asarray(value)
    super().__init__(args=[], shape=value.shape, dtype=value.dtype)

  def _simplified(self):
    if not self.value.any():
      return zeros_like(self)
    for i, sh in enumerate(self.shape):
      # Find and replace invariant axes with InsertAxis. Since `self.value.any()`
      # is False for arrays with a zero-length axis, we can arrive here only if all
      # axes have at least length one, hence the following statement should work.
      first, *others = numpy.rollaxis(self.value, i)
      if all(numpy.equal(first, other).all() for other in others):
        return insertaxis(Constant(first), i, sh)

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

  @property
  def _isunit(self):
    return numpy.equal(self.value, 1).all()

  def _transpose(self, axes):
    return Constant(self.value.transpose(axes))

  def _sum(self, axis):
    return Constant(numpy.sum(self.value, axis))

  def _add(self, other):
    if isinstance(other, Constant):
      return Constant(numpy.add(self.value, other.value))

  def _inverse(self, axis1, axis2):
    value = numpy.transpose(self.value, tuple(i for i in range(self.ndim) if i != axis1 and i != axis2) + (axis1, axis2))
    return Constant(numpy.linalg.inv(value))

  def _product(self):
    return Constant(self.value.prod(-1))

  def _multiply(self, other):
    if self._isunit:
      return other
    if isinstance(other, Constant):
      return Constant(numpy.multiply(self.value, other.value))

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    return Constant(numpy.einsum('...kk->...k', numpy.transpose(self.value,
      list(range(axis1)) + list(range(axis1+1, axis2)) + list(range(axis2+1, self.ndim)) + [axis1, axis2])))

  def _take(self, index, axis):
    if index.isconstant:
      index_ = index.eval()
      return Constant(self.value.take(index_, axis))

  def _power(self, n):
    if isinstance(n, Constant):
      return Constant(numeric.power(self.value, n.value))

  def _eig(self, symmetric):
    eigval, eigvec = (numpy.linalg.eigh if symmetric else numpy.linalg.eig)(self.value)
    return Tuple((Constant(eigval), Constant(eigvec)))

  def _sign(self):
    return Constant(numeric.sign(self.value))

  def _unravel(self, axis, shape):
    shape = self.value.shape[:axis] + shape + self.value.shape[axis+1:]
    return Constant(self.value.reshape(shape))

  def _determinant(self, axis1, axis2):
    value = numpy.transpose(self.value, tuple(i for i in range(self.ndim) if i != axis1 and i != axis2) + (axis1, axis2))
    return Constant(numpy.linalg.det(value))

class InsertAxis(Array):

  __slots__ = 'func', 'length'

  @types.apply_annotations
  def __init__(self, func:asarray, length:asindex):
    self.func = func
    self.length = length
    super().__init__(args=[func, length], shape=func._axes+(Inserted(length),), dtype=func.dtype)

  def _simplified(self):
    return self.func._insertaxis(self.ndim-1, self.length)

  def evalf(self, func, length):
    if length == 1:
      return func[...,numpy.newaxis]
    try:
      return numpy.ndarray(buffer=func, dtype=func.dtype, shape=(*func.shape, length), strides=(*func.strides, 0))
    except ValueError: # non-contiguous data
      return numpy.repeat(func[...,numpy.newaxis], length, -1)

  def _derivative(self, var, seen):
    return insertaxis(derivative(self.func, var, seen), self.ndim-1, self.length)

  def _sum(self, i):
    if i == self.ndim - 1:
      return Multiply([self.func, _inflate_scalar(self.length, self.func.shape)])
    return InsertAxis(sum(self.func, i), self.length)

  def _product(self):
    return Power(self.func, _inflate_scalar(self.length, self.func.shape))

  def _power(self, n):
    for axis in range(self.ndim):
      if isinstance(self._axes[axis], Inserted) and isinstance(n._axes[axis], Inserted):
        return insertaxis(Power(self._uninsert(axis), n._uninsert(axis)), axis, self.shape[axis])

  def _add(self, other):
    for axis in range(self.ndim):
      if isinstance(self._axes[axis], Inserted) and isinstance(other._axes[axis], Inserted):
        return insertaxis(Add([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

  def _multiply(self, other):
    for axis in range(self.ndim):
      if isinstance(self._axes[axis], Inserted) and isinstance(other._axes[axis], Inserted):
        return insertaxis(Multiply([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

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

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    assert axis < self.ndim-1
    return [(ind, InsertAxis(f, self.length)) for ind, f in self.func._desparsify(axis)]

  def _determinant(self, axis1, axis2):
    if axis1 < self.ndim-1 and axis2 < self.ndim-1:
      return InsertAxis(determinant(self.func, (axis1, axis2)), self.length)

  def _inverse(self, axis1, axis2):
    if axis1 < self.ndim-1 and axis2 < self.ndim-1:
      return InsertAxis(inverse(self.func, (axis1, axis2)), self.length)

  def _loopsum(self, index, length):
    return InsertAxis(LoopSum(self.func, index, length), self.length)

  @property
  def _assparse(self):
    return tuple((*(InsertAxis(idx, self.length) for idx in indices), prependaxes(Range(self.length), values.shape), InsertAxis(values, self.length)) for *indices, values in self.func._assparse)

class Transpose(Array):

  __slots__ = 'func', 'axes'
  __cache__ = '_invaxes'

  @classmethod
  @types.apply_annotations
  def _end(cls, array:asarray, axes, invert=False):
    axes = [numeric.normdim(array.ndim, axis) for axis in axes]
    if all(a == b for a, b in enumerate(axes, start=array.ndim-len(axes))):
      return array
    trans = [i for i in range(array.ndim) if i not in axes]
    trans.extend(axes)
    if len(trans) != array.ndim:
      raise Exception('duplicate axes')
    return cls(array, numpy.argsort(trans) if invert else trans)

  @classmethod
  def from_end(cls, array, *axes):
    return cls._end(array, axes, invert=True)

  @classmethod
  def to_end(cls, array, *axes):
    return cls._end(array, axes, invert=False)

  @types.apply_annotations
  def __init__(self, func:asarray, axes:types.tuple[types.strictint]):
    assert sorted(axes) == list(range(func.ndim))
    self.func = func
    self.axes = axes
    super().__init__(args=[func], shape=[func._axes[n] for n in axes], dtype=func.dtype)

  @property
  def _invaxes(self):
    return tuple(numpy.argsort(self.axes))

  def _simplified(self):
    if self.axes == tuple(range(self.ndim)):
      return self.func
    return self.func._transpose(self.axes)

  def evalf(self, arr):
    return arr.transpose(self.axes)

  @property
  def _node_details(self):
    return ','.join(map(str, self.axes))

  def _transpose(self, axes):
    newaxes = [self.axes[i] for i in axes]
    return Transpose(self.func, newaxes)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    orig1, orig2 = sorted(self.axes[axis] for axis in [axis1, axis2])
    trytakediag = self.func._takediag(orig1, orig2)
    if trytakediag is not None:
      return Transpose(trytakediag, [ax-(ax>orig1)-(ax>orig2) for ax in self.axes[:axis1] + self.axes[axis1+1:axis2] + self.axes[axis2+1:]] + [self.ndim-2])

  def _sum(self, i):
    axis = self.axes[i]
    trysum = self.func._sum(axis)
    if trysum is not None:
      axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
      return Transpose(trysum, axes)

  def _derivative(self, var, seen):
    return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

  def _multiply(self, other):
    other_trans = other._transpose(self._invaxes)
    if other_trans is not None and not isinstance(other_trans, Transpose):
      # The second clause is to avoid infinite recursions; see
      # tests.test_evaluable.simplify.test_multiply_transpose.
      return Transpose(Multiply([self.func, other_trans]), self.axes)
    trymultiply = self.func._multiply(Transpose(other, self._invaxes))
    if trymultiply is not None:
      return Transpose(trymultiply, self.axes)

  def _add(self, other):
    if isinstance(other, Transpose) and self.axes == other.axes:
      return Transpose(Add([self.func, other.func]), self.axes)
    other_trans = other._transpose(self._invaxes)
    if other_trans is not None:
      return Transpose(Add([self.func, other_trans]), self.axes)

  def _take(self, indices, axis):
    trytake = self.func._take(indices, self.axes[axis])
    if trytake is not None:
      return Transpose(trytake, self._axes_for(indices.ndim, axis))

  def _axes_for(self, ndim, axis):
    funcaxis = self.axes[axis]
    axes = [ax+(ax>funcaxis)*(ndim-1) for ax in self.axes if ax != funcaxis]
    axes[axis:axis] = range(funcaxis, funcaxis + ndim)
    return axes

  def _power(self, n):
    n_trans = Transpose(n, self._invaxes)
    return Transpose(Power(self.func, n_trans), self.axes)

  def _sign(self):
    return Transpose(Sign(self.func), self.axes)

  def _unravel(self, axis, shape):
    orig_axis = self.axes[axis]
    tryunravel = self.func._unravel(orig_axis, shape)
    if tryunravel is not None:
      axes = [ax + (ax>orig_axis) for ax in self.axes]
      axes.insert(axis+1, orig_axis+1)
      return Transpose(tryunravel, axes)

  def _product(self):
    if self.axes[-1] == self.ndim-1:
      return Transpose(Product(self.func), self.axes[:-1])

  def _determinant(self, axis1, axis2):
    orig1, orig2 = self.axes[axis1], self.axes[axis2]
    trydet = self.func._determinant(orig1, orig2)
    if trydet:
      axes = [ax-(ax>orig1)-(ax>orig2) for ax in self.axes if ax != orig1 and ax != orig2]
      return Transpose(trydet, axes)

  def _inverse(self, axis1, axis2):
    tryinv = self.func._inverse(self.axes[axis1], self.axes[axis2])
    if tryinv:
      return Transpose(tryinv, self.axes)

  def _ravel(self, axis):
    if self.axes[axis] == self.ndim-2 and self.axes[axis+1] == self.ndim-1:
      return Transpose(Ravel(self.func), self.axes[:-1])

  def _inflate(self, dofmap, length, axis):
    i = self.axes[axis] if dofmap.ndim else self.func.ndim
    if self.axes[axis:axis+dofmap.ndim] == tuple(range(i,i+dofmap.ndim)):
      tryinflate = self.func._inflate(dofmap, length, i)
      if tryinflate is not None:
        axes = [ax-(ax>i)*(dofmap.ndim-1) for ax in self.axes]
        axes[axis:axis+dofmap.ndim] = i,
        return Transpose(tryinflate, axes)

  def _diagonalize(self, axis):
    trydiagonalize = self.func._diagonalize(self.axes[axis])
    if trydiagonalize is not None:
      return Transpose(trydiagonalize, self.axes + (self.ndim,))

  def _insertaxis(self, axis, length):
    return Transpose(InsertAxis(self.func, length), self.axes[:axis] + (self.ndim,) + self.axes[axis:])

  def _loopsum(self, index, length):
    return Transpose(LoopSum(self.func, index, length), self.axes)

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    return [(ind, Transpose(f, self._axes_for(ind.ndim, axis))) for ind, f in self.func._desparsify(self.axes[axis])]

  @property
  def _assparse(self):
    return tuple((*(indices[i] for i in self.axes), values) for *indices, values in self.func._assparse)

class Product(Array):

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=int if func.dtype == bool else func.dtype)

  def _simplified(self):
    if equalindex(self.func.shape[-1], 1):
      return get(self.func, self.ndim, 0)
    return self.func._product()

  def evalf(self, arr):
    assert arr.ndim == self.ndim+1
    return numpy.product(arr, axis=-1)

  def _derivative(self, var, seen):
    grad = derivative(self.func, var, seen)
    funcs = stack([util.product(self.func[...,j] for j in range(self.func.shape[-1]) if i != j) for i in range(self.func.shape[-1])], axis=self.ndim)
    return (grad * funcs[(...,)+(_,)*var.ndim]).sum(self.ndim)

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #ext = (...,)+(_,)*len(shape)
    #return self[ext] * (derivative(self.func,var,shape,seen) / self.func[ext]).sum(self.ndim)

  def _take(self, indices, axis):
    return Product(_take(self.func, indices, axis))

  def _takediag(self, axis1, axis2):
    return product(_takediag(self.func, axis1, axis2), self.ndim-2)

class ApplyTransforms(Array):

  __slots__ = 'trans', '_todims'

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], points, todims:types.strictint):
    self.trans = trans
    self._todims = todims
    super().__init__(args=[points, trans], shape=points.shape[:-1]+(todims,), dtype=float)

  def evalf(self, points, chain):
    return transform.apply(chain, points)

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords) and len(var) > 0:
      return prependaxes(LinearFrom(self.trans, self._todims, len(var)), self.shape[:-1])
    return zeros(self.shape+var.shape)

class LinearFrom(Array):

  __slots__ = 'todims', 'fromdims'

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], todims:types.strictint, fromdims:types.strictint):
    self.todims = todims
    self.fromdims = fromdims
    super().__init__(args=[trans], shape=(todims, fromdims), dtype=float)

  def evalf(self, chain):
    assert not chain or chain[0].todims == self.todims
    return transform.linearfrom(chain, self.fromdims)

class Inverse(Array):
  '''
  Matrix inverse of ``func`` over the last two axes.  All other axes are
  treated element-wise.
  '''

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert func.ndim >= 2 and equalindex(func.shape[-1], func.shape[-2])
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=float)

  def _simplified(self):
    return self.func._inverse(self.ndim-2, self.ndim-1)

  def evalf(self, arr):
    return numeric.inv(arr)

  def _derivative(self, var, seen):
    G = derivative(self.func, var, seen)
    n = var.ndim
    a = slice(None)
    return -sum(self[(...,a,a,_,_)+(_,)*n] * G[(...,_,a,a,_)+(a,)*n] * self[(...,_,_,a,a)+(_,)*n], [-2-n, -3-n])

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

class Interpolate(Array):
  'interpolate uniformly spaced data; stepwise for now'

  __slots__ = 'xp', 'fp', 'left', 'right'

  @types.apply_annotations
  def __init__(self, x:asarray, xp:types.arraydata, fp:types.arraydata, left:types.strictfloat=None, right:types.strictfloat=None):
    xp = numpy.asarray(xp)
    fp = numpy.asarray(fp)
    assert xp.ndim == fp.ndim == 1
    if not numpy.greater(numpy.diff(xp), 0).all():
      warnings.warn('supplied x-values are non-increasing')
    assert x.ndim == 0
    self.xp = xp
    self.fp = fp
    self.left = left
    self.right = right
    super().__init__(args=[x], shape=(), dtype=float)

  def evalf(self, x):
    return numpy.interp(x, self.xp, self.fp, self.left, self.right)

class Determinant(Array):

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert isarray(func) and func.ndim >= 2 and equalindex(func.shape[-1], func.shape[-2])
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-2], dtype=_jointdtype(func.dtype, float))

  def _simplified(self):
    return self.func._determinant(self.ndim, self.ndim+1)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.linalg.det(arr)

  def _derivative(self, var, seen):
    Finv = swapaxes(inverse(self.func), -2, -1)
    G = derivative(self.func, var, seen)
    ext = (...,)+(_,)*var.ndim
    return self[ext] * sum(Finv[ext] * G, axis=[-2-var.ndim,-1-var.ndim])

  def _take(self, index, axis):
    return Determinant(_take(self.func, index, axis))

  def _takediag(self, axis1, axis2):
    return determinant(_takediag(self.func, axis1, axis2), (self.ndim-2, self.ndim-1))

class Multiply(Array):

  __slots__ = 'funcs',

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert equalshape(func1.shape, func2.shape)
    axes = [axis1 if axis1 == axis2
       else axis1 if isinstance(axis1, Sparse)
       else axis2 if isinstance(axis2, Sparse)
       else Axis(axis1.length) for axis1, axis2 in zip(func1._axes, func2._axes)]
    super().__init__(args=self.funcs, shape=axes, dtype=_jointdtype(func1.dtype,func2.dtype))

  def _simplified(self):
    func1, func2 = self.funcs
    if isuniform(func1, 1):
      return func2
    if isuniform(func2, 1):
      return func1
    diagonals = {}
    for i, axis in enumerate(self._axes):
      if isinstance(axis, Inserted):
        return insertaxis(Multiply(func._uninsert(i) for func in self.funcs), i, axis.length)
      if isinstance(axis, Sparse):
        return self._resparsify(i)
      if isinstance(axis, Raveled):
        return ravel(Multiply(unravel(func, i, axis.shape) for func in self.funcs), i)
      if isinstance(axis, Diagonal):
        if axis in diagonals:
          return diagonalize(Multiply(takediag(func, diagonals[axis], i) for func in self.funcs), diagonals[axis], i)
        diagonals[axis] = i
    return func1._multiply(func2) or func2._multiply(func1)

  def _optimized_for_numpy(self):
    func1, func2 = self.funcs
    if isuniform(func1, -1) and func2.dtype != bool:
      return Negative(func2)
    if isuniform(func2, -1) and func1.dtype != bool:
      return Negative(func1)
    if func1 == sign(func2):
      return Absolute(func2)
    if func2 == sign(func1):
      return Absolute(func1)
    if not self.ndim:
      return
    keep = numpy.ones([2, self.ndim], dtype=bool)
    for i in reversed(range(self.ndim)):
      if isinstance(func1._axes[i], Inserted):
        keep[0, i] = False
        func1 = func1._uninsert(i)
      if isinstance(func2._axes[i], Inserted):
        keep[1, i] = False
        func2 = func2._uninsert(i)
    if not keep.any(0).all():
      warnings.warn('simplification failed for multiplication of InsertAxis', ExpensiveEvaluationWarning)
    return Einsum((func1, func2), tuple(mask.nonzero()[0] for mask in keep), tuple(range(self.ndim)))

  def evalf(self, arr1, arr2):
    return arr1 * arr2

  def _sum(self, axis):
    func1, func2 = self.funcs
    if equalindex(self.shape[axis], 1):
      return multiply(get(func1, axis, 0), get(func2, axis, 0))
    if isinstance(func1._axes[axis], Inserted):
      return multiply(func1._uninsert(axis), func2.sum(axis))
    if isinstance(func2._axes[axis], Inserted):
      return multiply(func1.sum(axis), func2._uninsert(axis))

  def _add(self, other):
    func1, func2 = self.funcs
    if other == func1:
      return Multiply([func1, Add([func2, ones_like(func2)])])
    if other == func2:
      return Multiply([func2, Add([func1, ones_like(func1)])])
    if isinstance(other, Multiply) and not self.funcs.isdisjoint(other.funcs):
      f = next(iter(self.funcs & other.funcs))
      return Multiply([f, Add(self.funcs + other.funcs - [f,f])])

  def _determinant(self, axis1, axis2):
    func1, func2 = self.funcs
    axis1, axis2 = sorted([axis1, axis2])
    if equalindex(self.shape[axis1], 1) and equalindex(self.shape[axis2], 1):
      return Multiply([determinant(func1, (axis1, axis2)), determinant(func2, (axis1, axis2))])
    if all(isinstance(func1._axes[axis], Inserted) for axis in (axis1, axis2)):
      return Multiply([func1._uninsert(axis2)._uninsert(axis1)**self.shape[axis1], determinant(func2, (axis1, axis2))])
    if all(isinstance(func2._axes[axis], Inserted) for axis in (axis1, axis2)):
      return Multiply([func2._uninsert(axis2)._uninsert(axis1)**self.shape[axis1], determinant(func1, (axis1, axis2))])

  def _product(self):
    func1, func2 = self.funcs
    return Multiply([Product(func1), Product(func2)])

  def _multiply(self, other):
    func1, func2 = self.funcs
    func1_other = func1._multiply(other)
    if func1_other is not None:
      return Multiply([func1_other, func2])
    func2_other = func2._multiply(other)
    if func2_other is not None:
      return Multiply([func1, func2_other])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*var.ndim
    return func1[ext] * derivative(func2, var, seen) \
         + func2[ext] * derivative(func1, var, seen)

  def _takediag(self, axis1, axis2):
    func1, func2 = self.funcs
    return Multiply([_takediag(func1, axis1, axis2), _takediag(func2, axis1, axis2)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Multiply([_take(func1, index, axis), _take(func2, index, axis)])

  def _sign(self):
    return Multiply([Sign(func) for func in self.funcs])

  def _unravel(self, axis, shape):
    return Multiply([unravel(func, axis, shape) for func in self.funcs])

  def _inverse(self, axis1, axis2):
    func1, func2 = self.funcs
    if all(isinstance(func1._axes[axis], Inserted) for axis in (axis1, axis2)):
      return divide(inverse(func2, (axis1, axis2)), func1)
    if all(isinstance(func2._axes[axis], Inserted) for axis in (axis1, axis2)):
      return divide(inverse(func1, (axis1, axis2)), func2)

  def _loopsum(self, index, length):
    func1, func2 = self.funcs
    if func1 != index and index not in func1.dependencies:
      return Multiply([func1, LoopSum(func2, index, length)])
    if func2 != index and index not in func2.dependencies:
      return Multiply([LoopSum(func1, index, length), func2])

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    func1, func2 = self.funcs
    if not isinstance(func1._axes[axis], Sparse):
      assert isinstance(func2._axes[axis], Sparse)
      func1, func2 = func2, func1
    assert isinstance(func1._axes[axis], Sparse)
    return [(ind, multiply(f1, _take(func2, ind, axis))) for ind, f1 in func1._desparsify(axis)]

  @property
  def _assparse(self):
    func1, func2 = self.funcs
    if all(isinstance(axis, Inserted) for axis in func1._axes):
      for i in reversed(range(func1.ndim)):
        func1 = func1._uninsert(i)
      return tuple((*indices, appendaxes(func1, values.shape)*values) for *indices, values in func2._assparse)
    elif all(isinstance(axis, Inserted) for axis in func2._axes):
      for i in reversed(range(func2.ndim)):
        func2 = func2._uninsert(i)
      return tuple((*indices, values*appendaxes(func2, values.shape)) for *indices, values in func1._assparse)
    else:
      return super()._assparse

class Add(Array):

  __slots__ = 'funcs',

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert equalshape(func1.shape, func2.shape)
    axes = [axis1 if axis1 == axis2 else Axis(axis1.length) for axis1, axis2 in zip(func1._axes, func2._axes)]
    sparse = [i for i, (axis1, axis2) in enumerate(zip(func1._axes, func2._axes)) if isinstance(axis1, Sparse) and isinstance(axis2, Sparse)]
    if len(sparse) > 1: # an addition of multiple sparse axes is always sparse
      for i in sparse:
        axes[i] = Sparse(axes[i].length)
    elif len(sparse) == 1: # an addition of a single sparse axis may have become dense
      i, = sparse
      mask = func1._axes[i].mask | func2._axes[i].mask # axis positions that are certainly filled
      if not mask.all():
        axes[i] = Sparse(axes[i].length, mask)
    super().__init__(args=self.funcs, shape=axes, dtype=_jointdtype(func1.dtype,func2.dtype))

  def _simplified(self):
    func1, func2 = self.funcs
    if func1 == func2:
      return multiply(func1, 2)
    return func1._add(func2) or func2._add(func1)

  def evalf(self, arr1, arr2=None):
    return arr1 + arr2

  def _sum(self, axis):
    return Add([sum(func, axis) for func in self.funcs])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    return derivative(func1, var, seen) + derivative(func2, var, seen)

  def _takediag(self, axis1, axis2):
    func1, func2 = self.funcs
    return Add([_takediag(func1, axis1, axis2), _takediag(func2, axis1, axis2)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Add([_take(func1, index, axis), _take(func2, index, axis)])

  def _add(self, other):
    func1, func2 = self.funcs
    func1_other = func1._add(other)
    if func1_other is not None:
      return Add([func1_other, func2])
    func2_other = func2._add(other)
    if func2_other is not None:
      return Add([func1, func2_other])

  def _unravel(self, axis, shape):
    return Add([unravel(func, axis, shape) for func in self.funcs])

  def _loopsum(self, index, length):
    if any(index not in func.arguments for func in self.funcs):
      return Add([LoopSum(func, index, length) for func in self.funcs])

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    return _gatherblocks(block for func in self.funcs for block in func._desparsify(axis))

  @property
  def _assparse(self):
    func1, func2 = self.funcs
    return _gathersparsechunks(itertools.chain(func1._assparse, func2._assparse))

class Einsum(Array):

  __slots__ = 'args', 'out_idx', 'args_idx', '_einsumfmt', '_has_summed_axes'

  @types.apply_annotations
  def __init__(self, args:asarrays, args_idx:types.tuple[types.tuple[types.strictint]], out_idx:types.tuple[types.strictint]):
    if len(args_idx) != len(args):
      raise ValueError('Expected one list of indices for every argument, but got {} and {}, respectively.'.format(len(args_idx), len(args)))
    for iarg, (idx, arg) in enumerate(zip(args_idx, args), 1):
      if len(idx) != arg.ndim:
        raise ValueError('Expected one index for every axis of argument {}, but got {} and {}, respectively.'.format(iarg, len(idx), arg.ndim))

    if len(out_idx) != len(set(out_idx)):
      raise ValueError('Repeated output indices.')
    lengths = {}
    for idx, arg in zip(args_idx, args):
      for i, length in zip(idx, arg.shape):
        if i not in lengths:
          lengths[i] = length
        elif not equalindex(lengths[i], length):
          raise ValueError('Axes with index {} have different lengths.'.format(i))
    try:
      shape = [lengths[i] for i in out_idx]
    except KeyError:
      raise ValueError('Output axis {} is not listed in any of the arguments.'.format(', '.join(i for i in out_idx if i not in lengths)))
    self.args = args
    self.args_idx = args_idx
    self.out_idx = out_idx
    self._einsumfmt = ','.join(''.join(chr(97+i) for i in idx) for idx in args_idx) + '->' + ''.join(chr(97+i) for i in out_idx)
    self._has_summed_axes = len(lengths) > len(out_idx)
    super().__init__(args=self.args, shape=shape, dtype=_jointdtype(*(arg.dtype for arg in args)))

  def evalf(self, *args):
    if self._has_summed_axes:
      args = tuple(numpy.asarray(arg, order='F') for arg in args)
    return numpy.core.multiarray.c_einsum(self._einsumfmt, *args)

  @property
  def _node_details(self):
    return self._einsumfmt

  def _simplified(self):
    for i, arg in enumerate(self.args):
      if isinstance(arg, Transpose): # absorb `Transpose`
        idx = tuple(map(self.args_idx[i].__getitem__, numpy.argsort(arg.axes)))
        return Einsum(self.args[:i]+(arg.func,)+self.args[i+1:], self.args_idx[:i]+(idx,)+self.args_idx[i+1:], self.out_idx)

  def _sum(self, axis):
    if not (0 <= axis < self.ndim):
      raise IndexError('Axis out of range.')
    return Einsum(self.args, self.args_idx, self.out_idx[:axis] + self.out_idx[axis+1:])

  def _takediag(self, axis1, axis2):
    if not (0 <= axis1 < axis2 < self.ndim):
      raise IndexError('Axis out of range.')
    ikeep, irm = self.out_idx[axis1], self.out_idx[axis2]
    args_idx = tuple(tuple(ikeep if i == irm else i for i in idx) for idx in self.args_idx)
    return Einsum(self.args, args_idx, self.out_idx[:axis1] + self.out_idx[axis1+1:axis2] + self.out_idx[axis2+1:] + (ikeep,))

class Sum(Array):

  __slots__ = 'func'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim == 0:
      raise Exception('cannot sum a scalar function')
    self.func = func
    shape = func._axes[:-1]
    super().__init__(args=[func], shape=shape, dtype=int if func.dtype == bool else func.dtype)

  def _simplified(self):
    if equalindex(self.func.shape[-1], 1):
      return Take(self.func, 0)
    return self.func._sum(self.ndim)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+1
    return numpy.sum(arr, -1)

  def _sum(self, axis):
    trysum = self.func._sum(axis)
    if trysum is not None:
      return Sum(trysum)

  def _derivative(self, var, seen):
    return sum(derivative(self.func, var, seen), self.ndim)

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    return [(ind, Sum(f)) for ind, f in self.func._desparsify(axis)]

  @property
  def _assparse(self):
    chunks = []
    for *indices, _rmidx, values in self.func._assparse:
      if values.dtype == bool:
        values = Int(values)
      while True:
        for axis in reversed(range(values.ndim)):
          if all(isinstance(idx._axes[axis], Inserted) for idx in indices):
            indices = tuple(idx._uninsert(axis) for idx in indices)
            values = sum(values, axis)
        else:
          break
      if self.ndim == 0:
        while values.ndim:
          values = Sum(values)
      chunks.append((*indices, values))
    return _gathersparsechunks(chunks)

class TakeDiag(Array):

  __slots__ = 'func'
  __cache__ = '_assparse'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('takediag requires an argument of dimension >= 2')
    if not equalindex(func.shape[-1], func.shape[-2]):
      raise Exception('takediag axes do not match')
    self.func = func
    shape = func._axes[:-2]+(Axis(func.shape[-1]),)
    super().__init__(args=[func], shape=shape, dtype=func.dtype)

  def _simplified(self):
    if equalindex(self.shape[-1], 1):
      return Take(self.func, 0)
    return self.func._takediag(self.ndim-1, self.ndim)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+1
    return numpy.einsum('...kk->...k', arr, optimize=False)

  def _derivative(self, var, seen):
    return takediag(derivative(self.func, var, seen), self.ndim-1, self.ndim)

  def _take(self, index, axis):
    if axis < self.ndim - 1:
      return TakeDiag(_take(self.func, index, axis))
    func = _take(Take(self.func, index), index, self.ndim-1)
    for i in reversed(range(self.ndim-1, self.ndim-1+index.ndim)):
      func = takediag(func, i, i+index.ndim)
    return func

  def _sum(self, axis):
    if axis != self.ndim - 1:
      return TakeDiag(sum(self.func, axis))

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    assert axis != self.ndim-1
    return [(ind, TakeDiag(f)) for ind, f in self.func._desparsify(axis)]

  @property
  def _assparse(self):
    chunks = []
    for *indices, values in self.func._assparse:
      if indices[-2] == indices[-1]:
        chunks.append((*indices[:-1], values))
      else:
        *indices, values = map(_flat, (*indices, values))
        mask = Equal(indices[-2], indices[-1])
        chunks.append(tuple(take(arr, mask, 0) for arr in (*indices[:-1], values)))
    return _gathersparsechunks(chunks)

class Take(Array):

  __slots__ = 'func', 'indices'

  @types.apply_annotations
  def __init__(self, func:asarray, indices:asarray):
    if func.ndim == 0:
      raise Exception('cannot take a scalar function')
    if indices.dtype != int:
      raise Exception('invalid indices argument for take')
    self.func = func
    self.indices = indices
    shape = (*func._axes[:-1], *(axis if isinstance(axis, Inserted) else Axis(axis.length) for axis in indices._axes))
    super().__init__(args=[func,indices], shape=shape, dtype=func.dtype)

  def _simplified(self):
    if self.indices.size == 0:
      return zeros_like(self)
    length = self.func.shape[-1]
    if self.indices == Range(length):
      return self.func
    if self.indices.ndim == 1 and isinstance(self.indices._axes[-1], Raveled):
      shape = self.indices._axes[-1].shape
      return Ravel(Take(self.func, Unravel(self.indices, *shape)))
    for iaxis in reversed(range(self.func.ndim-1, self.ndim)):
      axis = self._axes[iaxis]
      if isinstance(axis, Inserted):
        return insertaxis(self._uninsert(iaxis), iaxis, axis.length)
    return self.func._take(self.indices, self.func.ndim-1)

  def evalf(self, arr, indices):
    return arr[...,indices]

  def _derivative(self, var, seen):
    return _take(derivative(self.func, var, seen), self.indices, self.func.ndim-1)

  def _take(self, index, axis):
    if axis >= self.func.ndim-1:
      return Take(self.func, _take(self.indices, index, axis-self.func.ndim+1))
    trytake = self.func._take(index, axis)
    if trytake is not None:
      return Take(trytake, self.indices)

  def _sum(self, axis):
    if axis < self.func.ndim - 1:
      return Take(sum(self.func, axis), self.indices)

  def _uninsert(self, axis):
    if axis < self.func.ndim-1:
      return Take(self.func._uninsert(axis), self.indices)
    else:
      return Take(self.func, self.indices._uninsert(axis-self.func.ndim+1))

  def _desparsify(self, axis):
    assert axis < self.func.ndim-1 and isinstance(self._axes[axis], Sparse)
    return [(ind, Take(f, self.indices)) for ind, f in self.func._desparsify(axis)]

class Power(Array):

  __slots__ = 'func', 'power'

  @types.apply_annotations
  def __init__(self, func:asarray, power:asarray):
    assert equalshape(func.shape, power.shape)
    self.func = func
    self.power = power
    dtype = float if func.dtype == power.dtype == int else _jointdtype(func.dtype, power.dtype)
    super().__init__(args=[func,power], shape=func.shape, dtype=dtype)

  def _simplified(self):
    if iszero(self.power):
      return ones_like(self)
    return self.func._power(self.power)

  def _optimized_for_numpy(self):
    if isuniform(self.power, -1):
      return Reciprocal(self.func)
    elif isuniform(self.power, 2):
      return Square(self.func)
    elif isuniform(self.power, -2):
      return Reciprocal(Square(self.func))
    elif isuniform(self.power, 1):
      return self.func
    else:
      return self._simplified()

  def evalf(self, base, exp):
    return numeric.power(base, exp)

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    if self.power.isconstant:
      p = self.power.eval()
      p_decr = p - (p!=0)
      return multiply(p, power(self.func, p_decr))[ext] * derivative(self.func, var, seen)
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    return (self.power * power(self.func, self.power - 1))[ext] * derivative(self.func, var, seen) \
         + (ln(self.func) * self)[ext] * derivative(self.power, var, seen)

  def _power(self, n):
    func = self.func
    newpower = Multiply([self.power, n])
    if iszero(self.power % 2) and not iszero(newpower % 2):
      func = abs(func)
    return Power(func, newpower)

  def _takediag(self, axis1, axis2):
    return Power(_takediag(self.func, axis1, axis2), _takediag(self.power, axis1, axis2))

  def _take(self, index, axis):
    return Power(_take(self.func, index, axis), _take(self.power, index, axis))

  def _unravel(self, axis, shape):
    return Power(unravel(self.func, axis, shape), unravel(self.power, axis, shape))

  def _product(self):
    if isinstance(self._axes[-1], Inserted):
      return Power(Product(self.func), self.power._uninsert(self.ndim-1))

class Pointwise(Array):
  '''
  Abstract base class for pointwise array functions.
  '''

  __slots__ = 'args',

  deriv = None

  @types.apply_annotations
  def __init__(self, *args:asarrays):
    retval = self.evalf(*[numpy.ones((), dtype=arg.dtype) for arg in args])
    shape0 = args[0].shape
    assert all(equalshape(arg.shape, shape0) for arg in args[1:]), 'pointwise arguments have inconsistent shapes'
    shape = tuple(axes[0] if len(set(axes)) == 1 and not isinstance(axes[0], Sparse) else Axis(axes[0].length) for axes in zip(*(arg._axes for arg in args)))
    # TODO check shape definition
    self.args = args
    super().__init__(args=args, shape=shape, dtype=retval.dtype)

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
    if self.isconstant:
      retval = self.eval()
      return Constant(retval)
    if len(self.args) == 1 and isinstance(self.args[0], Transpose):
      arg, = self.args
      return Transpose(self.__class__(arg.func), arg.axes)
    for i in reversed(range(self.ndim)):
      if all(isinstance(arg._axes[i], Inserted) for arg in self.args):
        return insertaxis(self.__class__(*(arg._uninsert(i) for arg in self.args)), i, self.shape[i])

  def _derivative(self, var, seen):
    if self.deriv is None:
      return super()._derivative(var, seen)
    return util.sum(deriv(*self.args)[(...,)+(_,)*var.ndim] * derivative(arg, var, seen) for arg, deriv in zip(self.args, self.deriv))

  def _takediag(self, axis1, axis2):
    return self.__class__(*[_takediag(arg, axis1, axis2) for arg in self.args])

  def _take(self, index, axis):
    return self.__class__(*[_take(arg, index, axis) for arg in self.args])

  def _unravel(self, axis, shape):
    return self.__class__(*[unravel(arg, axis, shape) for arg in self.args])

class Reciprocal(Pointwise):
  __slots__ = ()
  evalf = numpy.reciprocal

class Negative(Pointwise):
  __slots__ = ()
  evalf = numpy.negative

class Square(Pointwise):
  __slots__ = ()
  evalf = numpy.square
  def _sum(self, axis):
    func, = self.args
    idx = tuple(range(func.ndim))
    return Einsum((func, func), (idx, idx), idx)._sum(axis)

class FloorDivide(Pointwise):
  __slots__ = ()
  evalf = numpy.floor_divide

class Absolute(Pointwise):
  __slots__ = ()
  evalf = numpy.absolute

class Cos(Pointwise):
  'Cosine, element-wise.'
  __slots__ = ()
  evalf = numpy.cos
  deriv = lambda x: -Sin(x),

class Sin(Pointwise):
  'Sine, element-wise.'
  __slots__ = ()
  evalf = numpy.sin
  deriv = Cos,

class Tan(Pointwise):
  'Tangent, element-wise.'
  __slots__ = ()
  evalf = numpy.tan
  deriv = lambda x: Cos(x)**-2,

class ArcSin(Pointwise):
  'Inverse sine, element-wise.'
  __slots__ = ()
  evalf = numpy.arcsin
  deriv = lambda x: reciprocal(sqrt(1-x**2)),

class ArcCos(Pointwise):
  'Inverse cosine, element-wise.'
  __slots__ = ()
  evalf = numpy.arccos
  deriv = lambda x: -reciprocal(sqrt(1-x**2)),

class ArcTan(Pointwise):
  'Inverse tangent, element-wise.'
  __slots__ = ()
  evalf = numpy.arctan
  deriv = lambda x: reciprocal(1+x**2),

class Exp(Pointwise):
  __slots__ = ()
  evalf = numpy.exp
  deriv = lambda x: Exp(x),

class Log(Pointwise):
  __slots__ = ()
  evalf = numpy.log
  deriv = lambda x: reciprocal(x),

class Mod(Pointwise):
  __slots__ = ()
  evalf = numpy.mod

class ArcTan2(Pointwise):
  __slots__ = ()
  evalf = numpy.arctan2
  deriv = lambda x, y: y / (x**2 + y**2), lambda x, y: -x / (x**2 + y**2)

class Greater(Pointwise):
  __slots__ = ()
  evalf = numpy.greater
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Equal(Pointwise):
  __slots__ = ()
  evalf = numpy.equal
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Less(Pointwise):
  __slots__ = ()
  evalf = numpy.less
  deriv = (lambda a, b: Zeros(a.shape, dtype=int),) * 2

class Minimum(Pointwise):
  __slots__ = ()
  evalf = numpy.minimum
  deriv = Less, lambda x, y: 1 - Less(x, y)

class Maximum(Pointwise):
  __slots__ = ()
  evalf = numpy.maximum
  deriv = lambda x, y: 1 - Less(x, y), Less

class Int(Pointwise):
  __slots__ = ()
  evalf = staticmethod(lambda a: a.astype(int))
  deriv = lambda a: Zeros(a.shape, int),

class Sign(Array):

  __slots__ = 'func',

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=func.dtype)

  def _simplified(self):
    return self.func._sign()

  def evalf(self, arr):
    return numpy.sign(arr)

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

class Sampled(Array):
  '''Basis-like identity operator.

  Basis-like function that for every point in a predefined set evaluates to the
  unit vector corresponding to its index.

  Args
  ----
  points : 1d :class:`Array`
      Present point coordinates.
  expect : 2d :class:`Array`
      Elementwise constant that evaluates to the predefined point coordinates;
      used for error checking and to inherit the shape.
  '''

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, points:asarray, expect:asarray):
    assert points.ndim == 2
    super().__init__(args=[points, expect], shape=(points.shape[0], expect.shape[0]), dtype=int)

  def evalf(self, points, expect):
    assert numpy.equal(points, expect).all(), 'illegal point set'
    return numpy.eye(len(points), dtype=int)

@types.apply_annotations
def Elemwise(data:types.tuple[types.arraydata], index:asarray, dtype:asdtype):
  unique, indices = util.unique(data)
  if len(unique) == 1:
    return Constant(unique[0])
  shape = [Take(s, index).simplified for s in numpy.array([d.shape for d in data]).T] # use original index to avoid potential inconsistencies with other arrays
  ndim = len(shape)
  if len(unique) < len(data):
    index = Take(indices, index)
  raveled = list(map(numpy.ravel, unique))
  concat = numpy.concatenate(raveled)
  if ndim == 0:
    return Take(concat, index)
  cumprod = list(shape)
  for i in reversed(range(ndim-1)):
    cumprod[i] *= cumprod[i+1] # work backwards so that the shape check matches in Unravel
  offsets = numpy.cumsum([0, *map(len, raveled[:-1])])
  elemwise = Take(concat, Range(cumprod[0], Take(offsets, index)))
  for i in range(ndim-1):
    elemwise = Unravel(elemwise, shape[i], cumprod[i+1])
  return elemwise

class Eig(Evaluable):

  __slots__ = 'symmetric', 'func', '_w_dtype', '_vt_dtype'

  @types.apply_annotations
  def __init__(self, func:asarray, symmetric:bool=False):
    assert func.ndim >= 2 and equalindex(func.shape[-1], func.shape[-2])
    self.symmetric = symmetric
    self.func = func
    self._w_dtype = float if symmetric else complex
    self._vt_dtype = _jointdtype(float, func.dtype if symmetric else complex)
    super().__init__(args=[func])

  def __len__(self):
    return 2

  def __iter__(self):
    yield ArrayFromTuple(self, index=0, shape=self.func.shape[:-1], dtype=self._w_dtype)
    yield ArrayFromTuple(self, index=1, shape=self.func.shape, dtype=self._vt_dtype)

  def _simplified(self):
    return self.func._eig(self.symmetric)

  def evalf(self, arr):
    w, vt = (numpy.linalg.eigh if self.symmetric else numpy.linalg.eig)(arr)
    w = w.astype(self._w_dtype, copy=False)
    vt = vt.astype(self._vt_dtype, copy=False)
    return (w, vt)

class ArrayFromTuple(Array):

  __slots__ = 'arrays', 'index'

  @types.apply_annotations
  def __init__(self, arrays:strictevaluable, index:types.strictint, shape:asshape, dtype:asdtype):
    self.arrays = arrays
    self.index = index
    super().__init__(args=[arrays], shape=shape, dtype=dtype)

  def evalf(self, arrays):
    assert isinstance(arrays, tuple)
    return arrays[self.index]

  def _node(self, cache, subgraph, times):
    if self in cache:
      return cache[self]
    elif hasattr(self.arrays, '_node_tuple'):
      cache[self] = node = self.arrays._node_tuple(cache, subgraph, times)[self.index]
      return node
    else:
      return super()._node(cache, subgraph, times)

class Zeros(Array):
  'zero'

  __slots__ = ()
  __cache__ = '_assparse'

  @types.apply_annotations
  def __init__(self, shape:asshape, dtype:asdtype):
    super().__init__(args=shape, shape=map(Sparse, shape), dtype=dtype)

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

  def _desparsify(self, axis):
    return []

  def _add(self, other):
    return other

  def _multiply(self, other):
    return self

  def _diagonalize(self, axis):
    return Zeros(self.shape+(self.shape[axis],), dtype=self.dtype)

  def _sum(self, axis):
    return Zeros(self.shape[:axis] + self.shape[axis+1:], dtype=int if self.dtype == bool else self.dtype)

  def _transpose(self, axes):
    shape = [self.shape[n] for n in axes]
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
    shape = list(self.shape)
    assert axis1 != axis2
    length, = set(map(shape.pop, sorted((axis1, axis2), reverse=True)))
    if iszero(length):
      return ones(shape, self.dtype)
    else:
      return Zeros(shape, self.dtype)

  @property
  def _assparse(self):
    return ()

class Inflate(Array):

  __slots__ = 'func', 'dofmap', 'length', 'warn'
  __cache__ = '_assparse'

  @types.apply_annotations
  def __init__(self, func:asarray, dofmap:asarray, length:asindex):
    if not equalshape(func.shape[func.ndim-dofmap.ndim:], dofmap.shape):
      raise Exception('invalid dofmap')
    self.func = func
    self.dofmap = dofmap
    self.length = length
    mask = not any(isinstance(ax, Sparse) for ax in func._axes[func.ndim-dofmap.ndim:]) and dofmap.isconstant and length.isconstant and numeric.asboolean(dofmap.eval().ravel(), int(length), ordered=False)
    shape = func._axes[:-1] + (Sparse(length, mask),)
    self.warn = not dofmap.isconstant
    super().__init__(args=[func,dofmap,length], shape=func._axes[:func.ndim-dofmap.ndim]+(Sparse(length, mask),), dtype=func.dtype)

  def _simplified(self):
    if self.dofmap == Range(self.length):
      return self.func
    for axis in range(self.dofmap.ndim):
      if equalindex(self.dofmap.shape[axis], 1):
        return Inflate(_take(self.func, 0, self.func.ndim-self.dofmap.ndim+axis), _take(self.dofmap, 0, axis), self.length)
      if isinstance(self.func._axes[self.func.ndim-self.dofmap.ndim+axis], Sparse):
        items = [Inflate(f, _take(self.dofmap, ind, axis), self.shape[-1]) for ind, f in self.func._desparsify(self.func.ndim-self.dofmap.ndim+axis)]
        return util.sum(items) if items else zeros_like(self)
    if self.dofmap.ndim == 0 and equalindex(self.dofmap, 0) and equalindex(self.length, 1):
      return InsertAxis(self.func, 1)
    return self.func._inflate(self.dofmap, self.length, self.ndim-1)

  def evalf(self, array, indices, length):
    assert indices.ndim == self.dofmap.ndim
    assert length.ndim == 0
    if self.warn and int(length) > indices.size:
      warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
    inflated = numpy.zeros(array.shape[:array.ndim-indices.ndim] + (length,), dtype=self.dtype)
    numpy.add.at(inflated, (slice(None),)*(self.ndim-1)+(indices,), array)
    return inflated

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    if axis == self.ndim-1:
      return [(self.dofmap, self.func)]
    return [(ind, Inflate(f, self.dofmap, self.length)) for ind, f in self.func._desparsify(axis)]

  def _inflate(self, dofmap, length, axis):
    if dofmap.ndim == 1 and axis == self.ndim-1:
      return Inflate(self.func, Take(dofmap, self.dofmap), length)
    if dofmap.ndim == 0 and dofmap == self.dofmap and length == self.length:
      return diagonalize(self, -1, axis)

  def _derivative(self, var, seen):
    return _inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.ndim-1)

  def _insertaxis(self, axis, length):
    return _inflate(insertaxis(self.func, axis+self.dofmap.ndim-1, length), self.dofmap, self.length, self.ndim-1 if axis == self.ndim else -1)

  def _multiply(self, other):
    return Inflate(Multiply([self.func, Take(other, self.dofmap)]), self.dofmap, self.length)

  def _add(self, other):
    if isinstance(other, Inflate) and self.dofmap == other.dofmap:
      return Inflate(Add([self.func, other.func]), self.dofmap, self.length)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    if axis2 == self.ndim-1:
      func = _take(self.func, self.dofmap, axis1)
      for i in range(self.dofmap.ndim):
        func = _takediag(func, axis1, axis2-i)
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
    else: # kronecker; newindex is all zeros (but of varying length)
      intersection = InsertAxis(self.func, newindex.shape[0])
    if index.ndim:
      swapped = Inflate(intersection, newdofmap, index.size)
      for i in range(index.ndim-1):
        swapped = Unravel(swapped, index.shape[i], util.product(index.shape[i+1:]))
    else: # get; newdofmap is all zeros (but of varying length)
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
    if self.dofmap.isconstant and _isunique(self.dofmap.eval()):
      return Inflate(Sign(self.func), self.dofmap, self.length)

  @property
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

class SwapInflateTake(Evaluable):

  def __init__(self, inflateidx, takeidx):
    self.inflateidx = inflateidx
    self.takeidx = takeidx
    super().__init__(args=[inflateidx, takeidx])

  def __iter__(self):
    shape = ArrayFromTuple(self, index=2, shape=(), dtype=int),
    return (ArrayFromTuple(self, index=index, shape=shape, dtype=int) for index in range(2))

  def evalf(self, inflateidx, takeidx):
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

class Diagonalize(Array):

  __slots__ = 'func'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim == 0:
      raise Exception('cannot diagonalize scalar function')
    self.func = func
    diagonal = Diagonal(func.shape[-1], object())
    super().__init__(args=[func], shape=func._axes[:-1]+(diagonal,diagonal), dtype=func.dtype)

  def _simplified(self):
    if self.shape[-1] == 1:
      return InsertAxis(self.func, 1)
    return self.func._diagonalize(self.ndim-2)

  def evalf(self, arr):
    result = numpy.zeros(arr.shape+(arr.shape[-1],), dtype=arr.dtype, order='F')
    diag = numpy.core.multiarray.c_einsum('...ii->...i', result)
    diag[:] = arr
    return result

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

  def _multiply(self, other):
    return Diagonalize(Multiply([self.func, TakeDiag(other)]))

  def _add(self, other):
    if isinstance(other, Diagonalize):
      return Diagonalize(Add([self.func, other.func]))

  def _sum(self, axis):
    if axis >= self.ndim - 2:
      return self.func
    return Diagonalize(sum(self.func, axis))

  def _insertaxis(self, axis, length):
    return diagonalize(insertaxis(self.func, min(axis, self.ndim-1), length), self.ndim-2+(axis<=self.ndim-2), self.ndim-1+(axis<=self.ndim-1))

  def _takediag(self, axis1, axis2):
    if axis1 == self.ndim-2: # axis2 == self.ndim-1
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

  def _loopsum(self, index, length):
    return Diagonalize(LoopSum(self.func, index, length))

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    assert axis < self.ndim-2
    return [(ind, Diagonalize(f)) for ind, f in self.func._desparsify(axis)]

  @property
  def _assparse(self):
    return tuple((*indices, indices[-1], values) for *indices, values in self.func._assparse)

class Guard(Array):
  'bar all simplifications'

  __slots__ = 'fun',

  @types.apply_annotations
  def __init__(self, fun:asarray):
    self.fun = fun
    super().__init__(args=[fun], shape=fun.shape, dtype=fun.dtype)

  @property
  def isconstant(self):
    return False # avoid simplifications based on fun being constant

  @staticmethod
  def evalf(dat):
    return dat

  def _derivative(self, var, seen):
    return Guard(derivative(self.fun, var, seen))

class TrigNormal(Array):
  'cos, sin'

  __slots__ = 'angle',

  @types.apply_annotations
  def __init__(self, angle:asarray):
    self.angle = angle
    super().__init__(args=[angle], shape=(*angle.shape, 2), dtype=float)

  def _derivative(self, var, seen):
    return appendaxes(TrigTangent(self.angle), var.shape) * insertaxis(derivative(self.angle, var, seen), self.ndim-1, 2)

  def evalf(self, angle):
    return numpy.stack([numpy.cos(angle), numpy.sin(angle)], axis=self.ndim-1)

  def _simplified(self):
    if iszero(self.angle):
      return prependaxes(Inflate(1, 0, 2), self.shape[:1])

class TrigTangent(Array):
  '-sin, cos'

  __slots__ = 'angle',

  @types.apply_annotations
  def __init__(self, angle:asarray):
    self.angle = angle
    super().__init__(args=[angle], shape=(*angle.shape, 2), dtype=float)

  def _derivative(self, var, seen):
    return -appendaxes(TrigNormal(self.angle), var.shape) * insertaxis(derivative(self.angle, var, seen), self.ndim-1, 2)

  def evalf(self, angle):
    return numpy.stack([-numpy.sin(angle), numpy.cos(angle)], axis=self.ndim-1)

  def _simplified(self):
    if iszero(self.angle):
      return prependaxes(Inflate(1, 1, 2), self.shape[:1])

class Find(Array):
  'indices of boolean index vector'

  __slots__ = 'where',

  @types.apply_annotations
  def __init__(self, where:asarray):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    self.where = where
    super().__init__(args=[where], shape=[where.sum()], dtype=int)

  def evalf(self, where):
    return where.nonzero()[0]

  def _simplified(self):
    if self.isconstant:
      return Constant(self.eval())

class DerivativeTargetBase(Array):
  'base class for derivative targets'

  __slots__ = ()

  @property
  def isconstant(self):
    return False

class Argument(DerivativeTargetBase):
  '''Array argument, to be substituted before evaluation.

  The :class:`Argument` is an :class:`Array` with a known shape, but whose
  values are to be defined later, before evaluation, e.g. using
  :func:`replace_arguments`.

  It is possible to take the derivative of an :class:`Array` to an
  :class:`Argument`:

  >>> from nutils import evaluable
  >>> a = evaluable.Argument('x', [])
  >>> b = evaluable.Argument('y', [])
  >>> f = a**3 + b**2
  >>> evaluable.derivative(f, a).simplified == (3*a**2).simplified
  True

  Args
  ----
  name : :class:`str`
      The Identifier of this argument.
  shape : :class:`tuple` of :class:`int`\\s
      The shape of this argument.
  '''

  __slots__ = '_name'

  @types.apply_annotations
  def __init__(self, name:types.strictstr, shape:asshape, dtype=float):
    self._name = name
    super().__init__(args=[EVALARGS], shape=shape, dtype=dtype)

  def evalf(self, evalargs):
    try:
      value = evalargs[self._name]
    except KeyError:
      raise ValueError('argument {!r} missing'.format(self._name))
    else:
      value = numpy.asarray(value)
      assert equalshape(value.shape, self.shape)
      value = value.astype(self.dtype, casting='safe', copy=False)
      return value

  def _derivative(self, var, seen):
    if isinstance(var, Argument) and var._name == self._name:
      result = _inflate_scalar(1., self.shape)
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

class LocalCoords(DerivativeTargetBase):
  'local coords derivative target'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__(args=[], shape=[ndims], dtype=float)

  def evalf(self):
    raise Exception('LocalCoords should not be evaluated')

class Ravel(Array):

  __slots__ = 'func'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('cannot ravel function of dimension < 2')
    self.func = func
    ax1, ax2 = func._axes[-2:]
    axisprop = Sparse(ax1.length * ax2.length) if isinstance(ax1, Sparse) or isinstance(ax2, Sparse) else Raveled((ax1.length, ax2.length))
    super().__init__(args=[func], shape=func._axes[:-2]+(axisprop,), dtype=func.dtype)

  def _simplified(self):
    if equalindex(self.func.shape[-2], 1):
      return get(self.func, -2, 0)
    if equalindex(self.func.shape[-1], 1):
      return get(self.func, -1, 0)
    if isinstance(self._axes[-1], Sparse):
      return self._resparsify(self.ndim-1)
    if all(isinstance(axis, Inserted) for axis in self._axes[-2:]):
      return InsertAxis(self.func._uninsert(self.ndim)._uninsert(self.ndim-1), self.shape[-1])
    return self.func._ravel(self.ndim-1)

  def evalf(self, f):
    return f.reshape(f.shape[:-2] + (f.shape[-2]*f.shape[-1],))

  def _multiply(self, other):
    if isinstance(other, Ravel) and equalshape(other.func.shape[-2:], self.func.shape[-2:]):
      return Ravel(Multiply([self.func, other.func]))
    return Ravel(Multiply([self.func, Unravel(other, *self.func.shape[-2:])]))

  def _add(self, other):
    if isinstance(other, Ravel) and equalshape(other.func.shape[-2:], self.func.shape[-2:]):
      return Ravel(Add([self.func, other.func]))

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

  def _take(self, index, axis):
    if axis != self.ndim-1:
      return Ravel(_take(self.func, index, axis))

  def _unravel(self, axis, shape):
    if axis != self.ndim-1:
      return Ravel(unravel(self.func, axis, shape))
    elif equalshape(shape, self.func.shape[-2:]):
      return self.func

  def _inflate(self, dofmap, length, axis):
    if axis < self.ndim-dofmap.ndim:
      return Ravel(_inflate(self.func, dofmap, length, axis))

  def _diagonalize(self, axis):
    if axis != self.ndim-1:
      return ravel(diagonalize(self.func, axis), self.ndim-1)

  def _insertaxis(self, axis, length):
    return ravel(insertaxis(self.func, axis+(axis==self.ndim), length), self.ndim-(axis==self.ndim))

  def _power(self, n):
    return Ravel(Power(self.func, Unravel(n, *self.func.shape[-2:])))

  def _sign(self):
    return Ravel(Sign(self.func))

  def _product(self):
    return Product(Product(self.func))

  def _loopsum(self, index, length):
    return Ravel(LoopSum(self.func, index, length))

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    if axis != self.ndim-1:
      return [(ind, Ravel(f)) for ind, f in self.func._desparsify(axis)]
    if isinstance(self.func._axes[-2], Sparse):
      items = [(ind1, Range(self.func.shape[-1]), f) for ind1, f in self.func._desparsify(self.ndim-1)]
    else:
      assert isinstance(self.func._axes[-1], Sparse)
      items = [(Range(self.func.shape[-2]), ind2, f) for ind2, f in self.func._desparsify(self.ndim)]
    return [(appendaxes(ind1, ind2.shape) * self.func.shape[-1] + prependaxes(ind2, ind1.shape), f) for ind1, ind2, f in items]

  @property
  def _assparse(self):
    return tuple((*indices[:-2], indices[-2]*self.func.shape[-1]+indices[-1], values) for *indices, values in self.func._assparse)

class Unravel(Array):

  __slots__ = 'func'

  @types.apply_annotations
  def __init__(self, func:asarray, sh1:asindex, sh2:asindex):
    if func.ndim == 0:
      raise Exception('cannot unravel scalar function')
    if not equalindex(func.shape[-1], sh1 * sh2):
      raise Exception('new shape does not match axis length')
    self.func = func
    super().__init__(args=[func, sh1, sh2], shape=func._axes[:-1]+(sh1, sh2), dtype=func.dtype)

  def _simplified(self):
    if equalindex(self.shape[-2], 1):
      return insertaxis(self.func, self.ndim-2, 1)
    if equalindex(self.shape[-1], 1):
      return insertaxis(self.func, self.ndim-1, 1)
    return self.func._unravel(self.ndim-2, self.shape[-2:])

  def _derivative(self, var, seen):
    return unravel(derivative(self.func, var, seen), axis=self.ndim-2, shape=self.shape[-2:])

  def evalf(self, f, sh1, sh2):
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

  def _desparsify(self, axis):
    assert isinstance(self._axes[axis], Sparse)
    assert axis < self.ndim-2
    return [(ind, Unravel(f, *self.shape[-2:])) for ind, f in self.func._desparsify(axis)]

  @property
  def _assparse(self):
    return tuple((*indices[:-1], *divmod(indices[-1], appendaxes(self.shape[-1], values.shape)), values) for *indices, values in self.func._assparse)

class Range(Array):

  __slots__ = 'length', 'offset'

  @types.apply_annotations
  def __init__(self, length:asindex, offset:asindex=Zeros((), int)):
    self.length = length
    self.offset = offset
    super().__init__(args=[length, offset], shape=[length], dtype=int)

  def _take(self, index, axis):
    if index.ndim == 1 and isinstance(index, Range) and self.isconstant and index.isconstant:
      assert (index.offset + index.length).eval() <= self.length.eval()
      return Range(index.length, self.offset + index.offset)
    return InRange(index, self.length) + self.offset

  def _add(self, offset):
    if isinstance(self._axes[0], Inserted):
      return Range(self.length, self.offset + offset._uninsert(0))

  def evalf(self, length, offset):
    return numpy.arange(offset, offset+length)

class InRange(Array):

  __slots__ = 'index', 'length'

  @types.apply_annotations
  def __init__(self, index:asarray, length:asarray):
    self.index = index
    self.length = length
    super().__init__(args=[index, length], shape=index.shape, dtype=int)

  def evalf(self, index, length):
    assert index.size == 0 or 0 <= index.min() and index.max() < length
    return index

class Polyval(Array):
  '''
  Computes the :math:`k`-dimensional array

  .. math:: j_0,\\dots,j_{k-1} \\mapsto \\sum_{\\substack{i_0,\\dots,i_{n-1}\\in\\mathbb{N}\\\\i_0+\\cdots+i_{n-1}\\le d}} p_0^{i_0} \\cdots p_{n-1}^{i_{n-1}} c_{j_0,\\dots,j_{k-1},i_0,\\dots,i_{n-1}},

  where :math:`p` are the :math:`n`-dimensional local coordinates and :math:`c`
  is the argument ``coeffs`` and :math:`d` is the degree of the polynomial,
  where :math:`d` is the length of the last :math:`n` axes of ``coeffs``.

  .. warning::

     All coefficients with a (combined) degree larger than :math:`d` should be
     zero.  Failing to do so won't raise an :class:`Exception`, but might give
     incorrect results.
  '''

  __slots__ = 'points_ndim', 'coeffs', 'points', 'ngrad'

  @types.apply_annotations
  def __init__(self, coeffs:asarray, points:asarray, ngrad:types.strictint=0):
    if points.ndim < 1:
      raise ValueError('argument `points` should have at least one axis')
    if not points.shape[-1].isconstant:
      raise ValueError('the last axis of argument `points` should be a constant integer')
    self.points_ndim = int(points.shape[-1])
    ndim = coeffs.ndim - self.points_ndim
    if ndim < 0:
      raise ValueError('argument `coeffs` should have at least one axis per spatial dimension')
    self.coeffs = coeffs
    self.points = points
    self.ngrad = ngrad
    super().__init__(args=[points, coeffs], shape=points.shape[:-1]+coeffs.shape[:ndim]+(self.points_ndim,)*ngrad, dtype=float)

  def evalf(self, points, coeffs):
    for igrad in range(self.ngrad):
      coeffs = numeric.poly_grad(coeffs, self.points_ndim)
    return numeric.poly_eval(coeffs, points)

  def _derivative(self, var, seen):
    # Derivative to argument `points`.
    dpoints = dot(
      appendaxes(Polyval(self.coeffs, self.points, self.ngrad+1), var.shape),
      derivative(Transpose.to_end(appendaxes(self.points, self.shape[self.points.ndim-1:]), self.points.ndim-1), var, seen),
      self.ndim)
    # Derivative to argument `coeffs`.  `trans` shuffles the coefficient axes
    # of `derivative(self.coeffs)` after the derivative axes.
    dcoeffs = Transpose.from_end(Polyval(Transpose.to_end(derivative(self.coeffs, var, seen), *range(self.coeffs.ndim)), self.points, self.ngrad), *range(self.points.ndim-1, self.ndim))
    return dpoints + dcoeffs

  def _take(self, index, axis):
    if axis < self.points.ndim - 1:
      return Polyval(self.coeffs, _take(self.points, index, axis), self.ngrad)
    elif axis < self.points.ndim - 1 + self.coeffs.ndim - self.points_ndim:
      return Polyval(_take(self.coeffs, index, axis - self.points.ndim + 1), self.points, self.ngrad)

  def _const_helper(self, *j):
    if len(j) == self.ngrad:
      coeffs = self.coeffs
      for i in reversed(range(self.points_ndim)):
        p = builtins.sum(k==i for k in j)
        coeffs = math.factorial(p)*get(coeffs, i+self.coeffs.ndim-self.points_ndim, p)
      return coeffs
    else:
      return stack([self._const_helper(*j, k) for k in range(self.points_ndim)], axis=self.coeffs.ndim-self.points_ndim+self.ngrad-len(j)-1)

  def _simplified(self):
    degree = 0 if self.points_ndim == 0 else self.coeffs.shape[-1]-1 if isinstance(self.coeffs.shape[-1], int) else float('inf')
    if iszero(self.coeffs) or self.ngrad > degree:
      return zeros_like(self)
    elif self.ngrad == degree:
      return prependaxes(self._const_helper(), self.points.shape[:-1])

class PolyOuterProduct(Array):

  def __init__(self, left, right):
    nleft = left.shape[1]
    assert all(n == nleft for n in left.shape[2:])
    nright = right.shape[1]
    assert all(n == nright for n in right.shape[2:])
    shape = (left.shape[0] * right.shape[0],) + (nleft + nright - 1,) * (left.ndim + right.ndim - 2)
    super().__init__(args=[left, right], shape=shape, dtype=float)

  def evalf(self, left, right):
    return numeric.poly_outer_product(left, right)

class AssertEqual(Array):

  def __init__(self, *args):
    self._args = args
    assert len(set(arg.shape for arg in args)) == 1
    assert len(set(arg.dtype for arg in args)) == 1
    super().__init__(self._args, shape=args[0].shape, dtype=args[0].dtype)

  def evalf(self, *args):
    arg0 = args[0]
    for arg in args[1:]:
      numpy.testing.assert_array_equal(arg, arg0)
    return arg0

  def _simplified(self):
    if len(set(self._args)) == 1:
      return self._args[0]

class RevolutionAngle(Array):
  '''
  Pseudo coordinates of a :class:`nutils.topology.RevolutionTopology`.
  '''

  __slots__ = ()

  def __init__(self):
    super().__init__(args=[EVALARGS], shape=[], dtype=float)

  def evalf(self, evalargs):
    raise Exception('RevolutionAngle should not be evaluated')

  def _derivative(self, var, seen):
    return (ones_like if isinstance(var, LocalCoords) and len(var) > 0 else zeros_like)(var)

  def _optimized_for_numpy(self):
    return Zeros(self.shape, float)

class Choose(Array):
  '''Function equivalent of :func:`numpy.choose`.'''

  @types.apply_annotations
  def __init__(self, index:asarray, choices:asarrays):
    if index.dtype != int:
      raise Exception('index must be integer valued')
    dtype = _jointdtype(*[choice.dtype for choice in choices])
    shape = index.shape
    if not all(equalshape(choice.shape, shape) for choice in choices):
      raise Exception('shapes vary')
    self.index = index
    self.choices = choices
    super().__init__(args=(index,)+choices, shape=shape, dtype=dtype)

  def evalf(self, index, *choices):
    return numpy.choose(index, choices)

  def _derivative(self, var, seen):
    return Choose(appendaxes(self.index, var.shape), [derivative(choice, var, seen) for choice in self.choices])

  def _simplified(self):
    if all(choice == self.choices[0] for choice in self.choices[1:]):
      return self.choices[0]

  def _multiply(self, other):
    if isinstance(other, Choose) and self.index == other.index:
      return Choose(self.index, map(multiply, self.choices, other.choices))

  def _get(self, i, item):
    return Choose(get(self.index, i, item), [get(choice, i, item) for choice in self.choices])

  def _sum(self, axis):
    if isinstance(self.index._axes[axis], Inserted):
      return Choose(self.index._uninsert(axis), [sum(choice, axis) for choice in self.choices])

  def _take(self, index, axis):
    return Choose(_take(self.index, index, axis), [_take(choice, index, axis) for choice in self.choices])

  def _takediag(self, axis, rmaxis):
    return Choose(takediag(self.index, axis, rmaxis), [takediag(choice, axis, rmaxis) for choice in self.choices])

  def _product(self):
    axis = self.index.ndim-1
    if isinstance(self.index._axes[axis], Inserted):
      return Choose(self.index._uninsert(axis), [Product(choice) for choice in self.choices])

class NormDim(Array):

  @types.apply_annotations
  def __init__(self, length: asarray, index: asarray):
    assert length.dtype == int
    assert index.dtype == int
    self.length = length
    self.index = index
    super().__init__(args=[length, index], shape=index.shape, dtype=index.dtype)

  def evalf(self, length, index):
    assert length.shape == index.shape
    assert length.dtype.kind == 'i'
    assert index.dtype.kind == 'i'
    result = numpy.empty(index.shape, dtype=int)
    for i in numpy.ndindex(index.shape):
      result[i] = numeric.normdim(length[i], index[i])
    return result

  def _simplified(self):
    if self.length.isconstant and self.index.isconstant:
      return Constant(self.eval())

class LoopSum(Array):

  __cache__ = '_serialized'

  def prepare_funcdata(arg):
    # separate shape from array to make it simplifiable (annotations are
    # treated as preprocessor, which means the processed value is returned by
    # self.__reduce__)
    if isinstance(arg, tuple):
      return arg
    arg = asarray(arg)
    return (arg, *arg.shape)

  @types.apply_annotations
  def __init__(self, funcdata:prepare_funcdata, index:types.strict[Argument], length:asindex):
    shape = Tuple(funcdata[1:])
    if index.dtype != int or index.ndim != 0:
      raise ValueError('expected an index with dtype int and dimension zero but got {}'.format(index))
    if index in shape.arguments:
      raise ValueError('the shape of the function must not depend on the index')
    if index in length.arguments:
      raise ValueError('the length of the loop must not depend on the index')
    self.func = funcdata[0]
    self.index = index
    self.length = length
    self._invariants, self._dependencies = _dependencies_sans_invariants(self.func, index)
    axes = tuple(Axis(axis.length) if isinstance(axis, Sparse) else axis for axis in self.func._axes)
    super().__init__(args=(shape, length, *self._invariants), shape=axes, dtype=self.func.dtype)

  @property
  def _serialized(self):
    indices = {d: i for i, d in enumerate(itertools.chain([self.index], self._invariants, self._dependencies))}
    return tuple((dep, tuple(map(indices.__getitem__, dep._Evaluable__args))) for dep in self._dependencies)

  def evalf(self, shape, length, *args):
    serialized = self._serialized
    result = numpy.zeros(shape, self.dtype)
    for index in range(length):
      values = [numpy.array(index)]
      values.extend(args)
      values.extend(op.evalf(*[values[i] for i in indices]) for op, indices in serialized)
      result += values[-1]
    return result

  def evalf_withtimes(self, times, shape, length, *args):
    serialized = self._serialized
    times[self] = subtimes = collections.defaultdict(_Stats)
    result = numpy.zeros(shape, self.dtype)
    for index in range(length):
      values = [numpy.array(index)]
      values.extend(args)
      values.extend(op.evalf_withtimes(subtimes, *[values[i] for i in indices]) for op, indices in serialized)
      result += values[-1]
    return result

  def _derivative(self, var, seen):
    return LoopSum(derivative(self.func, var, seen), self.index, self.length)

  def _node(self, cache, subgraph, times):
    if self in cache:
      return cache[self]
    subcache = {}
    for arg in self._Evaluable__args:
      subcache[arg] = arg._node(cache, subgraph, times)
    loopgraph = Subgraph('Loop', subgraph)
    subcache[self.index] = RegularNode('LoopIndex', (), dict(length=self.length._node(cache, subgraph, times)), (type(self).__name__, _Stats()), loopgraph)
    subtimes = times.get(self, collections.defaultdict(_Stats))
    sum_kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times) for i, n in enumerate(self.shape)}
    sum_kwargs['func'] = self.func._node(subcache, loopgraph, subtimes)
    cache[self] = node = RegularNode('LoopSum', (), sum_kwargs, (type(self).__name__, subtimes['sum']), loopgraph)
    return node

  def _simplified(self):
    if iszero(self.func):
      return zeros_like(self)
    elif self.index not in self.func.arguments:
      return self.func * self.length
    return self.func._loopsum(self.index, self.length)

  def _takediag(self, axis1, axis2):
    return LoopSum(_takediag(self.func, axis1, axis2), self.index, self.length)

  def _take(self, index, axis):
    return LoopSum(_take(self.func, index, axis), self.index, self.length)

  def _unravel(self, axis, shape):
    return LoopSum(unravel(self.func, axis, shape), self.index, self.length)

  def _sum(self, axis):
    return LoopSum(sum(self.func, axis), self.index, self.length)

  def _add(self, other):
    if isinstance(other, LoopSum) and other.length == self.length and other.index == self.index:
      return LoopSum(self.func + other.func, self.index, self.length)

  @property
  def _assparse(self):
    chunks = []
    for *elem_indices, elem_values in self.func._assparse:
      if self.ndim == 0:
        values = loop_concatenate(InsertAxis(elem_values, 1), self.index, self.length)
        while values.ndim:
          values = Sum(values)
        chunks.append((values,))
      else:
        if elem_values.ndim == 0:
          *elem_indices, elem_values = (InsertAxis(arr, 1) for arr in (*elem_indices, elem_values))
        else:
          # minimize ravels by transposing all variable length axes to the end
          variable = tuple(i for i, n in enumerate(elem_values.shape) if not numeric.isint(n) and self.index in n.arguments)
          *elem_indices, elem_values = (Transpose.to_end(arr, *variable) for arr in (*elem_indices, elem_values))
          for i in variable[:-1]:
            *elem_indices, elem_values = map(Ravel, (*elem_indices, elem_values))
          assert all(n.isconstant for n in elem_values.shape[:-1])
        chunks.append(tuple(loop_concatenate(arr, self.index, self.length) for arr in (*elem_indices, elem_values)))
    return tuple(chunks)

class _SizesToOffsets(Array):

  def __init__(self, sizes):
    assert sizes.ndim == 1
    assert sizes.dtype == int
    self._sizes = sizes
    super().__init__(args=[sizes], shape=(sizes.shape[0]+1,), dtype=int)

  def evalf(self, sizes):
    return numpy.cumsum([0, *sizes])

  def _simplified(self):
    if isinstance(self._sizes._axes[0], Inserted):
      return Range(self.shape[0]) * appendaxes(self._sizes._uninsert(0), self.shape[:1])

class LoopConcatenate(Array):

  @types.apply_annotations
  def __init__(self, funcdata:asarrays, index:types.strict[Argument], length:asindex):
    self.funcdata = funcdata
    self.func, self.start, stop, *shape = funcdata
    if not self.func.ndim:
      raise ValueError('expected an array with at least one axis')
    if any(index in n.arguments for n in shape):
      raise ValueError('the shape of the function must not depend on the index')
    if index in length.arguments:
      raise ValueError('the length of the loop must not depend on the index')
    self.index = index
    self.length = length
    self._lcc = LoopConcatenateCombined((self.funcdata,), index, length)
    axes = (*(Axis(axis.length) if isinstance(axis, Sparse) else axis for axis in self.func._axes[:-1]), Axis(shape[-1]))
    super().__init__(args=[self._lcc], shape=axes, dtype=self.func.dtype)

  def evalf(self, arg):
    return arg[0]

  def evalf_withtimes(self, times, arg):
    with times[self]:
      return arg[0]

  def _derivative(self, var, seen):
    return Transpose.from_end(loop_concatenate(Transpose.to_end(derivative(self.func, var, seen), self.ndim-1), self.index, self.length), self.ndim-1)

  def _node(self, cache, subgraph, times):
    if self in cache:
      return cache[self]
    else:
      cache[self] = node = self._lcc._node_tuple(cache, subgraph, times)[0]
      return node

  def _simplified(self):
    if iszero(self.func):
      return zeros_like(self)
    elif self.index not in self.func.arguments:
      return Ravel(Transpose.from_end(InsertAxis(self.func, self.length), -2))
    for iaxis, axis in enumerate(self.func._axes[:-1]):
      if isinstance(axis, Inserted) and self.index not in axis.length.arguments:
        return insertaxis(loop_concatenate(self.func._uninsert(iaxis), self.index, self.length), iaxis, axis.length)
    if isinstance(self.func._axes[-1], Inserted) and self.index not in self.func.shape[-1].arguments and not equalindex(self.func.shape[-1], 1):
      return Ravel(InsertAxis(loop_concatenate(InsertAxis(self.func._uninsert(self.func.ndim-1), 1), self.index, self.length), self.func.shape[-1]))

  def _takediag(self, axis1, axis2):
    if axis1 < self.ndim-1 and axis2 < self.ndim-1:
      return Transpose.from_end(loop_concatenate(Transpose.to_end(_takediag(self.func, axis1, axis2), -2), self.index, self.length), -2)

  def _take(self, index, axis):
    if axis < self.ndim-1:
      return loop_concatenate(_take(self.func, index, axis), self.index, self.length)

  def _unravel(self, axis, shape):
    if axis < self.ndim-1:
      return loop_concatenate(unravel(self.func, axis, shape), self.index, self.length)

  @property
  def _assparse(self):
    chunks = []
    for *indices, last_index, values in self.func._assparse:
      last_index = last_index + prependaxes(self.start, last_index.shape)
      chunks.append(tuple(loop_concatenate(_flat(arr), self.index, self.length) for arr in (*indices, last_index, values)))
    return tuple(chunks)

  @property
  def _loop_concatenate_deps(self):
    return (self,) + super()._loop_concatenate_deps

class LoopConcatenateCombined(Evaluable):

  __cache__ = '_serialized'

  @types.apply_annotations
  def __init__(self, funcdatas:types.tuple[asarrays], index:types.strict[Argument], length:asindex):
    self._funcdatas = funcdatas
    if index.dtype != int or index.ndim != 0:
      raise ValueError('expected an index with dtype int and dimension zero but got {}'.format(index))
    self._funcs = tuple(func for func, start, stop, *shape in funcdatas)
    if any(not func.ndim for func in self._funcs):
      raise ValueError('expected an array with at least one axis')
    shapes = [Tuple(shape) for func, start, stop, *shape in funcdatas]
    if any(index in shape.arguments for shape in shapes):
      raise ValueError('the shape of the function must not depend on the index')
    if index in length.arguments:
      raise ValueError('the length of the loop must not depend on the index')
    self._index = index
    self._length = length
    self._invariants, self._dependencies = _dependencies_sans_invariants(
      Tuple([Tuple([start, stop, func]) for func, start, stop, *shape in funcdatas]), index)
    super().__init__(args=(Tuple(shapes), length, *self._invariants))

  @property
  def _serialized(self):
    indices = {d: i for i, d in enumerate(itertools.chain([self._index], self._invariants, self._dependencies))}
    return tuple((dep, tuple(map(indices.__getitem__, dep._Evaluable__args))) for dep in self._dependencies)

  def evalf(self, shapes, length, *args):
    serialized = self._serialized
    results = [parallel.shempty(tuple(map(int, shape)), dtype=func.dtype) for func, shape in zip(self._funcs, shapes)]
    with parallel.ctxrange('loop', int(length)) as indices:
      for index in indices:
        values = [numpy.array(index)]
        values.extend(args)
        values.extend(op.evalf(*[values[i] for i in indices]) for op, indices in serialized)
        for result, (start, stop, block) in zip(results, values[-1]):
          result[...,start:stop] = block
    return tuple(results)

  def evalf_withtimes(self, times, shapes, length, *args):
    serialized = self._serialized
    times[self] = subtimes = collections.defaultdict(_Stats)
    results = [parallel.shempty(tuple(map(int, shape)), dtype=func.dtype) for func, shape in zip(self._funcs, shapes)]
    for index in range(length):
      values = [numpy.array(index)]
      values.extend(args)
      values.extend(op.evalf_withtimes(subtimes, *[values[i] for i in indices]) for op, indices in serialized)
      for func, result, (start, stop, block) in zip(self._funcs, results, values[-1]):
        with subtimes['concat', func]:
          result[...,start:stop] = block
    return tuple(results)

  def _node_tuple(self, cache, subgraph, times):
    if (self, 'tuple') in cache:
      return cache[self, 'tuple']
    subcache = {}
    for arg in self._invariants:
      subcache[arg] = arg._node(cache, subgraph, times)
    loopgraph = Subgraph('Loop', subgraph)
    subcache[self._index] = RegularNode('LoopIndex', (), dict(length=self._length._node(cache, subgraph, times)), (type(self).__name__, _Stats()), loopgraph)
    subtimes = times.get(self, collections.defaultdict(_Stats))
    concats = []
    for func, start, stop, *shape in self._funcdatas:
      concat_kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times) for i, n in enumerate(shape)}
      concat_kwargs['start'] = start._node(subcache, loopgraph, subtimes)
      concat_kwargs['stop'] = stop._node(subcache, loopgraph, subtimes)
      concat_kwargs['func'] = func._node(subcache, loopgraph, subtimes)
      concats.append(RegularNode('LoopConcatenate', (), concat_kwargs, (type(self).__name__, subtimes['concat', func]), loopgraph))
    cache[self, 'tuple'] = concats = tuple(concats)
    return concats

# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_ascending = lambda arg: numpy.greater(numpy.diff(arg), 0).all()
_normdims = lambda ndim, shapes: tuple(numeric.normdim(ndim,sh) for sh in shapes)

def _jointdtype(*dtypes):
  'determine joint dtype'

  type_order = bool, int, float, complex
  kind_order = 'bifc'
  itype = max(kind_order.index(dtype.kind) if isinstance(dtype,numpy.dtype)
           else type_order.index(dtype) for dtype in dtypes)
  return type_order[itype]

def _gatherblocks(blocks):
  return tuple((ind, util.sum(funcs)) for ind, funcs in util.gather(blocks))

def _gathersparsechunks(chunks):
  return tuple((*ind, util.sum(funcs)) for ind, funcs in util.gather((tuple(ind), func) for *ind, func in chunks))

def _numpy_align(*arrays):
  '''reshape arrays according to Numpy's broadcast conventions'''

  # NOTE: this routine largely repeats the logic of equalshape to reduce the
  # number of comparisons. Ultimately _numpy_align will be removed from
  # evaluable so this situation is temporary.

  arrays = [asarray(array) for array in arrays]
  if len(arrays) > 1:
    ndim = max([array.ndim for array in arrays])
    for idim in range(ndim):
      lengths = {}
      insert = []
      for iarray, array in enumerate(arrays):
        if array.ndim < ndim:
          insert.append(iarray) # mark array for axis insertion
        else:
          lengths.setdefault(array.shape[idim], []).append(iarray)
      if len(lengths) > 1: # shapes differ -> simplify to canonical form
        for length in list(lengths):
          simple = length.simplified
          if length is not simple:
            lengths.setdefault(simple, []).extend(lengths.pop(length))
      expand = () if len(lengths) == 1 \
          else lengths.pop(Constant(1), ()) # NOTE: this uses the fact that Constant(1) is simple
      assert len(lengths) == 1 or all(l.isconstant for l in lengths) and len(set(map(int, lengths))) == 1, \
        'incompatible shapes: {}'.format(' != '.join(str(l) for l in lengths))
      length, _ = lengths.popitem()
      for i in insert:
        arrays[i] = insertaxis(arrays[i], idim, length)
      for i in expand:
        arrays[i] = repeat(arrays[i], length, idim)
  return arrays

def _inflate_scalar(arg, shape):
  arg = asarray(arg)
  assert arg.ndim == 0
  for idim, length in enumerate(shape):
    arg = insertaxis(arg, idim, length)
  return arg

def _isunique(array):
  return numpy.unique(array).size == array.size

def _dependencies_sans_invariants(func, arg):
  invariants = []
  dependencies = []
  _populate_dependencies_sans_invariants(func, arg, invariants, dependencies, {arg})
  assert (dependencies or invariants or [arg])[-1] == func
  return tuple(invariants), tuple(dependencies)

def _populate_dependencies_sans_invariants(func, arg, invariants, dependencies, cache):
  if func in cache:
    return
  cache.add(func)
  if arg in func.arguments:
    for child in func._Evaluable__args:
      _populate_dependencies_sans_invariants(child, arg, invariants, dependencies, cache)
    dependencies.append(func)
  else:
    invariants.append(func)

class _Stats:

  __slots__ = 'ncalls', 'time', '_start'

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

def iszero(arg):
  return isinstance(arg.simplified, Zeros)

def zeros(shape, dtype=float):
  return Zeros(shape, dtype)

def zeros_like(arr):
  return zeros(arr.shape, arr.dtype)

def isuniform(arg, value):
  while isinstance(arg, (InsertAxis, Transpose)):
    arg = arg.func
  if isinstance(arg, Constant) and arg.ndim == 0:
    return arg.value[()] == value
  else:
    return False

def ones(shape, dtype=float):
  return _inflate_scalar(numpy.ones((), dtype=dtype), shape)

def ones_like(arr):
  return ones(arr.shape, arr.dtype)

def reciprocal(arg):
  return power(arg, -1)

def negative(arg):
  return multiply(arg, -1)

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

def exp(x):
  return Exp(x)

def ln(x):
  return Log(x)

def divmod(x, y):
  div = FloorDivide(*_numpy_align(x, y))
  mod = x - div * y
  return div, mod

def mod(arg1, arg2):
  return Mod(*_numpy_align(arg1, arg2))

def log2(arg):
  return ln(arg) / ln(2)

def log10(arg):
  return ln(arg) / ln(10)

def sqrt(arg):
  return power(arg, .5)

def arctan2(arg1, arg2):
  return ArcTan2(*_numpy_align(arg1, arg2))

def abs(arg):
  return arg * sign(arg)

def sinh(arg):
  return .5 * (exp(arg) - exp(-arg))

def cosh(arg):
  return .5 * (exp(arg) + exp(-arg))

def tanh(arg):
  return 1 - 2. / (exp(2*arg) + 1)

def arctanh(arg):
  return .5 * (ln(1+arg) - ln(1-arg))

def divide(arg1, arg2):
  return multiply(arg1, reciprocal(arg2))

def subtract(arg1, arg2):
  return add(arg1, negative(arg2))

def blocks(arg):
  return asarray(arg).simplified.blocks

@replace
def _bifurcate(arg, side):
  if isinstance(arg, (SelectChain, TransformChainFromSequence)):
    return SelectBifurcation(arg, side)

bifurcate1 = functools.partial(_bifurcate, side=True)
bifurcate2 = functools.partial(_bifurcate, side=False)

def insertaxis(arg, n, length):
  return Transpose.from_end(InsertAxis(arg, length), n)

def stack(args, axis=0):
  aligned = _numpy_align(*args)
  return Transpose.from_end(util.sum(Inflate(arg, i, len(args)) for i, arg in enumerate(args)), axis)

def repeat(arg, length, axis):
  arg = asarray(arg)
  assert equalindex(arg.shape[axis], 1)
  return insertaxis(get(arg, axis, 0), axis, length)

def get(arg, iax, item):
  if numeric.isint(item):
    if numeric.isint(arg.shape[iax]):
      item = numeric.normdim(arg.shape[iax], item)
    else:
      assert item >= 0
  return Take(Transpose.to_end(arg, iax), item)

def jacobian(geom, ndims):
  '''
  Return :math:`\\sqrt{|J^T J|}` with :math:`J` the gradient of ``geom`` to the
  local coordinate system with ``ndims`` dimensions (``localgradient(geom,
  ndims)``).
  '''

  assert geom.ndim >= 1
  J = localgradient(geom, ndims)
  cndims = int(geom.shape[-1])
  assert int(J.shape[-2]) == cndims and int(J.shape[-1]) == ndims, 'wrong jacobian shape: got {}, expected {}'.format((int(J.shape[-2]), int(J.shape[-1])), (cndims, ndims))
  assert cndims >= ndims, 'geometry dimension < topology dimension'
  detJ = abs(determinant(J)) if cndims == ndims \
    else ones(J.shape[:-2]) if ndims == 0 \
    else abs(determinant((J[...,:,:,_] * J[...,:,_,:]).sum(-3)))**.5
  return detJ

def determinant(arg, axes=(-2,-1)):
  return Determinant(Transpose.to_end(arg, *axes))

def inverse(arg, axes=(-2,-1)):
  return Transpose.from_end(Inverse(Transpose.to_end(arg, *axes)), *axes)

def takediag(arg, axis=-2, rmaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  rmaxis = numeric.normdim(arg.ndim, rmaxis)
  assert axis < rmaxis
  return Transpose.from_end(_takediag(arg, axis, rmaxis), axis)

def _takediag(arg, axis1=-2, axis2=-1):
  return TakeDiag(Transpose.to_end(arg, axis1, axis2))

def derivative(func, var, seen=None):
  'derivative'

  assert isinstance(var, DerivativeTargetBase), 'invalid derivative target {!r}'.format(var)
  if seen is None:
    seen = {}
  func = asarray(func)
  if func in seen:
    result = seen[func]
  else:
    result = func._derivative(var, seen)
    seen[func] = result
  assert equalshape(result.shape, func.shape+var.shape), 'bug in {}._derivative'.format(type(func).__name__)
  return result

def localgradient(arg, ndims):
  'local derivative'

  return derivative(arg, LocalCoords(ndims))

def diagonalize(arg, axis=-1, newaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  newaxis = numeric.normdim(arg.ndim+1, newaxis)
  assert axis < newaxis
  return Transpose.from_end(Diagonalize(Transpose.to_end(arg, axis)), axis, newaxis)

def sign(arg):
  arg = asarray(arg)
  return Sign(arg)

def eig(arg, axes=(-2,-1), symmetric=False):
  eigval, eigvec = Eig(Transpose.to_end(arg, *axes), symmetric)
  return Tuple(Transpose.from_end(v, *axes) for v in [diagonalize(eigval), eigvec])

@types.apply_annotations
def _takeslice(arg:asarray, s:types.strict[slice], axis:types.strictint):
  n = arg.shape[axis]
  if s.step == None or s.step == 1:
    start = 0 if s.start is None else s.start if s.start >= 0 else s.start + n
    stop = n if s.stop is None else s.stop if s.stop >= 0 else s.stop + n
    if start == 0 and stop == n:
      return arg
    index = Range(stop-start, start)
  elif n.isconstant:
    index = Constant(numpy.arange(*s.indices(arg.shape[axis])))
  else:
    raise Exception('a non-unit slice requires a constant-length axis')
  return take(arg, index, axis)

@types.apply_annotations
def take(arg:asarray, index:asarray, axis:types.strictint):
  assert index.ndim == 1
  length = arg.shape[axis]
  if index.dtype == bool:
    assert equalindex(index.shape[0], length)
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
      return _take(arg, Constant(index_ + ineg * int(length)), axis)
    elif numpy.greater_equal(index_, int(length)).any():
      raise IndexError('indices out of bounds: {} >= {}'.format(index_, int(length)))
    elif numpy.greater(numpy.diff(index_), 0).all():
      return mask(arg, numeric.asboolean(index_, int(length)), axis)
  return _take(arg, index, axis)

@types.apply_annotations
def _take(arg:asarray, index:asarray, axis:types.strictint):
  axis = numeric.normdim(arg.ndim, axis)
  return Transpose.from_end(Take(Transpose.to_end(arg, axis), index), *range(axis, axis+index.ndim))

@types.apply_annotations
def _inflate(arg:asarray, dofmap:asarray, length:asindex, axis:types.strictint):
  axis = numeric.normdim(arg.ndim+1-dofmap.ndim, axis)
  assert equalshape(dofmap.shape, arg.shape[axis:axis+dofmap.ndim])
  return Transpose.from_end(Inflate(Transpose.to_end(arg, *range(axis, axis+dofmap.ndim)), dofmap, length), axis)

def mask(arg, mask, axis=0):
  return take(arg, mask, axis)

def unravel(func, axis, shape):
  func = asarray(func)
  axis = numeric.normdim(func.ndim, axis)
  assert len(shape) == 2
  return Transpose.from_end(Unravel(Transpose.to_end(func, axis), *shape), axis, axis+1)

def ravel(func, axis):
  func = asarray(func)
  axis = numeric.normdim(func.ndim-1, axis)
  return Transpose.from_end(Ravel(Transpose.to_end(func, axis, axis+1)), axis)

def _flat(func):
  func = asarray(func)
  if func.ndim == 0:
    return InsertAxis(func, 1)
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
    func = insertaxis(func, func.ndim, n)
  return func

def _loop_concatenate_data(func, index, length):
  func = asarray(func)
  chunk_size = func.shape[-1]
  if chunk_size.isconstant:
    chunk_sizes = InsertAxis(chunk_size, length)
  else:
    chunk_sizes = loop_concatenate(InsertAxis(func.shape[-1], 1), index, length)
  offsets = _SizesToOffsets(chunk_sizes)
  start = Take(offsets, index)
  stop = Take(offsets, index+1)
  return (func, start, stop, *func.shape[:-1], Take(offsets, length))

def loop_concatenate(func, index, length):
  length = asindex(length)
  funcdata = _loop_concatenate_data(func, index, length)
  return LoopConcatenate(funcdata, index, length)

def loop_concatenate_combined(funcs, index, length):
  length = asindex(length)
  unique_funcs = []
  unique_funcs.extend(func for func in funcs if func not in unique_funcs)
  unique_func_data = tuple(_loop_concatenate_data(func, index, length) for func in unique_funcs)
  loop = LoopConcatenateCombined(unique_func_data, index, length)
  return tuple(ArrayFromTuple(loop, unique_funcs.index(func), shape, func.dtype) for func, start, stop, *shape in unique_func_data)

@replace
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
    assert equalshape(value.shape, v.shape)
    return v

if __name__ == '__main__':
  # Diagnostics for the development for simplify operations.
  simplify_priority = (
    Transpose, Ravel, # reinterpretation
    Inflate, Diagonalize, InsertAxis, # size increasing
    Multiply, Add, LoopSum, Sign, Power, Inverse, Unravel, # size preserving
    Product, Determinant, TakeDiag, Take, Sum) # size decreasing
  # The simplify priority defines the preferred order in which operations are
  # performed: shape decreasing operations such as Sum and Take should be done
  # as soon as possible, and shape increasing operations such as Inflate and
  # Diagonalize as late as possible. In shuffling the order of operations the
  # two classes might annihilate each other, for example when a Sum passes
  # through a Diagonalize. Any shape increasing operations that remain should
  # end up at the surface, exposing sparsity by means of the blocks method.
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

# vim:sw=2:sts=2:et
