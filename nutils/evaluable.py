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

isevaluable = lambda arg: isinstance(arg, Evaluable)

def strictevaluable(value):
  if not isinstance(value, Evaluable):
    raise ValueError('expected an object of type {!r} but got {!r} with type {!r}'.format(Evaluable.__qualname__, value, type(value).__qualname__))
  return value

def simplified(value):
  return strictevaluable(value).simplified

asdtype = lambda arg: arg if any(arg is dtype for dtype in (bool, int, float, complex)) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]

def asarray(arg):
  if hasattr(type(arg), 'as_evaluable_array'):
    return arg.as_evaluable_array
  if _containsarray(arg):
    return stack(arg, axis=0)
  else:
    return Constant(arg)

asarrays = types.tuple[asarray]

def asindex(arg):
  arg = asarray(arg)
  if arg.ndim or arg.dtype not in (int, bool): # NOTE: bool to be removed after introduction of Cast
    raise ValueError('argument is not an index: {}'.format(arg))
  if arg.dtype == bool:
    arg = Int(arg)
  elif arg._intbounds[0] < 0:
    raise ValueError('index must be non-negative')
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

  if N == M:
    return True
  if len(N) != len(M):
    return False
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
        lcs = combine.setdefault(lc.index, [])
        if lc not in lcs:
          lcs.append(lc)
          exclude.update(set(lc._loop_concatenate_deps) - {lc})
      # Combine top-level `LoopConcatenate` instances excluding those in
      # `exclude`.
      replacements = {}
      for index, lcs in combine.items():
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
        combined = LoopConcatenateCombined(data, index._name, index.length)
        for i, lc in enumerate(lcs):
          intbounds = dict(zip(('_lower', '_upper'), lc._intbounds)) if lc.dtype == int else {}
          replacements[lc] = ArrayFromTuple(combined, i, lc.shape, lc.dtype, **intbounds)
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

class EvaluableConstant(Evaluable):
  '''Evaluate to the given constant value.

  Parameters
  ----------
  value
      The return value of ``eval``.
  '''

  __slots__ = 'value'

  def __init__(self, value):
    self.value = value
    super().__init__(())

  def evalf(self):
    return self.value

  @property
  def _node_details(self):
    s = repr(self.value)
    if '\n' in s:
      s = s.split('\n', 1)[0] + '...'
    if len(s) > 20:
      s = s[:17] + '...'
    return s

class Tuple(Evaluable):

  __slots__ = 'items'

  @types.apply_annotations
  def __init__(self, items: types.tuple[strictevaluable]):
    self.items = items
    super().__init__(items)

  def evalf(self, *items):
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
    return ArrayFromTuple(self, index=0, shape=(), dtype=int, _lower=0, _upper=max(0, len(self._transforms)-1))

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
    arg = Transpose(arg, numpy.argsort(where))
  assert equalshape(arg.shape, shape)
  return arg

def unalign(*args):
  '''Remove (joint) inserted axes.

  Given one or more equally shaped array arguments, return the shortest common
  axis vector along with function arguments such that the original arrays can
  be recovered by :func:`align`.
  '''

  assert args
  if len(args) == 1:
    return args[0]._unaligned
  if any(arg.ndim != args[0].ndim for arg in args[1:]):
    raise ValueError('varying dimensions in unalign')
  nonins = functools.reduce(operator.or_, [set(arg._unaligned[1]) for arg in args])
  if len(nonins) == args[0].ndim:
    return (*args, tuple(range(args[0].ndim)))
  ret = []
  for arg in args:
    unaligned, where = arg._unaligned
    for i in sorted(nonins - set(where)):
      unaligned = InsertAxis(unaligned, args[0].shape[i])
      where += i,
    if not ret: # first argument
      commonwhere = where
    elif where != commonwhere:
      unaligned = Transpose(unaligned, map(where.index, commonwhere))
    ret.append(unaligned)
  return (*ret, commonwhere)

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

  __slots__ = 'shape', 'dtype', '__index'
  __cache__ = 'assparse', '_assparse', '_intbounds'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], shape:asshape, dtype:asdtype):
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
      array = get(array, axis, item=it) if numeric.isint(it) \
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
  _shape_str = lambda self, form: '{}:{}'.format(self.dtype.__name__[0] if hasattr(self, 'dtype') else '?', ','.join(str(int(length)) if length.isconstant else '?' for length in self.shape) if hasattr(self, 'shape') else '?')

  sum = sum
  prod = product
  dot = dot
  swapaxes = swapaxes
  transpose = transpose
  choose = lambda self, choices: Choose(self, choices)

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
  _loopsum = lambda self, loop_index: None # NOTE: type of `loop_index` is `_LoopIndex`

  @property
  def _unaligned(self):
    return self, tuple(range(self.ndim))

  _diagonals = ()
  _inflations = ()

  def _derivative(self, var, seen):
    if self.dtype in (bool, int) or var not in self.dependencies:
      return Zeros(self.shape + var.shape, dtype=self.dtype)
    raise NotImplementedError('derivative not defined for {}'.format(self.__class__.__name__))

  @property
  def as_evaluable_array(self):
    'return self'

    return self

  @property
  def _intbounds(self):
    # inclusive lower and upper bounds
    if self.ndim == 0 and self.dtype == int and self.isconstant:
      value = self.__index__()
      return value, value
    else:
      lower, upper = self._intbounds_impl()
      assert isinstance(lower, int) or lower == float('-inf') or lower == float('inf')
      assert isinstance(upper, int) or upper == float('-inf') or upper == float('inf')
      assert lower <= upper
      return lower, upper

  def _intbounds_impl(self):
    return float('-inf'), float('inf')

class NPoints(Array):
  'The length of the points axis.'

  __slots__ = ()

  def __init__(self):
    super().__init__(args=[EVALARGS], shape=(), dtype=int)

  def evalf(self, evalargs):
    points = evalargs['_points'].coords
    return types.frozenarray(points.shape[0])

  def _intbounds_impl(self):
    return 0, float('inf')

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
    unaligned, where = unalign(self.lgrad)
    for axis in self.ndim - 1, self.ndim:
      if axis not in where:
        unaligned = InsertAxis(unaligned, self.lgrad.shape[axis])
        where += axis,
    if len(where) < self.ndim + 1:
      if where[-2:] != (self.ndim - 1, self.ndim):
        unaligned = Transpose(unaligned, numpy.argsort(where))
        where = tuple(sorted(where))
      return align(Normal(unaligned), where[:-1], self.shape)

  def evalf(self, lgrad):
    n = lgrad[...,-1]
    # orthonormalize n to G
    G = lgrad[...,:-1]
    GG = numpy.einsum('...ki,...kj->...ij', G, G)
    v1 = numpy.einsum('...ij,...i->...j', G, n)
    v2 = numpy.linalg.solve(GG, v1)
    v3 = numpy.einsum('...ij,...j->...i', G, v2)
    return numeric.normalize(n - v3)

  def _derivative(self, var, seen):
    if equalindex(self.shape[-1], 1):
      return zeros(self.shape + var.shape)
    G = self.lgrad[...,:-1]
    invGG = inverse(einsum('Aki,Akj->Aij', G, G))
    return -einsum('Ail,Alj,Ak,AkjB->AiB', G, invGG, self, derivative(G, var, seen))

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

  def _intbounds_impl(self):
    if self.dtype == int and self.value.size:
      return int(self.value.min()), int(self.value.max())
    else:
      return super()._intbounds_impl()

class InsertAxis(Array):

  __slots__ = 'func', 'length'
  __cache__ = '_unaligned', '_inflations'

  @types.apply_annotations
  def __init__(self, func:asarray, length:asindex):
    self.func = func
    self.length = length
    super().__init__(args=[func, length], shape=(*func.shape, length), dtype=func.dtype)

  @property
  def _diagonals(self):
    return self.func._diagonals

  @property
  def _inflations(self):
    return tuple((axis, types.frozendict((dofmap, InsertAxis(func, self.length)) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

  @property
  def _unaligned(self):
    return self.func._unaligned

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
    unaligned1, unaligned2, where = unalign(self, n)
    if len(where) != self.ndim:
      return align(unaligned1 ** unaligned2, where, self.shape)

  def _add(self, other):
    unaligned1, unaligned2, where = unalign(self, other)
    if len(where) != self.ndim:
      return align(unaligned1 + unaligned2, where, self.shape)

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

  def _rtake(self, func, axis):
    return insertaxis(_take(func, self.func, axis), axis+self.ndim-1, self.length)

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

  def _loopsum(self, index):
    return InsertAxis(loop_sum(self.func, index), self.length)

  @property
  def _assparse(self):
    return tuple((*(InsertAxis(idx, self.length) for idx in indices), prependaxes(Range(self.length), values.shape), InsertAxis(values, self.length)) for *indices, values in self.func._assparse)

  def _intbounds_impl(self):
    return self.func._intbounds

class Transpose(Array):

  __slots__ = 'func', 'axes'
  __cache__ = '_invaxes', '_unaligned', '_diagonals', '_inflations'

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
    super().__init__(args=[func], shape=[func.shape[n] for n in axes], dtype=func.dtype)

  @property
  def _diagonals(self):
    return tuple(frozenset(self._invaxes[i] for i in axes) for axes in self.func._diagonals)

  @property
  def _inflations(self):
    return tuple((self._invaxes[axis], types.frozendict((dofmap, Transpose(func, self._axes_for(dofmap.ndim, self._invaxes[axis]))) for dofmap, func in parts.items())) for axis, parts in self.func._inflations)

  @property
  def _unaligned(self):
    unaligned, where = unalign(self.func)
    return unaligned, tuple(self._invaxes[i] for i in where)

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
    if axes == self._invaxes:
      # NOTE: While we could leave this particular simplification to be dealt
      # with by Transpose, the benefit of handling it directly is that _add and
      # _multiply can rely on _transpose for the right hand side without having
      # to separately account for the trivial case.
      return self.func
    newaxes = [self.axes[i] for i in axes]
    return Transpose(self.func, newaxes)

  def _takediag(self, axis1, axis2):
    assert axis1 < axis2
    orig1, orig2 = sorted(self.axes[axis] for axis in [axis1, axis2])
    if orig1 == self.ndim-2:
      return Transpose(TakeDiag(self.func), (*self.axes[:axis1], *self.axes[axis1+1:axis2], *self.axes[axis2+1:], self.ndim-2))
    trytakediag = self.func._takediag(orig1, orig2)
    if trytakediag is not None:
      return Transpose(trytakediag, [ax-(ax>orig1)-(ax>orig2) for ax in self.axes[:axis1] + self.axes[axis1+1:axis2] + self.axes[axis2+1:]] + [self.ndim-2])

  def _sum(self, i):
    axis = self.axes[i]
    trysum = self.func._sum(axis)
    if trysum is not None:
      axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
      return Transpose(trysum, axes)
    if axis == self.ndim - 1:
      return Transpose(Sum(self.func), self._axes_for(0, i))

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
      return Transpose(trytake, self._axes_for(indices.ndim, axis))
    if self.axes[axis] == self.ndim - 1:
      return Transpose(Take(self.func, indices), self._axes_for(indices.ndim, axis))

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

  def _loopsum(self, index):
    return Transpose(loop_sum(self.func, index), self.axes)

  @property
  def _assparse(self):
    return tuple((*(indices[i] for i in self.axes), values) for *indices, values in self.func._assparse)

  def _intbounds_impl(self):
    return self.func._intbounds

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
    funcs = Product(insertaxis(self.func, -2, self.func.shape[-1]) + Diagonalize(1 - self.func)) # replace diagonal entries by 1
    return einsum('Ai,AiB->AB', funcs, grad)

  def _take(self, indices, axis):
    return Product(_take(self.func, indices, axis))

  def _takediag(self, axis1, axis2):
    return product(_takediag(self.func, axis1, axis2), self.ndim-2)

class ApplyTransforms(Array):

  __slots__ = 'trans', '_points', '_todims'

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], points, todims:types.strictint):
    self.trans = trans
    self._points = points
    self._todims = todims
    super().__init__(args=[points, trans], shape=points.shape[:-1]+(todims,), dtype=float)

  def evalf(self, points, chain):
    return transform.apply(chain, points)

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords) and len(var) > 0:
      return prependaxes(LinearFrom(self.trans, self._todims, len(var)), self.shape[:-1])
    else:
      return einsum('ij,AjB->AiB', LinearFrom(self.trans, self._todims, self._points.shape[-1].__index__()), derivative(self._points, var, seen), B=var.ndim)

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
    result = self.func._inverse(self.ndim-2, self.ndim-1)
    if result is not None:
      return result
    if equalindex(self.func.shape[-1], 1):
      return reciprocal(self.func)

  def evalf(self, arr):
    return numeric.inv(arr)

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
    result = self.func._determinant(self.ndim, self.ndim+1)
    if result is not None:
      return result
    if equalindex(self.func.shape[-1], 1):
      return Take(Take(self.func, zeros((), int)), zeros((), int))

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.linalg.det(arr)

  def _derivative(self, var, seen):
    return einsum('A,Aji,AijB->AB', self, inverse(self.func), derivative(self.func, var, seen))

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
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def _simplified(self):
    func1, func2 = self.funcs
    if isuniform(func1, 1):
      return func2
    if isuniform(func2, 1):
      return func1
    unaligned1, unaligned2, where = unalign(func1, func2)
    if len(where) != self.ndim:
      return align(unaligned1 * unaligned2, where, self.shape)
    for axis1, axis2, *other in map(sorted, func1._diagonals or func2._diagonals):
      return diagonalize(Multiply(takediag(func, axis1, axis2) for func in self.funcs), axis1, axis2)
    for i, parts in func1._inflations:
      return util.sum(_inflate(f * _take(func2, dofmap, i), dofmap, self.shape[i], i) for dofmap, f in parts.items())
    for i, parts in func2._inflations:
      return util.sum(_inflate(_take(func1, dofmap, i) * f, dofmap, self.shape[i], i) for dofmap, f in parts.items())
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
    unaligned1, where1 = unalign(func1)
    unaligned2, where2 = unalign(func2)
    return Einsum((unaligned1, unaligned2), (where1, where2), tuple(range(self.ndim)))

  def evalf(self, arr1, arr2):
    return arr1 * arr2

  def _sum(self, axis):
    func1, func2 = self.funcs
    unaligned, where = unalign(func1)
    if axis not in where:
      return align(unaligned, [i-(i>axis) for i in where], self.shape[:axis]+self.shape[axis+1:]) * sum(func2, axis)
    unaligned, where = unalign(func2)
    if axis not in where:
      return sum(func1, axis) * align(unaligned, [i-(i>axis) for i in where], self.shape[:axis]+self.shape[axis+1:])

  def _add(self, other):
    func1, func2 = self.funcs
    if isinstance(other, Multiply):
      for common in self.funcs & other.funcs:
        return common * Add(self.funcs + other.funcs - [common, common])

  def _determinant(self, axis1, axis2):
    func1, func2 = self.funcs
    axis1, axis2 = sorted([axis1, axis2])
    if equalindex(self.shape[axis1], 1) and equalindex(self.shape[axis2], 1):
      return Multiply([determinant(func1, (axis1, axis2)), determinant(func2, (axis1, axis2))])
    unaligned1, where1 = unalign(func1)
    if {axis1, axis2}.isdisjoint(where1):
      d2 = determinant(func2, (axis1, axis2))
      d1 = align(unaligned1**self.shape[axis1], [i-(i>axis1)-(i>axis2) for i in where1 if i not in (axis1, axis2)], d2.shape)
      return d1 * d2
    unaligned2, where2 = unalign(func2)
    if {axis1, axis2}.isdisjoint(where2):
      d1 = determinant(func1, (axis1, axis2))
      d2 = align(unaligned2**self.shape[axis1], [i-(i>axis1)-(i>axis2) for i in where2 if i not in (axis1, axis2)], d1.shape)
      return d1 * d2

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
    # Reorder the multiplications such that the amount of flops is minimized.
    # The flops are counted based on the lower int bounds of the shape and loop
    # lengths, excluding common inserted axes and invariant loops of the inner
    # product.
    sizes = []
    unaligned = tuple(map(unalign, (func1, func2, other)))
    for (f1, w1), (f2, w2) in itertools.combinations(unaligned, 2):
      lengths = [self.shape[i] for i in set(w1) | set(w2)]
      lengths += [arg.length for arg in f1.arguments | f2.arguments if isinstance(arg, _LoopIndex)]
      sizes.append(util.product((max(1, length._intbounds[0]) for length in lengths), 1))
    min_size = min(sizes)
    if sizes[0] == min_size:
      return # status quo
    elif sizes[1] == min_size:
      return (func1 * other) * func2
    elif sizes[2] == min_size:
      return (func2 * other) * func1

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    return einsum('A,AB->AB', func1, derivative(func2, var, seen)) \
         + einsum('A,AB->AB', func2, derivative(func1, var, seen))

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
    if set(unalign(func1)[1]).isdisjoint((axis1, axis2)):
      return divide(inverse(func2, (axis1, axis2)), func1)
    if set(unalign(func2)[1]).isdisjoint((axis1, axis2)):
      return divide(inverse(func1, (axis1, axis2)), func2)

  def _loopsum(self, index):
    func1, func2 = self.funcs
    if func1 != index and index not in func1.dependencies:
      return Multiply([func1, loop_sum(func2, index)])
    if func2 != index and index not in func2.dependencies:
      return Multiply([loop_sum(func1, index), func2])

  @property
  def _assparse(self):
    func1, func2 = self.funcs
    uninserted1, where1 = unalign(func1)
    uninserted2, where2 = unalign(func2)
    if not set(where1) & set(where2):
      sparse = []
      for *indices1, values1 in uninserted1._assparse:
        for *indices2, values2 in uninserted2._assparse:
          indices = [None] * self.ndim
          for i, j in enumerate(where1):
            indices[j] = appendaxes(indices1[i], values2.shape)
          for i, j in enumerate(where2):
            indices[j] = prependaxes(indices2[i], values1.shape)
          assert all(indices)
          values = appendaxes(values1, values2.shape) * prependaxes(values2, values1.shape)
          sparse.append((*indices, values))
      return tuple(sparse)
    return super()._assparse

  def _intbounds_impl(self):
    func1, func2 = self.funcs
    extrema = [b1 and b2 and b1 * b2 for b1 in func1._intbounds for b2 in func2._intbounds]
    return min(extrema), max(extrema)

class Add(Array):

  __slots__ = 'funcs',
  __cache__ = '_inflations'

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert equalshape(func1.shape, func2.shape)
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  @property
  def _inflations(self):
    func1, func2 = self.funcs
    func2_inflations = dict(func2._inflations)
    inflations = []
    for axis, parts1 in func1._inflations:
      if axis not in func2_inflations:
        continue
      parts2 = func2_inflations[axis]
      dofmaps = set(parts1) | set(parts2)
      if (len(parts1) < len(dofmaps) and len(parts2) < len(dofmaps) # neither set is a subset of the other; total may be dense
          and self.shape[axis].isconstant and all(dofmap.isconstant for dofmap in dofmaps)):
        mask = numpy.zeros(int(self.shape[axis]), dtype=bool)
        for dofmap in dofmaps:
          mask[dofmap.eval()] = True
        if mask.all(): # axis adds up to dense
          continue
      inflations.append((axis, types.frozendict((dofmap, util.sum(parts[dofmap] for parts in (parts1, parts2) if dofmap in parts)) for dofmap in dofmaps)))
    return tuple(inflations)

  def _simplified(self):
    func1, func2 = self.funcs
    if func1 == func2:
      return multiply(func1, 2)
    for axes1 in func1._diagonals:
      for axes2 in func2._diagonals:
        if len(axes1 & axes2) >= 2:
          axes = sorted(axes1 & axes2)[:2]
          return diagonalize(takediag(func1, *axes) + takediag(func2, *axes), *axes)
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

  def _loopsum(self, index):
    if any(index not in func.arguments for func in self.funcs):
      return Add([loop_sum(func, index) for func in self.funcs])

  def _multiply(self, other):
    func1, func2 = self.funcs
    if func1._inflations and func2._inflations:
      # NOTE: As this operation is the precise opposite of Multiply._add, there
      # appears to be a great risk of recursion. However, since both factors
      # are sparse, we can be certain that subsequent simpifications will
      # irreversibly process the new terms before reaching this point.
      return (func1 * other) + (func2 * other)

  @property
  def _assparse(self):
    func1, func2 = self.funcs
    return _gathersparsechunks(itertools.chain(func1._assparse, func2._assparse))

  def _intbounds_impl(self):
    func1, func2 = self.funcs
    lower1, upper1 = func1._intbounds
    lower2, upper2 = func2._intbounds
    return lower1 + lower2, upper1 + upper2

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
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

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

  @property
  def _assparse(self):
    chunks = []
    for *indices, _rmidx, values in self.func._assparse:
      if values.dtype == bool:
        values = Int(values)
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

  __slots__ = 'func'
  __cache__ = '_assparse'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('takediag requires an argument of dimension >= 2')
    if not equalindex(func.shape[-1], func.shape[-2]):
      raise Exception('takediag axes do not match')
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

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

  def _intbounds_impl(self):
    return self.func._intbounds

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
    super().__init__(args=[func,indices], shape=func.shape[:-1]+indices.shape, dtype=func.dtype)

  def _simplified(self):
    if self.indices.size == 0:
      return zeros_like(self)
    trytake = self.func._take(self.indices, self.func.ndim-1) or \
              self.indices._rtake(self.func, self.func.ndim-1)
    if trytake:
      return trytake
    for axis, parts in self.func._inflations:
      if axis == self.func.ndim - 1:
        return util.sum(Inflate(func, dofmap, self.func.shape[-1])._take(self.indices, self.func.ndim - 1) for dofmap, func in parts.items())

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

  def _intbounds_impl(self):
    return self.func._intbounds

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
    elif isuniform(self.power, 1):
      return self.func
    elif isuniform(self.power, 2):
      return self.func * self.func
    else:
      return self.func._power(self.power)

  def _optimized_for_numpy(self):
    if isuniform(self.power, -1):
      return Reciprocal(self.func)
    elif isuniform(self.power, -2):
      return Reciprocal(self.func * self.func)
    else:
      return self._simplified()

  def evalf(self, base, exp):
    return numeric.power(base, exp)

  def _derivative(self, var, seen):
    if self.power.isconstant:
      p = self.power.eval()
      return einsum('A,A,AB->AB', p, power(self.func, p - (p!=0)), derivative(self.func, var, seen))
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    return einsum('A,A,AB->AB', self.power, power(self.func, self.power - 1), derivative(self.func, var, seen)) \
         + einsum('A,A,AB->AB', ln(self.func), self, derivative(self.power, var, seen))

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
    self.args = args
    super().__init__(args=args, shape=shape0, dtype=retval.dtype)

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
    *uninserted, where = unalign(*self.args)
    if len(where) != self.ndim:
      return align(self.__class__(*uninserted), where, self.shape)

  def _derivative(self, var, seen):
    if self.deriv is None:
      return super()._derivative(var, seen)
    return util.sum(einsum('A,AB->AB', deriv(*self.args), derivative(arg, var, seen)) for arg, deriv in zip(self.args, self.deriv))

  def _takediag(self, axis1, axis2):
    return self.__class__(*[_takediag(arg, axis1, axis2) for arg in self.args])

  def _take(self, index, axis):
    return self.__class__(*[_take(arg, index, axis) for arg in self.args])

  def _unravel(self, axis, shape):
    return self.__class__(*[unravel(arg, axis, shape) for arg in self.args])

class Reciprocal(Pointwise):
  __slots__ = ()
  evalf = functools.partial(numpy.reciprocal, dtype=float)

class Negative(Pointwise):
  __slots__ = ()
  evalf = numpy.negative

  def _intbounds_impl(self):
    lower, upper = self.args[0]._intbounds
    return -upper, -lower

class FloorDivide(Pointwise):
  __slots__ = ()
  evalf = numpy.floor_divide

class Absolute(Pointwise):
  __slots__ = ()
  evalf = numpy.absolute

  def _intbounds_impl(self):
    lower, upper = self.args[0]._intbounds
    extrema = builtins.abs(lower), builtins.abs(upper)
    if lower <= 0 and upper >= 0:
      return 0, max(extrema)
    else:
      return min(extrema), max(extrema)

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
  __slots__ = ()
  evalf = numpy.maximum
  deriv = lambda x, y: 1 - Less(x, y), Less

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

class Int(Pointwise):
  __slots__ = ()
  evalf = staticmethod(lambda a: a.astype(int))
  deriv = lambda a: Zeros(a.shape, int),

  def _intbounds_impl(self):
    if self.args[0].dtype == bool:
      return 0, 1
    else:
      return self.args[0]._intbounds

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

  def _intbounds_impl(self):
    lower, upper = self.func._intbounds
    return int(numpy.sign(lower)), int(numpy.sign(upper))

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
  if len(unique) < len(data):
    index = Take(indices, index)
  # Move all axes with constant shape to the left and ravel the remainder.
  is_constant = numpy.array([n.isconstant for n in shape], dtype=bool)
  nconstant = is_constant.sum()
  reorder = numpy.argsort(~is_constant)
  shape = [shape[i] for i in reorder]
  var_shape = shape[nconstant:]
  reshape = [n.__index__() for n in shape[:nconstant]] + [-1]
  raveled = [numpy.transpose(d, reorder).reshape(reshape) for d in unique]
  # Concatenate the raveled axis, take slices, unravel and reorder the axes to
  # the original position.
  concat = numpy.concatenate(raveled, axis=-1)
  if len(var_shape) == 0:
    assert tuple(reorder) == tuple(range(nconstant))
    return Take(concat, index)
  cumprod = list(var_shape)
  for i in reversed(range(len(var_shape)-1)):
    cumprod[i] *= cumprod[i+1] # work backwards so that the shape check matches in Unravel
  offsets = _SizesToOffsets(asarray([d.shape[-1] for d in raveled]))
  elemwise = Take(concat, Range(cumprod[0]) + Take(offsets, index))
  for i in range(len(var_shape)-1):
    elemwise = Unravel(elemwise, var_shape[i], cumprod[i+1])
  return Transpose(elemwise, tuple(numpy.argsort(reorder)))

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

  __slots__ = 'arrays', 'index', '_lower', '_upper'

  @types.apply_annotations
  def __init__(self, arrays:strictevaluable, index:types.strictint, shape:asshape, dtype:asdtype, *, _lower=float('-inf'), _upper=float('inf')):
    self.arrays = arrays
    self.index = index
    self._lower = _lower
    self._upper = _upper
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

  def _intbounds_impl(self):
    return self._lower, self._upper

class Zeros(Array):
  'zero'

  __slots__ = ()
  __cache__ = '_assparse', '_unaligned'

  @types.apply_annotations
  def __init__(self, shape:asshape, dtype:asdtype):
    super().__init__(args=shape, shape=shape, dtype=dtype)

  @property
  def _unaligned(self):
    return Zeros((), self.dtype), ()

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

  def _intbounds_impl(self):
    return 0, 0

class Inflate(Array):

  __slots__ = 'func', 'dofmap', 'length', 'warn'
  __cache__ = '_assparse', '_diagonals', '_inflations'

  @types.apply_annotations
  def __init__(self, func:asarray, dofmap:asarray, length:asindex):
    if not equalshape(func.shape[func.ndim-dofmap.ndim:], dofmap.shape):
      raise Exception('invalid dofmap')
    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.warn = not dofmap.isconstant
    super().__init__(args=[func,dofmap,length], shape=(*func.shape[:func.ndim-dofmap.ndim], length), dtype=func.dtype)

  @property
  def _diagonals(self):
    return tuple(axes for axes in self.func._diagonals if all(axis < self.ndim-1 for axis in axes))

  @property
  def _inflations(self):
    inflations = [(self.ndim-1, types.frozendict({self.dofmap: self.func}))]
    for axis, parts in self.func._inflations:
      inflations.append((axis, types.frozendict((dofmap, Inflate(func, self.dofmap, self.length)) for dofmap, func in parts.items())))
    return tuple(inflations)

  def _simplified(self):
    for axis in range(self.dofmap.ndim):
      if equalindex(self.dofmap.shape[axis], 1):
        return Inflate(_take(self.func, 0, self.func.ndim-self.dofmap.ndim+axis), _take(self.dofmap, 0, axis), self.length)
    for axis, parts in self.func._inflations:
      i = axis - (self.ndim-1)
      if i >= 0:
        return util.sum(Inflate(f, _take(self.dofmap, ind, i), self.length) for ind, f in parts.items())
    if self.dofmap.ndim == 0 and equalindex(self.dofmap, 0) and equalindex(self.length, 1):
      return InsertAxis(self.func, 1)
    return self.func._inflate(self.dofmap, self.length, self.ndim-1) \
       or self.dofmap._rinflate(self.func, self.length, self.ndim-1)

  def evalf(self, array, indices, length):
    assert indices.ndim == self.dofmap.ndim
    assert length.ndim == 0
    if self.warn and int(length) > indices.size:
      warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
    inflated = numpy.zeros(array.shape[:array.ndim-indices.ndim] + (length,), dtype=self.dtype)
    numpy.add.at(inflated, (slice(None),)*(self.ndim-1)+(indices,), array)
    return inflated

  def _inflate(self, dofmap, length, axis):
    if dofmap.ndim == 0 and dofmap == self.dofmap and length == self.length:
      return diagonalize(self, -1, axis)

  def _derivative(self, var, seen):
    return _inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.ndim-1)

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

  def _intbounds_impl(self):
    lower, upper = self.func._intbounds
    return min(lower, 0), max(upper, 0)

class SwapInflateTake(Evaluable):

  def __init__(self, inflateidx, takeidx):
    self.inflateidx = inflateidx
    self.takeidx = takeidx
    super().__init__(args=[inflateidx, takeidx])

  def __iter__(self):
    shape = ArrayFromTuple(self, index=2, shape=(), dtype=int, _lower=0),
    return (ArrayFromTuple(self, index=index, shape=shape, dtype=int, _lower=0) for index in range(2))

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
  __cache__ = '_diagonals'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim == 0:
      raise Exception('cannot diagonalize scalar function')
    self.func = func
    super().__init__(args=[func], shape=(*func.shape, func.shape[-1]), dtype=func.dtype)

  @property
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

  def _sum(self, axis):
    if axis >= self.ndim - 2:
      return self.func
    return Diagonalize(sum(self.func, axis))

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

  def _loopsum(self, index):
    return Diagonalize(loop_sum(self.func, index))

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
    return einsum('Ai,AB->AiB', TrigTangent(self.angle), derivative(self.angle, var, seen))

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
    return -einsum('Ai,AB->AiB', TrigNormal(self.angle), derivative(self.angle, var, seen))

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
    super().__init__(args=[where], shape=[Int(where).sum()], dtype=int)

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
    if isinstance(var, Argument) and var._name == self._name and self.dtype == float:
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
  __cache__ = '_inflations'

  @types.apply_annotations
  def __init__(self, func:asarray):
    if func.ndim < 2:
      raise Exception('cannot ravel function of dimension < 2')
    self.func = func
    super().__init__(args=[func], shape=(*func.shape[:-2], func.shape[-2] * func.shape[-1]), dtype=func.dtype)

  @property
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
    if equalindex(self.func.shape[-2], 1):
      return get(self.func, -2, 0)
    if equalindex(self.func.shape[-1], 1):
      return get(self.func, -1, 0)
    return self.func._ravel(self.ndim-1)

  def evalf(self, f):
    return f.reshape(f.shape[:-2] + (f.shape[-2]*f.shape[-1],))

  def _multiply(self, other):
    if isinstance(other, Ravel) and equalshape(other.func.shape[-2:], self.func.shape[-2:]):
      return Ravel(Multiply([self.func, other.func]))
    return Ravel(Multiply([self.func, Unravel(other, *self.func.shape[-2:])]))

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
    elif equalshape(shape, self.func.shape[-2:]):
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
    return ravel(insertaxis(self.func, axis+(axis==self.ndim), length), self.ndim-(axis==self.ndim))

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
    unaligned, where = unalign(self.func)
    for i in self.ndim - 1, self.ndim:
      if i not in where:
        unaligned = InsertAxis(unaligned, self.func.shape[i])
        where += i,
    if where[-2:] != (self.ndim - 1, self.ndim):
      unaligned = Transpose(unaligned, numpy.argsort(where))
      where = tuple(sorted(where))
    return Ravel(unaligned), where[:-1]

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
    super().__init__(args=[func, sh1, sh2], shape=(*func.shape[:-1], sh1, sh2), dtype=func.dtype)

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

  @property
  def _assparse(self):
    return tuple((*indices[:-1], *divmod(indices[-1], appendaxes(self.shape[-1], values.shape)), values) for *indices, values in self.func._assparse)

class RavelIndex(Array):

  @types.apply_annotations
  def __init__(self, ia:asarray, ib:asarray, na:asindex, nb:asindex):
    self._ia = ia
    self._ib = ib
    self._na = na
    self._nb = nb
    self._length = na * nb
    super().__init__(args=[ia, ib, nb], shape=ia.shape + ib.shape, dtype=int)

  def evalf(self, ia, ib, nb):
    return ia[(...,)+(numpy.newaxis,)*ib.ndim] * nb + ib

  def _take(self, index, axis):
    if axis < self._ia.ndim:
      return RavelIndex(_take(self._ia, index, axis), self._ib, self._na, self._nb)
    else:
      return RavelIndex(self._ia, _take(self._ib, index, axis - self._ia.ndim), self._na, self._nb)

  def _rtake(self, func, axis):
    if equalindex(func.shape[-1], self._length):
      return Take(_take(Unravel(func, self._na, self._nb), self._ia, -2), self._ib)

  def _rinflate(self, func, length, axis):
    if equalindex(length, self._length):
      return Ravel(Inflate(_inflate(func, self._ia, self._na, func.ndim - self.ndim), self._ib, self._nb))

  def _intbounds_impl(self):
    nbmin, nbmax = self._nb._intbounds
    iamin, iamax = self._ia._intbounds
    ibmin, ibmax = self._ib._intbounds
    return iamin * nbmin + ibmin, iamax * nbmax + ibmax

class Range(Array):

  __slots__ = 'length'

  @types.apply_annotations
  def __init__(self, length:asindex):
    self.length = length
    super().__init__(args=[length], shape=[length], dtype=int)

  def _take(self, index, axis):
    return InRange(index, self.length)

  def _rtake(self, func, axis):
    if self.length == func.shape[axis]:
      return func

  def _rinflate(self, func, length, axis):
    if length == self.length:
      return func

  def evalf(self, length):
    return numpy.arange(length)

  def _intbounds_impl(self):
    lower, upper = self.length._intbounds
    assert lower >= 0
    return 0, max(0, upper - 1)

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
    dpoints = einsum('ABi,AiD->ABD', Polyval(self.coeffs, self.points, self.ngrad+1), derivative(self.points, var, seen), A=self.points.ndim-1)
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
    index, *choices, where = unalign(self.index, *self.choices)
    if len(where) < self.ndim:
      return align(Choose(index, choices), where, self.shape)

  def _multiply(self, other):
    if isinstance(other, Choose) and self.index == other.index:
      return Choose(self.index, map(multiply, self.choices, other.choices))

  def _get(self, i, item):
    return Choose(get(self.index, i, item), [get(choice, i, item) for choice in self.choices])

  def _sum(self, axis):
    unaligned, where = unalign(self.index)
    if axis not in where:
      index = align(unaligned, [i-(i>axis) for i in where], self.shape[:axis]+self.shape[axis+1:])
      return Choose(index, [sum(choice, axis) for choice in self.choices])

  def _take(self, index, axis):
    return Choose(_take(self.index, index, axis), [_take(choice, index, axis) for choice in self.choices])

  def _takediag(self, axis, rmaxis):
    return Choose(takediag(self.index, axis, rmaxis), [takediag(choice, axis, rmaxis) for choice in self.choices])

  def _product(self):
    unaligned, where = unalign(self.index)
    if self.ndim-1 not in where:
      index = align(unaligned, where, self.shape[:-1])
      return Choose(index, [Product(choice) for choice in self.choices])

class NormDim(Array):

  @types.apply_annotations
  def __init__(self, length: asarray, index: asarray):
    assert length.dtype == int
    assert index.dtype == int
    assert equalshape(length.shape, index.shape)
    # The following corner cases makes the assertion fail, hence we can only
    # assert the bounds if the arrays are guaranteed to be unempty:
    #
    #     Take(func, NormDim(func.shape[-1], Range(0) + func.shape[-1]))
    if all(n._intbounds[0] > 0 for n in index.shape):
      assert -length._intbounds[1] <= index._intbounds[0] and index._intbounds[1] <= length._intbounds[1] - 1
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
    lower_length, upper_length = self.length._intbounds
    lower_index, upper_index = self.index._intbounds
    if 0 <= lower_index and upper_index < lower_length:
      return self.index
    if isinstance(lower_length, int) and lower_length == upper_length and -lower_length <= lower_index and upper_index < 0:
      return self.index + lower_length
    if self.length.isconstant and self.index.isconstant:
      return Constant(self.eval())

  def _intbounds_impl(self):
    lower_length, upper_length = self.length._intbounds
    lower_index, upper_index = self.index._intbounds
    if lower_index >= 0:
      return min(lower_index, upper_length - 1), min(upper_index, upper_length - 1)
    elif upper_index < 0 and isinstance(lower_length, int) and lower_length == upper_length:
      return max(lower_index + lower_length, 0), max(upper_index + lower_length, 0)
    else:
      return 0, upper_length - 1

class _LoopIndex(Argument):

  __slots__ = 'length'

  @types.apply_annotations
  def __init__(self, name: types.strictstr, length: asindex):
    self.length = length
    super().__init__(name, (), int)

  def __str__(self):
    try:
      length = self.length.__index__()
    except EvaluationError:
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
    if equalindex(self.length, 1):
      return Zeros((), int)

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
  def __init__(self, funcdata:prepare_funcdata, index_name:types.strictstr, length:asindex):
    shape = Tuple(funcdata[1:])
    self.index = loop_index(index_name, length)
    if self.index in shape.arguments:
      raise ValueError('the shape of the function must not depend on the index')
    self.func = funcdata[0]
    self._invariants, self._dependencies = _dependencies_sans_invariants(self.func, self.index)
    super().__init__(args=(shape, length, *self._invariants), shape=self.func.shape, dtype=self.func.dtype)

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
    subtimes = times.setdefault(self, collections.defaultdict(_Stats))
    result = numpy.zeros(shape, self.dtype)
    for index in range(length):
      values = [numpy.array(index)]
      values.extend(args)
      values.extend(op.evalf_withtimes(subtimes, *[values[i] for i in indices]) for op, indices in serialized)
      result += values[-1]
    return result

  def _derivative(self, var, seen):
    return loop_sum(derivative(self.func, var, seen), self.index)

  def _node(self, cache, subgraph, times):
    if self in cache:
      return cache[self]
    subcache = {}
    for arg in self._Evaluable__args:
      subcache[arg] = arg._node(cache, subgraph, times)
    loopgraph = Subgraph('Loop', subgraph)
    subtimes = times.get(self, collections.defaultdict(_Stats))
    sum_kwargs = {'shape[{}]'.format(i): n._node(cache, subgraph, times) for i, n in enumerate(self.shape)}
    sum_kwargs['func'] = self.func._node(subcache, loopgraph, subtimes)
    cache[self] = node = RegularNode('LoopSum', (), sum_kwargs, (type(self).__name__, subtimes['sum']), loopgraph)
    return node

  def _simplified(self):
    if iszero(self.func):
      return zeros_like(self)
    elif self.index not in self.func.arguments:
      return self.func * self.index.length
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

  @property
  def _assparse(self):
    chunks = []
    for *elem_indices, elem_values in self.func._assparse:
      if self.ndim == 0:
        values = loop_concatenate(InsertAxis(elem_values, 1), self.index)
        while values.ndim:
          values = Sum(values)
        chunks.append((values,))
      else:
        if elem_values.ndim == 0:
          *elem_indices, elem_values = (InsertAxis(arr, 1) for arr in (*elem_indices, elem_values))
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
    super().__init__(args=[sizes], shape=(sizes.shape[0]+1,), dtype=int)

  def evalf(self, sizes):
    return numpy.cumsum([0, *sizes])

  def _simplified(self):
    unaligned, where = unalign(self._sizes)
    if not where:
      return Range(self.shape[0]) * appendaxes(unaligned, self.shape[:1])

  def _intbounds_impl(self):
    n = self._sizes.size._intbounds[1]
    m = self._sizes._intbounds[1]
    return 0, (0 if n == 0 or m == 0 else n * m)

class LoopConcatenate(Array):

  @types.apply_annotations
  def __init__(self, funcdata:asarrays, index_name:types.strictstr, length:asindex):
    self.funcdata = funcdata
    self.func, self.start, stop, *shape = funcdata
    self.index = loop_index(index_name, length)
    if not self.func.ndim:
      raise ValueError('expected an array with at least one axis')
    if any(self.index in n.arguments for n in shape):
      raise ValueError('the shape of the function must not depend on the index')
    self._lcc = LoopConcatenateCombined((self.funcdata,), index_name, length)
    super().__init__(args=[self._lcc], shape=shape, dtype=self.func.dtype)

  def evalf(self, arg):
    return arg[0]

  def evalf_withtimes(self, times, arg):
    with times[self]:
      return arg[0]

  def _derivative(self, var, seen):
    return Transpose.from_end(loop_concatenate(Transpose.to_end(derivative(self.func, var, seen), self.ndim-1), self.index), self.ndim-1)

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
      return Ravel(Transpose.from_end(InsertAxis(self.func, self.index.length), -2))
    unaligned, where = unalign(self.func)
    if self.ndim-1 not in where:
      # reinsert concatenation axis, at unit length if possible so we can
      # insert the remainder outside of the loop
      unaligned = InsertAxis(unaligned, self.func.shape[-1] if self.index in self.func.shape[-1].arguments else 1)
      where += self.ndim-1,
    elif where[-1] != self.ndim-1:
      # bring concatenation axis to the end
      unaligned = Transpose(unaligned, numpy.argsort(where))
      where = tuple(sorted(where))
    f = loop_concatenate(unaligned, self.index)
    if not equalindex(self.shape[-1], f.shape[-1]):
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

  @property
  def _assparse(self):
    chunks = []
    for *indices, last_index, values in self.func._assparse:
      last_index = last_index + prependaxes(self.start, last_index.shape)
      chunks.append(tuple(loop_concatenate(_flat(arr), self.index) for arr in (*indices, last_index, values)))
    return tuple(chunks)

  @property
  def _loop_concatenate_deps(self):
    return (self,) + super()._loop_concatenate_deps

  def _intbounds_impl(self):
    return self.func._intbounds

class LoopConcatenateCombined(Evaluable):

  __cache__ = '_serialized'

  @types.apply_annotations
  def __init__(self, funcdatas:types.tuple[asarrays], index_name:types.strictstr, length:asindex):
    self._funcdatas = funcdatas
    self._funcs = tuple(func for func, start, stop, *shape in funcdatas)
    self._index = loop_index(index_name, length)
    if any(not func.ndim for func in self._funcs):
      raise ValueError('expected an array with at least one axis')
    shapes = [Tuple(shape) for func, start, stop, *shape in funcdatas]
    if any(self._index in shape.arguments for shape in shapes):
      raise ValueError('the shape of the function must not depend on the index')
    self._invariants, self._dependencies = _dependencies_sans_invariants(
      Tuple([Tuple([start, stop, func]) for func, start, stop, *shape in funcdatas]), self._index)
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
    subtimes = times.setdefault(self, collections.defaultdict(_Stats))
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
  unaligned, where = unalign(arg)
  return not where and isinstance(unaligned, Constant) and unaligned.value[()] == value

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

@replace
def _bifurcate(arg, side):
  if isinstance(arg, (SelectChain, TransformChainFromSequence)):
    return SelectBifurcation(arg, side)

bifurcate1 = functools.partial(_bifurcate, side=True)
bifurcate2 = functools.partial(_bifurcate, side=False)

def insertaxis(arg, n, length):
  return Transpose.from_end(InsertAxis(arg, length), n)

def stack(args, axis=0):
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
    else abs(determinant(einsum('Aki,Akj->Aij', J, J)))**.5
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
  if var.dtype != float:
    return Zeros(func.shape + var.shape, dtype=func.dtype)
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
    index = Range(stop-start) + start
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
    func = InsertAxis(func, n)
  return func

def loop_index(name, length):
  return _LoopIndex(name, length)

def loop_sum(func, index):
  func = asarray(func)
  index = types.strict[_LoopIndex](index)
  return LoopSum(func, index._name, index.length)

def _loop_concatenate_data(func, index):
  func = asarray(func)
  index = types.strict[_LoopIndex](index)
  chunk_size = func.shape[-1]
  if chunk_size.isconstant:
    chunk_sizes = InsertAxis(chunk_size, index.length)
  else:
    chunk_sizes = loop_concatenate(InsertAxis(func.shape[-1], 1), index)
  offsets = _SizesToOffsets(chunk_sizes)
  start = Take(offsets, index)
  stop = Take(offsets, index+1)
  return (func, start, stop, *func.shape[:-1], Take(offsets, index.length))

def loop_concatenate(func, index):
  funcdata = _loop_concatenate_data(func, index)
  return LoopConcatenate(funcdata, index._name, index.length)

def loop_concatenate_combined(funcs, index):
  unique_funcs = []
  unique_funcs.extend(func for func in funcs if func not in unique_funcs)
  unique_func_data = tuple(_loop_concatenate_data(func, index) for func in unique_funcs)
  loop = LoopConcatenateCombined(unique_func_data, index._name, index.length)
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

def einsum(fmt, *args, **dims):
  '''Multiply and/or contract arrays via format string.

  The format string consists of a comma separated list of axis labels, followed
  by ``->`` and the axis labels of the return value. For example, the following
  swaps the axes of a matrix:

  >>> einsum('ij->ji', ones([2,3]))
  nutils.evaluable.Transpose<f:3,2>

  Axis labels that do not occur in the return value are summed. For example,
  the following performs a dot product of three matrices:

  >>> einsum('ij,jk,kl->il', ones([2,3]), ones([3,4]), ones([4,5]))
  nutils.evaluable.Sum<f:2,5>

  In case the dimension of the input and output arrays may vary, a variable
  length axes group can be denoted by a capital. Its length is automatically
  established based on the dimension of the input arrays. The following example
  performs a tensor product of an array and a vector:

  >>> einsum('A,i->Ai', ones([2,3,4]), ones([5]))
  nutils.evaluable.Multiply<f:2,3,4,5>

  The format string may contain multiple variable length axes groups, but their
  lengths must be resolvable from left to right. In case this is not possible,
  lengths may be specified as keyword arguments.

  >>> einsum('AjB,i->AijB', ones([2,3,4]), ones([5]), B=1)
  nutils.evaluable.Multiply<f:2,5,3,4>
  '''

  sin, sout = fmt.split('->')
  sin = sin.split(',')

  if len(sin) != len(args):
    raise ValueError('number of arguments does not match format string')

  if any(len(s) != len(set(s)) for s in (*sin, sout)):
    raise ValueError('internal repetitions are not supported')

  if any(n < 0 for n in dims.values()):
    raise ValueError('axis group dimensions cannot be negative')

  for c in 'abcdefghijklmnopqrstuvwxyz':
    dims.setdefault(c, 1) # lowercase characters default to single dimension

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
      if not equalindex(shapes.setdefault(c, sh), sh):
        raise ValueError('shapes do not match for axis {0[0]}{0[1]}'.format(c))

  ret = None
  for s, arg in zip(sin, args):
    index = {c: i for i, c in enumerate(s)}
    for c in sall:
      if c not in index:
        index[c] = arg.ndim
        arg = InsertAxis(arg, shapes[c])
    v = Transpose(arg, [index[c] for c in sall])
    ret = v if ret is None else ret * v
  for i in range(len(sout), len(sall)):
    ret = Sum(ret)
  return ret

if __name__ == '__main__':
  # Diagnostics for the development for simplify operations.
  simplify_priority = (
    Transpose, Ravel, # reinterpretation
    InsertAxis, Inflate, Diagonalize, # size increasing
    Multiply, Add, LoopSum, Sign, Power, Inverse, Unravel, # size preserving
    Product, Determinant, TakeDiag, Take, Sum) # size decreasing
  # The simplify priority defines the preferred order in which operations are
  # performed: shape decreasing operations such as Sum and Take should be done
  # as soon as possible, and shape increasing operations such as Inflate and
  # Diagonalize as late as possible. In shuffling the order of operations the
  # two classes might annihilate each other, for example when a Sum passes
  # through a Diagonalize. Any shape increasing operations that remain should
  # end up at the surface, exposing sparsity by means of the assparse method.
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
