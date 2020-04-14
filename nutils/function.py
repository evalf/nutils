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

from . import util, types, numpy, numeric, cache, transform, transformseq, expression, warnings, _
import sys, itertools, functools, operator, inspect, numbers, builtins, re, types as builtin_types, abc, collections.abc, math, treelog as log

isevaluable = lambda arg: isinstance(arg, Evaluable)

def strictevaluable(value):
  if not isinstance(value, Evaluable):
    raise ValueError('expected an object of type {!r} but got {!r} with type {!r}'.format(Evaluable.__qualname__, value, type(value).__qualname__))
  return value

def simplified(value):
  return strictevaluable(value).simplified

asdtype = lambda arg: arg if any(arg is dtype for dtype in (bool, int, float, complex)) else {'f': float, 'i': int, 'b': bool, 'c': complex}[numpy.dtype(arg).kind]
asarray = lambda arg: arg if isarray(arg) else Constant(arg) if numeric.isarray(arg) or numpy.asarray(arg).dtype != object else stack(arg, axis=0)
asarrays = types.tuple[asarray]

def as_canonical_length(value):
  if isarray(value):
    if value.ndim != 0 or value.dtype != int:
      raise ValueError('length should be an `int` or `Array` with zero dimensions and dtype `int`, got {!r}'.format(value))
    value = value.simplified
    if value.isconstant:
      value = int(value.eval()) # Ensure this is an `int`, not `numpy.int64`.
  elif numeric.isint(value):
    value = int(value) # Ensure this is an `int`, not `numpy.int64`.
  else:
    raise ValueError('length should be an `int` or `Array` with zero dimensions and dtype `int`, got {!r}'.format(value))
  return value

asshape = types.tuple[as_canonical_length]

class ExpensiveEvaluationWarning(Warning): pass

class Evaluable(types.Singleton):
  'Base class'

  __slots__ = '__args',
  __cache__ = 'dependencies', 'ordereddeps', 'dependencytree', 'simplified', 'prepare_eval', 'optimized_for_numpy'

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable]):
    super().__init__()
    self.__args = args

  def evalf(self, *args):
    raise NotImplementedError('Evaluable derivatives should implement the evalf method')

  @property
  def dependencies(self):
    '''collection of all function arguments'''
    deps = list(self.__args)
    for func in self.__args:
      deps.extend(func.dependencies)
    return frozenset(deps)

  @property
  def isconstant(self):
    return EVALARGS not in self.dependencies

  @property
  def ordereddeps(self):
    '''collection of all function arguments such that the arguments to
    dependencies[i] can be found in dependencies[:i]'''
    return tuple([EVALARGS] + sorted(self.dependencies - {EVALARGS}, key=lambda f: len(f.dependencies)))

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

  def asciitree(self, richoutput=False):
    'string representation'

    if richoutput:
      select = '├ ', '└ '
      bridge = '│ ', '  '
    else:
      select = ': ', ': '
      bridge = '| ', '  '
    lines = []
    ordereddeps = list(self.ordereddeps) + [self]
    pool = [('', len(ordereddeps)-1)] # prefix, object tuples
    while pool:
      prefix, n = pool.pop()
      s = '%{}'.format(n)
      if prefix:
        s = prefix[:-2] + select[bridge.index(prefix[-2:])] + s # locally change prefix into selector
      if ordereddeps[n] is not None:
        s += ' = ' + ordereddeps[n]._asciitree_str()
        pool.extend((prefix + bridge[i==0], arg) for i, arg in enumerate(reversed(self.dependencytree[n])))
        ordereddeps[n] = None
      lines.append(s)
    return '\n'.join(lines)

  def _asciitree_str(self):
    return str(self)

  def __str__(self):
    return self.__class__.__name__

  def eval(self, **evalargs):
    values = [evalargs]
    for op, indices in self.serialized:
      try:
        args = [values[i] for i in indices]
        retval = op.evalf(*args)
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        excargs = etype, evalue, self, values
        raise EvaluationError(*excargs).with_traceback(traceback)
      values.append(retval)
    return values[-1]

  @log.withcontext
  def graphviz(self, dotpath='dot', imgtype='png'):
    'create function graph'

    import os, subprocess

    lines = []
    lines.append('digraph {')
    lines.append('graph [dpi=72];')
    lines.extend('{0:} [label="{0:}. {1:}"];'.format(i, name._asciitree_str()) for i, name in enumerate(self.ordereddeps+(self,)))
    lines.extend('{} -> {};'.format(j, i) for i, indices in enumerate(self.dependencytree) for j in indices)
    lines.append('}')

    with log.infofile('dot.'+imgtype, 'wb') as img:
      status = subprocess.run([dotpath,'-T'+imgtype], input='\n'.join(lines).encode(), stdout=subprocess.PIPE)
      if status.returncode:
        log.warning('graphviz failed for error code', status.returncode)
      img.write(status.stdout)

  def stackstr(self, nlines=-1):
    'print stack'

    lines = ['  %0 = EVALARGS']
    for op, indices in self.serialized:
      args = ['%{}'.format(idx) for idx in indices]
      try:
        code = op.evalf.__code__
        offset = 1 if getattr(op.evalf, '__self__', None) is not None else 0
        names = code.co_varnames[offset:code.co_argcount]
        names += tuple('{}[{}]'.format(code.co_varnames[code.co_argcount], n) for n in range(len(indices) - len(names)))
        args = ['{}={}'.format(*item) for item in zip(names, args)]
      except:
        pass
      lines.append('  %{} = {}({})'.format(len(lines), op._asciitree_str(), ', '.join(args)))
      if len(lines) == nlines+1:
        break
    return '\n'.join(lines)

  @property
  def simplified(self):
    return self.edit(lambda arg: arg.simplified if isevaluable(arg) else arg)

  @property
  def optimized_for_numpy(self):
    return self.edit(lambda arg: arg.optimized_for_numpy if isevaluable(arg) else arg)

  @util.positional_only
  def prepare_eval(self, kwargs=...):
    '''
    Return a function tree suitable for evaluation.
    '''

    return self.edit(lambda arg: arg.prepare_eval(**kwargs) if isevaluable(arg) else arg)

class EvaluationError(Exception):
  'evaluation error'

  def __init__(self, etype, evalue, evaluable, values):
    'constructor'

    self.etype = etype
    self.evalue = evalue
    self.evaluable = evaluable
    self.values = values

  def __repr__(self):
    return 'EvaluationError{}'.format(self)

  def __str__(self):
    'string representation'

    return '\n{} --> {}: {}'.format(self.evaluable.stackstr(nlines=len(self.values)), self.etype.__name__, self.evalue)

EVALARGS = Evaluable(args=())

class Points(Evaluable):
  __slots__ = ()
  def __init__(self):
    super().__init__(args=[EVALARGS])
  def evalf(self, evalargs):
    points = evalargs['_points']
    assert numeric.isarray(points) and points.ndim == 2
    return types.frozenarray(points)

POINTS = Points()

class Tuple(Evaluable):

  __slots__ = 'items', 'indices'
  __cache__ = 'simplified',

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

  @property
  def simplified(self):
    return Tuple([item.simplified if isevaluable(item) else item for item in self.items])

  def edit(self, op):
    return Tuple([op(item) for item in self.items])

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

  @util.positional_only
  def prepare_eval(self, *, opposite=False, kwargs=...):
    return SelectChain(1-self.n) if opposite else self

TRANS = SelectChain()

class PopHead(TransformChain):

  __slots__ = 'trans',

  @types.apply_annotations
  def __init__(self, todims:types.strictint, trans=TRANS):
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
    return numpy.array(index)[None], tail

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
  arg = asarray(arg)
  if axis is None:
    axis = numpy.arange(arg.ndim)
  elif numeric.isint(axis):
    axis = numeric.normdim(arg.ndim, axis),
  else:
    axis = _norm_and_sort(arg.ndim, axis)
    assert numpy.greater(numpy.diff(axis), 0).all(), 'duplicate axes in sum'
  summed = arg
  for ax in reversed(axis):
    summed = Sum(summed, ax)
  return summed

def product(arg, axis):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  shape = arg.shape[:axis] + arg.shape[axis+1:]
  trans = [i for i in range(arg.ndim) if i != axis] + [axis]
  return Product(transpose(arg, trans))

def power(arg, n):
  arg, n = _numpy_align(arg, n)
  return Power(arg, n)

def dot(a, b, axes=None):
  '''
  Contract ``a`` and ``b`` along ``axes``.
  '''
  if axes is None:
    a = asarray(a)
    b = asarray(b)
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    for idim in range(1, a.ndim):
      b = insertaxis(b, idim, a.shape[idim])
    axes = 0,
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

  __slots__ = 'shape', 'ndim', 'dtype'
  __cache__ = 'optimized_for_numpy'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  @types.apply_annotations
  def __init__(self, args:types.tuple[strictevaluable], shape:asshape, dtype:asdtype):
    self.shape = shape
    self.ndim = len(shape)
    self.dtype = dtype
    super().__init__(args=args)

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
      elif it is _:
        array = expand_dims(array, axis)
        axis += 1
      elif it == slice(None):
        axis += 1
      elif isinstance(it, slice):
        assert it.step == None or it.step == 1
        start = 0 if it.start is None else it.start if it.start >= 0 else it.start + array.shape[axis]
        stop = array.shape[axis] if it.stop is None else it.stop if it.stop >= 0 else it.stop + array.shape[axis]
        array = take(array, index=Range(stop-start, start), axis=axis)
        axis += 1
      else:
        array = take(array, index=it, axis=axis)
        axis += 1
    assert axis == array.ndim
    return array

  def __len__(self):
    if self.ndim == 0:
      raise TypeError('len() of unsized object')
    return self.shape[0]

  def __iter__(self):
    if not self.shape:
      raise TypeError('iteration over a 0-d array')
    return (self[i,...] for i in range(self.shape[0]))

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
  __str__ = __repr__ = lambda self: 'Array<{}>'.format(','.join(map(str, self.shape)) if hasattr(self, 'shape') else '?')

  sum = sum
  prod = product
  dot = dot
  normalized = lambda self, axis=-1: normalized(self, axis)
  normal = lambda self, exterior=False: normal(self, exterior)
  curvature = lambda self, ndims=-1: curvature(self, ndims)
  swapaxes = swapaxes
  transpose = transpose
  grad = lambda self, geom, ndims=0: grad(self, geom, ndims)
  laplace = lambda self, geom, ndims=0: grad(self, geom, ndims).div(geom, ndims)
  add_T = lambda self, axes=(-2,-1): add_T(self, axes)
  symgrad = lambda self, geom, ndims=0: symgrad(self, geom, ndims)
  div = lambda self, geom, ndims=0: div(self, geom, ndims)
  dotnorm = lambda self, geom, axis=-1: dotnorm(self, geom, axis)
  tangent = lambda self, vec: tangent(self, vec)
  ngrad = lambda self, geom, ndims=0: ngrad(self, geom, ndims)
  nsymgrad = lambda self, geom, ndims=0: nsymgrad(self, geom, ndims)

  def vector(self, ndims):
    if self.ndim != 1:
      raise Exception('only a scalar basis van be vectorized')
    return ravel(diagonalize(insertaxis(self, 1, ndims), 1, 2), 0)

  @property
  def blocks(self):
    return [(tuple(Range(n) for n in self.shape), self)]

  def _asciitree_str(self):
    return '{}({})'.format(type(self).__name__, ','.join(['?' if isarray(sh) else str(sh) for sh in self.shape]))

  # simplifications
  _multiply = lambda self, other: None
  _transpose = lambda self, axes: None
  _insertaxis = lambda self, axis, length: None
  _get = lambda self, i, item: None
  _power = lambda self, n: None
  _add = lambda self, other: None
  _sum = lambda self, axis: None
  _take = lambda self, index, axis: None
  _determinant = lambda self: None
  _inverse = lambda self: None
  _takediag = lambda self, axis, rmaxis: None
  _diagonalize = lambda self, axis, newaxis: None
  _product = lambda self: None
  _sign = lambda self: None
  _eig = lambda self, symmetric: None
  _inflate = lambda self, dofmap, length, axis: None
  _mask = lambda self, maskvec, axis: None
  _unravel = lambda self, axis, shape: None
  _ravel = lambda self, axis: None
  _kronecker = lambda self, axis, length, pos: None
  _inserted_axes = ()

  @property
  def optimized_for_numpy(self):
    if self.isconstant:
      const, = self.eval()
      return Constant(const)
    return super().optimized_for_numpy

  def _derivative(self, var, seen):
    if self.dtype in (bool, int) or var not in self.dependencies:
      return Zeros(self.shape + var.shape, dtype=self.dtype)
    raise NotImplementedError('derivative not defined for {}'.format(self.__class__.__name__))

class Normal(Array):
  'normal'

  __slots__ = 'lgrad',

  @types.apply_annotations
  def __init__(self, lgrad:asarray):
    assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
    self.lgrad = lgrad
    super().__init__(args=[lgrad], shape=(len(lgrad),), dtype=float)

  def evalf(self, lgrad):
    n = lgrad[...,-1]
    if n.shape[-1] == 1: # geom is 1D
      return numpy.sign(n)
    # orthonormalize n to G
    G = lgrad[...,:-1]
    GG = numeric.contract(G[:,:,_,:], G[:,:,:,_], axis=1)
    v1 = numeric.contract(G, n[:,:,_], axis=1)
    v2 = numpy.linalg.solve(GG, v1)
    v3 = numeric.contract(G, v2[:,_,:], axis=2)
    return numeric.normalize(n - v3)

  def _derivative(self, var, seen):
    if len(self) == 1:
      return zeros(self.shape + var.shape)
    G = self.lgrad[...,:-1]
    GG = matmat(G.T, G)
    Gder = derivative(G, var, seen)
    nGder = matmat(self, Gder)
    return -matmat(G, inverse(GG), nGder)

class Constant(Array):

  __slots__ = 'value',
  __cache__ = 'simplified', '_isunit'

  @types.apply_annotations
  def __init__(self, value:types.frozenarray):
    self.value = value
    super().__init__(args=[], shape=value.shape, dtype=value.dtype)

  @property
  def simplified(self):
    if not self.value.any():
      return zeros_like(self)
    # Find and replace invariant axes with InsertAxis.
    value = self.value
    invariant = []
    for i in reversed(range(self.ndim)):
      # Since `self.value.any()` is False for arrays with a zero-length axis,
      # we can arrive here only if all axes have at least length one, hence the
      # following statement should work.
      first = numeric.get(value, i, 0)
      if all(numpy.equal(first, numeric.get(value, i, j)).all() for j in range(1, value.shape[i])):
        invariant.append(i)
        value = first
    if invariant:
      value = Constant(value)
      for i in reversed(invariant):
        value = InsertAxis(value, i, self.shape[i])
      return value.simplified
    return self

  def evalf(self):
    return self.value[_]

  @property
  def _isunit(self):
    return numpy.equal(self.value, 1).all()

  def _transpose(self, axes):
    return Constant(self.value.transpose(axes))

  def _sum(self, axis):
    return Constant(numpy.sum(self.value, axis))

  def _get(self, i, item):
    if item.isconstant:
      item, = item.eval()
      return Constant(numeric.get(self.value, i, item))

  def _add(self, other):
    if isinstance(other, Constant):
      return Constant(numpy.add(self.value, other.value))

  def _inverse(self):
    return Constant(numpy.linalg.inv(self.value))

  def _product(self):
    return Constant(self.value.prod(-1))

  def _multiply(self, other):
    if self._isunit:
      return other
    if isinstance(other, Constant):
      return Constant(numpy.multiply(self.value, other.value))

  def _takediag(self, axis, rmaxis):
    return Constant(numeric.takediag(self.value, axis, rmaxis))

  def _take(self, index, axis):
    if isinstance(index, Constant):
      return Constant(self.value.take(index.value, axis))

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

  def _mask(self, maskvec, axis):
    return Constant(self.value[(slice(None),)*axis+(numpy.asarray(maskvec),)])

  def _determinant(self):
    # NOTE: numpy <= 1.12 cannot compute the determinant of an array with shape [...,0,0]
    return Constant(numpy.linalg.det(self.value) if self.value.shape[-1] else numpy.ones(self.value.shape[:-2]))

class DofMap(Array):

  __slots__ = 'dofs', 'index'

  @types.apply_annotations
  def __init__(self, dofs:types.tuple[types.frozenarray], index:asarray):
    assert index.ndim == 0 and index.dtype == int
    self.dofs = dofs
    self.index = index
    length = get([len(d) for d in dofs], iax=0, item=index)
    super().__init__(args=[index], shape=(length,), dtype=int)

  def evalf(self, index):
    index, = index
    return self.dofs[index][_]

class InsertAxis(Array):

  __slots__ = 'func', 'axis', 'length'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint, length:asarray):
    assert length.ndim == 0 and length.dtype == int
    assert 0 <= axis <= func.ndim
    self.func = func
    self.axis = axis
    self.length = length
    super().__init__(args=[func, length], shape=func.shape[:axis]+(length,)+func.shape[axis:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._insertaxis(self.axis, self.length)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return InsertAxis(func, self.axis, self.length)

  def evalf(self, func, length):
    # We would like to return an array with stride zero for the inserted axis,
    # but this appears to be *slower* (checked with examples/cylinderflow.py)
    # than the implementation below.
    length, = length
    func = numpy.asarray(func)[(slice(None),)*(self.axis+1)+(None,)]
    if length != 1:
      func = numpy.repeat(func, length, self.axis+1)
    return func

  def _derivative(self, var, seen):
    return insertaxis(derivative(self.func, var, seen), self.axis, self.length)

  def _get(self, i, item):
    if i == self.axis:
      if item.isconstant and self.length.isconstant:
        assert item.eval()[0] < self.length.eval()[0]
      return self.func
    return InsertAxis(Get(self.func, i-(i>self.axis), item), self.axis-(i<self.axis), self.length)

  def _sum(self, i):
    if i == self.axis:
      return Multiply([self.func, _inflate_scalar(self.length, self.func.shape)])
    return InsertAxis(Sum(self.func, i-(i>self.axis)), self.axis-(i<self.axis), self.length)

  def _product(self):
    if self.axis == self.ndim-1:
      return Power(self.func, _inflate_scalar(self.length, self.func.shape))
    return InsertAxis(Product(self.func), self.axis, self.length)

  def _power(self, n):
    for axis in n._inserted_axes:
      if axis in self._inserted_axes:
        return InsertAxis(Power(self._uninsert(axis), n._uninsert(axis)), axis, self.shape[axis])

  def _add(self, other):
    for axis in other._inserted_axes:
      if axis in self._inserted_axes:
        return InsertAxis(Add([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

  def _multiply(self, other):
    for axis in other._inserted_axes:
      if axis in self._inserted_axes:
        return InsertAxis(Multiply([self._uninsert(axis), other._uninsert(axis)]), axis, self.shape[axis])

  def _insertaxis(self, axis, length):
    if (not length.isconstant, axis) < (not self.length.isconstant, self.axis):
      return InsertAxis(InsertAxis(self.func, axis-(axis>self.axis), length), self.axis+(axis<=self.axis), self.length)

  def _take(self, index, axis):
    if axis == self.axis:
      return InsertAxis(self.func, self.axis, index.shape[0])
    return InsertAxis(Take(self.func, index, axis-(axis>self.axis)), self.axis, self.length)

  def _takediag(self, axis, rmaxis):
    if self.axis == rmaxis:
      return self.func
    elif self.axis == axis:
      return Transpose(self.func, list(range(axis))+[rmaxis-1]+list(range(axis, rmaxis-1))+list(range(rmaxis, self.func.ndim)))
    else:
      return InsertAxis(TakeDiag(self.func, axis-(self.axis<axis), rmaxis-(self.axis<rmaxis)), self.axis-(self.axis>rmaxis), self.length)

  def _mask(self, maskvec, axis):
    if axis == self.axis:
      assert len(maskvec) == self.shape[self.axis]
      return InsertAxis(self.func, self.axis, maskvec.sum())
    return InsertAxis(Mask(self.func, maskvec, axis-(self.axis<axis)), self.axis, self.length)

  def _transpose(self, axes):
    i = axes.index(self.axis)
    return InsertAxis(Transpose(self.func, [ax-(ax>self.axis) for ax in axes[:i]+axes[i+1:]]), i, self.length)

  def _unravel(self, axis, shape):
    if axis == self.axis:
      return InsertAxis(InsertAxis(self.func, self.axis, shape[1]), self.axis, shape[0])
    else:
      return InsertAxis(Unravel(self.func, axis-(axis>self.axis), shape), self.axis+(axis<self.axis), self.length)

  @property
  def _inserted_axes(self):
    return tuple([self.axis] + [axis + (axis>=self.axis) for axis in self.func._inserted_axes])

  def _uninsert(self, axis):
    return self.func if axis == self.axis else InsertAxis(self.func._uninsert(axis-(axis>self.axis)), self.axis-(axis<self.axis), self.length)

  def _sign(self):
    return InsertAxis(Sign(self.func), self.axis, self.length)

  def _inverse(self):
    if self.axis < self.ndim-2:
      return InsertAxis(Inverse(self.func), self.axis, self.length)

  def _determinant(self):
    if self.axis < self.ndim-2:
      return InsertAxis(Determinant(self.func), self.axis, self.length)

  @property
  def blocks(self):
    return tuple((ind[:self.axis]+(Range(self.length),)+ind[self.axis:], InsertAxis(f, self.axis, self.length)) for ind, f in self.func.blocks)

class Transpose(Array):

  __slots__ = 'func', 'axes'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axes:types.tuple[types.strictint]):
    assert sorted(axes) == list(range(func.ndim))
    self.func = func
    self.axes = axes
    super().__init__(args=[func], shape=[func.shape[n] for n in axes], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if self.axes == tuple(range(self.ndim)):
      return func
    retval = func._transpose(self.axes)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Transpose(func, self.axes)

  def evalf(self, arr):
    return arr.transpose([0] + [n+1 for n in self.axes])

  def _transpose(self, axes):
    newaxes = [self.axes[i] for i in axes]
    return Transpose(self.func, newaxes)

  def _takediag(self, axis, rmaxis):
    if self.axes[axis] < self.axes[rmaxis]:
      axes = self.axes
    else:
      axes = list(self.axes)
      axes[axis], axes[rmaxis] = axes[rmaxis], axes[axis]
    assert axes[axis] < axes[rmaxis]
    return Transpose(TakeDiag(self.func, axes[axis], axes[rmaxis]), [ax-(ax>axes[rmaxis]) for ax in axes[:rmaxis]+axes[rmaxis+1:]])

  def _get(self, i, item):
    axis = self.axes[i]
    axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
    return Transpose(Get(self.func, axis, item), axes)

  def _sum(self, i):
    axis = self.axes[i]
    axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
    return Transpose(Sum(self.func, axis), axes)

  def _derivative(self, var, seen):
    return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

  def _multiply(self, other):
    other_trans = other._transpose(_invtrans(self.axes))
    if other_trans is not None:
      return Transpose(Multiply([self.func, other_trans]), self.axes)

  def _add(self, other):
    if isinstance(other, Transpose) and self.axes == other.axes:
      return Transpose(Add([self.func, other.func]), self.axes)
    other_trans = other._transpose(_invtrans(self.axes))
    if other_trans is not None:
      return Transpose(Add([self.func, other_trans]), self.axes)

  def _take(self, indices, axis):
    return Transpose(Take(self.func, indices, self.axes[axis]), self.axes)

  def _mask(self, maskvec, axis):
    return Transpose(Mask(self.func, maskvec, self.axes[axis]), self.axes)

  def _power(self, n):
    n_trans = n._transpose(_invtrans(self.axes))
    return Transpose(Power(self.func, n_trans), self.axes)

  def _sign(self):
    return Transpose(Sign(self.func), self.axes)

  def _unravel(self, axis, shape):
    orig_axis = self.axes[axis]
    axes = [ax + (ax>orig_axis) for ax in self.axes]
    axes.insert(axis+1, orig_axis+1)
    return Transpose(Unravel(self.func, orig_axis, shape), axes)

  def _product(self):
    if self.axes[-1] == self.ndim-1:
      return Transpose(Product(self.func), self.axes[:-1])

  def _determinant(self):
    if sorted(self.axes[-2:]) == [self.ndim-2, self.ndim-1]:
      return Transpose(Determinant(self.func), self.axes[:-2])

  def _inverse(self):
    if sorted(self.axes[-2:]) == [self.ndim-2, self.ndim-1]:
      return Transpose(Inverse(self.func), self.axes)

  @property
  def blocks(self):
    return tuple((tuple(ind[n] for n in self.axes), Transpose(f, self.axes)) for ind, f in self.func.blocks)

class Get(Array):

  __slots__ = 'func', 'axis', 'item'
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint, item:asarray):
    assert item.ndim == 0 and item.dtype == int
    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    if item.isconstant and numeric.isint(func.shape[axis]):
      assert 0 <= item.eval()[0] < func.shape[axis], 'item is out of bounds'
    super().__init__(args=[func, item], shape=func.shape[:axis]+func.shape[axis+1:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    item = self.item.simplified
    retval = func._get(self.axis, item)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Get(func, self.axis, item)

  def evalf(self, arr, item):
    if len(item) == 1:
      item, = item
      p = slice(None)
    elif len(arr) == 1:
      p = numpy.zeros(len(item), dtype=int)
    else:
      p = numpy.arange(len(item))
    return arr[(p,)+(slice(None),)*self.axis+(item,)]

  def _derivative(self, var, seen):
    f = derivative(self.func, var, seen)
    return get(f, self.axis, self.item)

  def _get(self, i, item):
    tryget = self.func._get(i+(i>=self.axis), item)
    if tryget is not None:
      return Get(tryget, self.axis-(i<self.axis), self.item)

class Product(Array):

  __slots__ = 'func',
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

  @property
  def simplified(self):
    if self.func.shape[-1] == 1:
      return get(self.func, self.ndim, 0).simplified
    func = self.func.simplified
    retval = func._product()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Product(func)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.product(arr, axis=-1)

  def _derivative(self, var, seen):
    grad = derivative(self.func, var, seen)
    funcs = stack([util.product(self.func[...,j] for j in range(self.func.shape[-1]) if i != j) for i in range(self.func.shape[-1])], axis=self.ndim)
    return (grad * funcs[(...,)+(_,)*var.ndim]).sum(self.ndim)

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #ext = (...,)+(_,)*len(shape)
    #return self[ext] * (derivative(self.func,var,shape,seen) / self.func[ext]).sum(self.ndim)

  def _get(self, i, item):
    func = Get(self.func, i, item)
    return Product(func)

  def _take(self, indices, axis):
    return Product(Take(self.func, indices, axis))

  def _mask(self, maskvec, axis):
    return Product(Mask(self.func, maskvec, axis))

  def _takediag(self, axis, rmaxis):
    return Product(TakeDiag(self.func, axis, rmaxis))

class ApplyTransforms(Array):

  __slots__ = 'trans',

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], points:strictevaluable=POINTS):
    self.trans = trans
    super().__init__(args=[points, trans], shape=[trans.todims], dtype=float)

  def evalf(self, points, chain):
    return transform.apply(chain, points)

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords) and len(var) > 0:
      return LinearFrom(self.trans, len(var))
    return zeros(self.shape+var.shape)

class LinearFrom(Array):

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, trans:types.strict[TransformChain], fromdims:types.strictint):
    super().__init__(args=[trans], shape=(trans.todims, fromdims), dtype=float)

  def evalf(self, chain):
    todims, fromdims = self.shape
    assert not chain or chain[0].todims == todims
    return transform.linearfrom(chain, fromdims)[_]

class Inverse(Array):
  '''
  Matrix inverse of ``func`` over the last two axes.  All other axes are
  treated element-wise.
  '''

  __slots__ = 'func',
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=float)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._inverse()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Inverse(func)

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

  def _determinant(self):
    return reciprocal(Determinant(self.func))

  def _get(self, i, item):
    if i < self.ndim - 2:
      return Inverse(Get(self.func, i, item))

  def _take(self, indices, axis):
    if axis < self.ndim - 2:
      return Inverse(Take(self.func, indices, axis))

  def _mask(self, maskvec, axis):
    if axis < self.ndim - 2:
      return Inverse(Mask(self.func, maskvec, axis))

  def _takediag(self, axis, rmaxis):
    if axis < rmaxis < self.ndim - 2:
      return Inverse(TakeDiag(self.func, axis, rmaxis))

  def _unravel(self, axis, shape):
    if axis < self.ndim-2:
      return Inverse(Unravel(self.func, axis, shape))

class Concatenate(Array):

  __slots__ = 'funcs', 'axis'
  __cache__ = '_withslices', 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.tuple[asarray], axis:types.strictint=0):
    ndim = funcs[0].ndim
    assert all(func.ndim == ndim for func in funcs)
    assert 0 <= axis < ndim
    assert all(func.shape[:axis] == funcs[0].shape[:axis] and func.shape[axis+1:] == funcs[0].shape[axis+1:] for func in funcs[1:])
    length = util.sum(func.shape[axis] for func in funcs)
    shape = funcs[0].shape[:axis] + (length,) + funcs[0].shape[axis+1:]
    dtype = _jointdtype(*[func.dtype for func in funcs])
    self.funcs = funcs
    self.axis = axis
    super().__init__(args=funcs, shape=shape, dtype=dtype)

  def edit(self, op):
    return Concatenate([op(func) for func in self.funcs], self.axis)

  @property
  def _slices(self):
    shapes = [func.shape[self.axis] for func in self.funcs]
    return tuple(map(Range, shapes, util.cumsum(shapes)))

  @property
  def _withslices(self):
    return tuple(zip(self._slices, self.funcs))

  @property
  def simplified(self):
    funcs = tuple(func.simplified for func in self.funcs if func.shape[self.axis] != 0)
    if all(iszero(func) for func in funcs):
      return zeros_like(self)
    if len(funcs) == 1:
      return funcs[0]
    if all(isinstance(func, Inflate) or iszero(func) for func in funcs):
      (dofmap, axis), *other = set((func.dofmap, func.axis) for func in funcs if isinstance(func, Inflate))
      if not other and axis != self.axis:
        # This is an Inflate-specific simplification that shouldn't appear
        # here, but currently cannot appear anywhere else due to design
        # choices. We need it here to fix a regression while awaiting a full
        # rewrite of this module to fundamentally take care of the issue.
        concat_blocks = Concatenate([Take(func, dofmap, axis) if iszero(func) else func.func for func in funcs], self.axis)
        return Inflate(concat_blocks, dofmap=dofmap, length=self.shape[axis], axis=axis).simplified
    return Concatenate(funcs, self.axis)

  def evalf(self, *arrays):
    shape = list(builtins.max(arrays, key=len).shape)
    shape[self.axis+1] = builtins.sum(array.shape[self.axis+1] for array in arrays)
    retval = numpy.empty(shape, dtype=self.dtype)
    n0 = 0
    for array in arrays:
      n1 = n0 + array.shape[self.axis+1]
      retval[(slice(None),)*(self.axis+1)+(slice(n0,n1),)] = array
      n0 = n1
    assert n0 == retval.shape[self.axis+1]
    return retval

  @property
  def blocks(self):
    blocks = []
    for (ind1, ind2), ind_f in util.gather(((ind[:self.axis], ind[self.axis+1:]), (ind[self.axis]+n, f))
      for n, func in zip(util.cumsum(func.shape[self.axis] for func in self.funcs), self.funcs)
        for ind, f in func.blocks):
      if len(ind_f) == 1:
        ind, f = ind_f[0]
      else:
        ind = Concatenate([ind for ind, f in ind_f], axis=0)
        f = Concatenate([f for ind, f in ind_f], axis=len(ind1))
      blocks.append((ind1+(ind.simplified,)+ind2, f))
    return tuple(blocks)

  def _get(self, i, item):
    if i != self.axis:
      axis = self.axis - (self.axis > i)
      return Concatenate([Get(f, i, item) for f in self.funcs], axis=axis)
    if item.isconstant:
      item, = item.eval()
      for f in self.funcs:
        if item < f.shape[i]:
          return Get(f, i, item)
        item -= f.shape[i]
      raise Exception

  def _derivative(self, var, seen):
    funcs = [derivative(func, var, seen) for func in self.funcs]
    return concatenate(funcs, axis=self.axis)

  def _multiply(self, other):
    funcs = [Multiply([func, Take(other, s, self.axis)]) for s, func in self._withslices]
    return Concatenate(funcs, self.axis)

  def _add(self, other):
    if isinstance(other, Concatenate) and self.axis == other.axis and [f1.shape[self.axis] for f1 in self.funcs] == [f2.shape[self.axis] for f2 in other.funcs]:
      other_funcs = other.funcs
    else:
      other_funcs = [Take(other, s, self.axis) for s in self._slices]
    return Concatenate([Add(f12) for f12 in zip(self.funcs, other_funcs)], self.axis)

  def _sum(self, axis):
    funcs = [Sum(func, axis) for func in self.funcs]
    if axis == self.axis:
      while len(funcs) > 1:
        funcs[-2:] = Add(funcs[-2:]),
      return funcs[0]
    return Concatenate(funcs, self.axis - (axis<self.axis))

  def _transpose(self, axes):
    funcs = [Transpose(func, axes) for func in self.funcs]
    axis = axes.index(self.axis)
    return Concatenate(funcs, axis)

  def _insertaxis(self, axis, length):
    funcs = [InsertAxis(func, axis, length) for func in self.funcs]
    return Concatenate(funcs, self.axis+(axis<=self.axis))

  def _takediag(self, axis, rmaxis):
    if self.axis == axis:
      funcs = [TakeDiag(Take(func, s, rmaxis), axis, rmaxis) for s, func in self._withslices]
      return Concatenate(funcs, axis=axis)
    elif self.axis == rmaxis:
      funcs = [TakeDiag(Take(func, s, axis), axis, rmaxis) for s, func in self._withslices]
      return Concatenate(funcs, axis=axis)
    else:
      return Concatenate([TakeDiag(f, axis, rmaxis) for f in self.funcs], axis=self.axis-(self.axis>rmaxis))

  def _take(self, indices, axis):
    if axis != self.axis:
      return Concatenate([Take(func, indices, axis) for func in self.funcs], self.axis)

  def _power(self, n):
    return Concatenate([Power(func, Take(n, s, self.axis)) for s, func in self._withslices], self.axis)

  def _diagonalize(self, axis, newaxis):
    if self.axis != axis:
      return Concatenate([Diagonalize(func, axis, newaxis) for func in self.funcs], self.axis+(newaxis<=self.axis))

  def _mask(self, maskvec, axis):
    if axis != self.axis:
      return Concatenate([Mask(func,maskvec,axis) for func in self.funcs], self.axis)
    if all(s.isconstant for s, func in self._withslices):
      return Concatenate([Mask(func, maskvec[s.eval()[0]], axis) for s, func in self._withslices], axis)

  def _unravel(self, axis, shape):
    if axis != self.axis:
      return Concatenate([Unravel(func, axis, shape) for func in self.funcs], self.axis+(self.axis>axis))

class Interpolate(Array):
  'interpolate uniformly spaced data; stepwise for now'

  __slots__ = 'xp', 'fp', 'left', 'right'

  @types.apply_annotations
  def __init__(self, x:asarray, xp:types.frozenarray, fp:types.frozenarray, left:types.strictfloat=None, right:types.strictfloat=None):
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
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray):
    assert isarray(func) and func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-2], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._determinant()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Determinant(func)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+3
    # NOTE: numpy <= 1.12 cannot compute the determinant of an array with shape [...,0,0]
    return numpy.linalg.det(arr) if arr.shape[-1] else numpy.ones(arr.shape[:-2])

  def _derivative(self, var, seen):
    Finv = swapaxes(inverse(self.func), -2, -1)
    G = derivative(self.func, var, seen)
    ext = (...,)+(_,)*var.ndim
    return self[ext] * sum(Finv[ext] * G, axis=[-2-var.ndim,-1-var.ndim])

  def _get(self, axis, item):
    return Determinant(Get(self.func, axis, item))

  def _take(self, index, axis):
    return Determinant(Take(self.func, index, axis))

  def _mask(self, maskvec, axis):
    return Determinant(Mask(self.func, maskvec, axis))

  def _takediag(self, axis, rmaxis):
    return Determinant(TakeDiag(self.func, axis, rmaxis))

class Multiply(Array):

  __slots__ = 'funcs',
  __cache__ = 'simplified', 'optimized_for_numpy', 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert func1.shape == func2.shape
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def edit(self, op):
    return Multiply([op(func) for func in self.funcs])

  @property
  def simplified(self):
    func1, func2 = [func.simplified for func in self.funcs]
    retval = func1._multiply(func2)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._multiply(func1)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Multiply([func1, func2])

  @property
  def optimized_for_numpy(self):
    func1, func2 = [func.optimized_for_numpy for func in self.funcs]
    mask = [3] * self.ndim
    for axis in func1._inserted_axes:
      mask[axis] &= 2
      func1 = func1.func
    for axis in func2._inserted_axes:
      mask[axis] &= 1
      func2 = func2.func
    if all(mask): # should always be the case after simplify
      return Einsum(func1, func2, mask)
    return Multiply([func1, func2])

  def evalf(self, arr1, arr2):
    return arr1 * arr2

  def _sum(self, axis):
    func1, func2 = self.funcs
    if self.shape[axis] == 1:
      return multiply(get(func1, axis, 0), get(func2, axis, 0))
    if axis in func1._inserted_axes:
      return multiply(func1._uninsert(axis), func2.sum(axis))
    if axis in func2._inserted_axes:
      return multiply(func1.sum(axis), func2._uninsert(axis))

  def _get(self, axis, item):
    func1, func2 = self.funcs
    return Multiply([Get(func1, axis, item), Get(func2, axis, item)])

  def _add(self, other):
    func1, func2 = self.funcs
    if other == func1:
      return Multiply([func1, Add([func2, ones_like(func2)])])
    if other == func2:
      return Multiply([func2, Add([func1, ones_like(func1)])])
    if isinstance(other, Multiply) and not self.funcs.isdisjoint(other.funcs):
      f = next(iter(self.funcs & other.funcs))
      return Multiply([f, Add(self.funcs + other.funcs - [f,f])])

  def _determinant(self):
    func1, func2 = self.funcs
    if self.shape[-2:] == (1,1):
      return Multiply([Determinant(func1), Determinant(func2)])

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

  def _takediag(self, axis, rmaxis):
    func1, func2 = self.funcs
    return Multiply([TakeDiag(func1, axis, rmaxis), TakeDiag(func2, axis, rmaxis)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Multiply([Take(func1, index, axis), Take(func2, index, axis)])

  def _mask(self, maskvec, axis):
    func1, func2 = self.funcs
    return Multiply([Mask(func1, maskvec, axis), Mask(func2, maskvec, axis)])

  def _sign(self):
    return Multiply([Sign(func) for func in self.funcs])

  def _unravel(self, axis, shape):
    return Multiply([Unravel(func, axis, shape) for func in self.funcs])

  def _inverse(self):
    func1, func2 = self.funcs
    invaxes = {self.ndim-2, self.ndim-1}
    if invaxes.issubset(func1._inserted_axes):
      return divide(Inverse(func2), func1)
    if invaxes.issubset(func2._inserted_axes):
      return divide(Inverse(func1), func2)

  @property
  def blocks(self):
    func1, func2 = self.funcs
    blocks = []
    for ind1, f1_ in func1.blocks:
      for ind2, f2 in func2.blocks:
        f1 = f1_
        indices = []
        for i, sh in enumerate(self.shape):
          if ind1[i] == ind2[i]:
            ind = ind1[i]
          elif ind1[i] == Range(sh):
            ind = ind2[i]
            f1 = Take(f1, ind, axis=i)
          elif ind2[i] == Range(sh):
            ind = ind1[i]
            f2 = Take(f2, ind, axis=i)
          elif ind1[i].isconstant and ind2[i].isconstant:
            ind, subind1, subind2 = numpy.intersect1d(ind1[i].eval()[0], ind2[i].eval()[0], return_indices=True)
            if not ind.size:
              break # blocks do not overlap
            f1 = take(f1, subind1, i)
            f2 = take(f2, subind2, i)
          else:
            warnings.warn('failed to isolate sparsely multiplied blocks', ExpensiveEvaluationWarning)
            return super().blocks
          indices.append(asarray(ind))
        else:
          assert f1.shape == f2.shape == tuple(ind.shape[0] for ind in indices)
          blocks.append((tuple(indices), Multiply([f1, f2])))
    return _gatherblocks(blocks)

class Add(Array):

  __slots__ = 'funcs',
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, funcs:types.frozenmultiset[asarray]):
    self.funcs = funcs
    func1, func2 = funcs
    assert func1.shape == func2.shape
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def edit(self, op):
    return Add([op(func) for func in self.funcs])

  @property
  def simplified(self):
    func1, func2 = [func.simplified for func in self.funcs]
    if func1 == func2:
      return multiply(func1, 2).simplified
    retval = func1._add(func2)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._add(func1)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Add([func1, func2])

  def evalf(self, arr1, arr2=None):
    return arr1 + arr2

  def _sum(self, axis):
    return Add([Sum(func, axis) for func in self.funcs])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    return derivative(func1, var, seen) + derivative(func2, var, seen)

  def _get(self, axis, item):
    func1, func2 = self.funcs
    return Add([Get(func1, axis, item), Get(func2, axis, item)])

  def _takediag(self, axis, rmaxis):
    func1, func2 = self.funcs
    return Add([TakeDiag(func1, axis, rmaxis), TakeDiag(func2, axis, rmaxis)])

  def _take(self, index, axis):
    func1, func2 = self.funcs
    return Add([Take(func1, index, axis), Take(func2, index, axis)])

  def _add(self, other):
    func1, func2 = self.funcs
    func1_other = func1._add(other)
    if func1_other is not None:
      return Add([func1_other, func2])
    func2_other = func2._add(other)
    if func2_other is not None:
      return Add([func1, func2_other])

  def _mask(self, maskvec, axis):
    func1, func2 = self.funcs
    return Add([Mask(func1, maskvec, axis), Mask(func2, maskvec, axis)])

  def _unravel(self, axis, shape):
    return Add([Unravel(func, axis, shape) for func in self.funcs])

  @property
  def blocks(self):
    return _gatherblocks(block for func in self.funcs for block in func.blocks)

class Einsum(Array):

  __slots__ = 'func1', 'func2', 'mask', '_einsumfmt'

  @types.apply_annotations
  def __init__(self, func1:asarray, func2:asarray, mask:types.tuple[types.strictint]):
    self.func1 = func1
    self.func2 = func2
    self.mask = mask # tuple of bit mask integers in {0,1,2,3}
    # - the 1-bit indicates that the axis of func1 carries over to the result
    # - the 2-bit indicates that the axis of func2 carries over to the result
    # - if neither bit is set (mask==0) this means axes are contracted
    shape = []
    i1 = i2 = 0
    for m in mask:
      assert 0 <= m <= 3, 'invalid bit mask value'
      assert 0 < m < 3 or func1.shape[i1] == func2.shape[i2], 'non matching shapes'
      if m:
        shape.append(func1.shape[i1] if m == 1 else func2.shape[i2])
      i1 += m != 2
      i2 += m != 1
    assert i1 == func1.ndim and i2 == func2.ndim
    axes = [(chr(ord('a')+i+1), m) for i, m in enumerate(mask)]
    self._einsumfmt = 'a{},a{}->a{}'.format(*[''.join(c for c, m in axes if m != ex) for ex in (2,1,0)])
    super().__init__(args=[func1, func2], shape=shape, dtype=_jointdtype(func1.dtype, func2.dtype))

  def evalf(self, arr1, arr2):
    return numpy.core.multiarray.c_einsum(self._einsumfmt, arr1, arr2)

class Sum(Array):

  __slots__ = 'axis', 'func'
  __cache__ = 'simplified', 'optimized_for_numpy', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint):
    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    super().__init__(args=[func], shape=shape, dtype=int if func.dtype == bool else func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._sum(self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Sum(func, self.axis)

  @property
  def optimized_for_numpy(self):
    func = self.func.optimized_for_numpy
    if isinstance(func, Einsum):
      mask = numpy.array(func.mask)
      axes, = mask.nonzero()
      axis = axes[self.axis]
      if mask[axis] == 3:
        mask[axis] = 0
        return Einsum(func.func1, func.func2, mask)
    return Sum(func, self.axis)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numpy.sum(arr, self.axis+1)

  def _sum(self, axis):
    trysum = self.func._sum(axis+(axis>=self.axis))
    if trysum is not None:
      return Sum(trysum, self.axis-(axis<self.axis))

  def _get(self, axis, item):
    return Sum(Get(self.func, axis+(axis>=self.axis), item), self.axis-(axis<self.axis))

  def _derivative(self, var, seen):
    return sum(derivative(self.func, var, seen), self.axis)

  @property
  def blocks(self):
    return _gatherblocks((ind[:self.axis] + ind[self.axis+1:], f.sum(self.axis)) for ind, f in self.func.blocks)

class TakeDiag(Array):

  __slots__ = 'func', 'axis', 'rmaxis'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint, rmaxis:types.strictint):
    assert func.shape[axis] == func.shape[rmaxis]
    assert 0 <= axis < rmaxis < func.ndim
    self.func = func
    self.axis = axis
    self.rmaxis = rmaxis
    super().__init__(args=[func], shape=func.shape[:rmaxis]+func.shape[rmaxis+1:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if self.shape[self.axis] == 1:
      return get(func, self.rmaxis, 0).simplified
    retval = func._takediag(self.axis, self.rmaxis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return TakeDiag(func, self.axis, self.rmaxis)

  def evalf(self, arr):
    assert arr.ndim == self.ndim+2
    return numeric.takediag(arr, self.axis+1, self.rmaxis+1)

  def _derivative(self, var, seen):
    return TakeDiag(derivative(self.func, var, seen), self.axis, self.rmaxis)

  def _get(self, axis, item):
    if axis == self.axis:
      return Get(Get(self.func, self.rmaxis, item), self.axis, item)
    return TakeDiag(Get(self.func, axis+(axis>=self.rmaxis), item), self.axis-(axis<self.axis), self.rmaxis-(axis<self.rmaxis))

  def _take(self, index, axis):
    if axis == self.axis:
      func = Take(Take(self.func, index, self.axis), index, self.rmaxis)
    else:
      func = Take(self.func, index, axis+(axis>=self.rmaxis))
    return TakeDiag(func, self.axis, self.rmaxis)

  def _mask(self, maskvec, axis):
    if axis == self.axis:
      func = Mask(Mask(self.func, maskvec, self.axis), maskvec, self.rmaxis)
    else:
      func = Mask(self.func, maskvec, axis+(axis>=self.rmaxis))
    return TakeDiag(func, self.axis, self.rmaxis)

  def _sum(self, axis):
    if axis != self.axis:
      return TakeDiag(Sum(self.func, axis+(axis>=self.rmaxis)), self.axis-(axis<self.axis), self.rmaxis-(axis<self.rmaxis))

  @property
  def blocks(self):
    blocks = []
    for ind, f in self.func.blocks:
      if ind[self.axis] != ind[self.rmaxis]:
        if ind[self.rmaxis] == Range(self.func.shape[self.rmaxis]):
          f = Take(f, ind[self.axis], axis=self.rmaxis)
        elif ind[self.axis] == Range(self.func.shape[self.axis]):
          f = Take(f, ind[self.rmaxis], axis=self.axis)
          ind = ind[:self.axis] + (ind[self.rmaxis],) + ind[self.axis+1:]
        elif ind[self.axis].isconstant and ind[self.rmaxis].isconstant:
          newind, subind1, subind2 = numpy.intersect1d(ind[self.axis].eval()[0], ind[self.rmaxis].eval()[0], return_indices=True)
          if not newind.size:
            break # blocks do not overlap
          f = Take(Take(f, subind1, axis=self.axis), subind2, axis=self.rmaxis)
          ind = ind[:self.axis] + (asarray(newind),) + ind[self.axis+1:]
        else:
          warnings.warn('failed to preserve takediag sparsity', ExpensiveEvaluationWarning)
          return super().blocks
      blocks.append((ind[:self.rmaxis] + ind[self.rmaxis+1:], TakeDiag(f, self.axis, self.rmaxis)))
    return tuple(blocks)

class Take(Array):

  __slots__ = 'func', 'axis', 'indices'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, indices:asarray, axis:types.strictint):
    assert indices.ndim == 1 and indices.dtype == int
    assert 0 <= axis < func.ndim
    self.func = func
    self.axis = axis
    self.indices = indices
    shape = func.shape[:axis] + indices.shape + func.shape[axis+1:]
    super().__init__(args=[func,indices], shape=shape, dtype=func.dtype)

  @property
  def simplified(self):
    if self.shape[self.axis] == 0:
      return zeros(self.shape, dtype=self.dtype)
    func = self.func.simplified
    indices = self.indices.simplified
    length = self.func.shape[self.axis]
    if indices == Range(length):
      return func
    if indices.isconstant:
      indices_, = indices.eval()
      ineg, = numpy.less(indices_, 0).nonzero()
      if numeric.isint(length):
        if len(ineg):
          indices_ = indices_.copy()
          indices_[ineg] += length
          indices = Constant(types.frozenarray(indices_, copy=False))
        if numpy.less(indices_[ineg], 0).any() or numpy.greater_equal(indices_, length).any():
          raise IndexError('indices out of bounds: 0 !< {} !<= {}'.format(indices_, length))
        if numpy.greater(numpy.diff(indices_), 0).all():
          mask = numpy.zeros(length, dtype=bool)
          mask[indices_] = True
          return Mask(func, mask, self.axis).simplified
      elif len(ineg):
        raise IndexError('negative indices only allowed for constant-length axes')
    retval = func._take(indices, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Take(func, indices, self.axis)

  def evalf(self, arr, indices):
    if indices.shape[0] != 1:
      raise NotImplementedError('non element-constant indexing not supported yet')
    return types.frozenarray(numpy.take(arr, indices[0], self.axis+1), copy=False)

  def _derivative(self, var, seen):
    return take(derivative(self.func, var, seen), self.indices, self.axis)

  def _get(self, axis, item):
    if axis == self.axis:
      return Get(self.func, axis, Get(self.indices, 0, item))
    return Take(Get(self.func, axis, item), self.indices, self.axis-(axis<self.axis))

  def _take(self, index, axis):
    if axis == self.axis:
      return Take(self.func, self.indices[index], axis)
    trytake = self.func._take(index, axis)
    if trytake is not None:
      return Take(trytake, self.indices, self.axis)

  def _sum(self, axis):
    if axis != self.axis:
      return Take(Sum(self.func, axis), self.indices, self.axis-(axis<self.axis))

  @property
  def blocks(self):
    fullrange = Range(self.shape[self.axis])
    if not all(ind[self.axis] == fullrange for ind, f in self.func.blocks):
      return super().blocks
    return tuple((ind[:self.axis]+(fullrange,)+ind[self.axis+1:], Take(f, self.indices, self.axis)) for ind, f in self.func.blocks)

class Power(Array):

  __slots__ = 'func', 'power'
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray, power:asarray):
    assert func.shape == power.shape
    self.func = func
    self.power = power
    super().__init__(args=[func,power], shape=func.shape, dtype=float)

  @property
  def simplified(self):
    func = self.func.simplified
    power = self.power.simplified
    if iszero(power):
      return ones_like(self).simplified
    retval = func._power(power)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Power(func, power)

  def evalf(self, base, exp):
    return numeric.power(base, exp)

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    if self.power.isconstant:
      p, = self.power.eval()
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

  def _get(self, axis, item):
    return Power(Get(self.func, axis, item), Get(self.power, axis, item))

  def _takediag(self, axis, rmaxis):
    return Power(TakeDiag(self.func, axis, rmaxis), TakeDiag(self.power, axis, rmaxis))

  def _take(self, index, axis):
    return Power(Take(self.func, index, axis), Take(self.power, index, axis))

  def _mask(self, maskvec, axis):
    return Power(Mask(self.func, maskvec, axis), Mask(self.power, maskvec, axis))

  def _unravel(self, axis, shape):
    return Power(Unravel(self.func, axis, shape), Unravel(self.power, axis, shape))

  def _product(self):
    if self.ndim-1 in self.power._inserted_axes:
      return Power(Product(self.func), self.power._uninsert(self.ndim-1))

class Pointwise(Array):
  '''
  Abstract base class for pointwise array functions.
  '''

  __slots__ = 'args',
  __cache__ = 'simplified',

  deriv = None

  @types.apply_annotations
  def __init__(self, *args:asarrays):
    retval = self.evalf(*[numpy.ones((), dtype=arg.dtype) for arg in args])
    shapes = set(arg.shape for arg in args)
    assert len(shapes) == 1, 'pointwise arguments have inconsistent shapes'
    shape, = shapes
    self.args = args
    super().__init__(args=args, shape=shape, dtype=retval.dtype)

  @property
  def simplified(self):
    args = [arg.simplified for arg in self.args]
    if all(arg.isconstant for arg in args):
      retval, = self.evalf(*[arg.eval() for arg in args])
      return Constant(retval).simplified
    return self.__class__(*args)

  def _derivative(self, var, seen):
    if self.deriv is None:
      raise NotImplementedError('derivative is not defined for this operator')
    return util.sum(deriv(*self.args)[(...,)+(_,)*var.ndim] * derivative(arg, var, seen) for arg, deriv in zip(self.args, self.deriv))

  def _takediag(self, axis, rmaxis):
    return self.__class__(*[TakeDiag(arg, axis, rmaxis) for arg in self.args])

  def _get(self, axis, item):
    return self.__class__(*[Get(arg, axis, item) for arg in self.args])

  def _take(self, index, axis):
    return self.__class__(*[Take(arg, index, axis) for arg in self.args])

  def _mask(self, maskvec, axis):
    return self.__class__(*[Mask(arg, maskvec, axis) for arg in self.args])

  def _unravel(self, axis, shape):
    return self.__class__(*[Unravel(arg, axis, shape) for arg in self.args])

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
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._sign()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Sign(func)

  def evalf(self, arr):
    return numpy.sign(arr)

  def _takediag(self, axis, rmaxis):
    return Sign(TakeDiag(self.func, axis, rmaxis))

  def _get(self, axis, item):
    return Sign(Get(self.func, axis, item))

  def _take(self, index, axis):
    return Sign(Take(self.func, index, axis))

  def _mask(self, maskvec, axis):
    return Sign(Mask(self.func, maskvec, axis))

  def _sign(self):
    return self

  def _unravel(self, axis, shape):
    return Sign(Unravel(self.func, axis, shape))

  def _derivative(self, var, seen):
    return Zeros(self.shape + var.shape, dtype=self.dtype)

  @property
  def blocks(self):
    return tuple((ind, Sign(f)) for ind, f in self.func.blocks)

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
    assert points.ndim == 1 and expect.ndim == 2 and expect.shape[1] == points.shape[0]
    super().__init__(args=[points, expect], shape=expect.shape[:1], dtype=int)

  def evalf(self, points, expect):
    assert numpy.equal(points, expect).all(), 'illegal point set'
    return numpy.eye(len(points), dtype=int)

class Elemwise(Array):

  __slots__ = 'data',
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, data:types.tuple[types.frozenarray], index:asarray, dtype:asdtype):
    self.data = data
    ndim = self.data[0].ndim
    shape = tuple(get([d.shape[i] for d in self.data], iax=0, item=index) for i in range(ndim))
    super().__init__(args=[index], shape=shape, dtype=dtype)

  def evalf(self, index):
    index, = index
    return self.data[index][_]

  @property
  def simplified(self):
    if all(map(numeric.isint, self.shape)) and all(numpy.equal(self.data[0], self.data[i]).all() for i in range(1, len(self.data))):
      return Constant(self.data[0])
    return self

class Eig(Evaluable):

  __slots__ = 'symmetric', 'func'
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, func:asarray, symmetric:bool=False):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.symmetric = symmetric
    self.func = func
    super().__init__(args=[func])

  def __len__(self):
    return 2

  def __iter__(self):
    yield ArrayFromTuple(self, index=0, shape=self.func.shape[:-1], dtype=complex if not self.symmetric else float)
    yield ArrayFromTuple(self, index=1, shape=self.func.shape, dtype=complex if not self.symmetric or self.func.dtype == complex else float)

  @property
  def simplified(self):
    func = self.func.simplified
    retval = func._eig(self.symmetric)
    if retval is not None:
      assert len(retval) == 2
      return retval.simplified
    return Eig(func, self.symmetric)

  def evalf(self, arr):
    return (numpy.linalg.eigh if self.symmetric else numpy.linalg.eig)(arr)

class ArrayFromTuple(Array):

  __slots__ = 'arrays', 'index'

  @types.apply_annotations
  def __init__(self, arrays:strictevaluable, index:types.strictint, shape:asshape, dtype:asdtype):
    assert 0 <= index < len(arrays)
    self.arrays = arrays
    self.index = index
    super().__init__(args=[arrays], shape=shape, dtype=dtype)

  def evalf(self, arrays):
    assert isinstance(arrays, tuple)
    return arrays[self.index]

class Zeros(Array):
  'zero'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, shape:asshape, dtype:asdtype):
    super().__init__(args=[asarray(sh) for sh in shape], shape=shape, dtype=dtype)

  def evalf(self, *shape):
    if shape:
      shape, = zip(*shape)
    return numpy.zeros((1,)+shape, dtype=self.dtype)

  @property
  def blocks(self):
    return ()

  def edit(self, op):
    return Zeros(tuple(map(op, self.shape)), self.dtype)

  def _add(self, other):
    return other

  def _multiply(self, other):
    return self

  def _diagonalize(self, axis, newaxis):
    return Zeros(self.shape[:newaxis]+(self.shape[axis],)+self.shape[newaxis:], dtype=self.dtype)

  def _sum(self, axis):
    return Zeros(self.shape[:axis] + self.shape[axis+1:], dtype=int if self.dtype == bool else self.dtype)

  def _transpose(self, axes):
    shape = [self.shape[n] for n in axes]
    return Zeros(shape, dtype=self.dtype)

  def _insertaxis(self, axis, length):
    return Zeros(self.shape[:axis]+(length,)+self.shape[axis:], self.dtype)

  def _get(self, i, item):
    return Zeros(self.shape[:i] + self.shape[i+1:], dtype=self.dtype)

  def _takediag(self, axis, rmaxis):
    return Zeros(self.shape[:rmaxis]+self.shape[rmaxis+1:], dtype=self.dtype)

  def _take(self, index, axis):
    return Zeros(self.shape[:axis] + index.shape + self.shape[axis+1:], dtype=self.dtype)

  def _inflate(self, dofmap, length, axis):
    return Zeros(self.shape[:axis] + (length,) + self.shape[axis+1:], dtype=self.dtype)

  def _power(self, n):
    return self

  def _mask(self, maskvec, axis):
    return Zeros(self.shape[:axis] + (maskvec.sum(),) + self.shape[axis+1:], dtype=self.dtype)

  def _unravel(self, axis, shape):
    shape = self.shape[:axis] + shape + self.shape[axis+1:]
    return Zeros(shape, dtype=self.dtype)

  def _ravel(self, axis):
    return Zeros(self.shape[:axis] + (self.shape[axis]*self.shape[axis+1],) + self.shape[axis+2:], self.dtype)

  def _determinant(self):
    if self.shape[-1] == 0:
      return ones(self.shape[:-2], self.dtype)
    else:
      return Zeros(self.shape[:-2], self.dtype)

  def _kronecker(self, axis, length, pos):
    return Zeros(self.shape[:axis] + (length,) + self.shape[axis:], self.dtype)

class Inflate(Array):

  __slots__ = 'func', 'dofmap', 'length', 'axis'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, dofmap:asarray, length:types.strictint, axis:types.strictint):
    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.axis = axis
    assert 0 <= axis < func.ndim
    assert func.shape[axis] == dofmap.shape[0]
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    super().__init__(args=[func,dofmap], shape=shape, dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    dofmap = self.dofmap.simplified
    retval = func._inflate(dofmap, self.length, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Inflate(func, dofmap, self.length, self.axis)

  def evalf(self, array, indices):
    assert indices.shape[0] == 1
    indices, = indices
    assert array.ndim == self.ndim+1
    warnings.warn('using explicit inflation; this is usually a bug.', ExpensiveEvaluationWarning)
    shape = list(array.shape)
    shape[self.axis+1] = self.length
    inflated = numpy.zeros(shape, dtype=self.dtype)
    numpy.add.at(inflated, (slice(None),)*(self.axis+1)+(indices,), array)
    return inflated

  @property
  def blocks(self):
    return tuple((ind[:self.axis] + (Take(self.dofmap, ind[self.axis], axis=0).simplified,) + ind[self.axis+1:], f) for ind, f in self.func.blocks)

  def _mask(self, maskvec, axis):
    if axis != self.axis:
      return Inflate(Mask(self.func, maskvec, axis), self.dofmap, self.length, self.axis)
    newlength = maskvec.sum()
    selection = Take(maskvec, self.dofmap, axis=0)
    renumber = numpy.empty(len(maskvec), dtype=int)
    renumber[:] = newlength # out of bounds
    renumber[numpy.asarray(maskvec)] = numpy.arange(newlength)
    newdofmap = Take(renumber, Take(self.dofmap, Find(selection), axis=0), axis=0)
    newfunc = Take(self.func, Find(selection), axis=self.axis)
    return Inflate(newfunc, newdofmap, newlength, self.axis)

  def _inflate(self, dofmap, length, axis):
    if axis == self.axis:
      return Inflate(self.func, Take(dofmap, self.dofmap, 0), length, axis)
    if axis < self.axis:
      return Inflate(Inflate(self.func, dofmap, length, axis), self.dofmap, self.length, self.axis)

  def _derivative(self, var, seen):
    return Inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.axis)

  def _transpose(self, axes):
    axis = axes.index(self.axis)
    return Inflate(Transpose(self.func, axes), self.dofmap, self.length, axis)

  def _insertaxis(self, axis, length):
    return Inflate(InsertAxis(self.func, axis, length), self.dofmap, self.length, self.axis+(axis<=self.axis))

  def _get(self, axis, item):
    if axis != self.axis:
      return Inflate(Get(self.func,axis,item), self.dofmap, self.length, self.axis-(axis<self.axis))
    if self.dofmap.isconstant and item.isconstant:
      dofmap, = self.dofmap.eval()
      item, = item.eval()
      return Get(self.func, axis, tuple(dofmap).index(item)) if item in dofmap \
        else Zeros(self.shape[:axis]+self.shape[axis+1:], self.dtype)

  def _multiply(self, other):
    return Inflate(Multiply([self.func, Take(other, self.dofmap, self.axis)]), self.dofmap, self.length, self.axis)

  def _add(self, other):
    if isinstance(other, Inflate) and self.axis == other.axis and self.dofmap == other.dofmap:
      return Inflate(Add([self.func, other.func]), self.dofmap, self.length, self.axis)

  def _power(self, n):
    return Inflate(Power(self.func, Take(n, indices=self.dofmap, axis=self.axis)), self.dofmap, self.length, self.axis)

  def _takediag(self, axis, rmaxis):
    if self.axis == axis:
      return Inflate(TakeDiag(take(self.func, self.dofmap, rmaxis), axis, rmaxis), self.dofmap, self.length, axis)
    elif self.axis == rmaxis:
      return Inflate(TakeDiag(take(self.func, self.dofmap, axis), axis, rmaxis), self.dofmap, self.length, axis)
    else:
      return Inflate(TakeDiag(self.func, axis, rmaxis), self.dofmap, self.length, self.axis-(self.axis>rmaxis))

  def _take(self, index, axis):
    if axis != self.axis:
      return Inflate(Take(self.func, index, axis), self.dofmap, self.length, self.axis)
    if index == self.dofmap:
      return self.func

  def _diagonalize(self, axis, newaxis):
    if self.axis != axis:
      return Inflate(Diagonalize(self.func, axis, newaxis), self.dofmap, self.length, self.axis+(newaxis<=self.axis))

  def _sum(self, axis):
    arr = Sum(self.func, axis)
    if axis == self.axis:
      return arr
    return Inflate(arr, self.dofmap, self.length, self.axis-(axis<self.axis))

  def _unravel(self, axis, shape):
    if axis != self.axis:
      return Inflate(Unravel(self.func, axis, shape), self.dofmap, self.length, self.axis+(self.axis>axis))

  def _kronecker(self, axis, length, pos):
    return Inflate(Kronecker(self.func, axis, length, pos), self.dofmap, self.length, self.axis+(axis<=self.axis))

  def _sign(self):
    return Inflate(Sign(self.func), self.dofmap, self.length, self.axis)

  def _inverse(self):
    if self.axis < self.ndim-2:
      return Inflate(Inverse(self.func), self.dofmap, self.length, self.axis)

  def _determinant(self):
    if self.axis < self.ndim-2:
      return Inflate(Determinant(self.func), self.dofmap, self.length, self.axis)

  def _product(self):
    if self.axis < self.ndim-1:
      return Inflate(Product(self.func), self.dofmap, self.length, self.axis)

class Diagonalize(Array):

  __slots__ = 'func', 'axis', 'newaxis'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis=types.strictint, newaxis=types.strictint):
    assert 0 <= axis < newaxis <= func.ndim
    self.func = func
    self.axis = axis
    self.newaxis = newaxis
    super().__init__(args=[func], shape=func.shape[:newaxis]+(func.shape[axis],)+func.shape[newaxis:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if func.shape[self.axis] == 1:
      return insertaxis(func, self.newaxis, 1).simplified
    retval = func._diagonalize(self.axis, self.newaxis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Diagonalize(func, self.axis, self.newaxis)

  def evalf(self, arr):
    assert arr.ndim == self.ndim
    return numeric.diagonalize(arr, self.axis+1, self.newaxis+1)

  def _derivative(self, var, seen):
    return diagonalize(derivative(self.func, var, seen), self.axis, self.newaxis)

  def _get(self, i, item):
    if i != self.axis and i != self.newaxis:
      return Diagonalize(Get(self.func, i-(i>self.newaxis), item), self.axis-(i<self.axis), self.newaxis-(i<self.newaxis))
    return kronecker(Get(self.func, self.axis, item), axis=self.axis if i == self.newaxis else self.newaxis-1, length=self.shape[i], pos=item)

  def _inverse(self):
    if self.axis == self.func.ndim-1 and self.newaxis == self.ndim-1:
      return Diagonalize(reciprocal(self.func), self.axis, self.newaxis)

  def _determinant(self):
    if self.axis == self.func.ndim-1 and self.newaxis == self.ndim-1:
      return Product(Transpose(self.func, list(range(self.axis))+list(range(self.axis+1,self.func.ndim))+[self.axis]))

  def _multiply(self, other):
    return Diagonalize(Multiply([self.func, TakeDiag(other, self.axis, self.newaxis)]), self.axis, self.newaxis)

  def _add(self, other):
    if isinstance(other, Diagonalize) and other.axis == self.axis and other.newaxis == self.newaxis:
      return Diagonalize(Add([self.func, other.func]), self.axis, self.newaxis)

  def _sum(self, axis):
    if axis == self.newaxis:
      return self.func
    if axis == self.axis:
      return Transpose(self.func, list(range(self.axis))+list(range(self.axis+1,self.newaxis))+[self.axis]+list(range(self.newaxis,self.func.ndim)))
    return Diagonalize(Sum(self.func, axis-(axis>self.newaxis)), self.axis-(axis<self.axis), self.newaxis-(axis<self.newaxis))

  def _transpose(self, axes):
    axis = axes.index(self.axis)
    newaxis = axes.index(self.newaxis)
    if newaxis < axis:
      axes = list(axes)
      axes[axis] = self.newaxis
      axes[newaxis] = self.axis
      axis, newaxis = newaxis, axis
    newaxes = [ax-(ax>self.newaxis) for ax in axes[:newaxis]+axes[newaxis+1:]]
    return Diagonalize(Transpose(self.func, newaxes), axis, newaxis)

  def _insertaxis(self, axis, length):
    return Diagonalize(InsertAxis(self.func, axis-(axis>self.newaxis), length), self.axis+(axis<=self.axis), self.newaxis+(axis<=self.newaxis))

  def _takediag(self, axis, rmaxis):
    if self.axis == axis and self.newaxis == rmaxis:
      return self.func
    if self.newaxis == axis: # self.axis < self.newaxis = axis < rmaxis
      takeaxes = self.axis, rmaxis-1
      diagaxes = self.axis, self.newaxis
    elif self.newaxis == rmaxis:
      takeaxes = diagaxes = sorted([axis, self.axis])
    elif self.axis == rmaxis: # axis < rmaxis = self.axis < self.newaxis
      takeaxes = axis, rmaxis
      diagaxes = axis, self.newaxis-1
    elif self.newaxis > rmaxis: # axis < rmaxis < self.newaxis
      takeaxes = axis, rmaxis
      diagaxes = self.axis-(self.axis>=rmaxis), self.newaxis-1
    else: # self.axis < self.newaxis < rmaxis
      takeaxes = axis-(axis>self.newaxis), rmaxis-1
      diagaxes = self.axis, self.newaxis
    return Diagonalize(TakeDiag(self.func, *takeaxes), *diagaxes)

  def _take(self, index, axis):
    if axis not in (self.axis, self.newaxis):
      return Diagonalize(Take(self.func, index, axis-(axis>self.newaxis)), self.axis, self.newaxis)
    if numeric.isint(self.func.shape[self.axis]):
      diag = Diagonalize(Take(self.func, index, self.axis), self.axis, self.newaxis)
      return Inflate(diag, index, self.func.shape[self.axis], self.newaxis if axis == self.axis else self.axis)

  def _mask(self, maskvec, axis):
    if axis not in (self.axis, self.newaxis):
      return Diagonalize(Mask(self.func, maskvec, axis-(axis>self.newaxis)), self.axis, self.newaxis)
    indices, = numpy.where(maskvec)
    if not numpy.equal(numpy.diff(indices), 1).all():
      return
    # consecutive sub-block
    ax = self.axis if axis == self.newaxis else self.newaxis
    masked = Diagonalize(Mask(self.func, maskvec, self.axis), self.axis, self.newaxis)
    return Concatenate([Zeros(masked.shape[:ax] + (indices[0],) + masked.shape[ax+1:], dtype=self.dtype), masked, Zeros(masked.shape[:ax] + (self.shape[ax]-(indices[-1]+1),) + masked.shape[ax+1:], dtype=self.dtype)], axis=ax)

  def _unravel(self, axis, shape):
    if axis == self.axis or axis == self.newaxis:
      diag = Diagonalize(Diagonalize(Unravel(self.func, self.axis, shape), self.axis, self.newaxis+1), self.axis+1, self.newaxis+2)
      return Ravel(diag, self.newaxis+1 if axis == self.axis else self.axis)
    else:
      return Diagonalize(Unravel(self.func, axis-(axis>self.newaxis), shape), self.axis+(axis<self.axis), self.newaxis+(axis<self.newaxis))

  def _sign(self):
    return Diagonalize(Sign(self.func), self.axis, self.newaxis)

  def _power(self, n):
    return Diagonalize(Power(self.func, TakeDiag(n, self.axis, self.newaxis)), self.axis, self.newaxis)

  def _product(self):
    if self.newaxis < self.ndim-1:
      return Diagonalize(Product(self.func), self.axis, self.newaxis)
    elif numeric.isint(self.shape[self.axis]) and self.shape[self.axis] > 1:
      return Zeros(self.shape[:-1], dtype=self.dtype)

  @property
  def blocks(self):
    return tuple((ind[:self.newaxis] + (ind[self.axis],) + ind[self.newaxis:], Diagonalize(f, self.axis, self.newaxis)) for ind, f in self.func.blocks)

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
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return trigtangent(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf(self, angle):
    return numpy.array([numpy.cos(angle), numpy.sin(angle)]).T

class TrigTangent(Array):
  '-sin, cos'

  __slots__ = 'angle',

  @types.apply_annotations
  def __init__(self, angle:asarray):
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return -trignormal(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf(self, angle):
    return numpy.array([-numpy.sin(angle), numpy.cos(angle)]).T

class Find(Array):
  'indices of boolean index vector'

  __slots__ = 'where',

  @types.apply_annotations
  def __init__(self, where:asarray):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    self.where = where
    super().__init__(args=[where], shape=[where.sum()], dtype=int)

  def evalf(self, where):
    assert where.shape[0] == 1
    where, = where
    index, = where.nonzero()
    return index[_]

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

  >>> from nutils import function
  >>> a = function.Argument('x', [])
  >>> b = function.Argument('y', [])
  >>> f = a**3 + b**2
  >>> function.derivative(f, a).simplified == (3.*a**2).simplified
  True

  Furthermore, derivatives to the local cooardinates are remembered and applied
  to the replacement when using :func:`replace_arguments`:

  >>> from nutils import mesh
  >>> domain, x = mesh.rectilinear([2,2])
  >>> basis = domain.basis('spline', degree=2)
  >>> c = function.Argument('c', basis.shape)
  >>> replace_arguments(c.grad(x), dict(c=basis)) == basis.grad(x)
  True

  Args
  ----
  name : :class:`str`
      The Identifier of this argument.
  shape : :class:`tuple` of :class:`int`\\s
      The shape of this argument.
  nderiv : :class:`int`, non-negative
      Number of times a derivative to the local coordinates is taken.  Default:
      ``0``.
  '''

  __slots__ = '_name', '_nderiv'
  __cache__ = 'prepare_eval'

  @types.apply_annotations
  def __init__(self, name:types.strictstr, shape:asshape, nderiv:types.strictint=0):
    self._name = name
    self._nderiv = nderiv
    super().__init__(args=[EVALARGS], shape=shape, dtype=float)

  def evalf(self, evalargs):
    assert self._nderiv == 0
    try:
      value = evalargs[self._name]
    except KeyError:
      raise ValueError('argument {!r} missing'.format(self._name))
    else:
      assert numeric.isarray(value) and value.shape == self.shape
      return value[_]

  def _derivative(self, var, seen):
    if isinstance(var, Argument) and var._name == self._name:
      assert var._nderiv == 0 and self.shape[:self.ndim-self._nderiv] == var.shape
      if self._nderiv:
        return zeros(self.shape+var.shape)
      result = _inflate_scalar(1., self.shape)
      for i, sh in enumerate(self.shape):
        result = diagonalize(result, i, i+self.ndim)
      return result
    elif isinstance(var, LocalCoords):
      return Argument(self._name, self.shape+var.shape, self._nderiv+1)
    else:
      return zeros(self.shape+var.shape)

  def __str__(self):
    return '{} {!r} <{}>'.format(self.__class__.__name__, self._name, ','.join(map(str, self.shape)))

  @util.positional_only
  def prepare_eval(self, kwargs=...):
    return zeros_like(self) if self._nderiv > 0 else self

class LocalCoords(DerivativeTargetBase):
  'local coords derivative target'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__(args=[], shape=[ndims], dtype=float)

  def evalf(self):
    raise Exception('LocalCoords should not be evaluated')

class DelayedJacobian(Array):
  '''
  Placeholder for :func:`jacobian` until the dimension of the
  :class:`nutils.topology.Topology` where this functions is being evaluated is
  known.  The replacing is carried out by :meth:`Evaluable.prepare_eval`.
  '''

  __slots__ = '_geom', '_derivativestack'
  __cache__ = 'prepare_eval'

  @types.apply_annotations
  def __init__(self, geom:asarray, *derivativestack):
    self._geom = geom
    self._derivativestack = derivativestack
    super().__init__(args=[geom], shape=[n for var in derivativestack for n in var.shape], dtype=float)

  def evalf(self):
    raise Exception('DelayedJacobian should not be evaluated')

  def _derivative(self, var, seen):
    if iszero(derivative(self._geom, var, seen)):
      return zeros(self.shape + var.shape)
    return DelayedJacobian(self._geom, *self._derivativestack, var)

  @util.positional_only
  def prepare_eval(self, *, ndims, kwargs=...):
    jac = functools.reduce(derivative, self._derivativestack, asarray(jacobian(self._geom, ndims)))
    return jac.prepare_eval(ndims=ndims, **kwargs)

class Ravel(Array):

  __slots__ = 'func', 'axis'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint):
    assert 0 <= axis < func.ndim-1
    self.func = func
    self.axis = axis
    super().__init__(args=[func], shape=func.shape[:axis]+(func.shape[axis]*func.shape[axis+1],)+func.shape[axis+2:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if func.shape[self.axis] == 1:
      return get(func, self.axis, 0).simplified
    if func.shape[self.axis+1] == 1:
      return get(func, self.axis+1, 0).simplified
    retval = func._ravel(self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Ravel(func, self.axis)

  def evalf(self, f):
    return f.reshape(f.shape[:self.axis+1] + (f.shape[self.axis+1]*f.shape[self.axis+2],) + f.shape[self.axis+3:])

  def _multiply(self, other):
    if isinstance(other, Ravel) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return Ravel(Multiply([self.func, other.func]), self.axis)
    return Ravel(Multiply([self.func, Unravel(other, self.axis, self.func.shape[self.axis:self.axis+2])]), self.axis)

  def _add(self, other):
    if isinstance(other, Ravel) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return Ravel(Add([self.func, other.func]), self.axis)

  def _get(self, i, item):
    if i != self.axis:
      return Ravel(Get(self.func, i+(i>self.axis), item), self.axis-(i<self.axis))
    if item.isconstant and numeric.isint(self.func.shape[self.axis+1]):
      item, = item.eval()
      i, j = divmod(item, self.func.shape[self.axis+1])
      return Get(Get(self.func, self.axis, i), self.axis, j)

  def _sum(self, axis):
    if axis == self.axis:
      return Sum(Sum(self.func, axis), axis)
    return Ravel(Sum(self.func, axis+(axis>self.axis)), self.axis-(axis<self.axis))

  def _derivative(self, var, seen):
    return ravel(derivative(self.func, var, seen), axis=self.axis)

  def _transpose(self, axes):
    ravelaxis = axes.index(self.axis)
    funcaxes = [ax+(ax>self.axis) for ax in axes]
    funcaxes = funcaxes[:ravelaxis+1] + [self.axis+1] + funcaxes[ravelaxis+1:]
    return Ravel(Transpose(self.func, funcaxes), ravelaxis)

  def _takediag(self, axis, rmaxis):
    if not {self.axis, self.axis+1} & {axis, rmaxis}:
      return Ravel(TakeDiag(self.func, axis+(axis>self.axis), rmaxis+(rmaxis>self.axis)), self.axis-(self.axis>rmaxis))

  def _take(self, index, axis):
    if axis != self.axis:
      return Ravel(Take(self.func, index, axis+(axis>self.axis)), self.axis)

  def _unravel(self, axis, shape):
    if axis != self.axis:
      return Ravel(Unravel(self.func, axis+(axis>self.axis), shape), self.axis+(self.axis>axis))
    elif shape == self.func.shape[axis:axis+2]:
      return self.func

  def _mask(self, maskvec, axis):
    if axis != self.axis:
      return Ravel(Mask(self.func, maskvec, axis+(axis>self.axis)), self.axis)

  def _inflate(self, dofmap, length, axis):
    if axis != self.axis:
      return Ravel(Inflate(self.func, dofmap, length, axis=axis+(axis>self.axis)), self.axis)

  def _diagonalize(self, axis, newaxis):
    if axis != self.axis:
      return Ravel(Diagonalize(self.func, axis+(axis>self.axis), newaxis+(newaxis>self.axis)), self.axis+(newaxis<=self.axis))

  def _kronecker(self, axis, length, pos):
    return Ravel(Kronecker(self.func, axis+(axis>self.axis), length, pos), self.axis+(axis<=self.axis))

  def _insertaxis(self, axis, length):
    return Ravel(InsertAxis(self.func, axis+(axis>self.axis), length), self.axis+(axis<=self.axis))

  def _power(self, n):
    return Ravel(Power(self.func, Unravel(n, self.axis, self.func.shape[self.axis:self.axis+2])), self.axis)

  def _sign(self):
    return Ravel(Sign(self.func), self.axis)

  def _inverse(self):
    if self.axis < self.ndim-2:
      return Ravel(Inverse(self.func), self.axis)

  def _product(self):
    if self.axis == self.ndim-1:
      return Product(Product(self.func))
    return Ravel(Product(self.func), self.axis)

  def _determinant(self):
    if self.axis < self.ndim-2:
      return Ravel(Determinant(self.func), self.axis)

  @property
  def blocks(self):
    return tuple(((ind[:self.axis] + (ravel(ind[self.axis][:,_] * self.func.shape[self.axis+1] + ind[self.axis+1][_,:], axis=0),) + ind[self.axis+2:]), ravel(f, axis=self.axis)) for ind, f in self.func.blocks)

class Unravel(Array):

  __slots__ = 'func', 'axis', 'unravelshape'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint, shape:asshape):
    assert 0 <= axis < func.ndim
    assert func.shape[axis] == numpy.product(shape)
    assert len(shape) == 2
    self.func = func
    self.axis = axis
    self.unravelshape = shape
    super().__init__(args=[func]+[asarray(sh) for sh in shape], shape=func.shape[:axis]+shape+func.shape[axis+1:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if self.shape[self.axis] == 1:
      return InsertAxis(func, self.axis, 1).simplified
    if self.shape[self.axis+1] == 1:
      return InsertAxis(func, self.axis+1, 1).simplified
    retval = func._unravel(self.axis, self.unravelshape)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Unravel(func, self.axis, self.unravelshape)

  def _derivative(self, var, seen):
    return unravel(derivative(self.func, var, seen), axis=self.axis, shape=self.unravelshape)

  def evalf(self, f, sh1, sh2):
    sh1, = sh1
    sh2, = sh2
    return f.reshape(f.shape[:self.axis+1]+(sh1, sh2)+f.shape[self.axis+2:])

  def _get(self, axis, item):
    if not self.axis <= axis < self.axis+2:
      return Unravel(Get(self.func, axis-(axis>self.axis), item), self.axis-(axis<self.axis), self.unravelshape)

  def _determinant(self):
    if self.axis < self.func.ndim-2:
      return Unravel(Determinant(self.func), self.axis, self.unravelshape)

  def _takediag(self, axis, rmaxis):
    if not self.axis <= axis < self.axis+2 and not self.axis <= rmaxis < self.axis+2:
      return Unravel(TakeDiag(self.func, axis-(axis>self.axis), rmaxis-(rmaxis>self.axis)), self.axis-(rmaxis<self.axis), self.unravelshape)

  def _mask(self, maskvec, axis):
    if not self.axis <= axis < self.axis+2:
      return Unravel(Mask(self.func, maskvec, axis-(axis>self.axis)), self.axis, self.unravelshape)

  def _take(self, index, axis):
    if not self.axis <= axis < self.axis+2:
      return Unravel(Take(self.func, index, axis-(axis>self.axis)), self.axis, self.unravelshape)

  def _product(self):
    if self.axis < self.func.ndim-1:
      return Unravel(Product(self.func), self.axis, self.unravelshape)

  def _sum(self, axis):
    if not self.axis <= axis < self.axis+2:
      return Unravel(Sum(self.func, axis-(axis>self.axis)), self.axis-(axis<self.axis), self.unravelshape)

  @property
  def blocks(self):
    fullrange = Range(self.func.shape[self.axis])
    if not all(ind[self.axis] == fullrange for ind, f in self.func.blocks):
      return super().blocks
    ravelrange = Range(self.shape[self.axis]), Range(self.shape[self.axis+1])
    return tuple((ind[:self.axis]+ravelrange+ind[self.axis+1:], Unravel(f, self.axis, self.unravelshape)) for ind, f in self.func.blocks)

class Mask(Array):

  __slots__ = 'func', 'axis', 'mask'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, mask:types.frozenarray, axis:types.strictint):
    assert len(mask) == func.shape[axis]
    self.func = func
    self.axis = axis
    self.mask = mask
    super().__init__(args=[func], shape=func.shape[:axis]+(mask.sum(),)+func.shape[axis+1:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if self.mask.all():
      return func
    if not self.mask.any():
      return zeros_like(self)
    retval = func._mask(self.mask, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Mask(func, self.mask, self.axis)

  def evalf(self, func):
    return func[(slice(None),)*(self.axis+1)+(numpy.asarray(self.mask),)]

  def _derivative(self, var, seen):
    return mask(derivative(self.func, var, seen), self.mask, self.axis)

  def _get(self, i, item):
    if i != self.axis:
      return Mask(Get(self.func, i, item), self.mask, self.axis-(i<self.axis))
    if item.isconstant:
      item, = item.eval()
      where, = self.mask.nonzero()
      return Get(self.func, i, where[item])

  def _take(self, index, axis):
    if axis != self.axis:
      return Mask(Take(self.func, index, axis), self.mask, self.axis)
    where, = self.mask.nonzero()
    if numpy.equal(numpy.diff(where), 1).all():
      return Take(self.func, index+where[0], axis)

  def _mask(self, maskvec, axis):
    if axis == self.axis:
      newmask = numpy.zeros(len(self.mask), dtype=bool)
      newmask[numpy.asarray(self.mask)] = maskvec
      assert maskvec.sum() == newmask.sum()
      return Mask(self.func, newmask, self.axis)

  def _sum(self, axis):
    if axis != self.axis:
      return Mask(Sum(self.func, axis), self.mask, self.axis-(axis<self.axis))

  @property
  def blocks(self):
    blocks = []
    for ind, f in self.func.blocks:
      if ind[self.axis] == Range(self.func.shape[self.axis]):
        indi = Range(self.shape[self.axis])
        newf = Mask(f, self.mask, axis=self.axis)
      else:
        renumber = numpy.repeat(self.mask.size, self.mask.size) # initialize with out of bounds
        renumber[self.mask] = numpy.arange(self.mask.sum())
        subind = Find(Take(self.mask, ind[self.axis], axis=0))
        indi = Take(renumber, Take(ind[self.axis], subind, axis=0), axis=0)
        newf = Take(f, subind, axis=self.axis)
      blocks.append((ind[:self.axis] + (indi,) + ind[self.axis+1:], newf))
    return tuple(blocks)

class Range(Array):

  __slots__ = 'length', 'offset'

  @types.apply_annotations
  def __init__(self, length:asarray, offset:asarray=Zeros((), int)):
    assert length.ndim == 0 and length.dtype == int
    assert offset.ndim == 0 and offset.dtype == int
    self.length = length
    self.offset = offset
    super().__init__(args=[length, offset], shape=[length], dtype=int)

  def _take(self, index, axis):
    return add(index, self.offset)

  def _add(self, offset):
    if 0 in offset._inserted_axes:
      return Range(self.length, self.offset + offset._uninsert(0))

  def evalf(self, length, offset):
    length, = length
    offset, = offset
    return numpy.arange(offset, offset+length)[_]

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
  __cache__ = 'simplified',

  @types.apply_annotations
  def __init__(self, coeffs:asarray, points:asarray, ngrad:types.strictint=0):
    if points.ndim != 1:
      raise ValueError('argument `points` should have exactly one dimension')
    if not numeric.isint(points.shape[0]):
      raise ValueError('the shape of argument `points` should have be known, i.e. an `int`')
    self.points_ndim = points.shape[0]
    ndim = coeffs.ndim - self.points_ndim
    if coeffs.ndim < ndim:
      raise ValueError('argument `coeffs` should have at least one axis per spatial dimension')
    self.coeffs = coeffs
    self.points = points
    self.ngrad = ngrad
    super().__init__(args=[points, coeffs], shape=coeffs.shape[:ndim]+(self.points_ndim,)*ngrad, dtype=float)

  def evalf(self, points, coeffs):
    assert points.shape[1] == self.points_ndim
    points = types.frozenarray(points)
    coeffs = types.frozenarray(coeffs)
    for igrad in range(self.ngrad):
      coeffs = numeric.poly_grad(coeffs, self.points_ndim)
    return numeric.poly_eval(coeffs, points)

  def _derivative(self, var, seen):
    # Derivative to argument `points`.
    dpoints = dot(Polyval(self.coeffs, self.points, self.ngrad+1)[(...,*(_,)*var.ndim)], derivative(self.points, var, seen), self.ndim)
    # Derivative to argument `coeffs`.  `trans` shuffles the coefficient axes
    # of `derivative(self.coeffs)` after the derivative axes.
    shuffle = lambda a, b, c: (*range(0,a), *range(a+b,a+b+c), *range(a,a+b))
    pretrans = shuffle(self.coeffs.ndim-self.points_ndim, self.points_ndim, var.ndim)
    posttrans = shuffle(self.coeffs.ndim-self.points_ndim, var.ndim, self.ngrad)
    dcoeffs = Transpose(Polyval(Transpose(derivative(self.coeffs, var, seen), pretrans), self.points, self.ngrad), posttrans)
    return dpoints + dcoeffs

  def _take(self, index, axis):
    if axis < self.coeffs.ndim - self.points_ndim:
      return Polyval(take(self.coeffs, index, axis), self.points, self.ngrad)

  def _mask(self, maskvec, axis):
    if axis < self.coeffs.ndim - self.points_ndim:
      return Polyval(Mask(self.coeffs, maskvec, axis), self.points, self.ngrad)

  def _const_helper(self, *j):
    if len(j) == self.ngrad:
      coeffs = self.coeffs
      for i in reversed(range(self.points_ndim)):
        p = builtins.sum(k==i for k in j)
        coeffs = math.factorial(p)*Get(coeffs, axis=i+self.coeffs.ndim-self.points_ndim, item=p)
      return coeffs
    else:
      return stack([self._const_helper(*j, k) for k in range(self.points_ndim)], axis=self.coeffs.ndim-self.points_ndim+self.ngrad-len(j)-1)

  @property
  def simplified(self):
    self = self.edit(lambda arg: arg.simplified if isevaluable(arg) else arg)
    degree = 0 if self.points_ndim == 0 else self.coeffs.shape[-1]-1 if isinstance(self.coeffs.shape[-1], int) else float('inf')
    if iszero(self.coeffs) or self.ngrad > degree:
      return zeros_like(self)
    elif self.ngrad == degree:
      return self._const_helper().simplified
    else:
      return self

class RevolutionAngle(Array):
  '''
  Pseudo coordinates of a :class:`nutils.topology.RevolutionTopology`.
  '''

  __slots__ = ()
  __cache__ = 'prepare_eval'

  def __init__(self):
    super().__init__(args=[], shape=[], dtype=float)

  @property
  def isconstant(self):
    return False

  def evalf(self):
    raise Exception('RevolutionAngle should not be evaluated')

  def _derivative(self, var, seen):
    return (ones_like if isinstance(var, LocalCoords) and len(var) > 0 else zeros_like)(var)

  @util.positional_only
  def prepare_eval(self, kwargs=...):
    return zeros_like(self)

class Opposite(Array):

  __slots__ = '_value'
  __cache__ = 'simplified'

  @types.apply_annotations
  def __init__(self, value:asarray):
    self._value = value
    super().__init__(args=[value], shape=value.shape, dtype=value.dtype)

  def evalf(self, evalargs):
    raise Exception('Opposite should not be evaluated')

  @property
  def simplified(self):
    value = self._value.simplified
    if not any(isinstance(arg, SelectChain) for arg in value.dependencies):
      return value
    return Opposite(value)

  @util.positional_only
  def prepare_eval(self, *, opposite=False, kwargs=...):
    return self._value.prepare_eval(opposite=not opposite, **kwargs)

  def _derivative(self, var, seen):
    return Opposite(derivative(self._value, var, seen))

class Kronecker(Array):

  __slots__ = 'func', 'axis', 'length', 'pos'
  __cache__ = 'simplified', 'blocks'

  @types.apply_annotations
  def __init__(self, func:asarray, axis:types.strictint, length:asarray, pos:asarray):
    assert pos.ndim == 0 and pos.dtype == int
    assert length.ndim == 0 and length.dtype == int
    self.func = func
    self.axis = axis
    self.length = length
    self.pos = pos
    super().__init__(args=[func, length, pos], shape=func.shape[:axis]+(length,)+func.shape[axis:], dtype=func.dtype)

  @property
  def simplified(self):
    func = self.func.simplified
    if self.shape[self.axis] == 1:
      return InsertAxis(self.func, axis=self.axis, length=1) # we assume without checking that pos correctly evaluates to 0
    retval = func._kronecker(self.axis, self.length, self.pos)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Kronecker(func, self.axis, self.length, self.pos)

  def evalf(self, func, length, pos):
    length, = length
    pos, = pos
    retval = numpy.zeros(func.shape[:self.axis+1] + (length,) + func.shape[self.axis+1:], dtype=func.dtype)
    retval[(slice(None),)*(self.axis+1)+(pos,)] = func
    return retval

  def _derivative(self, var, seen):
    return Kronecker(derivative(self.func, var, seen), self.axis, self.length, self.pos)

  def _multiply(self, other):
    return Kronecker(multiply(self.func, get(other, self.axis, self.pos)), self.axis, self.length, self.pos)

  def _transpose(self, axes):
    i = axes.index(self.axis)
    return Kronecker(Transpose(self.func, [ax-(ax>self.axis) for ax in axes[:i]+axes[i+1:]]), i, self.length, self.pos)

  def _insertaxis(self, axis, length):
    return Kronecker(InsertAxis(self.func, axis-(axis>self.axis), length), self.axis+(axis<=self.axis), self.length, self.pos)

  def _get(self, axis, item):
    if axis != self.axis:
      return Kronecker(get(self.func, axis-(axis>self.axis), item), self.axis-(axis<=self.axis), self.length, self.pos)
    if item.simplified == self.pos.simplified or item.isconstant and self.pos.isconstant and item.eval() == self.pos.eval():
      return self.func
    if item.isconstant and self.pos.isconstant: # inequality is implied by the previous test
      return zeros_like(self.func)

  def _power(self, n):
    return Kronecker(power(self.func, get(n, self.axis, self.pos)), self.axis, self.length, self.pos)

  def _add(self, other):
    if isinstance(other, Kronecker) and self.axis == other.axis and self.pos == other.pos:
      assert self.length == other.length
      return Kronecker(add(self.func, other.func), self.axis, self.length, self.pos)

  def _sum(self, axis):
    if axis == self.axis:
      return self.func
    return Kronecker(sum(self.func, axis-(axis>self.axis)), self.axis-(axis<=self.axis), self.length, self.pos)

  def _take(self, index, axis):
    if axis != self.axis:
      return Kronecker(take(self.func, index, axis-(axis>self.axis)), self.axis, self.length, self.pos)
    if self.pos.isconstant and index.isconstant:
      pos, = self.pos.eval()
      index, = index.eval()
      if pos in index:
        assert tuple(index).count(pos) == 1
        newpos = tuple(index).index(pos)
        return Kronecker(self.func, self.axis, len(index), newpos)
      return Zeros(self.shape[:axis]+(len(index),)+self.shape[axis+1:], dtype=self.dtype)

  def _determinant(self):
    if self.axis < self.ndim-2:
      return Kronecker(determinant(self.func), self.axis, self.length, self.pos)
    elif self.length.isconstant and self.length.eval()[0] > 1:
      return Zeros(self.shape[:-2], dtype=self.dtype)

  def _inverse(self):
    if self.axis < self.ndim-2:
      return Kronecker(inverse(self.func), self.axis, self.length, self.pos)

  def _takediag(self, axis, rmaxis):
    assert axis < rmaxis
    if rmaxis == self.axis:
      return Kronecker(get(self.func, axis, self.pos), axis, self.length, self.pos)
    if axis == self.axis:
      return Kronecker(get(self.func, rmaxis-1, self.pos), axis, self.length, self.pos)
    return Kronecker(takediag(self.func, axis-(axis>self.axis), rmaxis-(rmaxis>self.axis)), self.axis-(rmaxis<self.axis), self.length, self.pos)

  def _product(self):
    if self.axis < self.ndim-1:
      return Kronecker(Product(self.func), self.axis, self.length, self.pos)
    if numeric.isint(self.shape[-1]) and self.shape[-1] > 1:
      return zeros_like(self.func)

  def _sign(self):
    return Kronecker(sign(self.func), self.axis, self.length, self.pos)

  def _mask(self, maskvec, axis):
    if axis != self.axis:
      return Kronecker(mask(self.func, maskvec, axis-(axis>self.axis)), self.axis, self.length, self.pos)
    if self.pos.isconstant:
      pos, = self.pos.eval()
      length = maskvec.sum()
      if maskvec[pos]:
        return Kronecker(self.func, self.axis, length, maskvec[:pos].sum())
      return Zeros(self.shape[:axis]+(length,)+self.shape[axis+1:], dtype=self.dtype)

  def _unravel(self, axis, shape):
    if axis != self.axis:
      return Kronecker(Unravel(self.func, axis-(axis>self.axis), shape), self.axis+(axis<self.axis)*(len(shape)-1), self.length, self.pos)

  def _diagonalize(self, axis, newaxis):
    if axis == self.axis:
      return Kronecker(self, newaxis, self.length, self.pos)
    return Kronecker(Diagonalize(self.func, axis-(axis>self.axis), newaxis-(newaxis>self.axis)), self.axis+(newaxis<=self.axis), self.length, self.pos)

  @property
  def blocks(self):
    return tuple((ind[:self.axis] + (self.pos[_],) + ind[self.axis:], InsertAxis(f, self.axis, 1)) for ind, f in self.func.blocks)

# BASES

class Basis(Array):
  '''Abstract base class for bases.

  A basis is a sequence of elementwise polynomial functions.

  Parameters
  ----------
  ndofs : :class:`int`
      The number of functions in this basis.
  transforms : :class:`nutils.transformseq.Transforms`
      The transforms on which this basis is defined.
  trans : :class:`TransformChain`

  Notes
  -----
  Subclasses must implement :meth:`get_dofs` and :meth:`get_coefficients` and
  if possible should redefine :meth:`get_support`.
  '''

  __slots__ = 'ndofs', 'transforms', '_index', '_points'
  __cache__ = '_computed_support'

  @types.apply_annotations
  def __init__(self, ndofs:types.strictint, transforms:transformseq.stricttransforms, trans:types.strict[TransformChain]=TRANS):
    self.ndofs = ndofs
    self.transforms = transforms

    self._index, tail = TransformsIndexWithTail(self.transforms, trans)
    self._points = ApplyTransforms(tail)
    super().__init__(args=(self._index, self._points), shape=(ndofs,), dtype=float)

  def evalf(self, index, points):
    warnings.warn('using explicit basis evaluation; this is usually a bug.', ExpensiveEvaluationWarning)
    index, = index
    values = numeric.poly_eval(self.get_coefficients(index)[None], points)
    inflated = numpy.zeros((points.shape[0], self.ndofs), float)
    numpy.add.at(inflated, (slice(None), self.get_dofs(index)), values)
    return inflated

  @property
  def _computed_support(self):
    support = [set() for i in range(self.ndofs)]
    for ielem in range(len(self.transforms)):
      for dof in self.get_dofs(ielem):
        support[dof].add(ielem)
    return tuple(types.frozenarray(numpy.fromiter(sorted(ielems), dtype=int), copy=False) for ielems in support)

  def get_support(self, dof):
    '''Return the support of basis function ``dof``.

    If ``dof`` is an :class:`int`, return the indices of elements that form the
    support of ``dof``.  If ``dof`` is an array, return the union of supports
    of the selected dofs as a unique array.  The returned array is always
    unique, i.e. strict monotonic increasing.

    Parameters
    ----------
    dof : :class:`int` or array of :class:`int` or :class:`bool`
        Index or indices of basis function or a mask.

    Returns
    -------
    support : sorted and unique :class:`numpy.ndarray`
        The elements (as indices) where function ``dof`` has support.
    '''

    if numeric.isint(dof):
      return self._computed_support[dof]
    elif numeric.isintarray(dof):
      if dof.ndim != 1:
        raise IndexError('dof has invalid number of dimensions')
      if len(dof) == 0:
        return numpy.array([], dtype=int)
      dof = numpy.unique(dof)
      if dof[0] < 0 or dof[-1] >= self.ndofs:
        raise IndexError('dof out of bounds')
      if self.get_support == __class__.get_support.__get__(self, __class__):
        return numpy.unique([ielem for ielem in range(len(self.transforms)) if numpy.in1d(self.get_dofs(ielem), dof, assume_unique=True).any()])
      else:
        return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_support, dof)), dtype=int))
    elif numeric.isboolarray(dof):
      if dof.shape != (self.ndofs,):
        raise IndexError('dof has invalid shape')
      return self.get_support(numpy.where(dof)[0])
    else:
      raise IndexError('invalid dof')

  @abc.abstractmethod
  def get_dofs(self, ielem):
    '''Return an array of indices of basis functions with support on element ``ielem``.

    If ``ielem`` is an :class:`int`, return the dofs on element ``ielem``
    matching the coefficients array as returned by :meth:`get_coefficients`.
    If ``ielem`` is an array, return the union of dofs on the selected elements
    as a unique array, i.e. a strict monotonic increasing array.

    Parameters
    ----------
    ielem : :class:`int` or array of :class:`int` or :class:`bool`
        Element number(s) or mask.

    Returns
    -------
    dofs : :class:`numpy.ndarray`
        A 1D Array of indices.
    '''

    if numeric.isint(ielem):
      raise NotImplementedError
    elif numeric.isintarray(ielem):
      if ielem.ndim != 1:
        raise IndexError('invalid ielem')
      if len(ielem) == 0:
        return numpy.array([], dtype=int)
      ielem = numpy.unique(ielem)
      if ielem[0] < 0 or ielem[-1] >= len(self.transforms):
        raise IndexError('ielem out of bounds')
      return numpy.unique(numpy.fromiter(itertools.chain.from_iterable(map(self.get_dofs, ielem)), dtype=int))
    elif numeric.isboolarray(ielem):
      if ielem.shape != (len(self.transforms),):
        raise IndexError('ielem has invalid shape')
      return self.get_dofs(numpy.where(ielem)[0])
    else:
      raise IndexError('invalid index')

  @abc.abstractmethod
  def get_coefficients(self, ielem):
    '''Return an array of coefficients for all basis functions with support on element ``ielem``.

    Parameters
    ----------
    ielem : :class:`int`
        Element number.

    Returns
    -------
    coefficients : :class:`nutils.types.frozenarray`
        Array of coefficients with shape ``(nlocaldofs,)+(degree,)*ndims``,
        where the first axis corresponds to the dofs returned by
        :meth:`get_dofs`.
    '''

    raise NotImplementedError

  @property
  def simplified(self):
    ielems = range(len(self.transforms))
    dofmap = DofMap(tuple(map(self.get_dofs, ielems)), index=self._index)
    coeffs = Elemwise(tuple(map(self.get_coefficients, ielems)), self._index, dtype=float)
    func = Polyval(coeffs, self._points)
    inflated = Inflate(func, dofmap, self.shape[0], axis=0)
    return inflated.simplified

  def _derivative(self, var, seen):
    return self.simplified._derivative(var, seen)

  def __getitem__(self, index):
    if numeric.isintarray(index) and index.ndim == 1 and numpy.all(numpy.greater(numpy.diff(index), 0)):
      return MaskedBasis(self, index)
    elif numeric.isboolarray(index) and index.shape == (self.ndofs,):
      return MaskedBasis(self, numpy.where(index)[0])
    else:
      return super().__getitem__(index)

strictbasis = types.strict[Basis]

class PlainBasis(Basis):
  '''A general purpose implementation of a :class:`Basis`.

  Use this class only if there exists no specific implementation of
  :class:`Basis` for the basis at hand.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  dofs : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The dofs corresponding to the ``coefficients`` argument.
  ndofs : :class:`int`
      The number of basis functions.
  transforms : :class:`nutils.transformseq.Transforms`
      The transforms on which this basis is defined.
  trans : :class:`TransformChain`
  '''

  __slots__ = '_coeffs', '_dofs'

  @types.apply_annotations
  def __init__(self, coefficients:types.tuple[types.frozenarray], dofs:types.tuple[types.frozenarray], ndofs:types.strictint, transforms:transformseq.stricttransforms, trans=TRANS):
    self._coeffs = coefficients
    self._dofs = dofs
    assert len(self._coeffs) == len(self._dofs) == len(transforms)
    assert all(c.ndim == 1+transforms.fromdims for c in self._coeffs)
    assert all(len(c) == len(d) for c, d in zip(self._coeffs, self._dofs))
    super().__init__(ndofs=ndofs, transforms=transforms, trans=trans)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    return self._dofs[ielem]

  def get_coefficients(self, ielem):
    return self._coeffs[ielem]

  @property
  def simplified(self):
    dofmap = DofMap(self._dofs, index=self._index)
    coeffs = Elemwise(self._coeffs, self._index, dtype=float)
    value = Polyval(coeffs, self._points)
    inflated = Inflate(value, dofmap, self.shape[0], axis=0)
    return inflated.simplified

class DiscontBasis(Basis):
  '''A discontinuous basis with monotonic increasing dofs.

  Parameters
  ----------
  coefficients : :class:`tuple` of :class:`nutils.types.frozenarray` objects
      The coefficients of the basis functions per transform.  The order should
      match the ``transforms`` argument.
  transforms : :class:`nutils.transformseq.Transforms`
      The transforms on which this basis is defined.
  trans : :class:`TransformChain`
  '''

  __slots__ = '_coeffs', '_offsets'

  @types.apply_annotations
  def __init__(self, coefficients:types.tuple[types.frozenarray], transforms:transformseq.stricttransforms, trans=TRANS):
    self._coeffs = coefficients
    assert len(self._coeffs) == len(transforms)
    assert all(c.ndim == 1+transforms.fromdims for c in self._coeffs)
    self._offsets = types.frozenarray(numpy.cumsum([0, *map(len, self._coeffs)]), copy=False)
    super().__init__(ndofs=self._offsets[-1], transforms=transforms, trans=trans)

  def get_support(self, dof):
    if not numeric.isint(dof):
      return super().get_support(dof)
    ielem = numpy.searchsorted(self._offsets[:-1], numeric.normdim(self.ndofs, dof), side='right')-1
    return numpy.array([ielem], dtype=int)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    ielem = numeric.normdim(len(self.transforms), ielem)
    return numpy.arange(self._offsets[ielem], self._offsets[ielem+1])

  def get_coefficients(self, ielem):
    return self._coeffs[ielem]

  @property
  def simplified(self):
    dofs = tuple(numpy.arange(self._offsets[i], self._offsets[i+1]) for i in range(len(self._coeffs)))
    dofmap = DofMap(dofs, index=self._index)
    coeffs = Elemwise(self._coeffs, self._index, dtype=float)
    value = Polyval(coeffs, self._points)
    inflated = Inflate(value, dofmap, self.shape[0], axis=0)
    return inflated.simplified

class MaskedBasis(Basis):
  '''An order preserving subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to mask.
  indices : array of :class:`int`\\s
      The strict monotonic increasing indices of ``parent`` basis functions to
      keep.
  trans : :class:`TransformChain`
  '''

  __slots__ = '_parent', '_indices'

  @types.apply_annotations
  def __init__(self, parent:strictbasis, indices:types.frozenarray[types.strictint], trans=TRANS):
    if indices.ndim != 1:
      raise ValueError('`indices` should have one dimension but got {}'.format(indices.ndim))
    if len(indices) and not numpy.all(numpy.greater(numpy.diff(indices), 0)):
      raise ValueError('`indices` should be strictly monotonic increasing')
    if len(indices) and (indices[0] < 0 or indices[-1] >= len(parent)):
      raise ValueError('`indices` out of range \x5b0,{}\x29'.format(0, len(parent)))
    self._parent = parent
    self._indices = indices
    super().__init__(ndofs=len(self._indices), transforms=parent.transforms, trans=trans)

  def get_dofs(self, ielem):
    return numeric.sorted_index(self._indices, self._parent.get_dofs(ielem), missing='mask')

  def get_coefficients(self, ielem):
    mask = numeric.sorted_contains(self._indices, self._parent.get_dofs(ielem))
    return self._parent.get_coefficients(ielem)[mask]

  def get_support(self, dof):
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return self._parent.get_support(self._indices[dof])

class StructuredBasis(Basis):
  '''A basis for class:`nutils.transformseq.StructuredTransforms`.

  Parameters
  ----------
  coeffs : :class:`tuple` of :class:`tuple`\\s of arrays
      Per dimension the coefficients of the basis functions per transform.
  start_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension the dof of the first entry in ``coeffs`` per transform.
  stop_dofs : :class:`tuple` of arrays of :class:`int`\\s
      Per dimension one plus the dof of the last entry  in ``coeffs`` per
      transform.
  dofs_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the dofs.
  transforms : :class:`nutils.transformseq.Transforms`
      The transforms on which this basis is defined.
  transforms_shape : :class:`tuple` of :class:`int`\\s
      The tensor shape of the transforms.
  trans : :class:`TransformChain`
  '''

  __slots__ = '_coeffs', '_start_dofs', '_stop_dofs', '_dofs_shape', '_transforms_shape'

  @types.apply_annotations
  def __init__(self, coeffs:types.tuple[types.tuple[types.frozenarray]], start_dofs:types.tuple[types.frozenarray[types.strictint]], stop_dofs:types.tuple[types.frozenarray[types.strictint]], dofs_shape:types.tuple[types.strictint], transforms:transformseq.stricttransforms, transforms_shape:types.tuple[types.strictint], trans=TRANS):
    self._coeffs = coeffs
    self._start_dofs = start_dofs
    self._stop_dofs = stop_dofs
    self._dofs_shape = dofs_shape
    self._transforms_shape = transforms_shape
    super().__init__(ndofs=util.product(dofs_shape), transforms=transforms, trans=trans)

  def _get_indices(self, ielem):
    ielem = numeric.normdim(len(self.transforms), ielem)
    indices = []
    for n in reversed(self._transforms_shape):
      ielem, index = divmod(ielem, n)
      indices.insert(0, index)
    if ielem != 0:
      raise IndexError
    return tuple(indices)

  def get_dofs(self, ielem):
    if not numeric.isint(ielem):
      return super().get_dofs(ielem)
    indices = self._get_indices(ielem)
    dofs = numpy.array(0)
    for start_dofs_i, stop_dofs_i, ndofs_i, index_i in zip(self._start_dofs, self._stop_dofs, self._dofs_shape, indices):
      dofs_i = numpy.arange(start_dofs_i[index_i], stop_dofs_i[index_i], dtype=int) % ndofs_i
      dofs = numpy.add.outer(dofs*ndofs_i, dofs_i)
    return types.frozenarray(dofs.ravel(), dtype=types.strictint, copy=False)

  def get_coefficients(self, ielem):
    return functools.reduce(numeric.poly_outer_product, map(operator.getitem, self._coeffs, self._get_indices(ielem)))

  def get_support(self, dof):
    if not numeric.isint(dof):
      return super().get_support(dof)
    dof = numeric.normdim(self.ndofs, dof)
    ndofs = 1
    ntrans = 1
    supports = []
    for start_dofs_i, stop_dofs_i, ndofs_i, ntrans_i in zip(reversed(self._start_dofs), reversed(self._stop_dofs), reversed(self._dofs_shape), reversed(self._transforms_shape)):
      dof, dof_i = divmod(dof, ndofs_i)
      supports_i = []
      while dof_i < stop_dofs_i[-1]:
        stop_ielem = numpy.searchsorted(start_dofs_i, dof_i, side='right')
        start_ielem = numpy.searchsorted(stop_dofs_i, dof_i, side='right')
        supports_i.append(numpy.arange(start_ielem, stop_ielem, dtype=int))
        dof_i += ndofs_i
      supports.append(numpy.unique(numpy.concatenate(supports_i)) * ntrans)
      ndofs *= ndofs_i
      ntrans *= ntrans_i
    assert dof == 0
    return types.frozenarray(functools.reduce(numpy.add.outer, reversed(supports)).ravel(), copy=False, dtype=types.strictint)

class PrunedBasis(Basis):
  '''A subset of another :class:`Basis`.

  Parameters
  ----------
  parent : :class:`Basis`
      The basis to prune.
  transmap : one-dimensional array of :class:`int`\\s
      The indices of transforms in ``parent`` that form this subset.
  '''

  __slots__ = '_parent', '_transmap', '_dofmap'

  @types.apply_annotations
  def __init__(self, parent:strictbasis, transmap:types.frozenarray[types.strictint], trans=TRANS):
    self._parent = parent
    self._transmap = transmap
    self._dofmap = parent.get_dofs(self._transmap)
    super().__init__(len(self._dofmap), parent.transforms[transmap], trans)

  def get_dofs(self, ielem):
    if numeric.isintarray(ielem) and ielem.ndim == 1 and numpy.any(numpy.less(ielem, 0)):
      raise IndexError('dof out of bounds')
    return types.frozenarray(numpy.searchsorted(self._dofmap, self._parent.get_dofs(self._transmap[ielem])), copy=False)

  def get_coefficients(self, ielem):
    return self._parent.get_coefficients(self._transmap[ielem])

  def get_support(self, dof):
    if numeric.isintarray(dof) and dof.ndim == 1 and numpy.any(numpy.less(dof, 0)):
      raise IndexError('dof out of bounds')
    return numeric.sorted_index(self._transmap, self._parent.get_support(self._dofmap[dof]), missing='mask')

# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_ascending = lambda arg: numpy.greater(numpy.diff(arg), 0).all()
_normdims = lambda ndim, shapes: tuple(numeric.normdim(ndim,sh) for sh in shapes)

def _jointdtype(*dtypes):
  'determine joint dtype'

  type_order = bool, int, float
  kind_order = 'bif'
  itype = builtins.max(kind_order.index(dtype.kind) if isinstance(dtype,numpy.dtype)
           else type_order.index(dtype) for dtype in dtypes)
  return type_order[itype]

def _matchndim(*arrays):
  'introduce singleton dimensions to match ndims'

  arrays = [asarray(array) for array in arrays]
  ndim = builtins.max(array.ndim for array in arrays)
  return tuple(array[(_,)*(ndim-array.ndim)] for array in arrays)

def _invtrans(trans):
  trans = numpy.asarray(trans)
  assert trans.dtype == int
  invtrans = numpy.empty(len(trans), dtype=int)
  invtrans[trans] = numpy.arange(len(trans))
  return tuple(invtrans)

def _norm_and_sort(ndim, args):
  'norm axes, sort, and assert unique'

  normargs = tuple(sorted(numeric.normdim(ndim, arg) for arg in args))
  assert _ascending(normargs) # strict
  return normargs

def _gatherblocks(blocks):
  return tuple((ind, util.sum(funcs)) for ind, funcs in util.gather(blocks))

def _numpy_align(*arrays):
  '''reshape arrays according to Numpy's broadcast conventions'''
  arrays = [asarray(array) for array in arrays]
  if len(arrays) > 1:
    ndim = builtins.max([array.ndim for array in arrays])
    for idim in range(ndim):
      lengths = [array.shape[idim] for array in arrays if array.ndim == ndim and array.shape[idim] != 1]
      length = lengths[0] if lengths else 1
      assert all(l == length for l in lengths), 'incompatible shapes: {}'.format(' != '.join(str(l) for l in lengths))
      for i, a in enumerate(arrays):
        if a.ndim < ndim:
          arrays[i] = insertaxis(a, idim, length)
        elif a.shape[idim] != length:
          arrays[i] = repeat(a, length, idim)
  return arrays

def _inflate_scalar(arg, shape):
  arg = asarray(arg)
  assert arg.ndim == 0
  for idim, length in enumerate(shape):
    arg = insertaxis(arg, idim, length)
  return arg

def replace(func):
  '''decorator for deep object replacement

  Generates a deep replacement method for Immutable objects based on a callable
  that is applied (recursively) on individual constructor arguments.

  Args
  ----
  func
      callable which maps (obj, ...) onto replaced_obj

  Returns
  -------
  :any:`callable`
      The method that searches the object to perform the replacements.
  '''

  @functools.wraps(func)
  def wrapped(target, *funcargs, **funckwargs):
    cache = {}
    def op(obj):
      try:
        replaced = cache[obj]
      except TypeError: # unhashable
        replaced = obj
      except KeyError:
        replaced = func(obj, *funcargs, **funckwargs)
        if replaced is None:
          replaced = obj.edit(op) if isinstance(obj, types.Immutable) else obj
        cache[obj] = replaced
      return replaced
    retval = op(target)
    del op
    return retval

  return wrapped

# FUNCTIONS

def isarray(arg):
  return isinstance(arg, Array)

def iszero(arg):
  return isinstance(arg.simplified, Zeros)

def zeros(shape, dtype=float):
  return Zeros(shape, dtype)

def zeros_like(arr):
  return zeros(arr.shape, arr.dtype)

def ones(shape, dtype=float):
  return _inflate_scalar(numpy.ones((), dtype=dtype), shape)

def ones_like(arr):
  return ones(arr.shape, arr.dtype)

def reciprocal(arg):
  return power(arg, -1)

def grad(arg, coords, ndims=0):
  return asarray(arg).grad(coords, ndims)

def symgrad(arg, coords, ndims=0):
  return asarray(arg).symgrad(coords, ndims)

def div(arg, coords, ndims=0):
  return asarray(arg).div(coords, ndims)

def negative(arg):
  return multiply(arg, -1)

def nsymgrad(arg, coords):
  return (symgrad(arg,coords) * coords.normal()).sum(-1)

def ngrad(arg, coords):
  return (grad(arg,coords) * coords.normal()).sum(-1)

def sin(x):
  return Sin(x)

def cos(x):
  return Cos(x)

def rotmat(arg):
  return stack([trignormal(arg), trigtangent(arg)], 0)

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

def greater(arg1, arg2):
  return Greater(*_numpy_align(arg1, arg2))

def equal(arg1, arg2):
  return Equal(*_numpy_align(arg1, arg2))

def less(arg1, arg2):
  return Less(*_numpy_align(arg1, arg2))

def min(a, b):
  return Minimum(*_numpy_align(a, b))

def max(a, b):
  return Maximum(*_numpy_align(a, b))

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

def piecewise(level, intervals, *funcs):
  return Get(stack(funcs, axis=0), axis=0, item=util.sum(Int(greater(level, interval)) for interval in intervals))

def partition(f, *levels):
  '''Create a partition of unity for a scalar function f.

  When ``n`` levels are specified, ``n+1`` indicator functions are formed that
  evaluate to one if and only if the following condition holds::

      indicator 0: f < levels[0]
      indicator 1: levels[0] < f < levels[1]
      ...
      indicator n-1: levels[n-2] < f < levels[n-1]
      indicator n: f > levels[n-1]

  At the interval boundaries the indicators evaluate to one half, in the
  remainder of the domain they evaluate to zero such that the whole forms a
  partition of unity. The partitions can be used to create a piecewise
  continuous function by means of multiplication and addition.

  The following example creates a topology consiting of three elements, and a
  function ``f`` that is zero in the first element, parabolic in the second,
  and zero again in the third element.

  >>> from nutils import mesh
  >>> domain, x = mesh.rectilinear([3])
  >>> left, center, right = partition(x[0], 1, 2)
  >>> f = (1 - (2*x[0]-3)**2) * center

  Args
  ----
  f : :class:`Array`
      Scalar-valued function
  levels : scalar constants or :class:`Array`\\s
      The interval endpoints.

  Returns
  -------
  :class:`list` of scalar :class:`Array`\\s
      The indicator functions.
  '''

  signs = [Sign(f - level) for level in levels]
  steps = map(subtract, signs[:-1], signs[1:])
  return [.5 - .5 * signs[0]] + [.5 * step for step in steps] + [.5 + .5 * signs[-1]]

def trace(arg, n1=-2, n2=-1):
  return sum(takediag(arg, n1, n2), numeric.normdim(arg.ndim, n1))

def normalized(arg, axis=-1):
  return divide(arg, expand_dims(norm2(arg, axis=axis), axis))

def norm2(arg, axis=-1):
  return sqrt(sum(multiply(arg, arg), axis))

def heaviside(arg):
  return Int(greater(arg, 0))

def divide(arg1, arg2):
  return multiply(arg1, reciprocal(arg2))

def subtract(arg1, arg2):
  return add(arg1, negative(arg2))

def mean(arg):
  return .5 * (arg + opposite(arg))

def jump(arg):
  return opposite(arg) - arg

def add_T(arg, axes=(-2,-1)):
  return swapaxes(arg, *axes) + arg

def blocks(arg):
  return asarray(arg).simplified.blocks

def rootcoords(ndims):
  return ApplyTransforms(PopHead(ndims))

def opposite(arg):
  return Opposite(arg)

@replace
def _bifurcate(arg, side):
  if isinstance(arg, SelectChain):
    return SelectBifurcation(arg, side)

bifurcate1 = functools.partial(_bifurcate, side=True)
bifurcate2 = functools.partial(_bifurcate, side=False)

def bifurcate(arg1, arg2):
  return bifurcate1(arg1), bifurcate2(arg2)

def curvature(geom, ndims=-1):
  return geom.normal().div(geom, ndims=ndims)

def laplace(arg, geom, ndims=0):
  return arg.grad(geom, ndims).div(geom, ndims)

def symgrad(arg, geom, ndims=0):
  return multiply(.5, add_T(arg.grad(geom, ndims)))

def div(arg, geom, ndims=0):
  return trace(arg.grad(geom, ndims))

def tangent(geom, vec):
  return subtract(vec, multiply(dot(vec, normal(geom), -1)[...,_], normal(geom)))

def ngrad(arg, geom, ndims=0):
  return dotnorm(grad(arg, geom, ndims), geom)

def nsymgrad(arg, geom, ndims=0):
  return dotnorm(symgrad(arg, geom, ndims), geom)

def expand_dims(arg, n):
  return InsertAxis(arg, numeric.normdim(arg.ndim+1, n), 1)

def trignormal(angle):
  angle = asarray(angle)
  assert angle.ndim == 0
  if iszero(angle):
    return kronecker(1, axis=0, length=2, pos=0)
  return TrigNormal(angle)

def trigtangent(angle):
  angle = asarray(angle)
  assert angle.ndim == 0
  if iszero(angle):
    return kronecker(1, axis=0, length=2, pos=1)
  return TrigTangent(angle)

def eye(n, dtype=float):
  return diagonalize(ones([n], dtype=dtype))

def insertaxis(arg, n, length):
  arg = asarray(arg)
  n = numeric.normdim(arg.ndim+1, n)
  return InsertAxis(arg, n, length)

def stack(args, axis=0):
  aligned = _numpy_align(*args)
  axis = numeric.normdim(aligned[0].ndim+1, axis)
  return Concatenate([InsertAxis(arg, axis, 1) for arg in aligned], axis)

def chain(funcs):
  'chain'

  funcs = [asarray(func) for func in funcs]
  shapes = [func.shape[0] for func in funcs]
  return [concatenate([func if i==j else zeros((sh,) + func.shape[1:])
             for j, sh in enumerate(shapes)], axis=0)
               for i, func in enumerate(funcs)]

def vectorize(args):
  '''
  Combine scalar-valued bases into a vector-valued basis.

  Args
  ----
  args : iterable of 1-dimensional :class:`nutils.function.Array` objects

  Returns
  -------
  :class:`Array`
  '''

  return concatenate([kronecker(arg, axis=-1, length=len(args), pos=iarg) for iarg, arg in enumerate(args)])

def repeat(arg, length, axis):
  arg = asarray(arg)
  assert arg.shape[axis] == 1
  return insertaxis(get(arg, axis, 0), axis, length)

def get(arg, iax, item):
  arg = asarray(arg)
  item = asarray(item)
  iax = numeric.normdim(arg.ndim, iax)
  sh = arg.shape[iax]
  if numeric.isint(sh) and item.isconstant:
    item = numeric.normdim(sh, item.eval()[0])
  return Get(arg, iax, item)

def jacobian(geom, ndims):
  '''
  Return :math:`\\sqrt{|J^T J|}` with :math:`J` the gradient of ``geom`` to the
  local coordinate system with ``ndims`` dimensions (``localgradient(geom,
  ndims)``).
  '''

  assert geom.ndim == 1
  J = localgradient(geom, ndims)
  cndims, = geom.shape
  assert J.shape == (cndims,ndims), 'wrong jacobian shape: got {}, expected {}'.format(J.shape, (cndims, ndims))
  assert cndims >= ndims, 'geometry dimension < topology dimension'
  detJ = abs(determinant(J)) if cndims == ndims \
    else 1. if ndims == 0 \
    else abs(determinant((J[:,:,_] * J[:,_,:]).sum(0)))**.5
  return detJ

def matmat(arg0, *args):
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = asarray(arg0)
  for arg in args:
    arg = asarray(arg)
    assert retval.shape[-1] == arg.shape[0], 'incompatible shapes'
    retval = dot(retval[(...,)+(_,)*(arg.ndim-1)], arg[(_,)*(retval.ndim-1)], retval.ndim-1)
  return retval

def determinant(arg, axes=(-2,-1)):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  arg = transpose(arg, trans)
  return Determinant(arg)

def inverse(arg, axes=(-2,-1)):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  arg = transpose(arg, trans)
  return transpose(Inverse(arg), _invtrans(trans))

def takediag(arg, axis=-2, rmaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  rmaxis = numeric.normdim(arg.ndim, rmaxis)
  assert axis < rmaxis
  return TakeDiag(arg, axis, rmaxis)

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
  assert result.shape == func.shape+var.shape, 'bug in {}._derivative'.format(func)
  return result

def localgradient(arg, ndims):
  'local derivative'

  return derivative(arg, LocalCoords(ndims))

def dotnorm(arg, coords):
  'normal component'

  return sum(arg * coords.normal(), -1)

def normal(geom):
  return geom.normal()

def kronecker(arg, axis, length, pos):
  arg = asarray(arg)
  return Kronecker(arg, axis=numeric.normdim(arg.ndim+1, axis), length=length, pos=pos)

def diagonalize(arg, axis=-1, newaxis=-1):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  newaxis = numeric.normdim(arg.ndim+1, newaxis)
  assert axis < newaxis
  return Diagonalize(arg, axis, newaxis)

def concatenate(args, axis=0):
  args = _matchndim(*args)
  axis = numeric.normdim(args[0].ndim, axis)
  return Concatenate(args, axis)

def cross(arg1, arg2, axis):
  arg1, arg2 = _numpy_align(arg1, arg2)
  axis = numeric.normdim(arg1.ndim, axis)
  assert arg1.shape[axis] == 3
  i = types.frozenarray([1, 2, 0])
  j = types.frozenarray([2, 0, 1])
  return take(arg1, i, axis) * take(arg2, j, axis) - take(arg2, i, axis) * take(arg1, j, axis)

def outer(arg1, arg2=None, axis=0):
  'outer product'

  if arg2 is None:
    arg2 = arg1
  elif arg1.ndim != arg2.ndim:
    raise ValueError('arg1 and arg2 have different dimensions')
  axis = numeric.normdim(arg1.ndim, axis)
  return expand_dims(arg1,axis+1) * expand_dims(arg2,axis)

def sign(arg):
  arg = asarray(arg)
  return Sign(arg)

def eig(arg, axes=(-2,-1), symmetric=False):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  transposed = transpose(arg, trans)
  eigval, eigvec = Eig(transposed, symmetric)
  return Tuple([transpose(diagonalize(eigval), _invtrans(trans)), transpose(eigvec, _invtrans(trans))])

@types.apply_annotations
def elemwise(transforms:transformseq.stricttransforms, values:types.tuple[types.frozenarray]):
  index, tail = TransformsIndexWithTail(transforms, TRANS)
  return Elemwise(values, index, dtype=float)

def take(arg, index, axis):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  index = asarray(index)
  assert index.ndim == 1
  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    if index.isconstant:
      mask, = index.eval()
      return Mask(arg, mask, axis)
    index = find(index)
  return Take(arg, index, axis)

def find(arg):
  'find'

  arg = asarray(arg)
  assert arg.ndim == 1 and arg.dtype == bool

  if arg.isconstant:
    arg, = arg.eval()
    index, = arg.nonzero()
    return asarray(index)

  return Find(arg)

def mask(arg, mask, axis=0):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  assert numeric.isarray(mask) and mask.ndim == 1 and mask.dtype == bool
  assert arg.shape[axis] == len(mask)
  return Mask(arg, mask, axis)

def J(geometry, ndims=None):
  '''
  Return :math:`\\sqrt{|J^T J|}` with :math:`J` the gradient of ``geometry`` to
  the local coordinate system with ``ndims`` dimensions (``localgradient(geom,
  ndims)``).
  '''
  if ndims is None:
    return DelayedJacobian(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian(geometry, ndims)

def unravel(func, axis, shape):
  func = asarray(func)
  axis = numeric.normdim(func.ndim, axis)
  shape = tuple(shape)
  assert func.shape[axis] == numpy.product(shape)
  return Unravel(func, axis, tuple(shape))

def ravel(func, axis):
  func = asarray(func)
  axis = numeric.normdim(func.ndim-1, axis)
  return Ravel(func, axis)

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
    assert value.shape[:value.ndim-value._nderiv] == v.shape
    for ndims in value.shape[value.ndim-value._nderiv:]:
      v = localgradient(v, ndims)
    return v

def _eval_ast(ast, functions):
  '''evaluate ``ast`` generated by :func:`nutils.expression.parse`'''

  op, *args = ast
  if op is None:
    value, = args
    return value

  args = (_eval_ast(arg, functions) for arg in args)
  if op == 'group':
    array, = args
    return array
  elif op == 'arg':
    name, *shape = args
    return Argument(name, shape)
  elif op == 'substitute':
    array, *arg_value_pairs = args
    subs = {}
    assert len(arg_value_pairs) % 2 == 0
    for arg, value in zip(arg_value_pairs[0::2], arg_value_pairs[1::2]):
      assert isinstance(arg, Argument) and arg._nderiv == 0
      assert arg._name not in subs
      subs[arg._name] = value
    return replace_arguments(array, subs)
  elif op == 'call':
    func, *args = args
    return functions[func](*args)
  elif op == 'jacobian':
    geom, ndims = args
    return J(geom, ndims)
  elif op == 'eye':
    length, = args
    return eye(length)
  elif op == 'normal':
    geom, = args
    return normal(geom)
  elif op == 'getitem':
    array, dim, index = args
    return get(array, dim, index)
  elif op == 'trace':
    array, n1, n2 = args
    return trace(array, n1, n2)
  elif op == 'sum':
    array, axis = args
    return sum(array, axis)
  elif op == 'concatenate':
    return concatenate(args, axis=0)
  elif op == 'grad':
    array, geom = args
    return grad(array, geom)
  elif op == 'surfgrad':
    array, geom = args
    return grad(array, geom, len(geom)-1)
  elif op == 'derivative':
    func, target = args
    return derivative(func, target)
  elif op == 'append_axis':
    array, length = args
    return insertaxis(array, -1, length)
  elif op == 'transpose':
    array, trans = args
    return transpose(array, trans)
  elif op == 'jump':
    array, = args
    return jump(array)
  elif op == 'mean':
    array, = args
    return mean(array)
  elif op == 'neg':
    array, = args
    return -asarray(array)
  elif op in ('add', 'sub', 'mul', 'truediv', 'pow'):
    left, right = args
    return getattr(operator, '__{}__'.format(op))(asarray(left), asarray(right))
  else:
    raise ValueError('unknown opcode: {!r}'.format(op))

class Namespace:
  '''Namespace for :class:`Array` objects supporting assignments with tensor expressions.

  The :class:`Namespace` object is used to store :class:`Array` objects.

  >>> from nutils import function
  >>> ns = function.Namespace()
  >>> ns.A = function.zeros([3, 3])
  >>> ns.x = function.zeros([3])
  >>> ns.c = 2

  In addition to the assignment of :class:`Array` objects, it is also possible
  to specify an array using a tensor expression string — see
  :func:`nutils.expression.parse` for the syntax.  All attributes defined in
  this namespace are available as variables in the expression.  If the array
  defined by the expression has one or more dimensions the indices of the axes
  should be appended to the attribute name.  Examples:

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
  nutils.expression.ExpressionSyntaxError: Length of axis cannot be determined from the expression.
  δ_ij
    ^

  There are two ways to inform the namespace of the correct lengths.  The first is to
  assign fixed lengths to certain indices via keyword argument ``length_<indices>``:

  >>> ns_fixed = function.Namespace(length_ij=2)
  >>> ns_fixed.eval_ij('δ_ij')
  Array<2,2>

  Note that evaluating an expression with an incompatible length raises an
  exception:

  >>> ns = function.Namespace(length_i=2)
  >>> ns.a = numpy.array([1,2,3])
  >>> 'a_i' @ ns
  Traceback (most recent call last):
  ...
  nutils.expression.ExpressionSyntaxError: Length of index i is fixed at 2 but the expression has length 3.
  a_i
    ^

  The second is to define a fallback length via the ``fallback_length`` argument:

  >>> ns_fallback = function.Namespace(fallback_length=2)
  >>> ns_fallback.eval_ij('δ_ij')
  Array<2,2>

  When evaluating an expression through this namespace the following functions
  are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan2``, ``arctanh``, ``exp``, ``abs``,
  ``ln``, ``log``, ``log2``, ``log10``, ``sqrt`` and ``sign``.

  Args
  ----
  default_geometry_name : :class:`str`
      The name of the default geometry.  This argument is passed to
      :func:`nutils.expression.parse`.  Default: ``'x'``.
  fallback_length : :class:`int`, optional
      The fallback length of an axis if the length cannot be determined from
      the expression.
  length_<indices> : :class:`int`
      The fixed length of ``<indices>``.  All axes in the expression marked
      with one of the ``<indices>`` are asserted to have the specified length.

  Attributes
  ----------
  arg_shapes : :class:`dict`
      A readonly map of argument names and shapes.
  default_geometry_name : :class:`str`
      The name of the default geometry.  See argument with the same name.
  '''

  __slots__ = '_attributes', '_arg_shapes', 'default_geometry_name', '_fixed_lengths', '_fallback_length'

  _re_assign = re.compile('^([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-z]+)?$')

  _functions = dict(
    opposite=opposite, sin=sin, cos=cos, tan=tan, sinh=sinh, cosh=cosh,
    tanh=tanh, arcsin=arcsin, arccos=arccos, arctan=arctan, arctan2=arctan2, arctanh=arctanh,
    exp=exp, abs=abs, ln=ln, log=ln, log2=log2, log10=log10, sqrt=sqrt,
    sign=sign,
  )
  _functions_nargs = {k: len(inspect.signature(v).parameters) for k, v in _functions.items()}

  @types.apply_annotations
  def __init__(self, *, default_geometry_name='x', fallback_length:types.strictint=None, **kwargs):
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
    super().__init__()

  def __getstate__(self):
    'Pickle instructions'
    attrs = '_arg_shapes', '_attributes', 'default_geometry_name', '_fixed_lengths', '_fallback_length'
    return {k: getattr(self, k) for k in attrs}

  def __setstate__(self, d):
    'Unpickle instructions'
    for k, v in d.items(): super().__setattr__(k, v)

  @property
  def arg_shapes(self):
    return builtin_types.MappingProxyType(self._arg_shapes)

  @property
  def default_geometry(self):
    ''':class:`nutils.function.Array`: The default geometry, shorthand for ``getattr(ns, ns.default_geometry_name)``.'''
    return getattr(self, self.default_geometry_name)

  def __call__(*args, **subs):
    '''Return a copy with arguments replaced by ``subs``.

    Return a copy of this namespace with :class:`Argument` objects replaced
    according to ``subs``.

    Args
    ----
    **subs : :class:`dict` of :class:`str` and :class:`nutils.function.Array` objects
        Replacements of the :class:`Argument` objects, identified by their names.

    Returns
    -------
    ns : :class:`Namespace`
        The copy of this namespace with replaced :class:`Argument` objects.
    '''

    if len(args) != 1:
      raise TypeError('{} instance takes 1 positional argument but {} were given'.format(type(self).__name__, len(args)))
    self, = args
    ns = Namespace(default_geometry_name=self.default_geometry_name)
    for k, v in self._attributes.items():
      setattr(ns, k, replace_arguments(v, subs))
    return ns

  def copy_(self, *, default_geometry_name=None):
    '''Return a copy of this namespace.'''

    if default_geometry_name is None:
      default_geometry_name = self.default_geometry_name
    ns = Namespace(default_geometry_name=default_geometry_name, fallback_length=self._fallback_length, **{'length_{i}': l for i, l in self._fixed_lengths.items()})
    for k, v in self._attributes.items():
      setattr(ns, k, v)
    return ns

  def __getattr__(self, name):
    '''Get attribute ``name``.'''

    if name.startswith('eval_'):
      return lambda expr: _eval_ast(expression.parse(expr, variables=self._attributes, functions=self._functions_nargs, indices=name[5:], arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0], self._functions)
    try:
      return self._attributes[name]
    except KeyError:
      pass
    raise AttributeError(name)

  def __setattr__(self, name, value):
    '''Set attribute ``name`` to ``value``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    m = self._re_assign.match(name)
    if not m or m.group(2) and len(set(m.group(2))) != len(m.group(2)):
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))
    else:
      name, indices = m.groups()
      indices = indices[1:] if indices else ''
      if isinstance(value, str):
        ast, arg_shapes = expression.parse(value, variables=self._attributes, functions=self._functions_nargs, indices=indices, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)
        value = _eval_ast(ast, self._functions)
        self._arg_shapes.update(arg_shapes)
      else:
        assert not indices
      self._attributes[name] = asarray(value)

  def __delattr__(self, name):
    '''Delete attribute ``name``.'''

    if name in self.__slots__:
      raise AttributeError('readonly')
    elif name in self._attributes:
      del self._attributes[name]
    else:
      raise AttributeError('{!r} object has no attribute {!r}'.format(type(self), name))

  def __rmatmul__(self, expr):
    '''Evaluate zero or one dimensional ``expr`` or a list of expressions.'''

    if isinstance(expr, (tuple, list)):
      return tuple(map(self.__rmatmul__, expr))
    if not isinstance(expr, str):
      return NotImplemented
    try:
      ast = expression.parse(expr, variables=self._attributes, functions=self._functions_nargs, indices=None, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name, fixed_lengths=self._fixed_lengths, fallback_length=self._fallback_length)[0]
    except expression.AmbiguousAlignmentError:
      raise ValueError('`expression @ Namespace` cannot be used because the expression has more than one dimension.  Use `Namespace.eval_...(expression)` instead')
    return _eval_ast(ast, self._functions)

def normal(arg, exterior=False):
  assert arg.ndim == 1
  if not exterior:
    lgrad = localgradient(arg, len(arg))
    return Normal(lgrad)
  lgrad = localgradient(arg, len(arg)-1)
  if len(arg) == 2:
    return asarray([lgrad[1,0], -lgrad[0,0]]).normalized()
  if len(arg) == 3:
    return cross(lgrad[:,0], lgrad[:,1], axis=0).normalized()
  raise NotImplementedError

def grad(self, geom, ndims=0):
  assert geom.ndim == 1
  if ndims <= 0:
    ndims += geom.shape[0]
  J = localgradient(geom, ndims)
  if J.shape[0] == J.shape[1]:
    Jinv = inverse(J)
  elif J.shape[0] == J.shape[1] + 1: # gamma gradient
    G = dot(J[:,:,_], J[:,_,:], 0)
    Ginv = inverse(G)
    Jinv = dot(J[_,:,:], Ginv[:,_,:], -1)
  else:
    raise Exception('cannot invert {}x{} jacobian'.format(J.shape))
  return dot(localgradient(self, ndims)[...,_], Jinv, -2)

def dotnorm(arg, geom, axis=-1):
  axis = numeric.normdim(arg.ndim, axis)
  assert geom.ndim == 1 and geom.shape[0] == arg.shape[axis]
  return dot(arg, normal(geom)[(slice(None),)+(_,)*(arg.ndim-axis-1)], axis)

if __name__ == '__main__':
  # Diagnostics for the development for simplify operations.
  simplify_priority = (
    Ravel, Inflate, Kronecker, Diagonalize, InsertAxis, Transpose, Multiply, Add, Sign, Power, Inverse, Unravel, # size preserving
    Product, Determinant, TakeDiag, Mask, Take, Sum, Get) # size decreasing
  # The simplify priority defines the preferred order in which operations are
  # performed: shape decreasing operations such as Sum and Get should be done
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
