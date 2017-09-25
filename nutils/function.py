# -*- coding: utf8 -*-
#
# Module FUNCTION
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

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

from . import util, numpy, numeric, log, core, cache, transform, expression, _
import sys, warnings, itertools, functools, operator, inspect, numbers, builtins, re, types, collections.abc

CACHE = 'Cache'
TRANS = 'Trans'
OPPTRANS = 'OppTrans'
POINTS = 'Points'
ARGUMENTS = 'Arguments'

TOKENS = CACHE, TRANS, OPPTRANS, POINTS, ARGUMENTS

class orderedset(frozenset):
  __slots__ = '_items',

  def __init__(self, items):
    self._items = tuple(items)

  def __iter__(self):
    return iter(self._items)

  def __len__(self):
    return len(self._items)

class Evaluable( cache.Immutable ):
  'Base class'

  def __init__(self, args:tuple):
    assert all(isevaluable(arg) or arg in TOKENS for arg in args)
    self.__args = args

  def evalf( self, *args ):
    raise NotImplementedError( 'Evaluable derivatives should implement the evalf method' )

  @property
  def isconstant( self ):
    return all( arg not in TOKENS and arg.isconstant for arg in self.__args )

  @cache.property
  def serialized( self ):
    '''returns (ops,inds), where len(ops) = len(inds)-1'''

    myops = list( TOKENS )
    myinds = []
    indices = []
    for arg in self.__args:
      try:
        index = myops.index( arg )
      except ValueError:
        argops, arginds = arg.serialized
        renumber = list( range( len(TOKENS) ) )
        for op, ind in zip( argops, arginds ):
          try:
            n = myops.index( op )
          except ValueError:
            n = len(myops)
            myops.append( op )
            myinds.append( numpy.take(renumber,ind) )
          renumber.append( n )
        index = len(myops)
        myops.append( arg )
        myinds.append( numpy.take(renumber,arginds[-1]) )
      indices.append( index )
    myinds.append( indices )
    return tuple(myops[len(TOKENS):]), tuple(myinds)

  def asciitree( self, seen=None ):
    'string representation'

    if seen is None:
      seen = [ None ] * len(TOKENS)
    try:
      index = seen.index( self )
    except ValueError:
      pass
    else:
      return '%{}'.format( index )
    asciitree = self._asciitree_str()
    if core.getprop( 'richoutput', False ):
      select = '├ ', '└ '
      bridge = '│ ', '  '
    else:
      select = ': ', ': '
      bridge = '| ', '  '
    for iarg, arg in enumerate( self.__args ):
      n = iarg >= len(self.__args) - 1
      asciitree += '\n' + select[n] + ( ('\n' + bridge[n]).join( arg.asciitree( seen ).splitlines() ) if isevaluable(arg) else '<{}>'.format(arg) )
    index = len(seen)
    seen.append( self )
    return '%{} = {}'.format( index, asciitree )

  def _asciitree_str(self):
    return str(self)

  def __str__( self ):
    return self.__class__.__name__

  def eval( self, elem=None, ischeme=None, fcache=cache.WrapperDummyCache(), arguments=None ):
    'evaluate'
    
    if elem is None:
      assert self.isconstant
      trans = opptrans = None
      points = None
    elif isinstance( elem, transform.TransformChain ):
      trans = opptrans = elem
      points = ischeme
    elif isinstance( elem, tuple ):
      trans, opptrans = elem
      points = ischeme
    else:
      trans = elem.transform
      opptrans = elem.opposite
      if isinstance(ischeme, collections.abc.Mapping):
        ischeme = ischeme[elem]
      if isinstance( ischeme, str ):
        points, weights = fcache[elem.reference.getischeme]( ischeme )
      elif isinstance( ischeme, tuple ):
        points, weights = ischeme
        assert points.shape[-1] == elem.ndims
        assert points.shape[:-1] == weights.shape, 'non matching shapes: points.shape=%s, weights.shape=%s' % ( points.shape, weights.shape )
      elif numeric.isarray(ischeme):
        points = numeric.const(ischeme, dtype=float)
        assert points.shape[-1] == elem.ndims
      elif ischeme is None:
        points = None
      else:
        raise Exception( 'invalid integration scheme of type %r' % type(ischeme) )

    if trans is not None:
      if opptrans is not None:
        assert trans.fromdims == opptrans.fromdims
      if points is not None:
        assert points.ndim == 2 and points.shape[1] == trans.fromdims

    if arguments is not None:
      assert all(numeric.isarray(value) and value.dtype.kind in 'bif' for value in arguments.values())

    simple = self.simplified
    ops, inds = simple.serialized
    assert TOKENS == ( CACHE, TRANS, OPPTRANS, POINTS, ARGUMENTS )
    values = [ fcache, trans, opptrans, points, arguments or {} ]
    for op, indices in zip( list(ops)+[simple], inds ):
      args = [ values[i] for i in indices ]
      try:
        retval = op.evalf( *args )
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        excargs = etype, evalue, simple, values
        raise EvaluationError(*excargs).with_traceback( traceback )
      values.append( retval )
    return values[-1]

  @log.title
  def graphviz( self ):
    'create function graph'

    import os, subprocess, hashlib

    dotpath = core.getprop( 'dot', True )
    if not isinstance( dotpath, str ):
      dotpath = 'dot'

    ops, inds = self.serialized

    lines = []
    lines.append( 'digraph {' )
    lines.append( 'graph [ dpi=72 ];' )
    lines.extend( '%d [label="%d. %s"];' % (i, i, name._asciitree_str() if isinstance(name, Array) else name) for i, name in enumerate( TOKENS + ops + (self,) ) )
    lines.extend( '%d -> %d;' % (j,i) for i, indices in enumerate( ([],)*len(TOKENS) + inds ) for j in indices )
    lines.append( '}' )
    imgdata = '\n'.join(lines).encode()

    imgtype = core.getprop( 'imagetype', 'png' )
    imgpath = 'dot_{}.{}'.format(hashlib.sha1(imgdata).hexdigest(), imgtype)
    if not os.path.exists( imgpath ):
      with core.open_in_outdir( imgpath, 'w' ) as img:
        with subprocess.Popen( [dotpath,'-T'+imgtype], stdin=subprocess.PIPE, stdout=img ) as dot:
          dot.communicate( imgdata )

    log.info( imgpath )

  def stackstr( self, nlines=-1 ):
    'print stack'

    ops, inds = self.serialized
    lines = []
    for name in TOKENS:
      lines.append( '  %%%d = %s' % ( len(lines), name ) )
    for op, indices in zip( list(ops)+[self], inds ):
      args = [ '%%%d' % idx for idx in indices ]
      try:
        code = op.evalf.__code__
        offset = 1 if getattr( op.evalf, '__self__', None ) is not None else 0
        names = code.co_varnames[ offset:code.co_argcount ]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(indices) - len(names) ) )
        args = [ '%s=%s' % item for item in zip( names, args ) ]
      except:
        pass
      lines.append( '  %%%d = %s( %s )' % ( len(lines), op._asciitree_str() if isarray(op) else op, ', '.join( args ) ) )
      if len(lines) == nlines+1:
        break
    return '\n'.join( lines )

  @cache.property
  def simplified(self):
    return self.edit(lambda arg: arg.simplified if isevaluable(arg) else arg)

class EvaluationError( Exception ):
  'evaluation error'

  def __init__( self, etype, evalue, evaluable, values ):
    'constructor'

    self.etype = etype
    self.evalue = evalue
    self.evaluable = evaluable
    self.values = values

  def __repr__( self ):
    return 'EvaluationError%s' % self

  def __str__( self ):
    'string representation'

    return '\n%s --> %s: %s' % ( self.evaluable.stackstr( nlines=len(self.values) ), self.etype.__name__, self.evalue )

class Tuple( Evaluable ):

  def __init__(self, items:tuple):
    self.items = items
    args = []
    indices = []
    for i, item in enumerate(self.items):
      if isevaluable(item):
        args.append(item)
        indices.append(i)
    self.indices = tuple(indices)
    super().__init__(args)

  @cache.property
  def simplified(self):
    return Tuple([item.simplified if isevaluable(item) else item for item in self.items])

  def edit(self, op):
    return Tuple([op(item) for item in self.items])

  def evalf( self, *items ):
    'evaluate'

    T = list(self.items)
    for index, item in zip( self.indices, items ):
      T[index] = item
    return tuple( T )

  def __iter__( self ):
    'iterate'

    return iter(self.items)

  def __len__( self ):
    'length'

    return len(self.items)

  def __getitem__( self, item ):
    'get item'

    return self.items[item]

  def __add__( self, other ):
    'add'

    return Tuple( self.items + tuple(other) )

  def __radd__( self, other ):
    'add'

    return Tuple( tuple(other) + self.items )

class SelectChain( Evaluable ):
  def __init__(self, trans, first:bool):
    self.trans = trans
    self.first = first
    super().__init__(args=[trans])
  def evalf( self, trans ):
    assert isinstance( trans, transform.TransformChain )
    bf = trans[0]
    assert isinstance( bf, transform.Bifurcate )
    ftrans = bf.trans1 if self.first else bf.trans2
    return transform.TransformChain( ftrans + trans[1:] )

class Promote(Evaluable):
  def __init__(self, ndims:int, trans):
    self.ndims = ndims
    super().__init__(args=[trans])
  def evalf(self, trans):
    head, tail = trans.canonical.promote(self.ndims)
    return transform.TransformChain(head + tail)

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
  elif not util.isiterable(axis):
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
  if axes is None:
    a = asarray(a)
    b = asarray(b)
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    for idim in range(1, a.ndim):
      b = insertaxis(b, idim, a.shape[idim])
    axes = 0,
  else:
    a, b = _numpy_align(a, b)
  if not util.isiterable(axes):
    axes = axes,
  axes = _norm_and_sort(a.ndim, axes)
  return Dot([a, b], axes)

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

asdtype = lambda arg: arg if arg in (bool, int, float) else {'f': float, 'i': int, 'b': bool}[numpy.dtype(arg).kind]
asarray = lambda arg: arg if isarray(arg) else Constant(arg) if numeric.isarray(arg) or numpy.asarray(arg).dtype != object else stack(arg, axis=0)

class Array( Evaluable ):
  'array function'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__(self, args:tuple, shape:tuple, dtype:asdtype):
    assert all(numeric.isint(sh) or isarray(sh) and sh.ndim == 0 and sh.dtype == int for sh in shape)
    self.shape = tuple(sh if not isarray(sh) else sh.eval()[0] if sh.isconstant else sh.simplified for sh in shape)
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
    return ( self[i,...] for i in range(self.shape[0]) )

  size = property(lambda self: util.product(self.shape) if self.ndim else 1)
  T = property(lambda self: transpose(self))

  __add__ = __radd__ = add
  __sub__ = lambda self, other: subtract(self, other)
  __rsub__ = lambda self, other: subtract(other, self)
  __mul__ = __rmul__ = multiply
  __truediv__ = lambda self, other: divide(self, other)
  __rtruediv__ = lambda self, other: divide(other, self)
  __neg__ = lambda self: negative(self)
  __pow__ = power
  __abs__ = lambda self: abs(self)
  __mod__  = lambda self, other: mod(self, other)
  __str__ = __repr__ = lambda self: 'Array<{}>'.format(','.join(map(str, self.shape)) if hasattr(self, 'shape') else '?')

  sum = sum
  prod = product
  vector = lambda self, ndims: vectorize([self] * ndims)
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

  @property
  def blocks(self):
    return [(tuple(Range(n) for n in self.shape), self)]

  def _asciitree_str(self):
    return '{}({})'.format(type(self).__name__, ','.join(['?' if isarray(sh) else str(sh) for sh in self.shape]))

  # simplifications
  _multiply = lambda self, other: None
  _transpose = lambda self, axes: None
  _insertaxis = lambda self, axis, length: None
  _dot = lambda self, other, axes: None
  _get = lambda self, i, item: None
  _power = lambda self, n: None
  _add = lambda self, other: None
  _sum = lambda self, axis: None
  _take = lambda self, index, axis: None
  _determinant = lambda self: None
  _inverse = lambda self: None
  _takediag = lambda self: None
  _kronecker = lambda self, axis, length, pos: None
  _diagonalize = lambda self: None
  _product = lambda self: None
  _choose = lambda self, choices: None
  _cross = lambda self, other, axis: None
  _pointwise = lambda self, evalf, deriv, dtype: None
  _sign = lambda self: None
  _eig = lambda self, symmetric: None
  _inflate = lambda self, dofmap, length, axis: None
  _mask = lambda self, maskvec, axis: None
  _unravel = lambda self, axis, shape: None
  _ravel = lambda self, axis: None

class Normal( Array ):
  'normal'

  def __init__(self, lgrad:asarray):
    assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
    self.lgrad = lgrad
    super().__init__(args=[lgrad], shape=(len(lgrad),), dtype=float)

  def evalf( self, lgrad ):
    n = lgrad[...,-1]
    if n.shape[-1] == 1: # geom is 1D
      return numpy.sign(n)
    # orthonormalize n to G
    G = lgrad[...,:-1]
    GG = numeric.contract( G[:,:,_,:], G[:,:,:,_], axis=1 )
    v1 = numeric.contract( G, n[:,:,_], axis=1 )
    v2 = numpy.linalg.solve( GG, v1 )
    v3 = numeric.contract( G, v2[:,_,:], axis=2 )
    return numeric.normalize( n - v3 )

  def _derivative(self, var, seen):
    if len(self) == 1:
      return zeros(self.shape + var.shape)
    G = self.lgrad[...,:-1]
    GG = matmat(G.T, G)
    Gder = derivative(G, var, seen)
    nGder = matmat(self, Gder)
    return -matmat(G, inverse(GG), nGder)

class ArrayFunc( Array ):
  'deprecated ArrayFunc alias'

  def __init__(self, args:tuple, shape:tuple):
    warnings.warn( 'function.ArrayFunc is deprecated; use function.Array instead', DeprecationWarning )
    super().__init__(args=args, shape=shape, dtype=float)

class Constant( Array ):

  def __init__(self, value:numeric.const):
    self.value = value
    super().__init__(args=[], shape=value.shape, dtype=value.dtype)

  @cache.property
  def simplified(self):
    if not self.value.any():
      return zeros_like(self)
    return self

  def evalf( self ):
    return self.value[_]

  @cache.property
  def _isunit( self ):
    return numpy.equal(self.value, 1).all()

  def _derivative(self, var, seen):
    return zeros(self.shape + var.shape)

  def _transpose(self, axes):
    return asarray(self.value.transpose(axes))

  def _sum( self, axis ):
    return asarray( numpy.sum( self.value, axis ) )

  def _get( self, i, item ):
    if item.isconstant:
      item, = item.eval()
      return asarray(numeric.get(self.value, i, item))

  def _add( self, other ):
    if isinstance( other, Constant ):
      return asarray( numpy.add( self.value, other.value ) )

  def _inverse( self ):
    return asarray( numpy.linalg.inv( self.value ) )

  def _product(self):
    return asarray(self.value.prod(-1))

  def _multiply( self, other ):
    if self._isunit:
      return other
    if isinstance( other, Constant ):
      return asarray( numpy.multiply( self.value, other.value ) )

  def _takediag( self ):
    return asarray( numeric.takediag( self.value ) )

  def _take( self, index, axis ):
    if isinstance( index, Constant ):
      return asarray( self.value.take( index.value, axis ) )

  def _power( self, n ):
    if isinstance( n, Constant ):
      return asarray( numeric.power( self.value, n.value ) )

  def _dot( self, other, axes ):
    if self._isunit:
      return sum(other, axes)
    if isinstance( other, Constant ):
      return asarray( numeric.contract( self.value, other.value, axes ) )

  def _cross( self, other, axis ):
    if isinstance( other, Constant ):
      return asarray( numeric.cross( self.value, other.value, axis ) )

  def _pointwise( self, evalf, deriv, dtype ):
    retval = evalf( *self.value )
    assert retval.dtype == dtype
    return asarray( retval )

  def _eig(self, symmetric):
    eigval, eigvec = (numpy.linalg.eigh if symmetric else numpy.linalg.eig)(self.value)
    return Tuple((asarray(eigval), asarray(eigvec)))

  def _sign( self ):
    return asarray( numeric.sign( self.value ) )

  def _choose( self, choices ):
    if all( isinstance( choice, Constant ) for choice in choices ):
      return asarray( numpy.choose( self.value, [ choice.value for choice in choices ] ) )

  def _kronecker( self, axis, length, pos ):
    return asarray( numeric.kronecker( self.value, axis, length, pos ) )

  def _unravel( self, axis, shape ):
    shape = self.value.shape[:axis] + shape + self.value.shape[axis+1:]
    return asarray( self.value.reshape(shape) )

  def _mask( self, maskvec, axis ):
    return asarray( self.value[(slice(None),)*axis+(maskvec,)] )

class DofMap(Array):

  def __init__(self, dofs:tuple, index:asarray):
    assert index.ndim == 0 and index.dtype == int
    self.dofs = dofs
    self.index = index
    length = get([len(d) for d in dofs] + [0], iax=0, item=index)
    super().__init__(args=[index], shape=(length,), dtype=int)

  @property
  def dofmap(self):
    return self.index.asdict(self.dofs)

  def evalf(self, index):
    index, = index
    return (self.dofs[index] if index < len(self.dofs) else numpy.empty([0], dtype=int))[_]

class ElementSize( Array):
  'dimension of hypercube with same volume as element'

  def __init__(self, geometry:asarray, ndims:int=0):
    assert geometry.ndim == 1
    self.ndims = len(geometry)+ndims if ndims <= 0 else ndims
    iwscale = jacobian( geometry, self.ndims )
    super().__init__(args=[iwscale], shape=(), dtype=float)

  def evalf( self, iwscale ):
    volume = iwscale.sum()
    return numeric.power( volume, 1/self.ndims )[_]

class InsertAxis(Array):

  def __init__(self, func:asarray, axis:int, length:asarray):
    assert length.ndim == 0 and length.dtype == int
    assert 0 <= axis <= func.ndim
    self.func = func
    self.axis = axis
    self.length = length
    super().__init__(args=[func, length], shape=func.shape[:axis]+(length,)+func.shape[axis:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._insertaxis(self.axis, self.length)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return InsertAxis(func, self.axis, self.length)

  def evalf(self, func, length):
    length, = length
    return numeric.const(func).insertaxis(self.axis+1, length)

  def _derivative(self, var, seen):
    return insertaxis(derivative(self.func, var, seen), self.axis, self.length)

  def _get(self, i, item):
    if i == self.axis:
      if item.isconstant and self.length.isconstant:
        assert item.eval()[0] < self.length.eval()[0]
      return self.func
    return insertaxis(get(self.func, i-(i>self.axis), item), self.axis-(i<self.axis), self.length)

  def _sum(self, i):
    if i == self.axis:
      return multiply(self.func, self.length)
    return insertaxis(sum(self.func, i-(i>self.axis)), self.axis-(i<self.axis), self.length)

  def _product(self):
    if self.axis == self.ndim-1:
      return power(self.func, self.length)
    return insertaxis(product(self.func, -1), self.axis, self.length)

  def _power(self, n):
    if isinstance(n, InsertAxis) and self.axis == n.axis:
      assert n.length == self.length
      return insertaxis(power(self.func, n.func), self.axis, self.length)

  def _add(self, other):
    if isinstance(other, InsertAxis) and self.axis == other.axis:
      assert self.length == other.length
      return insertaxis(add(self.func, other.func), self.axis, self.length)

  def _multiply(self, other):
    if isinstance(other, InsertAxis) and self.axis == other.axis:
      assert self.length == other.length
      return insertaxis(multiply(self.func, other.func), self.axis, self.length)

  def _insertaxis(self, axis, length):
    if (not length.isconstant, axis) < (not self.length.isconstant, self.axis):
      return insertaxis(insertaxis(self.func, axis-(axis>self.axis), length), self.axis+(axis<=self.axis), self.length)

  def _take(self, index, axis):
    if axis == self.axis:
      return insertaxis(self.func, self.axis, index.shape[0])
    return insertaxis(take(self.func, index, axis-(axis>self.axis)), self.axis, self.length)

  def _takediag(self):
    if self.axis >= self.ndim-2:
      assert self.func.shape[-1] == self.shape[self.axis]
      return self.func
    return insertaxis(takediag(self.func), self.axis, self.length)

  def _dot(self, other, axes):
    if self.axis in axes:
      assert other.shape[self.axis] == self.shape[self.axis]
      return dot(self.func, sum(other, self.axis), [ax-(ax>self.axis) for ax in axes if ax != self.axis])

  def _mask(self, maskvec, axis):
    if axis == self.axis:
      assert len(maskvec) == self.shape[self.axis]
      return insertaxis(self.func, self.axis, maskvec.sum())
    return insertaxis(mask(self.func, maskvec, axis-(self.axis<axis)), self.axis, self.length)

  def _transpose(self, axes):
    i = axes.index(self.axis)
    return insertaxis(self.func.transpose([ax-(ax>self.axis) for ax in axes[:i]+axes[i+1:]]), i, self.length)

class Transpose(Array):

  def __init__(self, func:asarray, axes:tuple):
    assert sorted(axes) == list(range(func.ndim))
    self.func = func
    self.axes = axes
    super().__init__(args=[func], shape=[func.shape[n] for n in axes], dtype=func.dtype)

  @cache.property
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
    return transpose(self.func, newaxes)

  def _takediag( self ):
    if self.axes[-2:] in [(self.ndim-2,self.ndim-1), (self.ndim-1,self.ndim-2)]:
      axes = self.axes[:-2] + (self.ndim-2,)
      return transpose(takediag(self.func), axes)

  def _get(self, i, item):
    axis = self.axes[i]
    axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
    return transpose(get(self.func, axis, item), axes)

  def _sum(self, i):
    axis = self.axes[i]
    axes = [ax-(ax>axis) for ax in self.axes if ax != axis]
    return transpose(sum(self.func, axis), axes)

  def _derivative(self, var, seen):
    return transpose(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)))

  def _multiply(self, other):
    if isinstance(other, Transpose) and self.axes == other.axes:
      return transpose(multiply(self.func, other.func), self.axes)
    other_trans = other._transpose(_invtrans(self.axes))
    if other_trans is not None:
      return transpose(multiply(self.func, other_trans), self.axes)

  def _add(self, other):
    if isinstance(other, Transpose) and self.axes == other.axes:
      return transpose(add(self.func, other.func), self.axes)
    other_trans = other._transpose(_invtrans(self.axes))
    if other_trans is not None:
      return transpose(add(self.func, other_trans), self.axes)

  def _take(self, indices, axis):
    return transpose(take(self.func, indices, self.axes[axis]), self.axes)

  def _dot(self, other, axes):
    sumaxes = [self.axes[axis] for axis in axes]
    trydot = self.func._dot(transpose(other, _invtrans(self.axes)), sumaxes)
    if trydot is not None:
      trans = [axis - builtins.sum(ax<axis for ax in sumaxes) for axis in self.axes if axis not in sumaxes]
      return transpose(trydot, trans)

  def _mask(self, maskvec, axis):
    return transpose(mask(self.func, maskvec, self.axes[axis]), self.axes)

class Get(Array):

  def __init__(self, func:asarray, axis:int, item:asarray):
    assert item.ndim == 0 and item.dtype == int
    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    if item.isconstant and numeric.isint(func.shape[axis]):
      assert 0 <= item.eval()[0] < func.shape[axis], 'item is out of bounds'
    super().__init__(args=[func, item], shape=func.shape[:axis]+func.shape[axis+1:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    item = self.item.simplified
    retval = func._get(self.axis, item)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Get(func, self.axis, item)

  def evalf(self, arr, item):
    item, = item
    return arr[(slice(None),)*(self.axis+1)+(item,)]

  def _derivative(self, var, seen):
    f = derivative(self.func, var, seen)
    return get(f, self.axis, self.item)

  def _get( self, i, item ):
    tryget = self.func._get(i+(i>=self.axis), item)
    if tryget is not None:
      return get( tryget, self.axis, self.item )

  def _take( self, indices, axis ):
    return get( take( self.func, indices, axis+(axis>=self.axis) ), self.axis, self.item )

class Product( Array ):

  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._product()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Product(func)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numpy.product( arr, axis=-1 )

  def _derivative(self, var, seen):
    grad = derivative(self.func, var, seen)
    funcs = stack([util.product(self.func[...,j] for j in range(self.func.shape[-1]) if i != j) for i in range(self.func.shape[-1])], axis=-1)
    return (grad * funcs[(...,)+(_,)*var.ndim]).sum(self.ndim)

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #ext = (...,)+(_,)*len(shape)
    #return self[ext] * ( derivative(self.func,var,shape,seen) / self.func[ext] ).sum( self.ndim )

  def _get( self, i, item ):
    func = get( self.func, i, item )
    return product( func, -1 )

class RootCoords( Array ):

  def __init__(self, ndims:int, trans=TRANS):
    self.trans = trans
    super().__init__(args=[POINTS,trans], shape=[ndims], dtype=float)

  def evalf( self, points, chain ):
    'evaluate'

    ndims = len(self)
    head, tail = chain.promote( ndims )
    while head and head[0].todims != ndims:
      head = head[1:]
    return transform.apply( head + tail, points )

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords) and len(var) > 0:
      return RootTransform(len(self), len(var), self.trans)
    return zeros(self.shape+var.shape)

class RootTransform( Array ):

  def __init__(self, ndims:int, nvars:int, trans):
    super().__init__(args=[Promote(ndims, trans)], shape=(ndims,nvars), dtype=float)

  def evalf(self, chain):
    todims, fromdims = self.shape
    while chain and chain[0].todims != todims:
      chain = chain[1:]
    return transform.linearfrom(chain, fromdims)[_]

  def _derivative(self, var, seen):
    return zeros(self.shape+var.shape)

class Function( Array ):

  def __init__(self, stds:tuple, depth:int, trans, index:asarray, derivs:tuple=()):
    assert index.ndim == 0 and index.dtype == int
    self.stds = stds
    self.depth = depth
    self.trans = trans
    self.index = index
    nshapes = get([std.nshapes for std in stds] + [0], iax=0, item=index)
    super().__init__(args=(CACHE,POINTS,trans,index), shape=(nshapes,)+derivs, dtype=float)

  @property
  def stdmap(self):
    return self.index.asdict(self.stds)

  def evalf(self, cache, points, trans, index):
    index, = index
    if index == len(self.stds):
      return numpy.empty((1,0)+self.shape[1:])
    tail = trans[self.depth:]
    if tail:
      points = cache[transform.apply](tail, points)
    fvals = cache[self.stds[index].eval](points, self.ndim-1)
    assert fvals.ndim == self.ndim+1
    if tail:
      for i, ndims in enumerate(self.shape[1:]):
        linear = cache[transform.linearfrom](tail, ndims)
        fvals = numeric.dot(fvals, linear, axis=i+2)
    return fvals

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords):
      return Function(self.stds, self.depth, self.trans, self.index, self.shape[1:]+var.shape)
    return zeros(self.shape+var.shape, dtype=self.dtype)

class Choose( Array ):

  def __init__(self, level:asarray, choices:tuple):
    assert all(isarray(choice) and choice.shape == level.shape for choice in choices)
    self.level = level
    self.choices = choices
    super().__init__(args=(level,)+choices, shape=level.shape, dtype=_jointdtype(*[choice.dtype for choice in choices]))

  def edit(self, op):
    return Choose(op(self.level), [op(func) for func in self.choices])

  @cache.property
  def simplified(self):
    level = self.level.simplified
    choices = tuple(choice.simplified for choice in self.choices)
    if all(iszero(choice) for choice in choices):
      return zeros(self.shape)
    retval = level._choose(choices)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Choose(level, choices)

  def evalf(self, level, *choices):
    return numpy.choose(level, choices)

  def _derivative(self, var, seen):
    grads = [derivative(choice, var, seen) for choice in self.choices]
    if not any(grads): # all-zero special case; better would be allow merging of intervals
      return zeros(self.shape + var.shape)
    return choose(self.level[(...,)+(_,)*var.ndim], grads)

class Choose2D( Array ):
  'piecewise function'

  def __init__(self, coords:asarray, contour:asarray, fin:asarray, fout:asarray):
    assert fin.shape == fout.shape
    self.contour = contour
    super().__init__(args=(coords,contour,fin,fout), shape=fin.shape, dtype=_jointdtype(fin.dtype,fout.dtype))

  @staticmethod
  def evalf( self, xy, fin, fout ):
    'evaluate'

    from matplotlib import nxutils
    mask = nxutils.points_inside_poly( xy.T, self.contour )
    out = numpy.empty( fin.shape or fout.shape, dtype=self.dtype )
    out[...,mask] = fin[...,mask] if fin.shape else fin
    out[...,~mask] = fout[...,~mask] if fout.shape else fout
    return out

class Inverse( Array ):

  def __init__(self, func:asarray):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=float)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._inverse()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Inverse(func)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+1
    try:
      inv = numpy.linalg.inv( arr )
    except numpy.linalg.LinAlgError:
      inv = numpy.empty_like( arr )
      flat = (-1,) + arr.shape[-2:]
      for arri, invi in zip( arr.reshape(flat), inv.reshape(flat) ):
        try:
          invi[...] = numpy.linalg.inv(arri)
        except numpy.linalg.LinAlgError:
          invi[...] = numpy.nan
    return inv

  def _derivative(self, var, seen):
    G = derivative(self.func, var, seen)
    n = var.ndim
    a = slice(None)
    return -sum(self[(...,a,a,_,_)+(_,)*n] * G[(...,_,a,a,_)+(a,)*n] * self[(...,_,_,a,a)+(_,)*n], [-2-n, -3-n])

  def _eig(self, symmetric):
    eigval, eigvec = Eig(self.func, symmetric)
    return Tuple((reciprocal(eigval), eigvec))

class Concatenate(Array):

  def __init__(self, funcs:tuple, axis:int=0):
    ndim = funcs[0].ndim
    assert all(isarray(func) and func.ndim == ndim for func in funcs)
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

  @cache.property
  def _withslices(self):
    return tuple((Range(func.shape[self.axis], n), func) for n, func in zip(util.cumsum(func.shape[self.axis] for func in self.funcs), self.funcs))

  @cache.property
  def simplified(self):
    funcs = tuple(func.simplified for func in self.funcs if func.shape[self.axis] != 0)
    if all(iszero(func) for func in funcs):
      return zeros_like(self)
    if len(funcs) == 1:
      return funcs[0]
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

  @cache.property
  def blocks(self):
    return _concatblocks(((ind[:self.axis], ind[self.axis+1:]), (ind[self.axis]+n, f))
      for n, func in zip(util.cumsum(func.shape[self.axis] for func in self.funcs), self.funcs)
        for ind, f in func.blocks)

  def _get( self, i, item ):
    if i != self.axis:
      axis = self.axis - (self.axis > i)
      return concatenate([get(f, i, item) for f in self.funcs], axis=axis)
    if item.isconstant:
      item, = item.eval()
      for f in self.funcs:
        if item < f.shape[i]:
          return get(f, i, item)
        item -= f.shape[i]
      raise Exception

  def _derivative(self, var, seen):
    funcs = [derivative(func, var, seen) for func in self.funcs]
    return concatenate(funcs, axis=self.axis)

  def _multiply( self, other ):
    funcs = [multiply(func, take(other, s, self.axis)) for s, func in self._withslices]
    return concatenate(funcs, self.axis)

  def _cross( self, other, axis ):
    if axis == self.axis:
      n = 1, 2, 0
      m = 2, 0, 1
      return take(self,n,axis) * take(other,m,axis) - take(self,m,axis) * take(other,n,axis)
    funcs = [cross(func, take(other, s, self.axis), axis) for s, func in self._withslices]
    return concatenate(funcs, self.axis)

  def _add( self, other ):
    if isinstance(other, Concatenate) and self.axis == other.axis:
      if [f1.shape[self.axis] for f1 in self.funcs] == [f2.shape[self.axis] for f2 in other.funcs]:
        funcs = [add(f1, f2) for f1, f2 in zip(self.funcs, other.funcs)]
      else:
        if isarray(self.shape[self.axis]):
          raise NotImplementedError
        funcs = []
        beg1 = 0
        for func1 in self.funcs:
          end1 = beg1 + func1.shape[self.axis]
          beg2 = 0
          for func2 in other.funcs:
            end2 = beg2 + func2.shape[self.axis]
            if end1 > beg2 and end2 > beg1:
              mask = numpy.zeros(self.shape[self.axis], dtype=bool)
              mask[builtins.max(beg1, beg2):builtins.min(end1, end2)] = True
              funcs.append(add(Mask(func1, mask[beg1:end1], self.axis), Mask(func2, mask[beg2:end2], self.axis)))
            beg2 = end2
          beg1 = end1
    else:
      funcs = [add(func, take(other, s, self.axis)) for s, func in self._withslices]
    return concatenate( funcs, self.axis )

  def _sum( self, axis ):
    if axis == self.axis:
      return util.sum( sum( func, axis ) for func in self.funcs )
    funcs = [ sum( func, axis ) for func in self.funcs ]
    axis = self.axis - (axis<self.axis)
    return concatenate( funcs, axis )

  def _transpose(self, axes):
    funcs = [transpose(func, axes) for func in self.funcs]
    axis = axes.index(self.axis)
    return concatenate(funcs, axis)

  def _insertaxis(self, axis, length):
    funcs = [insertaxis(func, axis, length) for func in self.funcs]
    return concatenate(funcs, self.axis+(axis<=self.axis))

  def _takediag( self ):
    if self.axis < self.ndim-2:
      return concatenate( [ takediag(f) for f in self.funcs ], axis=self.axis )
    axis = self.ndim-self.axis-3 # -1=>-2, -2=>-1
    funcs = [takediag(take(func, s, axis)) for s, func in self._withslices]
    return concatenate(funcs, axis=-1)

  def _take( self, indices, axis ):
    if axis != self.axis:
      return concatenate([take(func, indices, axis) for func in self.funcs], self.axis)
    if not indices.isconstant:
      return
    indices, = indices.eval()
    assert numpy.logical_and(numpy.greater_equal(indices, 0), numpy.less(indices, self.shape[axis])).all()
    ifuncs = numpy.hstack([ numpy.repeat(ifunc,func.shape[axis]) for ifunc, func in enumerate(self.funcs) ])[indices]
    splits, = numpy.nonzero( numpy.diff(ifuncs) != 0 )
    funcs = []
    for i, j in zip( numpy.hstack([ 0, splits+1 ]), numpy.hstack([ splits+1, len(indices) ]) ):
      ifunc = ifuncs[i]
      assert numpy.equal(ifuncs[i:j], ifunc).all()
      offset = builtins.sum( func.shape[axis] for func in self.funcs[:ifunc] )
      funcs.append( take( self.funcs[ifunc], indices[i:j] - offset, axis ) )
    if len( funcs ) == 1:
      return funcs[0]
    return concatenate( funcs, axis=axis )

  def _dot( self, other, axes ):
    funcs = [dot(func, take(other, s, self.axis), axes) for s, func in self._withslices]
    if self.axis in axes:
      return util.sum(funcs)
    return concatenate(funcs, self.axis - builtins.sum(axis < self.axis for axis in axes))

  def _power( self, n ):
    return concatenate([power(func, take(n, s, self.axis)) for s, func in self._withslices], self.axis)

  def _diagonalize( self ):
    if self.axis < self.ndim-1:
      return concatenate( [ diagonalize(func) for func in self.funcs ], self.axis )

  def _kronecker( self, axis, length, pos ):
    return concatenate( [ kronecker(func,axis,length,pos) for func in self.funcs ], self.axis+(axis<=self.axis) )

  def _mask( self, maskvec, axis ):
    if axis != self.axis:
      return concatenate([mask(func,maskvec,axis) for func in self.funcs], self.axis)
    if all(s.isconstant for s, func in self._withslices):
      return concatenate([mask(func, maskvec[s.eval()[0]], axis) for s, func in self._withslices], axis)

class Interpolate( Array ):
  'interpolate uniformly spaced data; stepwise for now'

  def __init__(self, x:asarray, xp:numeric.const, fp:numeric.const, left=None, right=None):
    assert xp.ndim == fp.ndim == 1
    if not numpy.greater(numpy.diff(xp), 0).all():
      warnings.warn( 'supplied x-values are non-increasing' )
    assert x.ndim == 0
    self.xp = xp
    self.fp = fp
    self.left = left
    self.right = right
    super.__init__(args=[x], shape=(), dtype=float)

  def evalf( self, x ):
    return numpy.interp( x, self.xp, self.fp, self.left, self.right )

class Cross( Array ):

  def __init__(self, func1:asarray, func2:asarray, axis:int):
    assert func1.shape == func2.shape
    assert 0 <= axis < func1.ndim and func2.shape[axis] == 3
    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    super().__init__(args=(func1,func2), shape=func1.shape, dtype=_jointdtype(func1.dtype, func2.dtype))

  @cache.property
  def simplified(self):
    func1 = self.func1.simplified
    func2 = self.func2.simplified
    retval = func1._cross(func2, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._cross(func1, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return negative(retval).simplified
    return Cross(func1, func2, self.axis)

  def evalf( self, a, b ):
    assert a.ndim == b.ndim == self.ndim+1
    return numeric.cross( a, b, self.axis+1 )

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    return cross(self.func1[ext], derivative(self.func2, var, seen), axis=self.axis) \
         - cross(self.func2[ext], derivative(self.func1, var, seen), axis=self.axis)

  def _take( self, index, axis ):
    if axis != self.axis:
      return cross(take(self.func1, index, axis), take(self.func2, index, axis), self.axis)

class Determinant( Array ):

  def __init__(self, func:asarray):
    assert isarray(func) and func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-2], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._determinant()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Determinant(func)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+3
    return numpy.linalg.det( arr )

  def _derivative(self, var, seen):
    Finv = swapaxes(inverse(self.func), -2, -1)
    G = derivative(self.func, var, seen)
    ext = (...,)+(_,)*var.ndim
    return self[ext] * sum(Finv[ext] * G, axis=[-2-var.ndim,-1-var.ndim])

class Multiply(Array):

  def __init__(self, funcs:orderedset):
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2) and func1.shape == func2.shape
    self.funcs = func1, func2
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def edit(self, op):
    return Multiply([op(func) for func in self.funcs])

  @cache.property
  def simplified(self):
    func1 = self.funcs[0].simplified
    func2 = self.funcs[1].simplified
    if func1 == func2:
      return power(func1, 2).simplified
    retval = func1._multiply(func2)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._multiply(func1)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Multiply([func1, func2])

  def evalf( self, arr1, arr2 ):
    return arr1 * arr2

  def _sum( self, axis ):
    func1, func2 = self.funcs
    return dot( func1, func2, [axis] )

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    return multiply(get(func1, axis, item), get(func2, axis, item))

  def _add( self, other ):
    func1, func2 = self.funcs
    if other == func1:
      return func1 * (func2+1)
    if other == func2:
      return func2 * (func1+1)
    if isinstance( other, Multiply ):
      common = _findcommon( self.funcs, other.funcs )
      if common:
        f, (g1,g2) = common
        return f * add( g1, g2 )

  def _determinant( self ):
    func1, func2 = self.funcs
    if self.shape[-2:] == (1,1):
      return multiply(determinant(func1), determinant(func2))
    if 1 in func1.shape[-2:]:
      func1, func2 = func2, func1 # swap singleton-axis argument into func2
    if 1 in func1.shape[-2:]: # tensor product
      return zeros(())
    if 1 in func2.shape[-2:]:
      det2 = power(func2[...,0,0], self.shape[-1]) if func2.shape[-2:] == (1,1) \
        else product(func2.sum( -1 if func2.shape[-1] == 1 else -2 ), axis=0)
      return multiply(determinant(func1), det2)

  def _product(self):
    func1, func2 = self.funcs
    return multiply(product(func1, -1), product(func2, -1))

  def _multiply( self, other ):
    func1, func2 = self.funcs
    func1_other = func1._multiply(other)
    if func1_other is not None:
      return multiply( func1_other, func2 )
    func2_other = func2._multiply(other)
    if func2_other is not None:
      return multiply( func1, func2_other )

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*var.ndim
    return func1[ext] * derivative(func2, var, seen) \
         + func2[ext] * derivative(func1, var, seen)

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) * takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return multiply(take(func1, index, axis), take(func2, index, axis))

  def _power( self, n ):
    func1, func2 = self.funcs
    func1pow = func1._power(n)
    func2pow = func2._power(n)
    if func1pow is not None and func2pow is not None:
      return multiply( func1pow, func2pow )

  def _dot(self, other, axes):
    func1, func2 = self.funcs
    trydot1 = func1._dot(multiply(func2, other), axes)
    if trydot1 is not None:
      return trydot1
    trydot2 = func2._dot(multiply(func1, other), axes)
    if trydot2 is not None:
      return trydot2

class Add(Array):

  def __init__(self, funcs:orderedset):
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2) and func1.shape == func2.shape
    self.funcs = func1, func2
    super().__init__(args=self.funcs, shape=func1.shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def edit(self, op):
    return Add([op(func) for func in self.funcs])

  @cache.property
  def simplified(self):
    func1 = self.funcs[0].simplified
    func2 = self.funcs[1].simplified
    if iszero(func1):
      return func2
    if iszero(func2):
      return func1
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

  def evalf( self, arr1, arr2=None ):
    return arr1 + arr2

  def _sum( self, axis ):
    sum1 = sum( self.funcs[0], axis )
    sum2 = sum( self.funcs[1], axis )
    n1 = self.funcs[0].shape[axis]
    n2 = self.funcs[1].shape[axis]
    return sum1 + sum2 if n1 == n2 else sum1 * n2 + sum2 * n1

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    return derivative(func1, var, seen) + derivative(func2, var, seen)

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    return add(get(func1, axis, item), get(func2, axis, item))

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) + takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return add(take(func1, index, axis), take(func2, index, axis))

  def _add( self, other ):
    func1, func2 = self.funcs
    func1_other = func1._add(other)
    if func1_other is not None:
      return add( func1_other, func2 )
    func2_other = func2._add(other)
    if func2_other is not None:
      return add( func1, func2_other )

  def _mask(self, maskvec, axis):
    func1, func2 = self.funcs
    return add(mask(func1, maskvec, axis), mask(func2, maskvec, axis))

class BlockAdd( Array ):
  'block addition (used for DG)'

  def __init__(self, funcs:orderedset):
    self.funcs = funcs
    shapes = set(func.shape for func in funcs)
    assert len(shapes) == 1, 'multiple shapes in BlockAdd'
    shape, = shapes
    super().__init__(args=funcs, shape=shape, dtype=_jointdtype(*[func.dtype for func in self.funcs]))

  def edit(self, op):
    return BlockAdd([op(func) for func in self.funcs])

  @cache.property
  def simplified(self):
    funcs = []
    for func in self.funcs:
      func = func.simplified
      if isinstance(func, BlockAdd):
        funcs.extend(func.funcs)
      elif not iszero(func):
        funcs.append(func)
    return BlockAdd(funcs) if len(funcs) > 1 else funcs[0] if funcs else zeros_like(self)

  def evalf(self, *args):
    return util.sum(args)

  def _add( self, other ):
    return BlockAdd(tuple(self.funcs) + tuple(other.funcs if isinstance(other, BlockAdd) else [other]))

  def _dot( self, other, axes ):
    return BlockAdd([dot(func, other, axes) for func in self.funcs])

  def _sum( self, axis ):
    return BlockAdd([sum(func, axis) for func in self.funcs])

  def _derivative(self, var, seen):
    return BlockAdd([derivative(func, var, seen) for func in self.funcs])

  def _get( self, i, item ):
    return BlockAdd([get(func, i, item) for func in self.funcs])

  def _takediag( self ):
    return BlockAdd([takediag(func) for func in self.funcs])

  def _take( self, indices, axis ):
    return BlockAdd([take(func, indices, axis) for func in self.funcs])

  def _transpose(self, axes):
    return BlockAdd([transpose(func, axes) for func in self.funcs])

  def _insertaxis(self, axis, length):
    return BlockAdd([insertaxis(func, axis, length) for func in self.funcs])

  def _multiply( self, other ):
    return BlockAdd([multiply(func, other) for func in self.funcs])

  def _kronecker( self, axis, length, pos ):
    return BlockAdd([kronecker(func, axis, length, pos) for func in self.funcs])

  def _mask(self, maskvec, axis):
    return BlockAdd([mask(func, maskvec, axis) for func in self.funcs])

  @cache.property
  def blocks(self):
    gathered = tuple((ind, util.sum(f)) for ind, f in util.gather(block for func in self.funcs for block in func.blocks))
    if len(gathered) > 1:
      for idim in range(self.ndim):
        gathered = _concatblocks(((ind[:idim], ind[idim+1:]), (ind[idim], f)) for ind, f in gathered)
    return gathered

class Dot(Array):

  def __init__(self, funcs:orderedset, axes:tuple):
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2) and func1.shape == func2.shape
    self.funcs = func1, func2
    self.axes = axes
    assert all(0 <= ax < func1.ndim for ax in axes)
    assert all(ax1 < ax2 for ax1, ax2 in zip(axes[:-1], axes[1:]))
    shape = func1.shape
    self.axes_complement = list(range(func1.ndim))
    for ax in reversed(self.axes):
      shape = shape[:ax] + shape[ax+1:]
      del self.axes_complement[ax]
    _abc = numeric._abc[:func1.ndim+1]
    self._einsumfmt = '{0},{0}->{1}'.format(_abc, ''.join(a for i, a in enumerate(_abc) if i-1 not in axes))
    super().__init__(args=funcs, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  def edit(self, op):
    return Dot([op(func) for func in self.funcs], self.axes)

  @cache.property
  def simplified(self):
    func1 = self.funcs[0].simplified
    func2 = self.funcs[1].simplified
    if len(self.axes) == 0:
      return multiply(func1, func2).simplified
    if iszero(func1) or iszero(func2):
      return zeros(self.shape)
    for i, axis in enumerate(self.axes):
      if func1.shape[axis] == 1:
        return dot(sum(func1,axis), sum(func2,axis), self.axes[:i] + tuple(axis-1 for axis in self.axes[i+1:])).simplified
    retval = func1._dot(func2, self.axes)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._dot(func1, self.axes)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Dot([func1, func2], self.axes)

  def evalf( self, arr1, arr2 ):
    return numpy.einsum(self._einsumfmt, arr1, arr2)

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    funcaxis = self.axes_complement[axis]
    return dot(get(func1, funcaxis, item), get(func2, funcaxis, item), [ax-(ax>=funcaxis) for ax in self.axes])

  def _derivative(self, var, seen):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*var.ndim
    return dot(derivative(func1, var, seen), func2[ext], self.axes) \
         + dot(func1[ext], derivative(func2, var, seen), self.axes)

  def _add( self, other ):
    if isinstance( other, Dot ) and self.axes == other.axes:
      common = _findcommon( self.funcs, other.funcs )
      if common:
        f, (g1,g2) = common
        return dot( f, g1 + g2, self.axes )

  def _takediag( self ):
    func1, func2 = self.funcs
    if self.axes[-1] < func1.ndim-2:
      return dot( takediag(func1), takediag(func2), self.axes )

  def _sum( self, axis ):
    funcaxis = self.axes_complement[axis]
    func1, func2 = self.funcs
    return dot( func1, func2, self.axes + (funcaxis,) )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    funcaxis = self.axes_complement[axis]
    return dot(take(func1, index, funcaxis), take(func2, index, funcaxis), self.axes)

class Sum( Array ):

  def __init__(self, func:asarray, axis:int):
    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    super().__init__(args=[func], shape=shape, dtype=int if func.dtype == bool else func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._sum(self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Sum(func, self.axis)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numpy.sum(arr, self.axis+1)

  def _sum( self, axis ):
    trysum = self.func._sum(axis+(axis>=self.axis))
    if trysum is not None:
      return sum(trysum, self.axis-(axis<self.axis))

  def _derivative(self, var, seen):
    return sum(derivative(self.func, var, seen), self.axis)

class Debug( Array ):
  'debug'

  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=func.dtype)

  def evalf( self, arr ):
    'debug'

    assert arr.ndim == self.ndim+1
    log.debug( 'debug output:\n%s' % arr )
    return arr

  def __str__( self ):
    'string representation'

    return '{DEBUG}'

  def _derivative(self, var, seen):
    return Debug(derivative(self.func, var, seen))

class TakeDiag( Array ):

  def __init__(self, func:asarray):
    assert func.shape[-1] == func.shape[-2]
    self.func = func
    super().__init__(args=[func], shape=func.shape[:-1], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._takediag()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return TakeDiag(func)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numeric.takediag( arr )

  def _derivative(self, var, seen):
    fder = derivative(self.func, var, seen)
    return transpose(takediag(fder, self.func.ndim-2, self.func.ndim-1), tuple(range(self.func.ndim-2))+(-1,)+tuple(range(self.func.ndim-2,fder.ndim-2)))

  def _sum( self, axis ):
    if axis != self.ndim-1:
      return takediag( sum( self.func, axis ) )

class Take( Array ):

  def __init__(self, func:asarray, indices:asarray, axis:int):
    assert indices.ndim == 1 and indices.dtype == int
    assert 0 <= axis < func.ndim
    self.func = func
    self.axis = axis
    self.indices = indices
    shape = func.shape[:axis] + indices.shape + func.shape[axis+1:]
    super().__init__(args=[func,indices], shape=shape, dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    indices = self.indices.simplified
    if self.shape[self.axis] == 0:
      return zeros(self.shape, dtype=self.dtype)
    if indices.isconstant:
      index_, = indices.eval()
      if len(index_) == func.shape[self.axis] and numpy.equal(numpy.diff(index_), 1).all():
        return func
    retval = func._take(indices, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Take(func, indices, self.axis)

  def evalf( self, arr, indices ):
    if indices.shape[0] != 1:
      raise NotImplementedError( 'non element-constant indexing not supported yet' )
    return numpy.take( arr, indices[0], self.axis+1 )

  def _derivative(self, var, seen):
    return take(derivative(self.func, var, seen), self.indices, self.axis)

  def _take( self, index, axis ):
    if axis == self.axis:
      return take( self.func, self.indices[index], axis )
    trytake = self.func._take(index, axis)
    if trytake is not None:
      return take( trytake, self.indices, self.axis )

class Power(Array):

  def __init__(self, func:asarray, power:asarray):
    assert func.shape == power.shape
    self.func = func
    self.power = power
    super().__init__(args=[func,power], shape=func.shape, dtype=float)

  @cache.property
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

  def evalf( self, base, exp ):
    return numeric.power( base, exp )

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    if self.power.isconstant:
      p, = self.power.eval()
      return zeros(self.shape + var.shape) if p == 0 \
        else multiply(p, power(self.func, p-1))[ext] * derivative(self.func, var, seen)
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    powerm1 = choose(equal(self.power, 0), [self.power-1, 0]) # avoid introducing negative powers where possible
    return (self.power * power(self.func, powerm1))[ext] * derivative(self.func, var, seen) \
         + (ln(self.func) * self)[ext] * derivative(self.power, var, seen)

  def _power( self, n ):
    func = self.func
    newpower = n * self.power
    if iszero( self.power % 2 ) and not iszero( newpower % 2 ):
      func = abs( func )
    return power( func, newpower )

  def _get( self, axis, item ):
    return power(get(self.func, axis, item), get(self.power, axis, item))

  def _sum( self, axis ):
    if self == (self.func**2):
      return dot( self.func, self.func, axis )

  def _takediag( self ):
    return power( takediag( self.func ), takediag( self.power ) )

  def _take( self, index, axis ):
    return power(take(self.func, index, axis), take(self.power, index, axis))

  def _multiply( self, other ):
    if isinstance( other, Power ) and self.func == other.func:
      return power( self.func, self.power + other.power )
    if other == self.func:
      return power( self.func, self.power + 1 )

  def _sign( self ):
    if iszero( self.power % 2 ):
      return ones_like(self)

class Pointwise( Array ):

  def __init__(self, args:asarray, evalfun, deriv, dtype:asdtype):
    assert args.ndim >= 1 and args.shape[0] >= 1
    shape = args.shape[1:]
    self.args = args
    self.evalfun = evalfun
    self.deriv = deriv
    super().__init__(args=[args], shape=shape, dtype=dtype)

  @cache.property
  def simplified(self):
    args = self.args.simplified
    retval = args._pointwise(self.evalfun, self.deriv, self.dtype)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Pointwise(args, self.evalfun, self.deriv, self.dtype)

  def evalf( self, args ):
    assert args.shape[1:] == self.args.shape
    return self.evalfun( *args.swapaxes(0,1) )

  def _derivative(self, var, seen):
    if self.deriv is None:
      raise NotImplementedError('derivative is not defined for this operator')
    return (self.deriv(self.args)[(...,)+(_,)*var.ndim] * derivative(self.args, var, seen)).sum( 0 )

  def _takediag( self ):
    return pointwise( takediag(self.args), self.evalfun, self.deriv, self.dtype )

  def _get( self, axis, item ):
    return pointwise( get( self.args, axis+1, item ), self.evalfun, self.deriv, self.dtype )

  def _take( self, index, axis ):
    return pointwise( take( self.args, index, axis+1 ), self.evalfun, self.deriv, self.dtype )

class Sign( Array ):

  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape, dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._sign()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Sign(func)

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+1
    return numpy.sign( arr )

  def _derivative(self, var, seen):
    return zeros(self.shape + var.shape)

  def _takediag( self ):
    return sign( takediag(self.func) )

  def _get( self, axis, item ):
    return sign( get( self.func, axis, item ) )

  def _take( self, index, axis ):
    return sign( take( self.func, index, axis ) )

  def _sign( self ):
    return self

  def _power( self, n ):
    if iszero( n % 2 ):
      return ones_like(self)

class Sampled( Array ):
  'sampled'

  def __init__(self, data:util.frozendict, trans=TRANS):
    self.data = data.copy()
    self.trans = trans
    items = iter(self.data.items())
    trans0, (values0,points0) = next(items)
    shape = values0.shape[1:]
    assert all( transi.fromdims == trans0.fromdims and valuesi.shape == pointsi.shape[:1]+shape for transi, (valuesi,pointsi) in items )
    super().__init__(args=[trans,POINTS], shape=shape, dtype=float)

  def evalf( self, trans, points ):
    (myvals,mypoints), tail = trans.lookup_item( self.data )
    evalpoints = tail.apply( points )
    assert mypoints.shape == evalpoints.shape and numpy.equal(mypoints, evalpoints).all(), 'Illegal point set'
    return myvals

class Elemwise( Array ):
  'elementwise constant data'

  def __init__(self, fmap:util.frozendict, shape:tuple, default=None, trans=TRANS):
    self.fmap = fmap
    self.default = default
    self.trans = trans
    super().__init__(args=[trans], shape=shape, dtype=float)

  def evalf( self, trans ):
    try:
      value, tail = trans.lookup_item( self.fmap )
    except KeyError:
      value = self.default
      if value is None:
        raise
    value = numpy.asarray( value )
    assert value.shape == self.shape, 'wrong shape: {} != {}'.format( value.shape, self.shape )
    return value[_]

  def _derivative(self, var, seen):
    return zeros(self.shape+var.shape)

class Eig( Evaluable ):

  def __init__(self, func:asarray, symmetric:bool=False):
    assert func.ndim >= 2 and func.shape[-1] == func.shape[-2]
    self.symmetric = symmetric
    self.func = func
    super().__init__(args=[func])

  def __len__(self):
    return 2

  def __iter__(self):
    yield ArrayFromTuple(self, index=0, shape=self.func.shape[:-1], dtype=float)
    yield ArrayFromTuple(self, index=1, shape=self.func.shape, dtype=float)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._eig(self.symmetric)
    if retval is not None:
      assert len(retval) == 2
      return retval.simplified
    return Eig(func, self.symmetric)

  def evalf(self, arr):
    return (numpy.linalg.eigh if self.symmetric else numeric.eig)(arr)

class ArrayFromTuple(Array):

  def __init__(self, arrays, index:int, shape:tuple, dtype:asdtype):
    assert isevaluable(arrays)
    assert 0 <= index < len(arrays)
    self.arrays = arrays
    self.index = index
    super().__init__(args=[arrays], shape=shape, dtype=dtype)

  def evalf(self, arrays):
    assert isinstance(arrays, tuple)
    return arrays[self.index]

class Zeros( Array ):
  'zero'

  def __init__(self, shape:tuple, dtype:asdtype):
    super().__init__(args=[asarray(sh) for sh in shape], shape=shape, dtype=dtype)

  def evalf(self, *shape):
    if shape:
      shape, = zip(*shape)
    return numpy.zeros((1,)+shape, dtype=self.dtype)

  @property
  def blocks(self):
    return ()

  def _derivative(self, var, seen):
    return zeros(self.shape+var.shape, dtype=self.dtype)

  def _add(self, other):
    return other

  def _multiply(self, other):
    return self

  def _dot(self, other, axes):
    shape = [sh for axis, sh in enumerate(self.shape) if axis not in axes]
    return zeros(shape, dtype=_jointdtype(self.dtype,other.dtype))

  def _cross(self, other, axis):
    return self

  def _diagonalize( self ):
    return zeros( self.shape + (self.shape[-1],), dtype=self.dtype )

  def _sum( self, axis ):
    return zeros( self.shape[:axis] + self.shape[axis+1:], dtype=self.dtype )

  def _transpose(self, axes):
    shape = [self.shape[n] for n in axes]
    return zeros(shape, dtype=self.dtype)

  def _insertaxis(self, axis, length):
    return zeros(self.shape[:axis]+(length,)+self.shape[axis:], self.dtype)

  def _get( self, i, item ):
    return zeros( self.shape[:i] + self.shape[i+1:], dtype=self.dtype )

  def _takediag( self ):
    sh = builtins.max( self.shape[-2], self.shape[-1] )
    return zeros( self.shape[:-2] + (sh,), dtype=self.dtype )

  def _take( self, index, axis ):
    return zeros( self.shape[:axis] + index.shape + self.shape[axis+1:], dtype=self.dtype )

  def _inflate( self, dofmap, length, axis ):
    assert not isinstance( self.shape[axis], int )
    return zeros( self.shape[:axis] + (length,) + self.shape[axis+1:], dtype=self.dtype )

  def _power( self, n ):
    return self

  def _pointwise( self, evalf, deriv, dtype ):
    value = evalf( *numpy.zeros(self.shape[0]) )
    assert value.dtype == dtype
    if value == 0:
      return zeros( self.shape[1:], dtype=dtype )
    return _inflate_scalar(value, self.shape[1:])

  def _kronecker( self, axis, length, pos ):
    return zeros( self.shape[:axis]+(length,)+self.shape[axis:], dtype=self.dtype )

  def _mask( self, maskvec, axis ):
    return zeros( self.shape[:axis] + (maskvec.sum(),) + self.shape[axis+1:], dtype=self.dtype )

  def _unravel( self, axis, shape ):
    shape = self.shape[:axis] + shape + self.shape[axis+1:]
    return zeros( shape, dtype=self.dtype )

class Inflate( Array ):

  def __init__(self, func:asarray, dofmap:asarray, length:int, axis:int):
    assert not dofmap.isconstant
    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.axis = axis
    assert 0 <= axis < func.ndim
    assert func.shape[axis] == dofmap.shape[0]
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    super().__init__(args=[func,dofmap], shape=shape, dtype=func.dtype)

  @cache.property
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
    warnings.warn( 'using explicit inflation; this is usually a bug.' )
    shape = list(array.shape)
    shape[self.axis+1] = self.length
    inflated = numpy.zeros(shape, dtype=self.dtype)
    inflated[(slice(None),)*(self.axis+1)+(indices,)] = array
    return inflated

  @property
  def blocks(self):
    for ind, f in self.func.blocks:
      assert ind[self.axis] == Range(self.func.shape[self.axis])
      yield (ind[:self.axis] + (self.dofmap,) + ind[self.axis+1:]), f

  def _mask( self, maskvec, axis ):
    if axis != self.axis:
      return inflate( mask( self.func, maskvec, axis ), self.dofmap, self.length, self.axis )
    newlength = maskvec.sum()
    selection = take( maskvec, self.dofmap, axis=0 )
    renumber = numpy.empty( len(maskvec), dtype=int )
    renumber[:] = newlength # out of bounds
    renumber[maskvec] = numpy.arange( newlength )
    newdofmap = take( renumber, take( self.dofmap, selection, axis=0 ), axis=0 )
    newfunc = take( self.func, selection, axis=self.axis )
    return inflate( newfunc, newdofmap, newlength, self.axis )

  def _inflate( self, dofmap, length, axis ):
    assert axis != self.axis
    if axis > self.axis:
      return
    return inflate( inflate( self.func, dofmap, length, axis ), self.dofmap, self.length, self.axis )

  def _derivative(self, var, seen):
    return inflate(derivative(self.func, var, seen), self.dofmap, self.length, self.axis)

  def _transpose(self, axes):
    axis = axes.index(self.axis)
    return inflate(transpose(self.func, axes), self.dofmap, self.length, axis)

  def _insertaxis(self, axis, length):
    return inflate(insertaxis(self.func, axis, length), self.dofmap, self.length, self.axis+(axis<=self.axis))

  def _get( self, axis, item ):
    assert axis != self.axis
    return inflate( get(self.func,axis,item), self.dofmap, self.length, self.axis-(axis<self.axis) )

  def _dot( self, other, axes ):
    if isinstance( other, Inflate ) and other.axis == self.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    else:
      other = take( other, self.dofmap, self.axis )
    arr = dot( self.func, other, axes )
    if self.axis in axes:
      return arr
    return inflate( arr, self.dofmap, self.length, self.axis - builtins.sum( axis < self.axis for axis in axes ) )

  def _multiply( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap and self.length == other.length
      take_other = other.func
    else:
      take_other = take( other, self.dofmap, self.axis )
    return inflate( multiply(self.func,take_other), self.dofmap, self.length, self.axis )

  def _add( self, other ):
    if isinstance(other, Inflate) and self.axis == other.axis and self.dofmap == other.dofmap:
      return inflate(add(self.func, other.func), self.dofmap, self.length, self.axis)
    return BlockAdd([self, other])

  def _cross( self, other, axis ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    else:
      other = take( other, self.dofmap, self.axis )
    return inflate( cross(self.func,other,axis), self.dofmap, self.length, self.axis )

  def _power( self, n ):
    return inflate( power(self.func,n), self.dofmap, self.length, self.axis )

  def _takediag( self ):
    assert self.axis < self.ndim-2
    return inflate( takediag(self.func), self.dofmap, self.length, self.axis )

  def _take( self, index, axis ):
    if axis != self.axis:
      return inflate( take( self.func, index, axis ), self.dofmap, self.length, self.axis )
    if index == self.dofmap:
      return self.func

  def _diagonalize( self ):
    assert self.axis < self.ndim-1
    return inflate( diagonalize(self.func), self.dofmap, self.length, self.axis )

  def _sum( self, axis ):
    arr = sum( self.func, axis )
    if axis == self.axis:
      return arr
    return inflate( arr, self.dofmap, self.length, self.axis-(axis<self.axis) )

  def _kronecker( self, axis, length, pos ):
    return inflate( kronecker(self.func,axis,length,pos), self.dofmap, self.length, self.axis+(axis<=self.axis) )

class Diagonalize( Array ):

  def __init__(self, func:asarray):
    self.func = func
    super().__init__(args=[func], shape=func.shape + func.shape[-1:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if func.shape[-1] == 1:
      return insertaxis(func, func.ndim, 1).simplified
    retval = func._diagonalize()
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Diagonalize(func)

  def evalf( self, arr):
    assert arr.ndim == self.ndim
    return numeric.diagonalize(arr)

  def _derivative(self, var, seen):
    result = derivative(self.func, var, seen)
    # move axis `self.ndim-1` to the end
    result = transpose(result, [i for i in range(result.ndim) if i != self.func.ndim-1] + [self.func.ndim-1])
    # diagonalize last axis
    result = diagonalize(result)
    # move diagonalized axes left of the derivatives axes
    return transpose(result, tuple(range(self.func.ndim-1)) + (result.ndim-2,result.ndim-1) + tuple(range(self.func.ndim-1,result.ndim-2)))

  def _get( self, i, item ):
    if i < self.ndim-2:
      return diagonalize(get(self.func, i, item))
    if item.isconstant:
      pos, = item.eval()
      return kronecker(get(self.func, -1, item), axis=-1, pos=pos, length=self.func.shape[-1])

  def _inverse( self ):
    return diagonalize( reciprocal( self.func ) )

  def _determinant( self ):
    return product( self.func, -1 )

  def _multiply( self, other ):
    return diagonalize(multiply(self.func, takediag(other)))

  def _dot( self, other, axes ):
    faxes = [ axis for axis in axes if axis < self.ndim-2 ]
    if len(faxes) < len(axes): # one of or both diagonalized axes are summed
      if len(axes) - len(faxes) == 2:
        faxes.append( self.func.ndim-1 )
      return dot( self.func, takediag(other), faxes )
    return diagonalize( dot( self.func, takediag(other), axes ) )

  def _add( self, other ):
    if isinstance( other, Diagonalize ):
      return diagonalize( self.func + other.func )

  def _sum( self, axis ):
    if axis >= self.ndim-2:
      return self.func
    return diagonalize( sum( self.func, axis ) )

  def _transpose(self, axes):
    if axes[-2:] in [(self.ndim-2,self.ndim-1), (self.ndim-1,self.ndim-2)]:
      return diagonalize(transpose(self.func, axes[:-2]+(self.ndim-2,)))

  def _insertaxis(self, axis, length):
    if axis >= self.func.ndim-1:
      return diagonalize(insertaxis(self.func, self.func.ndim-1, length)).swapaxes(self.func.ndim-1, axis)
    return diagonalize(insertaxis(self.func, axis, length))

  def _takediag( self ):
    return self.func

  def _take( self, index, axis ):
    if axis < self.ndim-2:
      return diagonalize( take( self.func, index, axis ) )
    if numeric.isint(self.func.shape[-1]):
      diag = diagonalize(take(self.func, index, self.func.ndim-1))
      return inflate(diag, index, self.func.shape[-1], self.ndim-1 if axis == self.ndim-2 else self.ndim-2)

  def _mask( self, maskvec, axis ):
    if axis < self.ndim-2:
      return diagonalize( mask( self.func, maskvec, axis ) )
    indices, = numpy.where( maskvec )
    if not numpy.equal(numpy.diff(indices), 1).all():
      return
    # consecutive sub-block
    rev = slice( None, None, 1 if axis == self.ndim-1 else -1 )
    return concatenate([
      zeros( self.func.shape[:-1] + (indices[0],len(indices))[rev] ),
      diagonalize( mask( self.func, maskvec, self.func.ndim-1 ) ),
      zeros( self.func.shape[:-1] + (self.shape[-1]-(indices[-1]+1),len(indices))[rev] ),
    ], axis=2*self.ndim-axis-3 )

class Guard( Array ):
  'bar all simplifications'

  def __init__(self, fun:asarray):
    self.fun = fun
    super().__init__(args=[fun], shape=fun.shape, dtype=fun.dtype)

  @staticmethod
  def evalf( dat ):
    return dat

  def _derivative(self, var, seen):
    return Guard(derivative(self.fun, var, seen))

class TrigNormal( Array ):
  'cos, sin'

  def __init__(self, angle:asarray):
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return trigtangent(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf( self, angle ):
    return numpy.array([ numpy.cos(angle), numpy.sin(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return asarray( 1 if isinstance(other,TrigNormal) else 0 )

class TrigTangent( Array ):
  '-sin, cos'

  def __init__(self, angle:asarray):
    assert angle.ndim == 0
    self.angle = angle
    super().__init__(args=[angle], shape=(2,), dtype=float)

  def _derivative(self, var, seen):
    return -trignormal(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf( self, angle ):
    return numpy.array([ -numpy.sin(angle), numpy.cos(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return asarray( 1 if isinstance(other,TrigTangent) else 0 )

class Find( Array ):
  'indices of boolean index vector'

  def __init__(self, where:asarray):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    self.where = where
    super().__init__(args=[where], shape=[where.sum()], dtype=int)

  def evalf( self, where ):
    assert where.shape[0] == 1
    where, = where
    index, = where.nonzero()
    return index[_]

class Kronecker( Array ):

  def __init__(self, func:asarray, axis:int, length:int, pos:int):
    assert 0 <= axis <= func.ndim
    assert 0 <= pos < length
    self.func = func
    self.axis = axis
    self.length = length
    self.pos = pos
    super().__init__(args=[func], shape=func.shape[:axis]+(length,)+func.shape[axis:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if self.length == 1:
      assert self.pos == 0
      return InsertAxis(func, axis=self.axis, length=1).simplified
    retval = func._kronecker(self.axis, self.length, self.pos)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Kronecker(func, self.axis, self.length, self.pos)

  def evalf( self, func ):
    return numeric.kronecker( func, self.axis+1, self.length, self.pos )

  def _derivative(self, var, seen):
    return kronecker(derivative(self.func, var, seen), self.axis, self.length, self.pos)

  def _get( self, i, item ):
    if i != self.axis:
      return kronecker( get(self.func,i-(i>self.axis),item), self.axis-(i<self.axis), self.length, self.pos )
    if item.isconstant:
      item, = item.eval()
      return self.func if item == self.pos else zeros(self.func.shape, self.dtype)

  def _add( self, other ):
    if isinstance( other, Kronecker ) and other.axis == self.axis and self.length == other.length and self.pos == other.pos:
      return kronecker( self.func + other.func, self.axis, self.length, self.pos )

  def _multiply(self, other):
    return kronecker(multiply(self.func, get(other, self.axis, self.pos)), self.axis, self.length, self.pos)

  def _dot( self, other, axes ):
    newaxis = self.axis
    newaxes = []
    for ax in axes:
      if ax < self.axis:
        newaxis -= 1
        newaxes.append( ax )
      elif ax > self.axis:
        newaxes.append( ax-1 )
    dotfunc = dot(self.func, get(other, self.axis, self.pos), newaxes )
    return dotfunc if len(newaxes) < len(axes) else kronecker( dotfunc, newaxis, self.length, self.pos )

  def _sum( self, axis ):
    if axis == self.axis:
      return self.func
    return kronecker( sum( self.func, axis-(axis>self.axis) ), self.axis-(axis<self.axis), self.length, self.pos )

  def _transpose(self, axes):
    newaxis = axes.index(self.axis)
    newaxes = [ax-(ax>self.axis) for ax in axes if ax != self.axis]
    return kronecker(transpose(self.func, newaxes), newaxis, self.length, self.pos)

  def _takediag( self ):
    if self.axis < self.ndim-2:
      return kronecker( takediag(self.func), self.axis, self.length, self.pos )
    return kronecker( get( self.func, self.func.ndim-1, self.pos ), self.func.ndim-1, self.length, self.pos )

  def _take( self, index, axis ):
    if axis != self.axis:
      return kronecker( take( self.func, index, axis-(axis>self.axis) ), self.axis, self.length, self.pos )
    # TODO select axis in index

  def _power( self, n ):
    return kronecker(power(self.func, get(n, self.axis, self.pos)), self.axis, self.length, self.pos)

  def _pointwise( self, evalf, deriv, dtype ):
    if self.axis == 0:
      return
    value = evalf( *numpy.zeros(self.shape[0]) )
    assert value.dtype == dtype
    if value == 0:
      return kronecker( pointwise( self.func, evalf, deriv, dtype ), self.axis-1, self.length, self.pos )

  def _mask( self, maskvec, axis ):
    if axis != self.axis:
      return kronecker( mask( self.func, maskvec, axis-(axis>self.axis) ), self.axis, self.length, self.pos )
    newlength = maskvec.sum()
    if not maskvec[self.pos]:
      return zeros( self.shape[:axis] + (newlength,) + self.shape[axis+1:], dtype=self.dtype )
    newpos = maskvec[:self.pos].sum()
    return kronecker( self.func, self.axis, newlength, newpos )

  def _insertaxis(self, axis, length):
    return kronecker(insertaxis(self.func, axis-(axis>self.axis), length), self.axis+(self.axis>=axis), self.length, self.pos)

class DerivativeTargetBase( Array ):
  'base class for derivative targets'

  @property
  def isconstant( self ):
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
  >>> function.derivative(f, a).simplified == (3*a**2).simplified
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

  def __init__(self, name, shape:tuple, nderiv:int=0):
    self._name = name
    self._nderiv = nderiv
    super().__init__(args=[ARGUMENTS], shape=shape, dtype=float)

  def evalf(self, args):
    assert self._nderiv == 0
    try:
      return args[self._name][_]
    except KeyError:
      raise ValueError('argument {!r} missing'.format(self._name))

  def _derivative(self, var, seen):
    if isinstance(var, Argument) and var._name == self._name:
      assert var._nderiv == 0 and self.shape[:self.ndim-self._nderiv] == var.shape
      if self._nderiv:
        return zeros(self.shape+var.shape)
      result = numpy.array(1)
      for i, sh in enumerate(self.shape):
        s = [numpy.newaxis] * self.ndim
        s[i] = slice(None)
        result = result * eye(sh)[tuple(s*2)]
      return result
    elif isinstance(var, LocalCoords):
      return Argument(self._name, self.shape+var.shape, self._nderiv+1)
    else:
      return zeros(self.shape+var.shape)

  def __str__(self):
    return '{} {!r} <{}>'.format(self.__class__.__name__, self._name, ','.join(map(str, self.shape)))

class LocalCoords( DerivativeTargetBase ):
  'local coords derivative target'

  def __init__(self, ndims:int):
    super().__init__(args=[], shape=[ndims], dtype=float)

  def evalf( self ):
    raise Exception( 'LocalCoords should not be evaluated' )

class Ravel( Array ):

  def __init__(self, func:asarray, axis:int):
    assert 0 <= axis < func.ndim-1
    self.func = func
    self.axis = axis
    super().__init__(args=[func], shape=func.shape[:axis]+(func.shape[axis]*func.shape[axis+1],)+func.shape[axis+2:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if func.shape[self.axis] == 1:
      return get(func, self.axis, 0 ).simplified
    if func.shape[self.axis+1] == 1:
      return get(func, self.axis+1, 0 ).simplified
    retval = func._ravel(self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Ravel(func, self.axis)

  def evalf( self, f ):
    return f.reshape( f.shape[:self.axis+1] + (f.shape[self.axis+1]*f.shape[self.axis+2],) + f.shape[self.axis+3:] )

  def _multiply( self, other ):
    if isinstance( other, Ravel ) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return ravel( multiply( self.func, other.func ), self.axis )

  def _add( self, other ):
    if isinstance( other, Ravel ) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return ravel( add( self.func, other.func ), self.axis )

  def _get( self, i, item ):
    if i != self.axis:
      return ravel( get( self.func, i+(i>self.axis), item ), self.axis-(i<self.axis) )
    if numeric.isint( self.func.shape[self.axis+1] ):
      i, j = divmod( item, self.func.shape[self.axis+1] )
      return get( get( self.func, self.axis, i ), self.axis, j )

  def _dot( self, other, axes ):
    newaxes = [ ax+(ax>self.axis) for ax in axes if ax != self.axis ]
    if len(newaxes) < len(axes): # self.axis in axes
      newaxes.extend([self.axis,self.axis+1])
    return dot( self.func, unravel( other, self.axis, self.func.shape[self.axis:self.axis+2] ), newaxes )

  def _sum( self, axis ):
    if axis == self.axis:
      return sum( self.func, [axis,axis+1] )
    return ravel( sum( self.func, axis+(axis>self.axis) ), self.axis-(axis<self.axis) )

  def _derivative(self, var, seen):
    return ravel(derivative(self.func, var, seen), axis=self.axis)

  def _transpose(self, axes):
    ravelaxis = axes.index(self.axis)
    funcaxes = [ax+(ax>self.axis) for ax in axes]
    funcaxes = funcaxes[:ravelaxis+1] + [self.axis+1] + funcaxes[ravelaxis+1:]
    return ravel(transpose(self.func, funcaxes), ravelaxis)

  def _kronecker( self, axis, length, pos ):
    return ravel( kronecker( self.func, axis+(axis>self.axis), length, pos ), self.axis+(axis<=self.axis) )

  def _takediag( self ):
    if self.axis < self.ndim-2:
      return ravel( takediag( self.func ), self.axis )

  def _unravel( self, axis, shape ):
    if axis == self.axis and shape == self.func.shape[axis:axis+2]:
      return self.func

  @property
  def blocks(self):
    for ind, f in self.func.blocks:
      newind = ravel(ind[self.axis][:,_] * self.func.shape[self.axis+1] + ind[self.axis+1][_,:], axis=0)
      yield (ind[:self.axis] + (newind,) + ind[self.axis+2:]), ravel(f, axis=self.axis)

class Unravel( Array ):

  def __init__(self, func:asarray, axis:int, shape:tuple):
    assert 0 <= axis < func.ndim
    assert func.shape[axis] == numpy.product(shape)
    self.func = func
    self.axis = axis
    self.unravelshape = shape
    super().__init__(args=[func], shape=func.shape[:axis]+shape+func.shape[axis+1:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if len(self.unravelshape) == 1:
      return func
    retval = func._unravel(self.axis, self.unravelshape)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Unravel(func, self.axis, self.unravelshape)

  def _derivative(self, var, seen):
    return unravel(derivative(self.func, var, seen), axis=self.axis, shape=self.unravelshape)

  def evalf( self, f ):
    return f.reshape( f.shape[0], *self.shape )

class Mask( Array ):

  def __init__(self, func:asarray, mask:numeric.const, axis:int):
    assert len(mask) == func.shape[axis]
    self.func = func
    self.axis = axis
    self.mask = mask
    super().__init__(args=[func], shape=func.shape[:axis]+(mask.sum(),)+func.shape[axis+1:], dtype=func.dtype)

  @cache.property
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

  def evalf( self, func ):
    return func[(slice(None),)*(self.axis+1)+(self.mask,)]

  def _derivative(self, var, seen):
    return mask(derivative(self.func, var, seen), self.mask, self.axis)

class FindTransform(Array):

  def __init__(self, transforms:tuple, trans):
    self.transforms = transforms
    bits = []
    bit = 1
    while bit <= len(transforms):
      bits.append(bit)
      bit <<= 1
    self.bits = numpy.array(bits[::-1])
    super().__init__(args=[trans], shape=(), dtype=int)

  def asdict(self, values):
    assert len(self.transforms) == len(values)
    return dict(zip(self.transforms, values))

  def evalf(self, trans):
    n = len(self.transforms)
    index = 0
    for bit in self.bits:
      i = index|bit
      if i <= n and trans >= self.transforms[i-1]:
        index = i
    index -= 1
    if index < 0 or trans[:len(self.transforms[index])] != self.transforms[index]:
      index = len(self.transforms)
    return numpy.array(index)[_]

class Range(Array):

  def __init__(self, length:asarray, offset:asarray=Zeros((), int)):
    assert length.ndim == 0 and length.dtype == int
    assert offset.ndim == 0 and offset.dtype == int
    self.length = length
    self.offset = offset
    super().__init__(args=[length, offset], shape=[length], dtype=int)

  def evalf(self, length, offset):
    length, = length
    offset, = offset
    return numpy.arange(offset, offset+length)[_]

# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_ascending = lambda arg: numpy.greater(numpy.diff(arg), 0).all()
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )

def _jointdtype( *dtypes ):
  'determine joint dtype'

  type_order = bool, int, float
  kind_order = 'bif'
  itype = builtins.max( kind_order.index(dtype.kind) if isinstance(dtype,numpy.dtype)
           else type_order.index(dtype) for dtype in dtypes )
  return type_order[itype]

def _matchndim( *arrays ):
  'introduce singleton dimensions to match ndims'

  arrays = [ asarray(array) for array in arrays ]
  ndim = builtins.max( array.ndim for array in arrays )
  return tuple(array[(_,)*(ndim-array.ndim)] for array in arrays)

def _obj2str( obj ):
  'convert object to string'

  if numeric.isarray(obj):
    if obj.size < 6:
      return _obj2str(obj.tolist())
    return 'array<%s>' % 'x'.join( str(n) for n in obj.shape )
  if isinstance( obj, list ):
    if len(obj) < 6:
      return '[%s]' % ','.join( _obj2str(o) for o in obj )
    return '[#%d]' % len(obj)
  if isinstance( obj, (tuple,set) ):
    if len(obj) < 6:
      return '(%s)' % ','.join( _obj2str(o) for o in obj )
    return '(#%d)' % len(obj)
  if isinstance(obj, collections.abc.Mapping):
    return '{#%d}' % len(obj)
  if isinstance( obj, slice ):
    I = ''
    if obj.start is not None:
      I += str(obj.start)
    if obj.step is not None:
      I += ':' + str(obj.step)
    I += ':'
    if obj.stop is not None:
      I += str(obj.stop)
    return I
  if obj is Ellipsis:
    return '...'
  return str(obj)

def _findcommon( a, b ):
  'find common item in 2x2 data'

  a1, a2 = a
  b1, b2 = b
  if a1 == b1:
    return a1, (a2,b2)
  if a1 == b2:
    return a1, (a2,b1)
  if a2 == b1:
    return a2, (a1,b2)
  if a2 == b2:
    return a2, (a1,b1)

def _invtrans(trans):
  trans = numpy.asarray(trans)
  assert trans.dtype == int
  invtrans = numpy.empty(len(trans), dtype=int)
  invtrans[trans] = numpy.arange(len(trans))
  return tuple(invtrans)

def _norm_and_sort( ndim, args ):
  'norm axes, sort, and assert unique'

  normargs = tuple( sorted( numeric.normdim( ndim, arg ) for arg in args ) )
  assert _ascending( normargs ) # strict
  return normargs

def _concatblocks(items):
  gathered = util.gather(items)
  order = [ind for ind12, ind_f in gathered for ind, f in ind_f]
  blocks = []
  for (ind1, ind2), ind_f in gathered:
    if len(ind_f) == 1:
      ind, f = ind_f[0]
    else:
      inds, fs = zip(*sorted(ind_f, key=lambda item: order.index(item[0])))
      ind = Concatenate(inds, axis=0)
      f = Concatenate(fs, axis=len(ind1))
    blocks.append(((ind1+(ind,)+ind2), f))
  return tuple(blocks)

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

# FUNCTIONS

isarray = lambda arg: isinstance( arg, Array )
iszero = lambda arg: isinstance( arg, Zeros )
isevaluable = lambda arg: isinstance( arg, Evaluable )
zeros = lambda shape, dtype=float: Zeros( shape, dtype )
zeros_like = lambda arr: zeros(arr.shape, arr.dtype)
ones = lambda shape, dtype=float: _inflate_scalar(numpy.ones((), dtype=dtype), shape)
ones_like = lambda arr: ones(arr.shape, arr.dtype)
grad = lambda arg, coords, ndims=0: asarray( arg ).grad( coords, ndims )
symgrad = lambda arg, coords, ndims=0: asarray( arg ).symgrad( coords, ndims )
div = lambda arg, coords, ndims=0: asarray( arg ).div( coords, ndims )
negative = lambda arg: multiply( arg, -1 )
nsymgrad = lambda arg, coords: ( symgrad(arg,coords) * coords.normal() ).sum(-1)
ngrad = lambda arg, coords: ( grad(arg,coords) * coords.normal() ).sum(-1)
sin = lambda arg: pointwise( [arg], numpy.sin, cos )
cos = lambda arg: pointwise( [arg], numpy.cos, lambda x: -sin(x) )
rotmat = lambda arg: asarray([ trignormal(arg), trigtangent(arg) ])
tan = lambda arg: pointwise( [arg], numpy.tan, lambda x: cos(x)**-2 )
arcsin = lambda arg: pointwise( [arg], numpy.arcsin, lambda x: reciprocal(sqrt(1-x**2)) )
arccos = lambda arg: pointwise( [arg], numpy.arccos, lambda x: -reciprocal(sqrt(1-x**2)) )
exp = lambda arg: pointwise( [arg], numpy.exp, exp )
ln = lambda arg: pointwise( [arg], numpy.log, reciprocal )
mod = lambda arg1, arg2: pointwise( [arg1,arg2], numpy.mod, dtype=_jointdtype(asarray(arg1).dtype,asarray(arg2).dtype) )
log2 = lambda arg: ln(arg) / ln(2)
log10 = lambda arg: ln(arg) / ln(10)
sqrt = lambda arg: power( arg, .5 )
reciprocal = lambda arg: power( arg, -1 )
argmin = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmin(numeric.stack(x),axis=0), zeros_like, dtype=int )
argmax = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmax(numeric.stack(x),axis=0), zeros_like, dtype=int )
arctan2 = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.arctan2, lambda x: stack([x[1],-x[0]]) / sum(power(x,2),0) )
greater = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.greater, zeros_like, dtype=bool )
equal = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.equal, zeros_like, dtype=bool )
less = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.less, zeros_like, dtype=bool )
min = lambda arg1, *args: choose( argmin( arg1 if not args else (arg1,)+args, axis=0 ), arg1 if not args else (arg1,)+args )
max = lambda arg1, *args: choose( argmax( arg1 if not args else (arg1,)+args, axis=0 ), arg1 if not args else (arg1,)+args )
abs = lambda arg: arg * sign(arg)
sinh = lambda arg: .5 * ( exp(arg) - exp(-arg) )
cosh = lambda arg: .5 * ( exp(arg) + exp(-arg) )
tanh = lambda arg: 1 - 2. / ( exp(2*arg) + 1 )
arctanh = lambda arg: .5 * ( ln(1+arg) - ln(1-arg) )
piecewise = lambda level, intervals, *funcs: choose( sum( greater( expand_dims(level,-1), intervals ), -1 ), funcs )
trace = lambda arg, n1=-2, n2=-1: sum( takediag( arg, n1, n2 ), -1 )
normalized = lambda arg, axis=-1: divide(arg, expand_dims(norm2(arg, axis=axis), axis))
norm2 = lambda arg, axis=-1: sqrt( sum( multiply( arg, arg ), axis ) )
heaviside = lambda arg: choose( greater( arg, 0 ), [0.,1.] )
divide = lambda arg1, arg2: multiply( arg1, reciprocal(arg2) )
subtract = lambda arg1, arg2: add( arg1, negative(arg2) )
mean = lambda arg: .5 * ( arg + opposite(arg) )
jump = lambda arg: opposite(arg) - arg
add_T = lambda arg, axes=(-2,-1): swapaxes( arg, axes ) + arg
blocks = lambda arg: asarray(arg).simplified.blocks
rootcoords = lambda ndims: RootCoords( ndims )
sampled = lambda data, ndims: Sampled( data )
opposite = cache.replace(initcache={TRANS: OPPTRANS, OPPTRANS: TRANS})
bifurcate1 = cache.replace(initcache={TRANS: SelectChain(TRANS, True), OPPTRANS: SelectChain(OPPTRANS, True)})
bifurcate2 = cache.replace(initcache={TRANS: SelectChain(TRANS, False), OPPTRANS: SelectChain(OPPTRANS, False)})
bifurcate = lambda arg1, arg2: ( bifurcate1(arg1), bifurcate2(arg2) )
curvature = lambda geom, ndims=-1: geom.normal().div(geom, ndims=ndims)
laplace = lambda arg, geom, ndims=0: arg.grad(geom, ndims).div(geom, ndims)
symgrad = lambda arg, geom, ndims=0: multiply(.5, add_T(arg.grad(geom, ndims)))
div = lambda arg, geom, ndims=0: trace(arg.grad(geom, ndims), -1, -2)
tangent = lambda geom, vec: subtract(vec, multiply(dot(vec, normal(geom), -1)[...,_], normal(geom)))
ngrad = lambda arg, geom, ndims=0: dotnorm(grad(arg, geom, ndims), geom)
nsymgrad = lambda arg, geom, ndims=0: dotnorm(symgrad(arg, geom, ndims), geom)
expand_dims = lambda arg, n: InsertAxis(arg, n, 1)

def trignormal( angle ):
  angle = asarray( angle )
  assert angle.ndim == 0
  if iszero( angle ):
    return kronecker( 1, axis=0, length=2, pos=0 )
  return TrigNormal( angle )

def trigtangent( angle ):
  angle = asarray( angle )
  assert angle.ndim == 0
  if iszero( angle ):
    return kronecker( 1, axis=0, length=2, pos=1 )
  return TrigTangent( angle )

eye = lambda n, dtype=float: diagonalize(ones([n], dtype=dtype))

def insertaxis(arg, n, length):
  arg = asarray(arg)
  n = numeric.normdim(arg.ndim+1, n)
  return InsertAxis(arg, n, length)

def stack(args, axis=0):
  assert len(args) > 0
  return insertaxis(args[0], axis, length=1) if len(args) == 1 \
    else builtins.sum(kronecker(arg, axis=axis, length=len(args), pos=iarg) for iarg, arg in enumerate(_numpy_align(*args)))

def chain( funcs ):
  'chain'

  funcs = [ asarray(func) for func in funcs ]
  shapes = [ func.shape[0] for func in funcs ]
  return [ concatenate( [ func if i==j else zeros( (sh,) + func.shape[1:] )
             for j, sh in enumerate(shapes) ], axis=0 )
               for i, func in enumerate(funcs) ]

def vectorize( args ):
  'vectorize'

  return stack( chain(args), axis=-1 )

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

def align(arg, axes, ndim):
  keep = numpy.zeros(ndim, dtype=bool)
  keep[list(axes)] = True
  renumber = keep.cumsum()-1
  transaxes = _invtrans(renumber[numpy.asarray(axes)])
  retval = transpose(arg, transaxes)
  for axis in numpy.where(~keep)[0]:
    retval = expand_dims(retval, axis)
  for i, j in enumerate(axes):
    assert arg.shape[i] == retval.shape[j]
  return retval

def bringforward( arg, axis ):
  'bring axis forward'

  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim,axis)
  if axis == 0:
    return arg
  return transpose( args, [axis] + range(axis) + range(axis+1,args.ndim) )

def jacobian( geom, ndims ):
  assert geom.ndim == 1
  J = localgradient( geom, ndims )
  cndims, = geom.shape
  assert J.shape == (cndims,ndims), 'wrong jacobian shape: got %s, expected %s' % ( J.shape, (cndims, ndims) )
  assert cndims >= ndims, 'geometry dimension < topology dimension'
  detJ = abs( determinant( J ) ) if cndims == ndims \
    else 1. if ndims == 0 \
    else abs( determinant( ( J[:,:,_] * J[:,_,:] ).sum(0) ) )**.5
  return detJ

def matmat( arg0, *args ):
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = asarray( arg0 )
  for arg in args:
    arg = asarray( arg )
    assert retval.shape[-1] == arg.shape[0], 'incompatible shapes'
    retval = dot( retval[(...,)+(_,)*(arg.ndim-1)], arg[(_,)*(retval.ndim-1)], retval.ndim-1 )
  return retval

def determinant(arg, axes=(-2,-1)):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  arg = transpose(arg, trans)
  return Determinant(arg)

def inverse(arg, axes=(-2,-1)):
  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  arg = transpose(arg, trans)
  return transpose(Inverse(arg), _invtrans(trans))

def takediag(arg, ax1=-2, ax2=-1):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort( arg.ndim, (ax1,ax2) )
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  arg = transpose(arg, trans)
  if arg.shape[-1] == 1:
    return get(arg, -1, 0)
  if arg.shape[-2] == 1:
    return get(arg, -2, 0)
  return TakeDiag(arg)

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

def dotnorm( arg, coords ):
  'normal component'

  return sum( arg * coords.normal(), -1 )

normal = lambda geom: geom.normal()

def kronecker(arg, axis, length, pos):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim+1, axis)
  assert 0 <= pos < length
  return Kronecker(arg, axis, length, pos)

def diagonalize(arg):
  arg = asarray(arg)
  return Diagonalize(arg)

def concatenate(args, axis=0):
  args = _matchndim(*args)
  axis = numeric.normdim(args[0].ndim, axis)
  return Concatenate(args, axis)

def choose(level, choices):
  level, *choices = _numpy_align(level, *choices)
  return Choose(level, tuple(choices))

def _condlist_to_level( *condlist ):
  level = 0
  mask = 1
  for i, condition in enumerate(condlist):
    condition = numpy.asarray(condition, bool)
    level += (i+1)*mask*condition
    mask *= 1-condition
  return level

def select( condlist, choicelist, default=0 ):
  'select'

  if not any(map(isarray, itertools.chain( condlist, choicelist, [default] ))):
    return asarray( numpy.select( condlist, choicelist, default=default ) )
  level = pointwise( condlist, _condlist_to_level, dtype=int )
  return choose( level, (default,)+tuple(choicelist) )

def cross(arg1, arg2, axis):
  arg1, arg2 = _numpy_align(arg1, arg2)
  axis = numeric.normdim(arg1.ndim, axis)
  assert arg1.shape[axis] == 3
  return Cross(arg1, arg2, axis)

def outer( arg1, arg2=None, axis=0 ):
  'outer product'

  if arg2 is not None and arg1.ndim != arg2.ndim:
    warnings.warn( 'varying ndims in function.outer; this will be forbidden in future', DeprecationWarning )
  arg1, arg2 = _matchndim( arg1, arg2 if arg2 is not None else arg1 )
  axis = numeric.normdim( arg1.ndim, axis )
  return expand_dims(arg1,axis+1) * expand_dims(arg2,axis)

def pointwise(args, evalf, deriv=None, dtype=float):
  args = asarray(args)
  return Pointwise(args, evalf, deriv, dtype)

def sign(arg):
  arg = asarray(arg)
  return Sign(arg)

def eig(arg, axes=(-2,-1), symmetric=False):
  arg = asarray(arg)
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict
  trans = [i for i in range(arg.ndim) if i not in (ax1, ax2)] + [ax1, ax2]
  transposed = transpose(arg, trans)
  eigval, eigvec = Eig(transposed, symmetric)
  return Tuple([transpose(diagonalize(eigval), _invtrans(trans)), transpose(eigvec, _invtrans(trans))])

def function(fmap, nmap, ndofs):
  transforms = sorted(fmap)
  depth, = set(len(transform) for transform in transforms)
  fromdims, = set(transform.fromdims for transform in transforms)
  promote = Promote(fromdims, trans=TRANS)
  index = FindTransform(transforms, promote)
  dofmap = DofMap([nmap[trans] for trans in transforms], index=index)
  func = Function(stds=[fmap[trans] for trans in transforms], depth=depth, trans=promote, index=index)
  return Inflate(func, dofmap, ndofs, axis=0)

def elemwise( fmap, shape, default=None ):
  return Elemwise( fmap=fmap, shape=shape, default=default )

def take(arg, index, axis):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)

  if isinstance(index, slice):
    if index == slice(None):
      return arg
    assert index.step == None or index.step == 1
    if numeric.isint(arg.shape[axis]):
      indexmask = numpy.zeros(arg.shape[axis], dtype=bool)
      indexmask[index] = True
      return mask(arg, indexmask, axis=axis)
    assert index.start == None or index.start >= 0
    assert index.stop != None and index.stop >= 0
    index = numpy.arange(index.start or 0, index.stop)

  if not isevaluable(index):
    index = numpy.array(index)
    assert index.ndim == 1
    if index.dtype == bool:
      return mask(arg, index, axis)
    assert index.dtype == int
    index[index < 0] += arg.shape[axis]
    assert numpy.logical_and(numpy.greater_equal(index, 0), numpy.less(index, arg.shape[axis])).all(), 'indices out of bounds'

  index = asarray(index)
  assert index.ndim == 1
  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    index = find(index)
  else:
    assert index.dtype == int

  return Take(arg, index, axis)

def find( arg ):
  'find'

  arg = asarray( arg )
  assert arg.ndim == 1 and arg.dtype == bool

  if arg.isconstant:
    arg, = arg.eval()
    index, = arg.nonzero()
    return asarray( index )

  return Find( arg )

def inflate(arg, dofmap, length, axis):
  arg = asarray(arg)
  dofmap = asarray(dofmap)
  axis = numeric.normdim(arg.ndim, axis)
  shape = arg.shape[:axis] + (length,) + arg.shape[axis+1:]
  if dofmap.isconstant:
    n = arg.shape[axis]
    assert numeric.isint(n), 'constant inflation only allowed over fixed-length axis'
    index, = dofmap.eval()
    assert len(index) == n
    assert numpy.greater_equal(index, 0).all() and numpy.less(index, length).all()
    assert numpy.equal(numpy.diff(index), 1).all(), 'constant inflation must be contiguous'
    if n == length:
      retval = arg
    else:
      parts = []
      if index[0] > 0:
        parts.append( zeros( arg.shape[:axis] + (index[0],) + arg.shape[axis+1:], dtype=arg.dtype ) )
      parts.append( arg )
      if index[0] + n < length:
        parts.append( zeros( arg.shape[:axis] + (length-index[0]-n,) + arg.shape[axis+1:], dtype=arg.dtype ) )
      retval = concatenate( parts, axis=axis )
    assert retval.shape == tuple(shape)
    return retval
  return Inflate(arg, dofmap, length, axis)

def mask(arg, mask, axis=0):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  assert numeric.isarray(mask) and mask.ndim == 1 and mask.dtype == bool
  assert arg.shape[axis] == len(mask)
  return Mask(arg, mask, axis)

def J( geometry, ndims=None ):
  if ndims is None:
    ndims = len(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian( geometry, ndims )

def unravel(func, axis, shape):
  func = asarray(func)
  axis = numeric.normdim(func.ndim, axis)
  shape = tuple(shape)
  assert func.shape[axis] == numpy.product(shape)
  return Unravel(func, axis, tuple(shape))

def ravel(func, axis):
  func = asarray(func)
  axis = numeric.normdim( func.ndim-1, axis )
  return Ravel(func, axis)

@cache.replace
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

@cache.replace
def zero_argument_derivatives(arg):
  if isinstance(arg, Argument) and arg._nderiv > 0:
    return zeros_like(arg)

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
    array, arg, value = args
    assert isinstance(arg, Argument) and arg._nderiv == 0
    return replace_arguments(array, {arg._name: value})
  elif op == 'call':
    func, arg = args
    return functions[func](arg)
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
    return repeat(asarray(array)[..., None], length, -1)
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

  When evaluating an expression through this namespace the following functions
  are available: ``opposite``, ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan2``, ``arctanh``, ``exp``, ``abs``,
  ``ln``, ``log``, ``log2``, ``log10``, ``sqrt`` and ``sign``.

  Args
  ----
  default_geometry_name : :class:`str`
      The name of the default geometry.  This argument is passed to
      :func:`nutils.expression.parse`.  Default: ``'x'``.

  Attributes
  ----------
  arg_shapes : :class:`types.MappingProxyType`
      A readonly map of argument names and shapes.
  default_geometry_name : :class:`str`
      The name of the default geometry.  See argument with the same name.
  '''

  __slots__ = '_attributes', '_arg_shapes', 'arg_shapes', 'default_geometry_name'

  _re_assign = re.compile('^([a-zA-Zα-ωΑ-Ω][a-zA-Zα-ωΑ-Ω0-9]*)(_[a-z]+)?$')

  _functions = dict(
    opposite=opposite, sin=sin, cos=cos, tan=tan, sinh=sinh, cosh=cosh,
    tanh=tanh, arcsin=arcsin, arccos=arccos, arctan2=arctan2, arctanh=arctanh,
    exp=exp, abs=abs, ln=ln, log=ln, log2=log2, log10=log10, sqrt=sqrt,
    sign=sign,
  )
  _functions_nargs = {k: len(inspect.signature(v).parameters) for k, v in _functions.items()}

  def __init__(self, *, default_geometry_name='x'):
    if not isinstance(default_geometry_name, str):
      raise ValueError('default_geometry_name: Expected a str, got {!r}.'.format(default_geometry_name))
    if '_' in default_geometry_name or not self._re_assign.match(default_geometry_name):
      raise ValueError('default_geometry_name: Invalid variable name: {!r}.'.format(default_geometry_name))
    super().__setattr__('_attributes', {})
    super().__setattr__('_arg_shapes', {})
    super().__setattr__('arg_shapes', types.MappingProxyType(self._arg_shapes))
    super().__setattr__('default_geometry_name', default_geometry_name)
    super().__init__()

  @property
  def default_geometry(self):
    ''':class:`nutils.function.Array`: The default geometry, shorthand for ``getattr(ns, ns.default_geometry_name)``.'''
    return getattr(self, self.default_geometry_name)

  def __or__(self, subs):
    '''Return a copy with arguments replaced by ``subs``.

    Return a copy of this namespace with :class:`Argument` objects replaced
    according to ``subs``.

    Args
    ----
    subs : :class:`dict` of :class:`str` and :class:`nutils.function.Array` objects
        Replacements of the :class:`Argument` objects, identified by their names.

    Returns
    -------
    ns : :class:`Namespace`
        The copy of this namespace with replaced :class:`Argument` objects.
    '''

    if not isinstance(subs, collections.abc.Mapping):
      return NotImplemented
    ns = Namespace(default_geometry_name=self.default_geometry_name)
    for k, v in self._attributes.items():
      setattr(ns, k, replace_arguments(v, subs))
    return ns

  def copy_(self, *, default_geometry_name=None):
    '''Return a copy of this namespace.'''

    if default_geometry_name is None:
      default_geometry_name = self.default_geometry_name
    ns = Namespace(default_geometry_name=default_geometry_name)
    for k, v in self._attributes.items():
      setattr(ns, k, v)
    return ns

  def __getattr__(self, name):
    '''Get attribute ``name``.'''

    if name.startswith('eval_'):
      return lambda expr: _eval_ast(expression.parse(expr, variables=self._attributes, functions=self._functions_nargs, indices=name[5:], arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name)[0], self._functions)
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
        ast, arg_shapes = expression.parse(value, variables=self._attributes, functions=self._functions_nargs, indices=indices, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name)
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
    '''Evaluate zero or one dimensional ``expr``.'''

    if not isinstance(expr, str):
      return NotImplemented
    try:
      ast = expression.parse(expr, variables=self._attributes, functions=self._functions_nargs, indices=None, arg_shapes=self._arg_shapes, default_geometry_name=self.default_geometry_name)[0]
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
    raise Exception( 'cannot invert %sx%s jacobian' % J.shape )
  return dot(localgradient(self, ndims)[...,_], Jinv, -2)

def dotnorm(arg, geom, axis=-1):
  axis = numeric.normdim(arg.ndim, axis)
  assert geom.ndim == 1 and geom.shape[0] == arg.shape[axis]
  return dot(arg, normal(geom)[(slice(None),)+(_,)*(arg.ndim-axis-1)], axis)

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
