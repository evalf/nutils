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

  def __init__( self, args ):
    'constructor'

    assert all( isevaluable(arg) or arg in TOKENS for arg in args )
    self.__args = tuple(args)

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
      if isinstance( ischeme, dict ):
        ischeme = ischeme[elem]
      if isinstance( ischeme, str ):
        points, weights = fcache[elem.reference.getischeme]( ischeme )
      elif isinstance( ischeme, tuple ):
        points, weights = ischeme
        assert points.shape[-1] == elem.ndims
        assert points.shape[:-1] == weights.shape, 'non matching shapes: points.shape=%s, weights.shape=%s' % ( points.shape, weights.shape )
      elif isinstance( ischeme, numpy.ndarray ):
        points = ischeme.astype( float )
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
      assert all(isinstance(value, numpy.ndarray) and value.dtype.kind in 'bif' for value in arguments.values())

    ops, inds = self.serialized
    assert TOKENS == ( CACHE, TRANS, OPPTRANS, POINTS, ARGUMENTS )
    values = [ fcache, trans, opptrans, points, arguments or {} ]
    for op, indices in zip( list(ops)+[self], inds ):
      args = [ values[i] for i in indices ]
      try:
        retval = op.evalf( *args )
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        excargs = etype, evalue, self, values
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

  def _edit( self, op ):
    raise NotImplementedError( '{} does not define an _edit method'.format( type(self).__name__ ) )

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
  'combine'

  def __init__( self, items ):
    'constructor'

    self.items = tuple( items )
    args = []
    indices = []
    for i, item in enumerate(self.items):
      if isevaluable( item ):
        args.append( item )
        indices.append( i )
    self.indices = tuple( indices )
    Evaluable.__init__( self, args )

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
  def __init__( self, trans, first ):
    assert isinstance( first, bool )
    self.trans = trans
    self.first = first
    Evaluable.__init__( self, args=[trans] )
  def evalf( self, trans ):
    assert isinstance( trans, transform.TransformChain )
    bf = trans[0]
    assert isinstance( bf, transform.Bifurcate )
    ftrans = bf.trans1 if self.first else bf.trans2
    return transform.TransformChain( ftrans + trans[1:] )
  def _edit( self, op ):
    return SelectChain( op(self.trans), self.first )

# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

def add(a, b):
  a, b = _matchndim(a, b)
  return Add(orderedset([a, b])).simplified

def multiply(a, b):
  a, b = _matchndim(a, b)
  return Multiply(orderedset([a, b])).simplified

def sum(arg, axis=None):
  arg = asarray(arg)
  if axis is None:
    axis = numpy.arange(arg.ndim)
  elif not util.isiterable(axis):
    axis = numeric.normdim(arg.ndim, axis),
  else:
    axis = _norm_and_sort(arg.ndim, axis)
    assert numpy.all(numpy.diff(axis) > 0), 'duplicate axes in sum'
  summed = arg
  for ax in reversed(axis):
    summed = Sum(summed, ax).simplified
  return summed

def power(arg, n):
  arg, n = _matchndim(arg, n)
  return Power(arg, n).simplified

def dot(a, b, axes=None):
  if axes is None:
    a = asarray(a)
    b = asarray(b)
    assert b.ndim == 1 and b.shape[0] == a.shape[0]
    while b.ndim < a.ndim:
      b = insert(b, b.ndim)
    axes = 0,
  else:
    a, b = _matchndim(a, b)
  if not util.isiterable(axes):
    axes = axes,
  axes = _norm_and_sort(a.ndim, axes)
  return Dot(orderedset([a, b]), axes).simplified

class Array( Evaluable ):
  'array function'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__(self, args, shape, dtype):
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    assert dtype is float or dtype is int or dtype is bool, 'invalid dtype {!r}'.format(dtype)
    self.dtype = dtype
    super().__init__(args=args)

  def __getitem__(self, item):
    if not isinstance(item, tuple):
      item = item,
    c = item.count(...)
    assert c <= 1, 'at most one ellipsis allowed'
    n = item.index(...) if c else len(item)
    array = self
    axis = 0
    for it in item[:n] + (slice(None),)*(self.ndim+item.count(_)+c-len(item)) + item[n+1:]:
      if numeric.isint(it):
        array = get(array, axis, item=it)
      elif it is _:
        array = insert(array, axis)
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

  size = property(lambda self: numpy.prod(self.shape, dtype=int))
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
  vector = lambda self, ndims: vectorize([self] * ndims)
  dot = dot
  normalized = lambda self, axis=-1: normalized(self, axis)
  normal = lambda self, exterior=False: normal(self, exterior)
  curvature = lambda self, ndims=-1: curvature(self, ndims)
  swapaxes = lambda self, axis1, axis2: swapaxes(self, axis1, axis2)
  transpose = lambda self, trans=None: transpose(self, trans)
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
    return (tuple(asarray(numpy.arange(n)) if numeric.isint(n) else None for n in self.shape), self),

  def _asciitree_str(self):
    return '{}<{}>'.format(type(self).__name__, ','.join(map(str, self.shape)))

  # simplifications
  _multiply = lambda self, other: None
  _align = lambda self, axes, ndim: None
  _dot = lambda self, other, axes: None
  _get = lambda self, i, item: None
  _power = lambda self, n: None
  _add = lambda self, other: None
  _concatenate = lambda self, other, axis: None
  _sum = lambda self, axis: None
  _take = lambda self, index, axis: None
  _repeat = lambda self, length, axis: None
  _determinant = lambda self: None
  _inverse = lambda self: None
  _takediag = lambda self: None
  _kronecker = lambda self, axis, length, pos: None
  _diagonalize = lambda self: None
  _product = lambda self, axis: None
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

  def __init__( self, lgrad ):
    assert lgrad.ndim == 2 and lgrad.shape[0] == lgrad.shape[1]
    self.lgrad = lgrad
    Array.__init__( self, args=[lgrad], shape=(len(lgrad),), dtype=float )

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

  def _edit( self, op ):
    return Normal( op(self.lgrad) )

class ArrayFunc( Array ):
  'deprecated ArrayFunc alias'

  def __init__( self, args, shape ):
    warnings.warn( 'function.ArrayFunc is deprecated; use function.Array instead', DeprecationWarning )
    Array.__init__( self, args=args, shape=shape, dtype=float )

class Constant( Array ):
  'constant'

  def __init__( self, value ):
    assert isinstance( value, numpy.ndarray ) and value.dtype != object
    self.value = value.copy()
    Array.__init__( self, args=[], shape=value.shape, dtype=_jointdtype(value.dtype) )

  def evalf( self ):
    return self.value[_]

  @cache.property
  def _isunit( self ):
    return numpy.all( self.value == 1 )

  def _derivative(self, var, seen):
    return zeros(self.shape + var.shape)

  def _align( self, axes, ndim ):
    return asarray( numeric.align( self.value, axes, ndim ) )

  def _sum( self, axis ):
    return asarray( numpy.sum( self.value, axis ) )

  def _get( self, i, item ):
    return asarray( numeric.get( self.value, i, item ) )

  def _add( self, other ):
    if isinstance( other, Constant ):
      return asarray( numpy.add( self.value, other.value ) )

  def _inverse( self ):
    return asarray( numpy.linalg.inv( self.value ) )

  def _product( self, axis ):
    return asarray( self.value.prod(axis) )

  def _multiply( self, other ):
    if self._isunit:
      shape = _jointshape( self.shape, other.shape )
      return expand( other, shape )
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

  def _edit( self, op ):
    return self

  def _dot( self, other, axes ):
    if self._isunit:
      shape = _jointshape( self.shape, other.shape )
      return sum( expand( other, shape ), axes )
    if isinstance( other, Constant ):
      return asarray( numeric.contract( self.value, other.value, axes ) )

  def _concatenate( self, other, axis ):
    if isinstance( other, Constant ):
      shape1 = list(self.shape)
      shape2 = list(other.shape)
      shape1[axis] = shape2[axis] = shape1[axis] + shape2[axis]
      shape = _jointshape( shape1, shape2 )
      retval = numpy.empty( shape, dtype=_jointdtype(self.dtype,other.dtype) )
      retval[(slice(None),)*axis+(slice(None,self.shape[axis]),)] = self.value
      retval[(slice(None),)*axis+(slice(self.shape[axis],None),)] = other.value
      return asarray( retval )

  def _cross( self, other, axis ):
    if isinstance( other, Constant ):
      return asarray( numeric.cross( self.value, other.value, axis ) )

  def _pointwise( self, evalf, deriv, dtype ):
    retval = evalf( *self.value )
    assert retval.dtype == dtype
    return asarray( retval )

  def _eig( self, symmetric ):
    eigval, eigvec = ( numpy.linalg.eigh if symmetric else numpy.linalg.eig )( self.value )
    return asarray( eigval ), asarray( eigvec )

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

class DofMap( Array ):
  'dof axis'

  def __init__( self, dofmap, length, trans=TRANS ):
    'new'

    self.trans = trans
    self.dofmap = dofmap
    Array.__init__( self, args=[trans], shape=(length,), dtype=int )

  def evalf( self, trans ):
    'evaluate'

    try:
      dofs, tail = trans.lookup_item( self.dofmap )
    except KeyError:
      dofs = numpy.empty( [0], dtype=int )
    return dofs[_]

  def _edit( self, op ):
    return DofMap( self.dofmap, self.shape[0], op(self.trans) )

class ElementSize( Array):
  'dimension of hypercube with same volume as element'

  def __init__( self, geometry, ndims=None ):
    assert geometry.ndim == 1
    self.ndims = len(geometry) if ndims is None else len(geometry)+ndims if ndims < 0 else ndims
    iwscale = jacobian( geometry, self.ndims )
    Array.__init__( self, args=[iwscale], shape=(), dtype=float )

  def evalf( self, iwscale ):
    volume = iwscale.sum()
    return numeric.power( volume, 1/self.ndims )[_]

class Align(Array):

  def __init__(self, func, axes, ndim):
    assert func.ndim == len(axes)
    self.func = func
    assert all(0 <= ax < ndim for ax in axes)
    self.axes = tuple(axes)
    shape = [1] * ndim
    for ax, sh in zip( self.axes, func.shape ):
      shape[ax] = sh
    self.negaxes = [ ax-ndim for ax in self.axes ]
    super().__init__(args=[func], shape=shape, dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if self.axes == tuple(range(self.ndim)):
      return func
    retval = func._align(self.axes, self.ndim)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Align(func, self.axes, self.ndim)

  def evalf(self, arr):
    return numeric.align(arr, [0]+[ax+1 for ax in self.axes], self.ndim+1)

  def _align( self, axes, ndim ):
    newaxes = [ axes[i] for i in self.axes ]
    return align( self.func, newaxes, ndim )

  def _takediag( self ):
    if self.ndim-1 not in self.axes:
      return align( self.func, self.axes, self.ndim-1 )
    if self.ndim-2 not in self.axes:
      axes = [ ax if ax != self.ndim-1 else self.ndim-2 for ax in self.axes ]
      return align( self.func, axes, self.ndim-1 )
    if self.axes[-2:] in [ (self.ndim-2,self.ndim-1), (self.ndim-1,self.ndim-2) ]:
      axes = self.axes[:-2] + (self.ndim-2,)
      return align( takediag( self.func ), axes, self.ndim-1 )

  def _get( self, i, item ):
    axes = [ ax - (ax>i) for ax in self.axes if ax != i ]
    if len(axes) == len(self.axes):
      return align( self.func, axes, self.ndim-1 )
    n = self.axes.index( i )
    return align( get( self.func, n, item ), axes, self.ndim-1 )

  def _sum( self, axis ):
    if axis in self.axes:
      idx = self.axes.index( axis )
      func = sum( self.func, idx )
    else:
      func = self.func
    trans = [ ax - (ax>axis) for ax in self.axes if ax != axis ]
    return align( func, trans, self.ndim-1 )

  def _derivative(self, var, seen):
    return align(derivative(self.func, var, seen), self.axes+tuple(range(self.ndim, self.ndim+var.ndim)), self.ndim+var.ndim)

  def _multiply( self, other ):
    if len(self.axes) == self.ndim:
      other_trans = other._align(_invtrans(self.axes), self.ndim)
      if other_trans is not None:
        return align( multiply( self.func, other_trans ), self.axes, self.ndim )
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func * other.func, self.axes, self.ndim )

  def _add( self, other ):
    if len(self.axes) == self.ndim:
      other_trans = other._align(_invtrans(self.axes), self.ndim)
      if other_trans is not None:
        return align( add( self.func, other_trans ), self.axes, self.ndim )
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func + other.func, self.axes, self.ndim )

  def _take( self, indices, axis ):
    try:
      n = self.axes.index( axis )
    except ValueError:
      return
    return align( take( self.func, indices, n ), self.axes, self.ndim )

  def _edit( self, op ):
    return align( op(self.func), self.axes, self.ndim )

  def _dot( self, other, axes ):
    if len(self.axes) == self.ndim:
      funcaxes = tuple( self.axes.index(axis) for axis in axes )
      trydot = self.func._dot(transpose(other,self.axes), funcaxes)
      if trydot is not None:
        keep = numpy.ones( self.ndim, dtype=bool )
        keep[list(axes)] = False
        axes = [ builtins.sum(keep[:axis]) for axis in self.axes if keep[axis] ]
        assert len(axes) == trydot.ndim
        return align( trydot, axes, len(axes) )

  def _mask( self, maskvec, axis ):
    funcaxis = self.axes.index( axis ) # must exist, otherwise situation should have been handled in def mask
    return align( mask( self.func, maskvec, funcaxis ), self.axes, self.ndim )

class Get(Array):

  def __init__(self, func, axis, item):
    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    assert 0 <= item
    if numeric.isint(func.shape[axis]):
      assert item < func.shape[axis], 'item is out of bounds'
    super().__init__(args=[func], shape=func.shape[:axis]+func.shape[axis+1:], dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    retval = func._get(self.axis, self.item)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Get(func, self.axis, self.item)

  def evalf(self, arr):
    return arr[(slice(None),)*(self.axis+1)+(self.item,)]

  def _derivative(self, var, seen):
    f = derivative(self.func, var, seen)
    return get(f, self.axis, self.item)

  def _get( self, i, item ):
    tryget = self.func._get(i+(i>=self.axis), item)
    if tryget is not None:
      return get( tryget, self.axis, self.item )

  def _take( self, indices, axis ):
    return get( take( self.func, indices, axis+(axis>=self.axis) ), self.axis, self.item )

  def _edit( self, op ):
    return get( op(self.func), self.axis, self.item )

class Product( Array ):
  'product'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] > 1
    self.func = func
    Array.__init__( self, args=[func], shape=func.shape[:-1], dtype=func.dtype )

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

  def _edit( self, op ):
    return product( op(self.func), -1 )

class RootCoords( Array ):
  'root coords'

  def __init__( self, ndims, trans=TRANS ):
    'constructor'

    self.trans = trans
    DerivativeTargetBase.__init__( self, args=[POINTS,trans], shape=[ndims], dtype=float )

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

  def _edit( self, op ):
    return RootCoords( len(self), op(self.trans) )

class RootTransform( Array ):
  'root transform'

  def __init__( self, ndims, nvars, trans ):
    'constructor'

    self.trans = trans
    Array.__init__( self, args=[trans], shape=(ndims,nvars), dtype=float )

  def evalf( self, chain ):
    'transform'

    todims, fromdims = self.shape
    head, tail = chain.promote( todims )
    while head and head[0].todims != todims:
      head = head[1:]
    return transform.linearfrom( head+tail, fromdims )[_]

  def _derivative(self, var, seen):
    return zeros(self.shape+var.shape)

  def _edit( self, op ):
    return RootTransform( self.shape[0], self.shape[1], op(self.trans) )

class Function( Array ):
  'function'

  def __init__( self, stdmap, shape, trans=TRANS ):
    'constructor'

    self.trans = trans
    self.stdmap = stdmap
    Array.__init__( self, args=(CACHE,POINTS,trans), shape=shape, dtype=float )

  def evalf( self, cache, points, trans ):
    'evaluate'

    try:
      std, tail = trans.lookup_item( self.stdmap )
    except KeyError:
      fvals = numpy.empty( (1,0)+(1,)*(self.ndim-1) )
    else:
      stdpoints = cache[ transform.apply ]( tail, points )
      fvals = cache[ std.eval ]( stdpoints, self.ndim-1 )
      assert fvals.ndim == self.ndim+1
      if tail:
        for i, ndims in enumerate(self.shape[1:]):
          linear = cache[ transform.linearfrom ]( tail, ndims )
          fvals = numeric.dot( fvals, linear, axis=i+2 )
    return fvals

  def _edit( self, op ):
    return Function( self.stdmap, self.shape, op(self.trans) )

  def _derivative(self, var, seen):
    if isinstance(var, LocalCoords):
      return Function(self.stdmap, self.shape+(len(var),), self.trans)
    return zeros(self.shape+var.shape, dtype=self.dtype)

class Choose( Array ):
  'piecewise function'

  def __init__( self, level, choices ):
    'constructor'

    self.level = level
    self.choices = tuple( choices )
    shape = _jointshape( level.shape, *[ choice.shape for choice in choices ] )
    dtype = _jointdtype( *[ choice.dtype for choice in choices ] )
    assert level.ndim == len( shape )
    self.ivar = [ i for i, choice in enumerate(choices) if isinstance(choice,Array) ]
    Array.__init__( self, args=[ level ] + [ choices[i] for i in self.ivar ], shape=shape, dtype=dtype )

  def evalf( self, level, *varchoices ):
    'choose'

    choices = [ choice[_] for choice in self.choices ]
    for i, choice in zip( self.ivar, varchoices ):
      choices[i] = choice
    assert all( choice.ndim == self.ndim+1 for choice in choices )
    return numpy.choose( level, choices )

  def _derivative(self, var, seen):
    grads = [derivative(choice, var, seen) for choice in self.choices]
    if not any(grads): # all-zero special case; better would be allow merging of intervals
      return zeros(self.shape + var.shape)
    return choose(self.level[(...,)+(_,)*var.ndim], grads)

  def _edit( self, op ):
    return choose( op(self.level), [ op(choice) for choice in self.choices ] )

class Choose2D( Array ):
  'piecewise function'

  def __init__( self, coords, contour, fin, fout ):
    'constructor'

    shape = _jointshape( fin.shape, fout.shape )
    self.contour = contour
    Array.__init__( self, args=(coords,contour,fin,fout), shape=shape, dtype=_jointdtype(fin.dtype,fout.dtype) )

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

  def __init__(self, func):
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

  def _edit( self, op ):
    return inverse( op(self.func) )

class Concatenate( Array ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    funcs = [ asarray(func) for func in funcs ]
    ndim = funcs[0].ndim
    assert all( func.ndim == ndim for func in funcs )
    axis = numeric.normdim( ndim, axis )
    lengths = [ func.shape[axis] for func in funcs ]
    if any( isinstance( n, str ) for n in lengths ):
      sh = '+'.join(str(n) for n in lengths)
    else:
      sh = builtins.sum( lengths )
    shape = _jointshape( *[ func.shape[:axis] + (sh,) + func.shape[axis+1:] for func in funcs ] )
    dtype = _jointdtype( *[ func.dtype for func in funcs ] )
    self.funcs = tuple(funcs)
    self.ivar = [ i for i, func in enumerate(funcs) if isevaluable(func) ]
    self.axis = axis
    self.axis_shiftright = axis-ndim
    Array.__init__( self, args=[ funcs[i] for i in self.ivar ], shape=shape, dtype=dtype )

  def evalf( self, *varargs ):
    'evaluate'

    arrays = [ func[_] for func in self.funcs ]
    for i, arg in zip( self.ivar, varargs ):
      arrays[i] = arg
    assert all( arr.ndim == self.ndim+1 for arr in arrays )

    iax = self.axis_shiftright
    ndim = numpy.max([ array.ndim for array in arrays ])
    axlen = util.sum( array.shape[iax] for array in arrays )
    shape = _jointshape( *[ (1,)*(ndim-array.ndim) + array.shape[:iax] + (axlen,) + ( array.shape[iax+1:] if iax != -1 else () ) for array in arrays ] )
    dtype = float if any( array.dtype == float for array in arrays ) else int
    retval = numpy.empty( shape, dtype=dtype )
    n0 = 0
    for array in arrays:
      n1 = n0 + array.shape[iax]
      retval[(slice(None),)*( iax if iax >= 0 else iax + ndim )+(slice(n0,n1),)] = array
      n0 = n1
    assert n0 == axlen
    return retval

  @cache.property
  def blocks(self):
    return _concatblocks(((ind[:self.axis], ind[self.axis+1:]), (ind[self.axis]+n, f))
      for n, func in zip(util.cumsum(func.shape[self.axis] for func in self.funcs), self.funcs)
        for ind, f in func.blocks)

  def _get( self, i, item ):
    if i == self.axis:
      for f in self.funcs:
        if item < f.shape[i]:
          fexp = expand( f, self.shape[:self.axis] + (f.shape[self.axis],) + self.shape[self.axis+1:] )
          return get( fexp, i, item )
        item -= f.shape[i]
      raise Exception
    axis = self.axis - (self.axis > i)
    return concatenate( [ get( aslength(f,self.shape[i],i), i, item ) for f in self.funcs ], axis=axis )

  def _derivative(self, var, seen):
    funcs = [derivative(func, var, seen) for func in self.funcs]
    return concatenate(funcs, axis=self.axis)

  def _multiply( self, other ):
    if other.shape[self.axis] == 1:
      funcs = [ multiply( func, other ) for func in self.funcs ]
    else:
      funcs = []
      n0 = 0
      for func in self.funcs:
        n1 = n0 + func.shape[ self.axis ]
        funcs.append( multiply( func, take( other, slice(n0,n1), self.axis ) ) )
        n0 = n1
      assert n0 == self.shape[ self.axis ]
    return concatenate( funcs, self.axis )

  def _cross( self, other, axis ):
    if axis == self.axis:
      n = 1, 2, 0
      m = 2, 0, 1
      return take(self,n,axis) * take(other,m,axis) - take(self,m,axis) * take(other,n,axis)
    if other.shape[self.axis] == 1:
      funcs = [ cross( func, other, axis ) for func in self.funcs ]
    else:
      funcs = []
      n0 = 0
      for func in self.funcs:
        n1 = n0 + func.shape[ self.axis ]
        funcs.append( cross( func, take( other, slice(n0,n1), self.axis ), axis ) )
        n0 = n1
      assert n0 == self.shape[ self.axis ]
    return concatenate( funcs, self.axis )

  def _add( self, other ):
    if isinstance( other, Concatenate ) and self.axis == other.axis:
      i = 0
      N1 = numpy.cumsum( [0] + [f1.shape[self.axis] for f1 in self.funcs] )
      N2 = numpy.cumsum( [0] + [f2.shape[self.axis] for f2 in other.funcs] )
      ifun1 = ifun2 = 0
      funcs = []
      while i < self.shape[self.axis]:
        j = builtins.min( N1[ifun1+1], N2[ifun2+1] )
        funcs.append( take( self.funcs[ifun1], slice(i-N1[ifun1],j-N1[ifun1]), self.axis )
                    + take( other.funcs[ifun2], slice(i-N2[ifun2],j-N2[ifun2]), self.axis ))
        i = j
        ifun1 += i >= N1[ifun1+1]
        ifun2 += i >= N2[ifun2+1]
      assert ifun1 == len(self.funcs)
      assert ifun2 == len(other.funcs)
      return concatenate( funcs, axis=self.axis )
    if other.shape[self.axis] == 1:
      funcs = [ add( func, other ) for func in self.funcs ]
    else:
      funcs = []
      n0 = 0
      for func in self.funcs:
        n1 = n0 + func.shape[ self.axis ]
        funcs.append( func + take( other, slice(n0,n1), self.axis ) )
        n0 = n1
      assert n0 == self.shape[ self.axis ]
    return concatenate( funcs, self.axis )

  def _sum( self, axis ):
    if axis == self.axis:
      return util.sum( sum( func, axis ) for func in self.funcs )
    funcs = [ sum( func, axis ) for func in self.funcs ]
    axis = self.axis - (axis<self.axis)
    return concatenate( funcs, axis )

  def _align( self, axes, ndim ):
    funcs = [ align( func, axes, ndim ) for func in self.funcs ]
    axis = axes[ self.axis ]
    return concatenate( funcs, axis )

  def _takediag( self ):
    if self.axis < self.ndim-2:
      return concatenate( [ takediag(f) for f in self.funcs ], axis=self.axis )
    axis = self.ndim-self.axis-3 # -1=>-2, -2=>-1
    n0 = 0
    funcs = []
    for func in self.funcs:
      n1 = n0 + func.shape[self.axis]
      funcs.append( takediag( take( func, slice(n0,n1), axis ) ) )
      n0 = n1
    assert n0 == self.shape[self.axis]
    return concatenate( funcs, axis=-1 )

  def _take( self, indices, axis ):
    if axis != self.axis:
      return concatenate( [ take(aslength(func,self.shape[axis],axis),indices,axis) for func in self.funcs ], self.axis )
    if not indices.isconstant:
      return
    indices, = indices.eval()
    assert numpy.all( (indices>=0) & (indices<self.shape[axis]) )
    ifuncs = numpy.hstack([ numpy.repeat(ifunc,func.shape[axis]) for ifunc, func in enumerate(self.funcs) ])[indices]
    splits, = numpy.nonzero( numpy.diff(ifuncs) != 0 )
    funcs = []
    for i, j in zip( numpy.hstack([ 0, splits+1 ]), numpy.hstack([ splits+1, len(indices) ]) ):
      ifunc = ifuncs[i]
      assert numpy.all( ifuncs[i:j] == ifunc )
      offset = builtins.sum( func.shape[axis] for func in self.funcs[:ifunc] )
      funcs.append( take( self.funcs[ifunc], indices[i:j] - offset, axis ) )
    if len( funcs ) == 1:
      return funcs[0]
    return concatenate( funcs, axis=axis )

  def _dot( self, other, axes ):
    if other.shape[self.axis] == 1:
      funcs = [ dot( f, other, axes ) for f in self.funcs ]
    else:
      n0 = 0
      funcs = []
      for f in self.funcs:
        n1 = n0 + f.shape[self.axis]
        funcs.append( dot( f, take( other, slice(n0,n1), self.axis ), axes ) )
        n0 = n1
    if self.axis in axes:
      return util.sum( funcs )
    return concatenate( funcs, self.axis - builtins.sum( axis < self.axis for axis in axes ) )

  def _power( self, n ):
    if n.shape[self.axis] != 1:
      raise NotImplementedError
    return concatenate( [ power( func, n ) for func in self.funcs ], self.axis )

  def _diagonalize( self ):
    if self.axis < self.ndim-1:
      return concatenate( [ diagonalize(func) for func in self.funcs ], self.axis )

  def _edit( self, op ):
    return concatenate( [ op(func) for func in self.funcs ], self.axis )

  def _kronecker( self, axis, length, pos ):
    return concatenate( [ kronecker(func,axis,length,pos) for func in self.funcs ], self.axis+(axis<=self.axis) )

  def _concatenate( self, other, axis ):
    if axis == self.axis:
      return concatenate( self.funcs + ( other.funcs if isinstance( other, Concatenate ) and other.axis == axis else (other,) ), axis )

  def _mask( self, maskvec, axis ):
    if axis != self.axis:
      return concatenate( [ mask(func,maskvec,axis) for func in self.funcs ], self.axis )
    s = numpy.cumsum( [0] + [ func.shape[axis] for func in self.funcs ] )
    assert s[-1] == self.shape[axis]
    return concatenate( [ mask( func, maskvec[s1:s2], axis ) for func, s1, s2 in zip( self.funcs, s[:-1], s[1:] ) ], axis )

class Interpolate( Array ):
  'interpolate uniformly spaced data; stepwise for now'

  def __init__( self, x, xp, fp, left=None, right=None ):
    'constructor'

    xp = numpy.array( xp )
    fp = numpy.array( fp )
    assert xp.ndim == fp.ndim == 1
    if not numpy.all( numpy.diff(xp) > 0 ):
      warnings.warn( 'supplied x-values are non-increasing' )

    assert x.ndim == 0
    Array.__init__( self, args=[x], shape=(), dtype=float )
    self.xp = xp
    self.fp = fp
    self.left = left
    self.right = right

  def evalf( self, x ):
    return numpy.interp( x, self.xp, self.fp, self.left, self.right )

class Cross( Array ):
  'cross product'

  def __init__( self, func1, func2, axis ):
    'contructor'

    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    shape = _jointshape( func1.shape, func2.shape )
    assert shape[axis] == 3
    dtype = _jointdtype( func1.dtype, func2.dtype )
    Array.__init__( self, args=(func1,func2), shape=shape, dtype=dtype )

  def evalf( self, a, b ):
    assert a.ndim == b.ndim == self.ndim+1
    return numeric.cross( a, b, self.axis+1 )

  def _derivative(self, var, seen):
    ext = (...,)+(_,)*var.ndim
    return cross(self.func1[ext], derivative(self.func2, var, seen), axis=self.axis) \
         - cross(self.func2[ext], derivative(self.func1, var, seen), axis=self.axis)

  def _take( self, index, axis ):
    if axis != self.axis:
      return cross( take(aslength(self.func1,self.shape[axis],axis),index,axis), take(aslength(self.func2,self.shape[axis],axis),index,axis), self.axis )

  def _edit( self, op ):
    return cross( op(self.func1), op(self.func2), self.axis )

class Determinant( Array ):

  def __init__(self, func):
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
    Finv = swapaxes(inverse(self.func))
    G = derivative(self.func, var, seen)
    ext = (...,)+(_,)*var.ndim
    return self[ext] * sum(Finv[ext] * G, axis=[-2-var.ndim,-1-var.ndim])

  def _edit( self, op ):
    return determinant( op(self.func) )

class Multiply(Array):

  def __init__(self, funcs):
    assert isinstance(funcs, orderedset)
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2)
    self.funcs = func1, func2
    super().__init__(args=self.funcs, shape=_jointshape(func1.shape, func2.shape), dtype=_jointdtype(func1.dtype,func2.dtype))

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
    return Multiply(orderedset([func1, func2]))

  def evalf( self, arr1, arr2 ):
    return arr1 * arr2

  def _sum( self, axis ):
    func1, func2 = self.funcs
    return dot( func1, func2, [axis] )

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    return multiply( get( aslength(func1,self.shape[axis],axis), axis, item ),
                     get( aslength(func2,self.shape[axis],axis), axis, item ) )

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

  def _product( self, axis ):
    func1, func2 = self.funcs
    prod1 = product( func1, axis ) if func1.shape[axis] != 1 else power( get(func1,axis,0), self.shape[axis] )
    prod2 = product( func2, axis ) if func2.shape[axis] != 1 else power( get(func2,axis,0), self.shape[axis] )
    return multiply( prod1, prod2 )

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
    return take( aslength(func1,self.shape[axis],axis), index, axis ) * take( aslength(func2,self.shape[axis],axis), index, axis )

  def _power( self, n ):
    func1, func2 = self.funcs
    func1pow = func1._power(n)
    func2pow = func2._power(n)
    if func1pow is not None and func2pow is not None:
      return multiply( func1pow, func2pow )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return multiply( op(func1), op(func2) )

  def _dot( self, other, axes ):
    func1, func2 = self.funcs
    s = [ slice(None) ] * self.ndim
    for axis in axes:
      s[axis] = 0
    s = tuple(s)
    if all( func1.shape[axis] == 1 for axis in axes ):
      return func1[s] * dot( func2, other, axes )
    if all( func2.shape[axis] == 1 for axis in axes ):
      return func2[s] * dot( func1, other, axes )

  def _inverse( self ):
    func1, func2 = self.funcs
    if self.shape[-2:] == (1,1):
      return multiply(inverse(func1), inverse(func2))
    if 1 in func1.shape[-2:]:
      func1, func2 = func2, func1
    if 1 in func1.shape[-2:]: # tensor product
      raise Exception( 'singular matrix' )
    if 1 in func2.shape[-2:]:
      return divide(inverse(func1), swapaxes(func2))

class Add(Array):

  def __init__(self, funcs):
    assert isinstance(funcs, orderedset)
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2)
    self.funcs = func1, func2
    super().__init__(args=self.funcs, shape=_jointshape(func1.shape, func2.shape), dtype=_jointdtype(func1.dtype,func2.dtype))

  @cache.property
  def simplified(self):
    func1 = self.funcs[0].simplified
    func2 = self.funcs[1].simplified
    if iszero(func1):
      return expand(func2, self.shape).simplified
    if iszero(func2):
      return expand(func1, self.shape).simplified
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
    return Add(orderedset([func1, func2]))

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
    return add( get( aslength(func1,self.shape[axis],axis), axis, item ),
                get( aslength(func2,self.shape[axis],axis), axis, item ) )

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) + takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return take( aslength(func1,self.shape[axis],axis), index, axis ) + take( aslength(func2,self.shape[axis],axis), index, axis )

  def _add( self, other ):
    func1, func2 = self.funcs
    func1_other = func1._add(other)
    if func1_other is not None:
      return add( func1_other, func2 )
    func2_other = func2._add(other)
    if func2_other is not None:
      return add( func1, func2_other )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return add( op(func1), op(func2) )

  def _mask( self, maskvec, axis ):
    func1, func2 = self.funcs
    if func1.shape[axis] != 1:
      func1 = mask( func1, maskvec, axis )
    if func2.shape[axis] != 1:
      func2 = mask( func2, maskvec, axis )
    return add( func1, func2 )

class BlockAdd( Array ):
  'block addition (used for DG)'

  def __init__( self, funcs ):
    'constructor'

    self.funcs = tuple( funcs )
    shape = _jointshape( *( func.shape for func in self.funcs ) )
    dtype = _jointdtype( *( func.dtype for func in self.funcs ) )
    if not isevaluable( funcs[-1] ):
      self.const = funcs[-1]
      funcs = funcs[:-1]
    else:
      self.const = 0
    Array.__init__( self, args=funcs, shape=shape, dtype=dtype )

  def evalf( self, *args ):
    assert all( arg.ndim == self.ndim+1 for arg in args )
    return functools.reduce( operator.add, args ) + self.const

  def _add( self, other ):
    return blockadd( self, other )

  def _dot( self, other, axes ):
    return blockadd( *( dot( func, other, axes ) for func in self.funcs ) )

  def _edit( self, op ):
    return blockadd( *map( op, self.funcs ) )

  def _sum( self, axis ):
    return blockadd( *( sum( func, axis ) for func in self.funcs ) )

  def _derivative(self, var, seen):
    return blockadd(*(derivative(func, var, seen) for func in self.funcs))

  def _get( self, i, item ):
    return blockadd( *( get( aslength(func,self.shape[i],i), i, item ) for func in self.funcs ) )

  def _takediag( self ):
    return blockadd( *( takediag( func ) for func in self.funcs ) )

  def _take( self, indices, axis ):
    return blockadd( *( take( aslength(func,self.shape[axis],axis), indices, axis ) for func in self.funcs ) )

  def _align( self, axes, ndim ):
    return blockadd( *( align( func, axes, ndim ) for func in self.funcs ) )

  def _multiply( self, other ):
    return blockadd( *( multiply( func, other ) for func in self.funcs ) )

  def _inflate( self, dofmap, length, axis ):
    return blockadd( *( inflate( func, dofmap, length, axis ) for func in self.funcs ) )

  def _kronecker( self, axis, length, pos ):
    return blockadd( *( kronecker( func, axis, length, pos ) for func in self.funcs ) )

  def _mask(self, maskvec, axis):
    return blockadd(*(mask(func, maskvec, axis) for func in self.funcs))

  @cache.property
  def blocks(self):
    gathered = tuple((ind, util.sum(f)) for ind, f in util.gather(block for func in self.funcs for block in func.blocks))
    if len(gathered) > 1:
      for idim in range(self.ndim):
        gathered = _concatblocks(((ind[:idim], ind[idim+1:]), (ind[idim], f)) for ind, f in gathered)
    return gathered

class Dot(Array):

  def __init__(self, funcs, axes):
    assert isinstance(funcs, orderedset) and isinstance(axes, tuple)
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2) and func1.ndim == func2.ndim
    self.funcs = func1, func2
    self.axes = axes
    assert all(0 <= ax < func1.ndim for ax in axes)
    assert all(ax1 < ax2 for ax1, ax2 in zip(axes[:-1], axes[1:]))
    shape = _jointshape(func1.shape, func2.shape)
    self.axes_complement = list(range(func1.ndim))
    for ax in reversed(self.axes):
      shape = shape[:ax] + shape[ax+1:]
      del self.axes_complement[ax]
    _abc = numeric._abc[:func1.ndim+1]
    self._einsumfmt = '{0},{0}->{1}'.format(_abc, ''.join(a for i, a in enumerate(_abc) if i-1 not in axes))
    super().__init__(args=funcs, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype))

  @cache.property
  def simplified(self):
    func1 = self.funcs[0].simplified
    func2 = self.funcs[1].simplified
    if len(self.axes) == 0:
      return multiply(func1, func2).simplified
    if iszero(func1) or iszero(func2):
      return zeros(self.shape)
    for i, axis in enumerate(self.axes):
      if func1.shape[axis] == 1 or func2.shape[axis] == 1:
        return dot(sum(func1,axis), sum(func2,axis), self.axes[:i] + tuple(axis-1 for axis in self.axes[i+1:])).simplified
    retval = func1._dot(func2, self.axes)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    retval = func2._dot(func1, self.axes)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Dot(orderedset([func1, func2]), self.axes)

  def evalf( self, arr1, arr2 ):
    return numpy.einsum(self._einsumfmt, arr1, arr2)

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    funcaxis = self.axes_complement[axis]
    return dot( get( aslength(func1,self.shape[axis],funcaxis), funcaxis, item ),
                get( aslength(func2,self.shape[axis],funcaxis), funcaxis, item ), [ ax-(ax>=funcaxis) for ax in self.axes ] )

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
    return dot( take( aslength(func1,self.shape[axis],funcaxis), index, funcaxis ), take( aslength(func2,self.shape[axis],funcaxis), index, funcaxis ), self.axes )

  def _concatenate( self, other, axis ):
    if isinstance( other, Dot ) and other.axes == self.axes:
      common = _findcommon( self.funcs, other.funcs )
      if common:
        f, (g1,g2) = common
        tryconcat = g1._concatenate(g2, self.axes_complement[axis])
        if tryconcat is not None:
          return dot( f, tryconcat, self.axes )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return dot( op(func1), op(func2), self.axes )

class Sum( Array ):

  def __init__(self, func, axis):
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

  def _edit( self, op ):
    return sum( op(self.func), axis=self.axis )

class Debug( Array ):
  'debug'

  def __init__( self, func ):
    'constructor'

    self.func = func
    Array.__init__( self, args=[func], shape=func.shape, dtype=func.dtype )

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

  def _edit( self, op ):
    return Debug( op(self.func) )

class TakeDiag( Array ):
  'extract diagonal'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    Array.__init__( self, args=[func], shape=func.shape[:-1], dtype=func.dtype )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numeric.takediag( arr )

  def _derivative(self, var, seen):
    fder = derivative(self.func, var, seen)
    return transpose(takediag(fder, self.func.ndim-2, self.func.ndim-1), tuple(range(self.func.ndim-2))+(-1,)+tuple(range(self.func.ndim-2,fder.ndim-2)))

  def _sum( self, axis ):
    if axis != self.ndim-1:
      return takediag( sum( self.func, axis ) )

  def _edit( self, op ):
    return takediag( op(self.func) )

class Take( Array ):

  def __init__(self, func, indices, axis):
    assert isarray(func)
    assert isarray(indices) and indices.ndim == 1 and indices.dtype == int
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
      if len(index_) == func.shape[self.axis] and numpy.all(numpy.diff(index_) == 1):
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

  def _edit( self, op ):
    return take( op(self.func), op(self.indices), self.axis )

class Power(Array):

  def __init__(self, func, power):
    self.func = func
    self.power = power
    super().__init__(args=[func,power], shape=_jointshape(func.shape, power.shape), dtype=float)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    power = self.power.simplified
    if iszero(power):
      return expand(numpy.ones([1]*self.ndim, dtype=float), self.shape)
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
        else p * power(self.func, p-1)[ext] * derivative(self.func, var, seen)
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
    return power( get( aslength(self.func, self.shape[axis],axis), axis, item ),
                  get( aslength(self.power,self.shape[axis],axis), axis, item ) )

  def _sum( self, axis ):
    if self == (self.func**2):
      return dot( self.func, self.func, axis )

  def _takediag( self ):
    return power( takediag( self.func ), takediag( self.power ) )

  def _take( self, index, axis ):
    return power( take( aslength(self.func,self.shape[axis],axis), index, axis ), take( aslength(self.power,self.shape[axis],axis), index, axis ) )

  def _multiply( self, other ):
    if isinstance( other, Power ) and self.func == other.func:
      return power( self.func, self.power + other.power )
    if other == self.func:
      return power( self.func, self.power + 1 )

  def _sign( self ):
    if iszero( self.power % 2 ):
      return expand( 1., self.shape )

  def _edit( self, op ):
    return power( op(self.func), op(self.power) )

class Pointwise( Array ):
  'pointwise transformation'

  def __init__( self, args, evalfun, deriv, dtype ):
    'constructor'

    assert isarray( args )
    assert args.ndim >= 1 and args.shape[0] >= 1
    shape = args.shape[1:]
    self.args = args
    self.evalfun = evalfun
    self.deriv = deriv
    Array.__init__( self, args=[args], shape=shape, dtype=dtype )

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

  def _edit( self, op ):
    return pointwise( op(self.args), self.evalfun, self.deriv, self.dtype )

class Sign( Array ):
  'sign'

  def __init__( self, func ):
    'constructor'

    assert isarray( func )
    self.func = func
    Array.__init__( self, args=[func], shape=func.shape, dtype=func.dtype )

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
      return expand( 1., self.shape )

  def _edit( self, op ):
    return sign( op(self.func) )

class Sampled( Array ):
  'sampled'

  def __init__ ( self, data, trans=TRANS ):
    assert isinstance(data,dict)
    self.data = data.copy()
    self.trans = trans
    items = iter(self.data.items())
    trans0, (values0,points0) = next(items)
    shape = values0.shape[1:]
    assert all( transi.fromdims == trans0.fromdims and valuesi.shape == pointsi.shape[:1]+shape for transi, (valuesi,pointsi) in items )
    Array.__init__( self, args=[trans,POINTS], shape=shape, dtype=float )

  def evalf( self, trans, points ):
    (myvals,mypoints), tail = trans.lookup_item( self.data )
    evalpoints = tail.apply( points )
    assert mypoints.shape == evalpoints.shape and numpy.all( mypoints == evalpoints ), 'Illegal point set'
    return myvals

  def _edit( self, op ):
    return Sampled( self.data, op(self.trans) )

class Elemwise( Array ):
  'elementwise constant data'

  def __init__( self, fmap, shape, default=None, trans=TRANS ):
    self.fmap = fmap
    self.default = default
    self.trans = trans
    Array.__init__( self, args=[trans], shape=shape, dtype=float )

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

  def _edit( self, op ):
    return Elemwise( self.fmap, self.shape, self.default, op(self.trans) )

class Eig( Evaluable ):
  'Eig'

  def __init__( self, func, symmetric=False, sort=False ):
    'contructor'

    Evaluable.__init__( self, args=[func] )
    self.symmetric = symmetric
    self.func = func
    self.shape = func.shape
    self.eig = numpy.linalg.eigh if symmetric else numeric.eig

  def evalf( self, arr ):
    assert arr.ndim == len(self.shape)+1
    return self.eig( arr )

  def _edit( self, op ):
    return Eig( op(self.func), self.symmetric )

class ArrayFromTuple( Array ):
  'array from tuple'

  def __init__( self, arrays, index, shape, dtype ):
    self.arrays = arrays
    self.index = index
    Array.__init__( self, args=[arrays], shape=shape, dtype=dtype )

  def evalf( self, arrays ):
    return arrays[ self.index ]

  def _edit( self, op ):
    return array_from_tuple( op(self.arrays), self.index, self.shape, self.dtype )

class Zeros( Array ):
  'zero'

  def __init__( self, shape, dtype ):
    'constructor'

    shape = tuple( shape )
    Array.__init__( self, args=[], shape=shape, dtype=dtype )

  def evalf( self ):
    'prepend point axes'

    assert not any( sh is None for sh in self.shape ), 'cannot evaluate zeros for shape %s' % (self.shape,)
    return numpy.zeros( (1,) + self.shape, dtype=self.dtype )

  @property
  def blocks(self):
    return ()

  def _repeat( self, length, axis ):
    assert self.shape[axis] == 1
    return zeros( self.shape[:axis] + (length,) + self.shape[axis+1:], dtype=self.dtype )

  def _derivative(self, var, seen):
    return zeros(self.shape+var.shape, dtype=self.dtype)

  def _add( self, other ):
    shape = _jointshape( self.shape, other.shape )
    return expand( other, shape )

  def _multiply( self, other ):
    shape = _jointshape( self.shape, other.shape )
    return zeros( shape, dtype=_jointdtype(self.dtype,other.dtype) )

  def _dot( self, other, axes ):
    shape = [ sh for axis, sh in enumerate( _jointshape( self.shape, other.shape ) ) if axis not in axes ]
    return zeros( shape, dtype=_jointdtype(self.dtype,other.dtype) )

  def _cross( self, other, axis ):
    shape = _jointshape( self.shape, other.shape )
    return zeros( shape, dtype=_jointdtype(self.dtype,other.dtype) )

  def _diagonalize( self ):
    return zeros( self.shape + (self.shape[-1],), dtype=self.dtype )

  def _sum( self, axis ):
    return zeros( self.shape[:axis] + self.shape[axis+1:], dtype=self.dtype )

  def _align( self, axes, ndim ):
    shape = [1] * ndim
    for ax, sh in zip( axes, self.shape ):
      shape[ax] = sh
    return zeros( shape, dtype=self.dtype )

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
    return expand( numpy.array(value)[(_,)*(self.ndim-1)], self.shape[1:] )

  def _kronecker( self, axis, length, pos ):
    return zeros( self.shape[:axis]+(length,)+self.shape[axis:], dtype=self.dtype )

  def _edit( self, op ):
    return self

  def _mask( self, maskvec, axis ):
    return zeros( self.shape[:axis] + (maskvec.sum(),) + self.shape[axis+1:], dtype=self.dtype )

  def _unravel( self, axis, shape ):
    shape = self.shape[:axis] + shape + self.shape[axis+1:]
    return zeros( shape, dtype=self.dtype )

class Inflate( Array ):
  'inflate'

  def __init__( self, func, dofmap, length, axis ):
    'constructor'

    assert not dofmap.isconstant
    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.axis = axis
    assert 0 <= axis < func.ndim
    assert func.shape[axis] == dofmap.shape[0]
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    self.axis_shiftright = axis-func.ndim
    Array.__init__( self, args=[func,dofmap], shape=shape, dtype=func.dtype )

  def evalf( self, array, indices ):
    'inflate'

    if indices.shape[0] != 1:
      raise NotImplementedError
    indices, = indices
    assert array.ndim == self.ndim+1
    warnings.warn( 'using explicit inflation; this is usually a bug.' )
    shape = list( array.shape )
    shape[self.axis_shiftright] = self.length
    inflated = numpy.zeros( shape, dtype=self.dtype )
    inflated[(Ellipsis,indices)+(slice(None),)*(-self.axis_shiftright-1)] = array
    return inflated

  @property
  def blocks(self):
    for ind, f in self.func.blocks:
      assert ind[self.axis] == None
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

  def _align( self, shuffle, ndims ):
    return inflate( align(self.func,shuffle,ndims), self.dofmap, self.length, shuffle[self.axis] )

  def _get( self, axis, item ):
    assert axis != self.axis
    return inflate( get(self.func,axis,item), self.dofmap, self.length, self.axis-(axis<self.axis) )

  def _dot( self, other, axes ):
    if isinstance( other, Inflate ) and other.axis == self.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    arr = dot( self.func, other, axes )
    if self.axis in axes:
      return arr
    return inflate( arr, self.dofmap, self.length, self.axis - builtins.sum( axis < self.axis for axis in axes ) )

  def _multiply( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap and self.length == other.length
      take_other = other.func
    elif other.shape[self.axis] == 1:
      take_other = other
    else:
      take_other = take( other, self.dofmap, self.axis )
    return inflate( multiply(self.func,take_other), self.dofmap, self.length, self.axis )

  def _add( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis and self.dofmap == other.dofmap:
      return inflate( add(self.func,other.func), self.dofmap, self.length, self.axis )
    return blockadd( self, other )

  def _cross( self, other, axis ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
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

  def _repeat( self, length, axis ):
    if axis != self.axis:
      return inflate( repeat(self.func,length,axis), self.dofmap, self.length, self.axis )

  def _edit( self, op ):
    return inflate( op(self.func), op(self.dofmap), self.length, self.axis )

  def _kronecker( self, axis, length, pos ):
    return inflate( kronecker(self.func,axis,length,pos), self.dofmap, self.length, self.axis+(axis<=self.axis) )

class Diagonalize( Array ):
  'diagonal matrix'

  def __init__( self, func ):
    'constructor'

    n = func.shape[-1]
    assert n != 1
    shape = func.shape + (n,)
    self.func = func
    Array.__init__( self, args=[func] if isinstance(func,Array) else [], shape=shape, dtype=func.dtype )

  def evalf( self, arr=None ):
    assert arr is None or arr.ndim == self.ndim
    return numeric.diagonalize( arr if arr is not None else self.func[_] )

  def _derivative(self, var, seen):
    result = derivative(self.func, var, seen)
    # move axis `self.ndim-1` to the end
    result = transpose(result, [i for i in range(result.ndim) if i != self.func.ndim-1] + [self.func.ndim-1])
    # diagonalize last axis
    result = diagonalize(result)
    # move diagonalized axes left of the derivatives axes
    return transpose(result, tuple(range(self.func.ndim-1)) + (result.ndim-2,result.ndim-1) + tuple(range(self.func.ndim-1,result.ndim-2)))

  def _get( self, i, item ):
    if i >= self.ndim-2:
      return kronecker( get( self.func, -1, item ), axis=-1, pos=item, length=self.func.shape[-1] )
    return diagonalize( get( self.func, i, item ) )

  def _inverse( self ):
    return diagonalize( reciprocal( self.func ) )

  def _determinant( self ):
    return product( self.func, -1 )

  def _multiply( self, other ):
    return diagonalize( self.func * takediag( other ) )

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

  def _align( self, axes, ndim ):
    axes = tuple( axes )
    if axes[-2:] in [ (ndim-2,ndim-1), (ndim-1,ndim-2) ]:
      return diagonalize( align( self.func, axes[:-2] + (ndim-2,), ndim-1 ) )
    if ndim > self.ndim: # push inserts below diagonalize so that remaining align becomes invertable
      newself = diagonalize(align(self.func, tuple(range(self.func.ndim-1)) + (ndim-2,), ndim-1))
      newaxes = axes[:self.ndim-2] + tuple(ax for ax in range(ndim) if ax not in axes) + axes[self.ndim-2:]
      assert len(newaxes) == ndim
      return align(newself, newaxes, ndim)

  def _edit( self, op ):
    return diagonalize( op(self.func) )

  def _takediag( self ):
    return self.func

  def _take( self, index, axis ):
    if axis < self.ndim-2:
      return diagonalize( take( self.func, index, axis ) )
    diag = diagonalize( take( self.func, index, self.func.ndim-1 ) )
    return inflate( diag, index, self.func.shape[-1], self.ndim-1 if axis == self.ndim-2 else self.ndim-2 )

  def _mask( self, maskvec, axis ):
    if axis < self.ndim-2:
      return diagonalize( mask( self.func, maskvec, axis ) )
    indices, = numpy.where( maskvec )
    if not numpy.all( numpy.diff(indices) == 1 ):
      return
    # consecutive sub-block
    rev = slice( None, None, 1 if axis == self.ndim-1 else -1 )
    return concatenate([
      zeros( self.func.shape[:-1] + (indices[0],len(indices))[rev] ),
      diagonalize( mask( self.func, maskvec, self.func.ndim-1 ) ),
      zeros( self.func.shape[:-1] + (self.shape[-1]-(indices[-1]+1),len(indices))[rev] ),
    ], axis=2*self.ndim-axis-3 )


class Repeat( Array ):

  def __init__(self, func, length, axis):
    assert isarray(func) and func.shape[axis] == 1
    assert 0 <= axis < func.ndim
    self.func = func
    self.axis = axis
    self.length = length
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    super().__init__(args=[func], shape=shape, dtype=func.dtype)

  @cache.property
  def simplified(self):
    func = self.func.simplified
    if self.length == 1:
      return func
    retval = func._repeat(self.length, self.axis)
    if retval is not None:
      assert retval.shape == self.shape
      return retval.simplified
    return Repeat(func, self.length, self.axis)

  def evalf( self, arr=None ):
    assert arr is None or arr.ndim == self.ndim+1
    return numeric.fastrepeat( arr if arr is not None else self.func[_], self.length, self.axis+1 )

  def _derivative(self, var, seen):
    return repeat(derivative(self.func, var, seen), self.length, self.axis)

  def _get( self, axis, item ):
    if axis == self.axis:
      assert 0 <= item
      if numeric.isint(self.length):
        assert item < self.length
      return get( self.func, axis, 0 )
    return repeat( get( self.func, axis, item ), self.length, self.axis-(axis<self.axis) )

  def _sum( self, axis ):
    if axis == self.axis:
      return get( self.func, axis, 0 ) * self.length
    return repeat( sum( self.func, axis ), self.length, self.axis-(axis<self.axis) )

  def _product( self, axis ):
    if axis == self.axis:
      return get( self.func, axis, 0 )**self.length
    return repeat( product( self.func, axis ), self.length, self.axis-(axis<self.axis) )

  def _power( self, n ):
    return aslength( power( self.func, n ), self.length, self.axis )

  def _add( self, other ):
    return aslength( self.func + other, self.length, self.axis )

  def _multiply( self, other ):
    return aslength( self.func * other, self.length, self.axis )

  def _align( self, shuffle, ndim ):
    return repeat( align(self.func,shuffle,ndim), self.length, shuffle[self.axis] )

  def _take( self, index, axis ):
    if axis == self.axis:
      return repeat( self.func, index.shape[0], self.axis )
    return repeat( take(self.func,index,axis), self.length, self.axis )

  def _takediag( self ):
    return repeat( takediag( self.func ), self.length, self.axis ) if self.axis < self.ndim-2 \
      else get( self.func, self.axis, 0 )

  def _cross( self, other, axis ):
    if axis != self.axis:
      return aslength( cross( self.func, other, axis ), self.length, self.axis )

  def _dot( self, other, axes ):
    func = dot( self.func, other, axes )
    if other.shape[self.axis] != 1:
      assert other.shape[self.axis] == self.length
      return func
    if self.axis in axes:
      return func * self.length
    return aslength( func, self.length, self.axis - builtins.sum( axis < self.axis for axis in axes ) )

  def _edit( self, op ):
    return repeat( op(self.func), self.length, self.axis )

  def _concatenate( self, other, axis ):
    if axis == self.axis:
      return
    if isinstance( other, Repeat ):
      return aslength( aslength( concatenate( [self.func,other.func], axis ), self.length, self.axis ), other.length, other.axis )
    return aslength( concatenate( [self.func,other], axis ), self.length, self.axis )

  def _kronecker( self, axis, length, pos ):
    return repeat( kronecker(self.func,axis,length,pos), self.length, self.axis+(axis<=self.axis) )

  def _mask( self, maskvec, axis ):
    if axis == self.axis:
      return repeat( self.func, maskvec.sum(), axis )
    return repeat( mask( self.func, maskvec, axis ), self.length, self.axis )

class Guard( Array ):
  'bar all simplifications'

  def __init__( self, fun ):
    self.fun = fun
    Array.__init__( self, args=[fun], shape=fun.shape, dtype=fun.dtype )

  @staticmethod
  def evalf( dat ):
    return dat

  def _edit( self, op ):
    return Guard( op(self.fun) )

  def _derivative(self, var, seen):
    return Guard(derivative(self.fun, var, seen))

class TrigNormal( Array ):
  'cos, sin'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    Array.__init__( self, args=[angle], shape=(2,), dtype=float )

  def _derivative(self, var, seen):
    return trigtangent(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf( self, angle ):
    return numpy.array([ numpy.cos(angle), numpy.sin(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return asarray( 1 if isinstance(other,TrigNormal) else 0 )

  def _edit( self, op ):
    return trignormal( op(self.angle) )

class TrigTangent( Array ):
  '-sin, cos'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    Array.__init__( self, args=[angle], shape=(2,), dtype=float )

  def _derivative(self, var, seen):
    return -trignormal(self.angle)[(...,)+(_,)*var.ndim] * derivative(self.angle, var, seen)

  def evalf( self, angle ):
    return numpy.array([ -numpy.sin(angle), numpy.cos(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return asarray( 1 if isinstance(other,TrigTangent) else 0 )

  def _edit( self, op ):
    return trigtangent( op(self.angle) )

class Find( Array ):
  'indices of boolean index vector'

  def __init__( self, where ):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    self.where = where
    Array.__init__( self, args=[where], shape=['~{}'.format(where.shape[0])], dtype=int )

  def evalf( self, where ):
    assert where.shape[0] == 1
    where, = where
    index, = where.nonzero()
    return index[_]

  def _edit( self, op ):
    return Find( op(self.where) )

class Kronecker( Array ):
  'kronecker'

  def __init__( self, func, axis, length, pos ):
    assert isarray( func )
    assert 0 <= axis <= func.ndim
    assert 0 <= pos < length
    self.func = func
    self.axis = axis
    self.length = length
    self.pos = pos
    Array.__init__( self, args=[func], shape=func.shape[:axis]+(length,)+func.shape[axis:], dtype=func.dtype )

  def evalf( self, func ):
    return numeric.kronecker( func, self.axis+1, self.length, self.pos )

  def _derivative(self, var, seen):
    return kronecker(derivative(self.func, var, seen), self.axis, self.length, self.pos)

  def _get( self, i, item ):
    if i != self.axis:
      return kronecker( get(self.func,i-(i>self.axis),item), self.axis-(i<self.axis), self.length, self.pos )
    if item != self.pos:
      return zeros( self.func.shape, self.dtype )
    return self.func

  def _add( self, other ):
    if isinstance( other, Kronecker ) and other.axis == self.axis and self.length == other.length and self.pos == other.pos:
      return kronecker( self.func + other.func, self.axis, self.length, self.pos )

  def _multiply( self, other ):
    getpos = 0 if other.shape[self.axis] == 1 else self.pos
    return kronecker( self.func * get( other, self.axis, getpos ), self.axis, self.length, self.pos )

  def _dot( self, other, axes ):
    getpos = 0 if other.shape[self.axis] == 1 else self.pos
    newother = get( other, self.axis, getpos )
    newaxis = self.axis
    newaxes = []
    for ax in axes:
      if ax < self.axis:
        newaxis -= 1
        newaxes.append( ax )
      elif ax > self.axis:
        newaxes.append( ax-1 )
    dotfunc = dot( self.func, newother, newaxes )
    return dotfunc if len(newaxes) < len(axes) else kronecker( dotfunc, newaxis, self.length, self.pos )

  def _sum( self, axis ):
    if axis == self.axis:
      return self.func
    return kronecker( sum( self.func, axis-(axis>self.axis) ), self.axis-(axis<self.axis), self.length, self.pos )

  def _align( self, axes, ndim ):
    newaxis = axes[self.axis]
    newaxes = [ ax-(ax>newaxis) for ax in axes if ax != newaxis ]
    return kronecker( align( self.func, newaxes, ndim-1 ), newaxis, self.length, self.pos )

  def _takediag( self ):
    if self.axis < self.ndim-2:
      return kronecker( takediag(self.func), self.axis, self.length, self.pos )
    return kronecker( get( self.func, self.func.ndim-1, self.pos ), self.func.ndim-1, self.length, self.pos )

  def _take( self, index, axis ):
    if axis != self.axis:
      return kronecker( take( self.func, index, axis-(axis>self.axis) ), self.axis, self.length, self.pos )
    # TODO select axis in index

  def _power( self, n ):
    assert n.ndim == self.ndim
    if n.shape[self.axis] == 1:
      return kronecker( power(self.func,get(n,self.axis,0)), self.axis, self.length, self.pos )

  def _pointwise( self, evalf, deriv, dtype ):
    if self.axis == 0:
      return
    value = evalf( *numpy.zeros(self.shape[0]) )
    assert value.dtype == dtype
    if value == 0:
      return kronecker( pointwise( self.func, evalf, deriv, dtype ), self.axis-1, self.length, self.pos )

  def _edit( self, op ):
    return kronecker( op(self.func), self.axis, self.length, self.pos )

  def _mask( self, maskvec, axis ):
    if axis != self.axis:
      return kronecker( mask( self.func, maskvec, axis-(axis>self.axis) ), self.axis, self.length, self.pos )
    newlength = maskvec.sum()
    if not maskvec[self.pos]:
      return zeros( self.shape[:axis] + (newlength,) + self.shape[axis+1:], dtype=self.dtype )
    newpos = maskvec[:self.pos].sum()
    return kronecker( self.func, self.axis, newlength, newpos )

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
  >>> function.derivative(f, a) == 3*a**2
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

  def __init__(self, name, shape, nderiv=0):
    self._name = name
    self._nderiv = nderiv
    DerivativeTargetBase.__init__(self, args=[ARGUMENTS], shape=shape, dtype=float)

  def evalf(self, args):
    assert self._nderiv == 0
    try:
      return args[self._name][_]
    except KeyError:
      raise ValueError('argument {!r} missing'.format(self._name))

  def _edit( self, op ):
    return self

  def _derivative(self, var, seen):
    if isinstance(var, Argument) and var._name == self._name:
      assert var._nderiv == 0 and self.shape[:self.ndim-self._nderiv] == var.shape
      if self._nderiv:
        return zeros(self.shape+var.shape)
      result = numpy.array(1)
      for i, n in enumerate( var.shape ):
        result = repeat(result[..., None], n, i)
      for i, sh in enumerate(self.shape):
        result = result * align(eye(sh), (i, self.ndim+i), self.ndim*2)
      return result
    elif isinstance(var, LocalCoords):
      return Argument(self._name, self.shape+var.shape, self._nderiv+1)
    else:
      return zeros(self.shape+var.shape)

  def __str__(self):
    return '{} {!r} <{}>'.format(self.__class__.__name__, self._name, ','.join(map(str, self.shape)))

class LocalCoords( DerivativeTargetBase ):
  'local coords derivative target'

  def __init__( self, ndims ):
    DerivativeTargetBase.__init__( self, args=[], shape=[ndims], dtype=float )

  def evalf( self ):
    raise Exception( 'LocalCoords should not be evaluated' )

class Ravel( Array ):
  'ravel'

  def __init__( self, func, axis ):
    self.func = func
    self.axis = axis
    assert 0 <= axis < func.ndim-1
    newlength = func.shape[axis] * func.shape[axis+1] if numeric.isint( func.shape[axis] ) and numeric.isint( func.shape[axis+1] ) \
           else '{}x{}'.format( func.shape[axis], func.shape[axis+1] )
    Array.__init__( self, args=[func], shape=func.shape[:axis]+(newlength,)+func.shape[axis+2:], dtype=func.dtype )

  def evalf( self, f ):
    return f.reshape( f.shape[:self.axis+1] + (f.shape[self.axis+1]*f.shape[self.axis+2],) + f.shape[self.axis+3:] )

  def _multiply( self, other ):
    if other.shape[self.axis] == 1:
      return ravel( multiply( self.func, insert(other,self.axis) ), self.axis )
    if isinstance( other, Ravel ) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return ravel( multiply( self.func, other.func ), self.axis )

  def _add( self, other ):
    if other.shape[self.axis] == 1:
      return ravel( add( self.func, insert(other,self.axis) ), self.axis )
    if isinstance( other, Ravel ) and other.axis == self.axis and other.func.shape[self.axis:self.axis+2] == self.func.shape[self.axis:self.axis+2]:
      return ravel( add( self.func, other.func ), self.axis )

  def _get( self, i, item ):
    if i != self.axis:
      return ravel( get( self.func, i+(i>self.axis), item ), self.axis-(i<self.axis) )
    if numeric.isint( self.func.shape[self.axis+1] ):
      i, j = divmod( item, self.func.shape[self.axis+1] )
      return get( get( self.func, self.axis, i ), self.axis, j )

  def _dot( self, other, axes ):
    if other.shape[self.axis] == 1:
      assert self.axis not in axes # should have been handled at higher level
      newaxes = [ ax+(ax>self.axis) for ax in axes ]
      return ravel( dot( self.func, insert(other,self.axis), newaxes ), self.axis )
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

  def _align( self, axes, ndim ):
    ravelaxis = axes[self.axis]
    funcaxes = [ ax+(ax>ravelaxis) for ax in axes ]
    funcaxes = funcaxes[:self.axis+1] + [ravelaxis+1] + funcaxes[self.axis+1:]
    return ravel( align( self.func, funcaxes, ndim+1 ), ravelaxis )

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

  def _edit( self, op ):
    return ravel( op(self.func), self.axis )

class Unravel( Array ):
  'unravel'

  def __init__( self, func, axis, shape ):
    shape = tuple(shape)
    assert func.shape[axis] == numpy.product(shape)
    self.func = func
    self.axis = axis
    self.unravelshape = shape
    Array.__init__( self, args=[func], shape=func.shape[:axis]+shape+func.shape[axis+1:], dtype=func.dtype )

  def _derivative(self, var, seen):
    return unravel(derivative(self.func, var, seen), axis=self.axis, shape=self.unravelshape)

  def evalf( self, f ):
    return f.reshape( f.shape[0], *self.shape )

  def _edit( self, op ):
    return unravel( op(self.func), self.axis, self.unravelshape )
    
class Mask( Array ):
  'mask'

  def __init__( self, func, mask, axis ):
    assert len(mask) == func.shape[axis]
    self.func = func
    self.axis = axis
    self.mask = mask
    Array.__init__( self, args=[func], shape=func.shape[:axis]+(mask.sum(),)+func.shape[axis+1:], dtype=func.dtype )

  def evalf( self, func ):
    return func[(slice(None),)*(self.axis+1)+(self.mask,)]

  def _derivative(self, var, seen):
    return mask(derivative(self.func, var, seen), self.mask, self.axis)

  def _edit( self, op ):
    return mask( op(self.func), self.mask, self.axis )

# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )

def _jointshape( shape, *shapes ):
  'determine shape after singleton expansion'

  if not shapes:
    return tuple(shape)
  other_shape, *remaining_shapes = shapes
  assert len(shape) == len(other_shape)
  combined_shape = [ sh1 if sh2 == 1 else sh2 for sh1, sh2 in zip( shape, other_shape ) ]
  assert all( shc == sh1 for shc, sh1 in zip( combined_shape, shape ) if sh1 != 1 )
  return _jointshape( combined_shape, *remaining_shapes )

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
  return [ array[(_,)*(ndim-array.ndim)] for array in arrays ]

def _obj2str( obj ):
  'convert object to string'

  if isinstance( obj, numpy.ndarray ):
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
  if isinstance( obj, dict ):
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

def _invtrans( trans ):
  trans = numpy.asarray(trans)
  assert trans.dtype == int
  invtrans = numpy.empty( len(trans), dtype=int )
  invtrans[trans] = numpy.arange(len(trans))
  return invtrans

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

# FUNCTIONS

isarray = lambda arg: isinstance( arg, Array )
iszero = lambda arg: isinstance( arg, Zeros )
isevaluable = lambda arg: isinstance( arg, Evaluable )
zeros = lambda shape, dtype=float: Zeros( shape, dtype )
zeros_like = lambda arr: zeros( arr.shape, _jointdtype(arr.dtype) )
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
piecewise = lambda level, intervals, *funcs: choose( sum( greater( insert(level,-1), intervals ), -1 ), funcs )
trace = lambda arg, n1=-2, n2=-1: sum( takediag( arg, n1, n2 ), -1 )
normalized = lambda arg, axis=-1: divide(arg, insert(norm2(arg, axis=axis), axis))
norm2 = lambda arg, axis=-1: sqrt( sum( multiply( arg, arg ), axis ) )
heaviside = lambda arg: choose( greater( arg, 0 ), [0.,1.] )
divide = lambda arg1, arg2: multiply( arg1, reciprocal(arg2) )
subtract = lambda arg1, arg2: add( arg1, negative(arg2) )
mean = lambda arg: .5 * ( arg + opposite(arg) )
jump = lambda arg: opposite(arg) - arg
add_T = lambda arg, axes=(-2,-1): swapaxes( arg, axes ) + arg
edit = lambda arg, f: arg._edit(f) if isevaluable(arg) else arg
blocks = lambda arg: asarray(arg).blocks
rootcoords = lambda ndims: RootCoords( ndims )
sampled = lambda data, ndims: Sampled( data )
bifurcate1 = lambda arg: SelectChain(arg,True ) if arg is TRANS or arg is OPPTRANS else edit( arg, bifurcate1 )
bifurcate2 = lambda arg: SelectChain(arg,False) if arg is TRANS or arg is OPPTRANS else edit( arg, bifurcate2 )
bifurcate = lambda arg1, arg2: ( bifurcate1(arg1), bifurcate2(arg2) )
curvature = lambda geom, ndims=-1: geom.normal().div(geom, ndims=ndims)
laplace = lambda arg, geom, ndims=0: arg.grad(geom, ndims).div(geom, ndims)
symgrad = lambda arg, geom, ndims=0: multiply(.5, add_T(arg.grad(geom, ndims)))
div = lambda arg, geom, ndims=0: trace(arg.grad(geom, ndims), -1, -2)
tangent = lambda geom, vec: subtract(vec, multiply(dot(vec, normal(geom), -1)[...,_], normal(geom)))
ngrad = lambda arg, geom, ndims=0: dotnorm(grad(arg, geom, ndims), geom)
nsymgrad = lambda arg, geom, ndims=0: dotnorm(symgrad(arg, geom, ndims), geom)

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

eye = lambda n: diagonalize( expand( [1.], (n,) ) )

def asarray( arg ):
  'convert to Array'

  if isarray(arg):
    return arg

  if isinstance( arg, numpy.ndarray ) or not util.isiterable( arg ):
    array = numpy.asarray( arg )
    assert array.dtype != object
    if numpy.all( array == 0 ):
      return zeros_like( array )
    return Constant( array )

  return stack( arg, axis=0 )

def insert( arg, n ):
  'insert axis'

  arg = asarray( arg )
  n = numeric.normdim( arg.ndim+1, n )
  I = numpy.arange( arg.ndim )
  return align( arg, I + (I>=n), arg.ndim+1 )

def stack( args, axis=0 ):
  'stack functions along new axis'

  length = len(args)
  assert length > 0
  stack = 0
  for iarg, arg in enumerate(args):
    stack += kronecker( arg, axis=axis, length=length, pos=iarg )
  return stack

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

def expand( arg, shape ):
  'expand'

  arg = asarray( arg )
  shape = tuple(shape)
  assert len(shape) == arg.ndim

  for i, sh in enumerate( shape ):
    arg = aslength( arg, sh, i )
  assert arg.shape == shape

  return arg

def aslength( arg, length, axis ):
  'as length'

  arg = asarray( arg )
  if arg.shape[axis] == length:
    return arg

  return repeat( arg, length, axis )

def repeat(arg, length, axis):
  arg = asarray(arg)
  axis = numeric.normdim(arg.ndim, axis)
  assert arg.shape[axis] == 1
  return Repeat(arg, length, axis).simplified

def get(arg, iax, item):
  assert numeric.isint(item)
  arg = asarray(arg)
  iax = numeric.normdim(arg.ndim, iax)
  sh = arg.shape[iax]
  if numeric.isint(sh):
    item = numeric.normdim(sh, item)
  return Get(arg, iax, item).simplified

def align(arg, axes, ndim):
  arg = asarray(arg)
  axes = _normdims(ndim, axes)
  return Align(arg, axes, ndim).simplified

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
  trans = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  arg = align(arg, trans, arg.ndim)
  return Determinant(arg)

def inverse(arg, axes=(-2,-1)):
  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort(arg.ndim, axes)
  assert ax2 > ax1 # strict
  trans = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  arg = align(arg, trans, arg.ndim)
  return transpose(Inverse(arg), trans).simplified

def takediag( arg, ax1=-2, ax2=-1 ):
  'takediag'

  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, (ax1,ax2) )
  assert ax2 > ax1 # strict

  axes = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  arg = align( arg, axes, arg.ndim )

  if arg.shape[-1] == 1:
    return get( arg, -1, 0 )

  if arg.shape[-2] == 1:
    return get( arg, -2, 0 )

  assert arg.shape[-1] == arg.shape[-2]
  shape = arg.shape[:-1]

  retval = arg._takediag()
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._takediag' % arg
    return retval

  return TakeDiag( arg )

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

def kronecker( arg, axis, length, pos ):
  'kronecker'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim+1, axis )
  assert 0 <= pos < length
  if length == 1:
    return insert( arg, axis )
  retval = arg._kronecker(axis, length, pos)
  if retval is not None:
    assert retval.shape == arg.shape[:axis]+(length,)+arg.shape[axis:], 'bug in %s._kronecker' % arg
    return retval
  return Kronecker( arg, axis, length, pos )

def diagonalize( arg ):
  'diagonalize'

  arg = asarray( arg )
  shape = arg.shape + (arg.shape[-1],)

  if arg.shape[-1] == 1:
    return arg[...,_]

  retval = arg._diagonalize()
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._diagonalize' % arg
    return retval

  return Diagonalize( arg )

def concatenate( args, axis=0 ):
  'concatenate'

  args = _matchndim( *args )
  axis = numeric.normdim( args[0].ndim, axis )
  args = [ arg for arg in args if arg.shape[axis] != 0 ]

  if all( iszero(arg) for arg in args ):
    shape = list( args[0].shape )
    axis = numeric.normdim( len(shape), axis )
    for arg in args[1:]:
      for i in range( len(shape) ):
        if i == axis:
          shape[i] += arg.shape[i]
        elif shape[i] == 1:
          shape[i] = arg.shape[i]
        else:
          assert arg.shape[i] in (shape[i],1)
    return zeros( shape, dtype=_jointdtype(*[arg.dtype for arg in args]) )

  i = 0
  while i+1 < len(args):
    arg1, arg2 = args[i:i+2]
    arg12 = arg1._concatenate(arg2, axis)
    if arg12 is None:
      i += 1
      continue
    args = args[:i] + [arg12] + args[i+2:]

  if len(args) == 1:
    return args[0]

  return Concatenate( args, axis )

def transpose( arg, trans=None ):
  'transpose'

  arg = asarray( arg )
  if not arg.ndim:
    assert not trans
    return arg

  invtrans = range( arg.ndim-1, -1, -1 ) if trans is None else _invtrans( trans )
  return align( arg, invtrans, arg.ndim )

def product( arg, axis ):
  'product'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  shape = arg.shape[:axis] + arg.shape[axis+1:]

  if arg.shape[axis] == 1:
    return get( arg, axis, 0 )

  trans = list(range(axis)) + [-1] + list(range(axis,arg.ndim-1))
  aligned_arg = align( arg, trans, arg.ndim )

  retval = aligned_arg._product(arg.ndim-1)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._product' % aligned_arg
    return retval

  return Product( aligned_arg )

def choose( level, choices ):
  'choose'

  level, *choices = _matchndim( level, *choices )
  shape = _jointshape( level.shape, *( choice.shape for choice in choices ) )
  if all( map( iszero, choices ) ):
    return zeros( shape )
  retval = level._choose(choices)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._choose' % level
    return retval
  return Choose( level, choices )

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

def cross( arg1, arg2, axis ):
  'cross product'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )
  axis = numeric.normdim( len(shape), axis )
  assert shape[axis] == 3

  retval = arg1._cross(arg2, axis)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._cross' % arg1
    return retval

  retval = arg2._cross(arg1, axis)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._cross' % arg2
    return -retval

  return Cross( arg1, arg2, axis )

def outer( arg1, arg2=None, axis=0 ):
  'outer product'

  if arg2 is not None and arg1.ndim != arg2.ndim:
    warnings.warn( 'varying ndims in function.outer; this will be forbidden in future', DeprecationWarning )
  arg1, arg2 = _matchndim( arg1, arg2 if arg2 is not None else arg1 )
  axis = numeric.normdim( arg1.ndim, axis )
  return insert(arg1,axis+1) * insert(arg2,axis)

def pointwise( args, evalf, deriv=None, dtype=float ):
  'general pointwise operation'

  args = asarray( _matchndim(*args) )
  retval = args._pointwise(evalf, deriv, dtype)
  if retval is not None:
    return retval
  return Pointwise( args, evalf, deriv, dtype )

def blockadd( *args ):
  args = tuple( itertools.chain( *( arg.funcs if isinstance( arg, BlockAdd ) else [arg] for arg in args ) ) )
  # group all `Inflate` objects with the same axis and dofmap
  inflates = util.OrderedDict()
  for arg in args:
    key = []
    while isinstance( arg, Inflate ):
      key.append( ( arg.dofmap, arg.length, arg.axis ) )
      arg = arg.func
    inflates.setdefault( tuple(key), [] ).append( arg )
  # add inflate args with the same axis and dofmap, blockadd the remainder
  args = []
  for key, values in inflates.items():
    if key is ():
      continue
    arg = functools.reduce( operator.add, values )
    for dofmap, length, axis in reversed( key ):
      arg = inflate( arg, dofmap, length, axis )
    args.append( arg )
  args.extend( inflates.get( (), () ) )
  if len( args ) == 1:
    return args[0]
  else:
    return BlockAdd( args )

def sign( arg ):
  'sign'

  arg = asarray( arg )

  if isinstance( arg, numpy.ndarray ):
    return numpy.sign( arg )

  retval = arg._sign()
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._sign' % arg
    return retval

  return Sign( arg )

def eig( arg, axes=(-2,-1), symmetric=False ):
  '''eig( arg, axes [ symmetric ] )
  Compute the eigenvalues and vectors of a matrix. The eigenvalues and vectors
  are positioned on the last axes.

  * tuple axes       The axis on which the eigenvalues and vectors are calculated
  * bool  symmetric  Is the matrix symmetric'''

  # Sort axis
  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict

  # Check if the matrix is square
  assert arg.shape[ax1] == arg.shape[ax2]

  # Move the axis with matrices
  trans = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  aligned_arg = align( arg, trans, arg.ndim )

  ret = aligned_arg._eig(symmetric)
  if ret is not None:
    # Check the shapes
    eigval, eigvec = ret
  else:
    eig = Eig( aligned_arg, symmetric=symmetric )
    eigval = array_from_tuple( eig, index=0, shape=aligned_arg.shape[:-1], dtype=float )
    eigvec = array_from_tuple( eig, index=1, shape=aligned_arg.shape, dtype=float )

  # Return the evaluable function objects in a tuple like numpy
  eigval = transpose( diagonalize( eigval ), trans )
  eigvec = transpose( eigvec, trans )
  assert eigvec.shape == arg.shape
  assert eigval.shape == arg.shape
  return eigval, eigvec

def array_from_tuple( arrays, index, shape, dtype ):
  if isinstance( arrays, Tuple ):
    array = arrays.items[index]
    assert array.shape == shape
    assert array.dtype == dtype
    return array
  else:
    return ArrayFromTuple( arrays, index, shape, dtype )

def swapaxes( arg, axis1=(-2,-1), axis2=None):
  if axis2 is None:
    axis1, axis2 = axis1
    warnings.warn('swapaxes(a,(axis1,axis2)) is deprecated; use swapaxes(a,axis1,axis2) instead', DeprecationWarning)
  arg = asarray(arg)
  trans = numpy.arange(arg.ndim)
  trans[axis1] = numeric.normdim(arg.ndim, axis2)
  trans[axis2] = numeric.normdim(arg.ndim, axis1)
  return align(arg, trans, arg.ndim)

def opposite( arg ):
  'evaluate jump over interface'

  return OPPTRANS if arg is TRANS \
    else TRANS if arg is OPPTRANS \
    else edit( arg, opposite )

def function( fmap, nmap, ndofs ):
  'create function on ndims-element'

  length = '~%d' % ndofs
  func = Function( fmap, shape=(length,) )
  dofmap = DofMap( nmap, length=length )
  return Inflate( func, dofmap, ndofs, axis=0 )

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
    assert numpy.all((index>=0) & (index<arg.shape[axis])), 'indices out of bounds'

  index = asarray(index)
  assert index.ndim == 1
  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    index = find(index)
  else:
    assert index.dtype == int

  return Take(arg, index, axis).simplified

def find( arg ):
  'find'

  arg = asarray( arg )
  assert arg.ndim == 1 and arg.dtype == bool

  if arg.isconstant:
    arg, = arg.eval()
    index, = arg.nonzero()
    return asarray( index )

  return Find( arg )

def inflate( arg, dofmap, length, axis ):
  'inflate'

  arg = asarray( arg )
  dofmap = asarray( dofmap )
  axis = numeric.normdim( arg.ndim, axis )
  shape = arg.shape[:axis] + (length,) + arg.shape[axis+1:]

  if dofmap.isconstant:
    n = arg.shape[axis]
    assert numeric.isint(n), 'constant inflation only allowed over fixed-length axis'
    index, = dofmap.eval()
    assert len(index) == n
    assert numpy.all( index >= 0 ) and numpy.all( index < length )
    assert numpy.all( numpy.diff(index) == 1 ), 'constant inflation must be contiguous'
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

  retval = arg._inflate(dofmap, length, axis)
  if retval is not None:
    assert retval.shape == tuple(shape)
    return retval

  return Inflate( arg, dofmap, length, axis )

def mask( arg, mask, axis=0 ):
  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  assert isinstance(mask,numpy.ndarray) and mask.ndim == 1 and mask.dtype == bool
  assert arg.shape[axis] == len(mask)
  if mask.all():
    return arg
  shape = arg.shape[:axis] + (mask.sum(),) + arg.shape[axis+1:]
  if not mask.any():
    return zeros( shape )

  retval = arg._mask(mask, axis)
  if retval is not None:
    assert retval.shape == shape
    return retval

  return Mask( arg, mask, axis )

def J( geometry, ndims=None ):
  if ndims is None:
    ndims = len(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian( geometry, ndims )

def unravel( func, axis, shape ):
  func = asarray( func )
  axis = numeric.normdim( func.ndim, axis )
  shape = tuple(shape)
  assert func.shape[axis] == numpy.product(shape)

  if len(shape) == 1:
    return func

  if shape[0] == 1:
    return insert( unravel( func, axis, shape[1:] ), axis )

  if shape[-1] == 1:
    return insert( unravel( func, axis, shape[:-1] ), axis+len(shape)-1 )

  retval = func._unravel(axis, shape)
  if retval is not None:
    return retval

  return Unravel( func, axis, shape )

def ravel( func, axis ):
  func = asarray( func )
  axis = numeric.normdim( func.ndim-1, axis )

  for i in axis, axis+1:
    if func.shape[i] == 1:
      return get( func, i, 0 )

  retval = func._ravel(axis)
  if retval is not None:
    return retval

  return Ravel( func, axis )

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
  d = {}
  def s(f):
    try:
      v = d[f]
    except KeyError:
      if isinstance(f, Argument) and f._name in arguments:
        v = arguments[f._name]
        assert f.shape[:f.ndim-f._nderiv] == v.shape
        for ndims in f.shape[f.ndim-f._nderiv:]:
          v = localgradient(v, ndims)
      else:
        v = edit(f, s)
      if isarray(f):
        assert v.shape == f.shape
      d[f] = v
    return v
  return s(value)

def zero_argument_derivatives(fun):
  d = {}
  def s(f):
    try:
      v = d[f]
    except KeyError:
      if isinstance(f, Argument) and f._nderiv > 0:
        v = zeros_like(f)
      else:
        v = edit(f, s)
      if isarray(f):
        assert v.shape == f.shape
      d[f] = v
    return v
  return s(fun)

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
