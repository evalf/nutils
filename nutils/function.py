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

from . import util, numpy, numeric, log, core, cache, transform, _
import sys, warnings, itertools, functools, operator, inspect, numbers, builtins

CACHE = 'Cache'
TRANS = 'Trans'
OPPTRANS = 'OppTrans'
POINTS = 'Points'

TOKENS = CACHE, TRANS, OPPTRANS, POINTS

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
    asciitree = str(self)
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

  def __str__( self ):
    return self.__class__.__name__

  def eval( self, elem=None, ischeme=None, fcache=cache.WrapperDummyCache() ):
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

    ops, inds = self.serialized
    assert TOKENS == ( CACHE, TRANS, OPPTRANS, POINTS )
    values = [ fcache, trans, opptrans, points ]
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
    lines.extend( '%d [label="%d. %s"];' % (i,i,name) for i, name in enumerate( TOKENS + ops + (self,) ) )
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
      lines.append( '  %%%d = %s( %s )' % ( len(lines), op, ', '.join( args ) ) )
      if len(lines) == nlines+1:
        break
    return '\n'.join( lines )

  def _edit( self, op ):
    raise NotImplementedError( '{} does not define an _edit method'.format( type(self).__name__ ) )

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

  __add__ = __radd__ = lambda self, other: add(self, other)
  __sub__ = lambda self, other: subtract(self, other)
  __rsub__ = lambda self, other: subtract(other, self)
  __mul__ = __rmul__ = lambda self, other: multiply(self, other)
  __truediv__ = lambda self, other: divide(self, other)
  __rtruediv__ = lambda self, other: divide(other, self)
  __neg__ = lambda self: negative(self)
  __pow__ = lambda self, n: power(self, n)
  __abs__ = lambda self: abs(self)
  __mod__  = lambda self, other: mod(self, other)
  __str__ = __repr__ = lambda self: '{}{}'.format(self.__class__.__name__, getattr(self, 'shape', '(uninitialized)'))

  sum = lambda self, axis: sum(self, axis)
  vector = lambda self, ndims: vectorize([self] * ndims)
  dot = lambda self, other, axes=None: dot(self, other, axes)
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
  def blocks( self ):
    return [( Tuple([ asarray(numpy.arange(n)) if numeric.isint(n) else None for n in self.shape ]), self )]

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

  simplified = property(lambda self: self)

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

class Align( Array ):
  'align axes'

  def __init__( self, func, axes, ndim ):
    'constructor'

    assert func.ndim == len(axes)
    self.func = func
    assert all( 0 <= ax < ndim for ax in axes )
    self.axes = tuple(axes)
    shape = [ 1 ] * ndim
    for ax, sh in zip( self.axes, func.shape ):
      shape[ax] = sh
    self.negaxes = [ ax-ndim for ax in self.axes ]
    Array.__init__( self, args=[func], shape=shape, dtype=func.dtype )

  def evalf( self, arr ):
    'align'

    assert arr.ndim == len(self.axes)+1
    extra = arr.ndim - len(self.negaxes)
    return numeric.align( arr, list(range(extra))+self.negaxes, self.ndim+extra )

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

class Get( Array ):
  'get'

  def __init__( self, func, axis, item ):
    'constructor'

    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    assert 0 <= item
    if numeric.isint( func.shape[axis] ):
      assert item < func.shape[axis], 'item is out of bounds'
    self.item_shiftright = (Ellipsis,item) + (slice(None),)*(func.ndim-axis-1)
    shape = func.shape[:axis] + func.shape[axis+1:]
    Array.__init__( self, args=[func], shape=shape, dtype=func.dtype )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return arr[ self.item_shiftright ]

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
  'inverse'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    Array.__init__( self, args=[func], shape=func.shape, dtype=float )

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
      assert all( isinstance( n, str ) for n in lengths )
      sh = ''.join(lengths)
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

  @property
  def blocks( self ):
    n = 0
    for func in self.funcs:
      for ind, f in blocks( func ):
        yield Tuple( ind[:self.axis] + (ind[self.axis]+n,) + ind[self.axis+1:] ), f
      n += func.shape[self.axis]

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
  'normal'

  def __init__( self, func ):
    'contructor'

    self.func = func
    Array.__init__( self, args=[func], shape=func.shape[:-2], dtype=func.dtype )

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

class Multiply( Array ):
  'multiply'

  def __init__( self, funcs ):
    'constructor'

    assert isinstance( funcs, Pair )
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2)
    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    Array.__init__( self, args=self.funcs, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype) )

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
    assert self.ndim >= 2 and self.shape != (1,1) # should have been handled at higher level
    func1, func2 = self.funcs
    if 1 in func1.shape[-2:]:
      func1, func2 = func2, func1 # swap singleton-axis argument into func2
    if 1 in func1.shape[-2:]: # tensor product
      return zeros( () )
    if 1 in func2.shape[-2:]:
      det2 = power( func2[...,0,0], self.shape[-1] ) if func2.shape[-2:] == (1,1) \
        else product( func2.sum( -1 if func2.shape[-1] == 1 else -2 ), axis=0 )
      return determinant(func1) * det2

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
    assert self.ndim >= 2 and self.shape != (1,1) # should have been handled at higher level
    func1, func2 = self.funcs
    if 1 in func1.shape[-2:]:
      func1, func2 = func2, func1
    if 1 in func1.shape[-2:]: # tensor product
      raise Exception( 'singular matrix' )
    if 1 in func2.shape[-2:]:
      return inverse(func1) / swapaxes(func2)

class Add( Array ):
  'add'

  def __init__( self, funcs ):
    'constructor'

    assert isinstance( funcs, Pair )
    func1, func2 = funcs
    assert isarray(func1) and isarray(func2)
    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    Array.__init__( self, args=self.funcs, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype) )

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

  @property
  def blocks( self ):
    for func in self.funcs:
      for ind, f in blocks( func ):
        yield ind, f

class Dot( Array ):
  'dot'

  def __init__( self, funcs, naxes ):
    'constructor'

    assert isinstance( funcs, Pair )
    func1, func2 = funcs
    assert isarray( func1 )
    assert naxes > 0
    self.naxes = naxes
    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )[:-naxes]
    Array.__init__( self, args=funcs, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype) )

  def evalf( self, arr1, arr2 ):
    assert arr1.ndim == self.ndim+1+self.naxes
    return numeric.contract_fast( arr1, arr2, self.naxes )

  @property
  def axes( self ):
    return list( range( self.ndim, self.ndim + self.naxes ) )

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    return dot( get( aslength(func1,self.shape[axis],axis), axis, item ),
                get( aslength(func2,self.shape[axis],axis), axis, item ), [ ax-1 for ax in self.axes ] )

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
    n1, n2 = self.ndim-2, self.ndim-1
    func1, func2 = self.funcs
    return dot( takediag( func1, n1, n2 ), takediag( func2, n1, n2 ), [ ax-2 for ax in self.axes ] )

  def _sum( self, axis ):
    func1, func2 = self.funcs
    return dot( func1, func2, self.axes + [axis] )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return dot( take( aslength(func1,self.shape[axis],axis), index, axis ), take( aslength(func2,self.shape[axis],axis), index, axis ), self.axes )

  def _concatenate( self, other, axis ):
    if isinstance( other, Dot ) and other.axes == self.axes:
      common = _findcommon( self.funcs, other.funcs )
      if common:
        f, (g1,g2) = common
        tryconcat = g1._concatenate(g2, axis)
        if tryconcat is not None:
          return dot( f, tryconcat, self.axes )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return dot( op(func1), op(func2), self.axes )

class Sum( Array ):
  'sum'

  def __init__( self, func, axis ):
    'constructor'

    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    self.axis_shiftright = axis-func.ndim
    Array.__init__( self, args=[func], shape=shape, dtype=func.dtype )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numpy.sum( arr, self.axis_shiftright )

  def _sum( self, axis ):
    trysum = self.func._sum(axis+(axis>=self.axis))
    if trysum is not None:
      return sum( trysum, self.axis )

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
  'generalization of numpy.take(), to accept lists, slices, arrays'

  def __init__( self, func, indices, axis ):
    'constructor'

    assert isarray(func) and func.shape[axis] != 1
    assert isarray(indices) and indices.ndim == 1 and indices.dtype == int
    assert 0 <= axis < func.ndim

    self.func = func
    self.axis = axis
    self.indices = indices

    shape = func.shape[:axis] + indices.shape + func.shape[axis+1:]
    Array.__init__( self, args=[func,indices], shape=shape, dtype=func.dtype )

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

class Power( Array ):
  'power'

  def __init__( self, func, power ):
    'constructor'

    self.func = func
    self.power = power
    shape = _jointshape( func.shape, power.shape )
    Array.__init__( self, args=[func,power], shape=shape, dtype=float )

  def evalf( self, base, exp ):
    return numeric.power( base, exp )

  def _derivative(self, var, seen):
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    ext = (...,)+(_,)*var.ndim
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
  def blocks( self ):
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
  def blocks( self ):
    for ind, f in blocks( self.func ):
      assert ind[self.axis] == None
      yield Tuple( ind[:self.axis] + (self.dofmap,) + ind[self.axis+1:] ), f

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
  'repeat singleton axis'

  def __init__( self, func, length, axis ):
    'constructor'

    assert length != 1
    assert func.shape[axis] == 1
    self.func = func
    self.axis = axis
    self.length = length
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    self.axis_shiftright = axis-func.ndim
    Array.__init__( self, args=[func] if isinstance(func,Array) else [], shape=shape, dtype=func.dtype )

  def evalf( self, arr=None ):
    assert arr is None or arr.ndim == self.ndim+1
    return numeric.fastrepeat( arr if arr is not None else self.func[_], self.length, self.axis_shiftright )

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
      return numpy.array( 1 if isinstance(other,TrigNormal) else 0 )

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
      return numpy.array( 1 if isinstance(other,TrigTangent) else 0 )

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

  @property
  def blocks( self ):
    for ind, f in blocks( self.func ):
      yield Tuple( ind[:self.axis] + (Constant(numpy.array([self.pos])),) + ind[self.axis:] ), insert( f, self.axis )

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

class DerivativeTarget( DerivativeTargetBase ):
  'helper class for computing derivatives'

  def __init__( self, shape ):
    DerivativeTargetBase.__init__( self, args=[], shape=shape, dtype=float )

  def evalf( self ):
    raise ValueError( 'unwrap {!r} before evaluation'.format( self ) )

  def _edit( self, op ):
    return self

  def _derivative(self, var, seen):
    if var is self:
      result = numpy.array(1)
      for i, n in enumerate( var.shape ):
        result = repeat(result[..., None], n, i)
      for i, sh in enumerate(self.shape):
        result = result * align(eye(sh), (i, self.ndim+i), self.ndim*2)
      return result
    else:
      return zeros(self.shape+var.shape)

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
  def blocks( self ):
    for ind, f in blocks( self.func ):
      newind = ravel( ind[self.axis][:,_] * self.func.shape[self.axis+1] + ind[self.axis+1][_,:], axis=0 )
      yield Tuple( ind[:self.axis] + (newind,) + ind[self.axis+2:] ), ravel( f, axis=self.axis )

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

def merge( funcs ):
  'Combines unchained funcs into one function object.'

  cascade = fmap = nmap = None
  offset = 0 # ndofs = builtins.sum( f.shape[0] for f in funcs )

  for inflated_func in funcs:
    (func, (dofmap,)), = inflated_func.blocks # Returns one scalar function.

    if fmap is None:
      fmap = func.stdmap.copy()
    else:
      targetlen = len( fmap ) + len( func.stdmap )
      fmap.update( func.stdmap )
      assert len( fmap ) == targetlen, 'Don`t allow overlap.'

    if nmap is None:
      nmap = dofmap.dofmap.copy()
    else:
      targetlen = len( nmap ) + len( dofmap.dofmap )
      nmap.update( dict( (key, val+offset) for key, val in dofmap.dofmap.items() ) )
      assert len( nmap ) == targetlen, 'Don`t allow overlap.'

    if cascade is None:
      cascade = func.cascade
    else:
      assert func.cascade == cascade, 'Functions have to be defined on domains of same dimension.'

    offset += inflated_func.shape[0]

  return function( fmap, nmap, offset, cascade.ndims )

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

def repeat( arg, length, axis ):
  'repeat'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  assert arg.shape[axis] == 1
  if length == 1:
    return arg

  retval = arg._repeat(length, axis)
  if retval is not None:
    shape = arg.shape[:axis] + (length,) + arg.shape[axis+1:]
    assert retval.shape == shape, 'bug in %s._repeat' % arg
    return retval

  return Repeat( arg, length, axis )

def get( arg, iax, item ):
  'get item'

  assert numeric.isint( item )

  arg = asarray( arg )
  iax = numeric.normdim( arg.ndim, iax )
  sh = arg.shape[iax]

  if numeric.isint( sh ):
    item = numeric.normdim( sh, item )
  assert item >= 0

  retval = arg._get(iax, item)
  if retval is not None:
    assert retval.shape == arg.shape[:iax] + arg.shape[iax+1:], 'bug in %s._get' % arg
    return retval

  return Get( arg, iax, item )

def align( arg, axes, ndim ):
  'align'

  arg = asarray( arg )

  assert ndim >= len(axes)
  assert len(axes) == arg.ndim
  axes = _normdims( ndim, axes )
  assert len(set(axes)) == len(axes), 'duplicate axes in align'

  if util.allequal( axes, range(ndim) ):
    return arg

  retval = arg._align(axes, ndim)
  if retval is not None:
    shape = [1] * ndim
    for i, axis in enumerate( axes ):
      shape[axis] = arg.shape[i]
    assert retval.shape == tuple(shape), 'bug in %s._align' % arg
    return retval

  return Align( arg, axes, ndim )

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

def sum( arg, axis=None ):
  'sum over one or multiply axes'

  assert axis is not None

  arg = asarray( arg )

  if util.isiterable(axis):
    if len(axis) == 0:
      return arg
    axis = _norm_and_sort( arg.ndim, axis )
    assert numpy.all( numpy.diff(axis) > 0 ), 'duplicate axes in sum'
    return sum( sum( arg, axis[1:] ), axis[0] )

  axis = numeric.normdim( arg.ndim, axis )

  retval = arg._sum(axis)
  if retval is not None:
    assert retval.shape == arg.shape[:axis] + arg.shape[axis+1:], 'bug in %s._sum' % arg
    return retval

  return Sum( arg, axis )

def dot( arg1, arg2, axes=None ):
  'dot product'

  if axes is None:
    arg1 = asarray(arg1)
    arg2 = asarray(arg2)
    assert arg2.ndim == 1 and arg2.shape[0] == arg1.shape[0]
    while arg2.ndim < arg1.ndim:
      arg2 = insert(arg2, arg2.ndim)
    axes = 0,
  else:
    arg1, arg2 = _matchndim( arg1, arg2 )

  if not util.isiterable(axes):
    axes = axes,

  if len(axes) == 0:
    return arg1 * arg2

  shape = _jointshape( arg1.shape, arg2.shape )
  axes = _norm_and_sort( len(shape), axes )
  assert numpy.all( numpy.diff(axes) > 0 ), 'duplicate axes in sum'

  dotshape = tuple( s for i, s in enumerate(shape) if i not in axes )

  if iszero( arg1 ) or iszero( arg2 ):
    return zeros( dotshape )

  for i, axis in enumerate( axes ):
    if arg1.shape[axis] == 1 or arg2.shape[axis] == 1:
      axes = axes[:i] + tuple( axis-1 for axis in axes[i+1:] )
      return dot( sum(arg1,axis), sum(arg2,axis), axes )

  for axis, sh1 in enumerate(arg1.shape):
    if sh1 == 1 and arg2.shape[axis] == 1:
      assert axis not in axes
      dotaxes = [ ax - (ax>axis) for ax in axes ]
      dotargs = dot( sum(arg1,axis), sum(arg2,axis), dotaxes )
      axis -= builtins.sum( ax<axis for ax in axes )
      return align( dotargs, [ ax + (ax>=axis) for ax in range(dotargs.ndim) ], dotargs.ndim+1 )

  retval = arg1._dot(arg2, axes)
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg1
    return retval

  retval = arg2._dot(arg1, axes)
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg2
    return retval

  shuffle = list( range( len(shape) ) )
  for ax in reversed( axes ):
    shuffle.append( shuffle.pop(ax) )

  return Dot( Pair( transpose(arg1,shuffle), transpose(arg2,shuffle) ), len(axes) )

def matmat( arg0, *args ):
  'helper function, contracts last axis of arg0 with first axis of arg1, etc'
  retval = asarray( arg0 )
  for arg in args:
    arg = asarray( arg )
    assert retval.shape[-1] == arg.shape[0], 'incompatible shapes'
    retval = dot( retval[(...,)+(_,)*(arg.ndim-1)], arg[(_,)*(retval.ndim-1)], retval.ndim-1 )
  return retval

def determinant( arg, axes=(-2,-1) ):
  'determinant'

  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict

  n = arg.shape[ax1]
  assert n == arg.shape[ax2]
  if n == 1:
    return get( get( arg, ax2, 0 ), ax1, 0 )

  trans = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  arg = align( arg, trans, arg.ndim )
  shape = arg.shape[:-2]

  retval = arg._determinant()
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._determinant' % arg
    return retval

  return Determinant( arg )

def inverse( arg, axes=(-2,-1) ):
  'inverse'

  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict

  n = arg.shape[ax1]
  assert arg.shape[ax2] == n
  if n == 1:
    return reciprocal( arg )

  trans = list(range(ax1)) + [-2] + list(range(ax1,ax2-1)) + [-1] + list(range(ax2-1,arg.ndim-2))
  arg = align( arg, trans, arg.ndim )

  retval = arg._inverse()
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._inverse' % arg
    return transpose( retval, trans )

  return transpose( Inverse(arg), trans )

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

def partial_derivative(func, arg_key):
  '''partial derivative of a function

  Compute the partial derivative of ``func`` with respect to argument
  ``arg_key``.

  Parameters
  ----------
  func : callable
  arg_key : int or str
      Reference to an argument of ``func``.  If ``arg_key`` is an :class:`int`,
      ``arg_key`` is the index of a positional argument of ``func``.  If
      ``arg_key`` is a :class:`str`, ``arg_key`` is the name of an argument of
      ``func``.

  Returns
  -------
  callable
      Partial derivative of ``func``.  The shape of this function is the
      concatenation of the shape of ``func`` and the shape of ``arg``,
      where ``arg`` is the argument refered to by ``arg_key``.
  '''

  if not isinstance(arg_key, (int, str)):
    raise ValueError('arg_key: expected an int or str, got {!r}'.format(arg_key))

  sig = inspect.signature(func)
  if isinstance(arg_key, str) and arg_key in sig.parameters:
    # convert `arg_key` to index if possible
    param = sig.parameters[arg_key]
    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
      for i, (n, p) in enumerate(self._signature.parameters.items()):
        if p.kind not in (p.POSITIONAL, p.POSITIONAL_OR_KEYWORD):
          break
        if n == arg_key:
          arg_key = i
          break

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    ba = sig.bind(*args, **kwargs)
    # add default arguments
    for param in sig.parameters.values():
      if (param.name not in ba.arguments and param.default is not param.empty):
        ba.arguments[param.name] = param.default

    # replace argument `arg_key` with a derivative helper
    args = list(ba.args)
    kwargs = dict(ba.kwargs)
    keyargs = args if isinstance(arg_key,int) else kwargs
    orig = keyargs[arg_key]
    var = DerivativeTarget(orig.shape)
    keyargs[arg_key] = var

    # compute derivative and replace derivative helper with original argument
    return replace(var, orig, derivative(func(*args, **kwargs), var))

  return wrapper

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

def multiply( arg1, arg2 ):
  'multiply'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if arg1 == arg2:
    return power( arg1, 2 )

  retval = arg1._multiply(arg2)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._multiply' % arg1
    return retval

  retval = arg2._multiply(arg1)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._multiply' % arg2
    return retval

  return Multiply( Pair(arg1,arg2) )

def add( arg1, arg2 ):
  'add'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if iszero( arg1 ):
    return expand( arg2, shape )

  if iszero( arg2 ):
    return expand( arg1, shape )

  if arg1 == arg2:
    return arg1 * 2

  for idim, sh in enumerate( shape ):
    if sh == 1:
      return insert( add( get(arg1,idim,0), get(arg2,idim,0) ), idim )

  retval = arg1._add(arg2)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._add' % arg1
    return retval

  retval = arg2._add(arg1)
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._add' % arg2
    return retval

  return Add( Pair(arg1,arg2) )

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

def power( arg, n ):
  'power'

  arg, n = _matchndim( arg, n )
  shape = _jointshape( arg.shape, n.shape )

  if iszero( n ):
    return numpy.ones( shape )

  retval = arg._power(n)
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._power' % arg
    return retval

  return Power( arg, n )

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

def take( arg, index, axis ):
  'take index'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if isinstance( index, slice ):
    if index == slice(None):
      return arg
    assert index.step == None or index.step == 1
    if numeric.isint( arg.shape[axis] ):
      indexmask = numpy.zeros( arg.shape[axis], dtype=bool )
      indexmask[index] = True
      return mask( arg, indexmask, axis=axis )
    assert index.start == None or index.start >= 0
    assert index.stop != None and index.stop >= 0
    index = numpy.arange( index.start or 0, index.stop )

  if not isevaluable( index ):
    index = numpy.array( index )
    assert index.ndim == 1
    if index.dtype == bool:
      return mask( arg, index, axis )
    assert index.dtype == int
    index[ index < 0 ] += arg.shape[axis]
    assert numpy.all( (index>=0) & (index<arg.shape[axis]) ), 'indices out of bounds'

  index = asarray( index )
  assert index.ndim == 1
  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    index = find( index )
  else:
    assert index.dtype == int

  shape = list(arg.shape)
  shape[axis] = index.shape[0]

  if 0 in shape:
    return zeros( shape, dtype=arg.dtype )

  if index.isconstant:
    index_, = index.eval()
    if len(index_) == 1:
      return insert( get( arg, axis, index_[0] ), axis )
    if len(index_) == arg.shape[axis] and numpy.all(numpy.diff(index_) == 1):
      return arg

  retval = arg._take(index, axis)
  if retval is not None:
    assert retval.shape == tuple(shape)
    return retval

  return Take( arg, index, axis )

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

@log.title
def fdapprox( func, w, dofs, delta=1.e-5 ):
  '''Finite difference approximation of the variation of func in directions w
  around dofs. Input arguments:
  * func,  the functional to differentiate
  * dofs,  DOF vector of linearization point
  * w,     the function space or a tuple of chained spaces
  * delta, finite difference step scaling of ||dofs||_inf'''

  if not isinstance( w, tuple ): w = w,
  x0 = tuple( wi.dot( dofs ) for wi in w )
  step = numpy.linalg.norm( dofs, numpy.inf )*delta
  ndofs = len( dofs )
  dfunc_fd = []
  for i in log.range( 'dof', ndofs ):
    pert = dofs.copy()
    pert[i] += step
    x1 = tuple( wi.dot( pert ) for wi in w )
    dfunc_fd.append( (func( *x1 ) - func( *x0 ))/step )
  return dfunc_fd

def supp( funcsp, indices ):
  'find support of selection of basis functions'

  warnings.warn( 'function.supp is deprecated; use domain.supp instead', DeprecationWarning )
  transforms = []
  def collect_transforms( f ):
    if isinstance( f, DofMap ):
      transforms.append( set(f.dofmap) )
    return edit( f, collect_transforms )
  ind_funcs = [ collect_transforms( ind[0] ) for ind, f in funcsp.blocks ]
  common_transforms = functools.reduce( set.intersection, transforms )
  return [ trans for trans in common_transforms if any( numpy.intersect1d( ind.eval(trans)[0], indices, assume_unique=True ).size for ind in ind_funcs ) ]

def J( geometry, ndims=None ):
  if ndims is None:
    ndims = len(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian( geometry, ndims )

class Pair:
  '''two-element container that is insensitive to order in equality testing'''
  def __init__( self, a, b ):
    self.items = a, b
  def __iter__( self ):
    return iter( self.items )
  def __hash__( self ):
    return hash( tuple( sorted( hash(item) for item in self.items ) ) )
  def __eq__( self, other ):
    return isinstance( other, Pair ) and self.items in ( other.items, other.items[::-1] )

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

def replace( old, new, arg ):
  assert isarray( old )
  new = asarray( new )
  assert new.shape == old.shape
  d = { old: new }
  def s( f ):
    try:
      v = d[f]
    except KeyError:
      v = edit( f, s )
      if isarray( f ):
        assert v.shape == f.shape
      d[f] = v
    return v
  return s( arg )

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
