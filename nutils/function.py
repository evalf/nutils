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

from . import util, numpy, numeric, log, core, cache, transform, rational, _
import sys, warnings, itertools, functools, operator, inspect, numbers

CACHE = 'Cache'
TRANS = 'Trans'
POINTS = 'Points'

TOKENS = CACHE, TRANS, POINTS

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
      trans = None
      points = None
    elif isinstance( elem, transform.TransformChain ):
      trans = elem, elem
      points = ischeme
    elif isinstance( elem, tuple ):
      trans = elem
      points = ischeme
    else:
      trans = elem.transform, elem.opposite
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
      assert trans[0].fromdims == trans[1].fromdims
    if points is not None:
      assert points.ndim == 2 and points.shape[1] == trans[0].fromdims

    ops, inds = self.serialized
    assert TOKENS == ( CACHE, TRANS, POINTS )
    values = [ fcache, trans, points ]
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

    import os, subprocess

    dotpath = core.getprop( 'dot', True )
    if not isinstance( dotpath, str ):
      dotpath = 'dot'

    imgtype = core.getprop( 'imagetype', 'png' )
    imgpath = util.getpath( 'dot{0:03x}.' + imgtype )

    ops, inds = self.serialized

    lines = []
    lines.append( 'digraph {' )
    lines.append( 'graph [ dpi=72 ];' )
    lines.extend( '%d [label="%d. %s"];' % (i,i,name) for i, name in enumerate( TOKENS + ops + (self,) ) )
    lines.extend( '%d -> %d;' % (j,i) for i, indices in enumerate( ([],)*len(TOKENS) + inds ) for j in indices )
    lines.append( '}' )

    with open( imgpath, 'w' ) as img:
      with subprocess.Popen( [dotpath,'-T'+imgtype], stdin=subprocess.PIPE, stdout=img ) as dot:
        dot.communicate( '\n'.join(lines).encode() )

    log.path( os.path.basename(imgpath) )

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
    return self

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

class PointShape( Evaluable ):
  'shape of integration points'

  def __init__( self ):
    'constructor'

    return Evaluable.__init__( self, args=[POINTS] )

  def evalf( self, points ):
    'evaluate'

    return points.shape[:-1]

class TransformChain( Evaluable ):
  'transform'

  def __init__( self, side, promote ):
    Evaluable.__init__( self, args=[TRANS] )
    self.side = side
    self.promote = promote

  def evalf( self, trans ):
    return trans[ self.side ].promote( self.promote )


# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

class Array( Evaluable ):
  'array function'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__( self, args, shape, dtype ):
    'constructor'

    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    assert dtype is float or dtype is int or dtype is bool, 'invalid dtype {!r}'.format(dtype)
    self.dtype = dtype
    Evaluable.__init__( self, args=args )

  # mathematical operators

  def _if_array_args( op ):
    @functools.wraps( op )
    def wrapper( self, other ):
      if isinstance( other, ( numbers.Number, numpy.number, numpy.ndarray, Array, tuple, list ) ):
        return op( self, other )
      else:
        return NotImplemented
    return wrapper

  __mul__  = _if_array_args( lambda self, other: multiply( self, other ) )
  __rmul__ = _if_array_args( lambda self, other: multiply( other, self ) )
  __div__  = _if_array_args( lambda self, other: divide( self, other ) )
  __truediv__ = __div__
  __rdiv__ = _if_array_args( lambda self, other: divide( other, self ) )
  __rtruediv__ = __rdiv__
  __add__  = _if_array_args( lambda self, other: add( self, other ) )
  __radd__ = _if_array_args( lambda self, other: add( other, self ) )
  __sub__  = _if_array_args( lambda self, other: subtract( self, other ) )
  __rsub__ = _if_array_args( lambda self, other: subtract( other, self ) )
  __neg__  = lambda self: negative( self )
  __pow__  = _if_array_args( lambda self, n: power( self, n ) )
  __abs__  = lambda self: abs( self )
  __len__  = lambda self: self.shape[0]
  __mod__  = lambda self, other: mod( self, other )
  sum      = lambda self, axis: sum( self, axis )
  del _if_array_args

  @property
  def size( self ):
    return numpy.prod( self.shape, dtype=int )

  # standalone methods

  @property
  def blocks( self ):
    return [( Tuple([ asarray(numpy.arange(n)) if numeric.isint(n) else None for n in self.shape ]), self )]

  def vector( self, ndims ):
    'vectorize'

    return vectorize( [self] * ndims )

  def dot( self, weights, axis=0 ):
    'array contraction'

    weights = asarray( weights )#, dtype=float )
    assert weights.ndim == 1
    s = [ numpy.newaxis ] * self.ndim
    s[axis] = slice(None)
    return dot( self, weights[tuple(s)], axes=axis )

  def __getitem__( self, item ):
    'get item, general function which can eliminate, add or modify axes.'

    if isinstance( item, str ):
      from . import index
      return index.wrap( self, item )
    myitem = list( item if isinstance( item, tuple ) else [item] )
    n = 0
    arr = self
    while myitem:
      it = myitem.pop(0)
      eqsafe = not isinstance( it, numpy.ndarray ) # it is not an array, safe to use == comparison
      if numeric.isint(it): # retrieve one item from axis
        arr = get( arr, n, it )
      elif eqsafe and it == _: # insert a singleton axis
        arr = insert( arr, n )
        n += 1
      elif eqsafe and it == slice(None): # select entire axis
        n += 1
      elif eqsafe and it == Ellipsis: # skip to end
        remaining_items = len(myitem) - myitem.count(_)
        skip = arr.ndim - n - remaining_items
        assert skip >= 0, 'shape=%s, item=%s' % ( self.shape, _obj2str(item) )
        n += skip
      else:
        arr = take( arr, it, n )
        n += 1
      assert n <= arr.ndim
    return arr

  def __iter__( self ):
    'split first axis'

    if not self.shape:
      raise TypeError( 'scalar function is not iterable' )

    return ( self[i,...] for i in range(self.shape[0]) )

  def find( self, elem, C ):#target, start, tol=1e-10, maxiter=999 ):
    'iteratively find x for f(x) = target, starting at x=start'

    raise NotImplementedError
    assert self.ndim == 1
    points = start
    Jinv = inverse( localgradient( self, elem.ndims ) )
    r = target - self( elem, points )
    niter = 0
    while numpy.any( numeric.contract( r, r, axis=-1 ) > tol ):
      niter += 1
      if niter >= maxiter:
        raise Exception( 'failed to converge in %d iterations' % maxiter )
      points = points.offset( numeric.contract( Jinv( elem, points ), r[:,_,:], axis=-1 ) )
      r = target - self( elem, points )
    return points

  def normalized( self ):
    'normalize last axis'

    return self / norm2( self, axis=-1 )

  def normal( self, ndims=-1 ):
    'normal'

    assert len(self.shape) == 1
    if ndims <= 0:
      ndims += self.shape[0]

    grad = localgradient( self, ndims )
    if grad.shape == (2,1):
      normal = concatenate([ grad[1,:], -grad[0,:] ]).normalized()
    elif grad.shape == (3,2):
      normal = cross( grad[:,0], grad[:,1], axis=0 ).normalized()
    elif grad.shape == (3,1):
      normal = cross( grad[:,0], self.normal(), axis=0 ).normalized()
    elif grad.shape == (1,0):
      normal = [1]
    else:
      raise NotImplementedError( 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims ) )
    return normal * Orientation( ndims )

  def curvature( self, ndims=-1 ):
    'curvature'

    return self.normal().div( self, ndims=ndims )

    #if ndims <= 0:
    #  ndims += self.shape[0]
    #assert ndims == 1 and self.shape == (2,)
    #J = localgradient( self, ndims )
    #H = localgradient( J, ndims )
    #dx, dy = J[:,0]
    #ddx, ddy = H[:,0,0]
    #return ( dx * ddy - dy * ddx ) / norm2( J[:,0], axis=0 )**3

  def swapaxes( self, n1, n2 ):
    'swap axes'

    return swapaxes( self, (n1,n2) )

  def transpose( self, trans=None ):
    'transpose'

    return transpose( self, trans )

  def grad( self, coords, ndims=0 ):
    'gradient'

    assert coords.ndim == 1
    if ndims <= 0:
      ndims += coords.shape[0]
    J = localgradient( coords, ndims )
    if J.shape[0] == J.shape[1]:
      Jinv = inverse( J )
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      G = ( J[:,:,_] * J[:,_,:] ).sum( 0 )
      Ginv = inverse( G )
      Jinv = ( J[_,:,:] * Ginv[:,_,:] ).sum( -1 )
    else:
      raise Exception( 'cannot invert %sx%s jacobian' % J.shape )
    return sum( localgradient( self, ndims )[...,_] * Jinv, axis=-2 )

  def laplace( self, coords, ndims=0 ):
    'laplacian'

    return self.grad(coords,ndims).div(coords,ndims)

  def add_T( self, axes=(-2,-1) ):
    'add transposed'

    return add_T( self, axes )

  def symgrad( self, coords, ndims=0 ):
    'gradient'

    return .5 * add_T( self.grad( coords, ndims ) )

  def div( self, coords, ndims=0 ):
    'gradient'

    return trace( self.grad( coords, ndims ), -1, -2 )

  def dotnorm( self, coords, ndims=0, axis=-1 ):
    'normal component'

    axis = numeric.normdim( self.ndim, axis )
    normal = coords.normal( ndims-1 )
    assert normal.shape == (self.shape[axis],)
    return ( self * normal[(slice(None),)+(_,)*(self.ndim-axis-1)] ).sum( axis )

  def tangent( self, vec ):
    normal = self.normal()
    return vec - ( vec * normal ).sum(-1)[...,_] * normal

  def ngrad( self, coords, ndims=0 ):
    'normal gradient'

    return dotnorm( self.grad( coords, ndims ), coords, ndims )

  def nsymgrad( self, coords, ndims=0 ):
    'normal gradient'

    return dotnorm( self.symgrad( coords, ndims ), coords, ndims )

  @property
  def T( self ):
    'transpose'

    return transpose( self )

  def __str__( self ):
    'string representation'

    return '%s<%s>' % ( self.__class__.__name__, ','.join( str(n) for n in self.shape ) )

  __repr__ = __str__

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

  def _derivative( self, var, axes, seen ):
    return zeros( self.shape + _taketuple(var.shape,axes) )

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
    return asarray( op(self.value) )

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

  def _revolved( self ):
    return self

  def _sign( self ):
    return asarray( numeric.sign( self.value ) )

  def _choose( self, choices ):
    if all( isinstance( choice, Constant ) for choice in choices ):
      return asarray( numpy.choose( self.value, [ choice.value for choice in choices ] ) )

class DofMap( Array ):
  'dof axis'

  def __init__( self, dofmap, length, side=0 ):
    'new'

    self.side = side
    self.dofmap = dofmap
    for trans in dofmap:
      break

    Array.__init__( self, args=[TransformChain(side,trans.fromdims)], shape=(length,), dtype=int )

  def evalf( self, trans ):
    'evaluate'

    return self.dofmap[ trans.lookup(self.dofmap) ][_]

  def _opposite( self ):
    return DofMap( self.dofmap, self.shape[0], 1-self.side )

class ElementSize( Array):
  'dimension of hypercube with same volume as element'

  def __init__( self, geometry, ndims=None ):
    assert geometry.ndim == 1
    self.ndims = len(geometry) if ndims is None else len(geometry)+ndims if ndims < 0 else ndims
    iwscale = jacobian( geometry, self.ndims ) * Iwscale(self.ndims)
    Array.__init__( self, args=[iwscale], shape=(), dtype=float )

  def evalf( self, iwscale ):
    volume = iwscale.sum()
    return numeric.power( volume, 1/self.ndims )[_]

class Orientation( Array ):
  'sign'

  def __init__( self, ndims, side=0 ):
    'constructor'

    Array.__init__( self, args=[TransformChain(side,ndims)], shape=(), dtype=float )
    self.side = side
    self.ndims = ndims

  def evalf( self, trans ):
    head, tail = trans.split( self.ndims )
    return numpy.array([ head.orientation ])

  def _opposite( self ):
    return Orientation( self.ndims, 1-self.side )

  def _derivative( self, var, axes, seen ):
    return zeros( _taketuple(var.shape,axes) )

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

  def _derivative( self, var, axes, seen ):
    return align( derivative( self.func, var, axes, seen ), self.axes+tuple(range(self.ndim, self.ndim+len(axes))), self.ndim+len(axes) )

  def _multiply( self, other ):
    if len(self.axes) == self.ndim:
      other_trans = _call( other, '_align', _invtrans(self.axes), self.ndim )
      if other_trans is not None:
        return align( multiply( self.func, other_trans ), self.axes, self.ndim )
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func * other.func, self.axes, self.ndim )

  def _add( self, other ):
    if len(self.axes) == self.ndim:
      other_trans = _call( other, '_align', _invtrans(self.axes), self.ndim )
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
      trydot = _call( self.func, '_dot', transpose(other,self.axes), funcaxes )
      if trydot is not None:
        keep = numpy.ones( self.ndim, dtype=bool )
        keep[list(axes)] = False
        axes = [ _sum(keep[:axis]) for axis in self.axes if keep[axis] ]
        assert len(axes) == trydot.ndim
        return align( trydot, axes, len(axes) )

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

  def _derivative( self, var, axes, seen ):
    f = derivative( self.func, var, axes, seen )
    return get( f, self.axis, self.item )

  def _get( self, i, item ):
    tryget = _call( self.func, '_get', i+(i>=self.axis), item )
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

  def _derivative( self, var, axes, seen ):
    grad = derivative( self.func, var, axes, seen )
    funcs = stack( [ util.product( self.func[...,j] for j in range(self.func.shape[-1]) if i != j ) for i in range( self.func.shape[-1] ) ], axis=-1 )
    return ( grad * funcs[(...,)+(_,)*len(axes)] ).sum( self.ndim )

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #ext = (...,)+(_,)*len(shape)
    #return self[ext] * ( derivative(self.func,var,shape,seen) / self.func[ext] ).sum( self.ndim )

  def _get( self, i, item ):
    func = get( self.func, i, item )
    return product( func, -1 )

  def _edit( self, op ):
    return product( op(self.func), -1 )

class Iwscale( Array ):
  'integration weights'

  def __init__( self, ndims ):
    'constructor'

    self.fromdims = ndims
    Array.__init__( self, args=[TransformChain(0,ndims)], shape=(), dtype=float )

  def evalf( self, trans ):
    'evaluate'

    assert trans.fromdims == self.fromdims
    return _abs( numpy.asarray( trans.split(self.fromdims)[1].det, dtype=float )[_] )

class Transform( Array ):
  'transform'

  def __init__( self, todims, fromdims, side ):
    'constructor'

    assert fromdims != todims
    self.fromdims = fromdims
    self.todims = todims
    self.side = side
    Array.__init__( self, args=[TransformChain(side,fromdims)], shape=(todims,fromdims), dtype=float )

  def evalf( self, trans ):
    'transform'

    trans = trans.split(self.fromdims)[0].split(self.todims)[1]
    matrix = trans.linear
    assert matrix.shape == (self.todims,self.fromdims)
    return matrix.astype( float )[_]

  def _derivative( self, var, axes, seen ):
    return zeros( self.shape+_taketuple(var.shape,axes) )

  def _opposite( self ):
    return Transform( self.todims, self.fromdims, 1-self.side )

class Function( Array ):
  'function'

  def __init__( self, ndims, stdmap, igrad, length, side=0 ):
    'constructor'

    self.side = side
    self.ndims = ndims
    self.stdmap = stdmap
    self.igrad = igrad
    self.localcoords = LocalCoords( self.ndims, side=self.side ) # only an implicit dependency for now
    for trans in stdmap:
      break
    Array.__init__( self, args=(CACHE,POINTS,TransformChain(side,trans.fromdims)), shape=(length,)+(ndims,)*igrad, dtype=float )

  def evalf( self, cache, points, trans ):
    'evaluate'

    fvals = []
    head = trans.lookup( self.stdmap )
    for std, keep in self.stdmap[head]:
      if std:
        transpoints = cache[trans.slicefrom(len(head)).apply]( points )
        F = cache[std.eval]( transpoints, self.igrad )
        assert F.ndim == self.igrad+2
        if keep is not None:
          F = F[(Ellipsis,keep)+(slice(None),)*self.igrad]
        if self.igrad:
          invlinear = head.split(head.fromdims)[1].invlinear
          if invlinear.ndim:
            for axis in range(-self.igrad,0):
              F = numeric.dot( F, invlinear, axis )
          elif invlinear != 1:
            F = F * (invlinear**self.igrad)
        fvals.append( F )
      head = head.sliceto(-1)
    return fvals[0] if len(fvals) == 1 else numpy.concatenate( fvals, axis=-1-self.igrad )

  def _opposite( self ):
    return Function( self.ndims, self.stdmap, self.igrad, self.shape[0], 1-self.side )

  def _derivative( self, var, axes, seen ):
    grad = Function( self.ndims, self.stdmap, self.igrad+1, self.shape[0], self.side )
    return ( grad[(...,)+(_,)*len(axes)] * derivative( self.localcoords, var, axes, seen ) ).sum( self.ndim )

  def _take( self, indices, axis ):
    if axis != 0:
      return
    stdmap = {}
    for trans, stdkeep in self.stdmap.items():
      ind, = indices.eval( trans )
      assert all( numpy.diff( ind ) > 0 )
      nshapes = _sum( 0 if not std else std.nshapes if keep is None else keep.sum() for std, keep in stdkeep )
      where = numpy.zeros( nshapes, dtype=bool )
      where[ind] = True
      newstdkeep = []
      for std, keep in stdkeep:
        if std:
          if keep is None:
            n = std.nshapes
            keep = where[:n]
          else:
            n = keep.sum()
            keep = keep.copy()
            keep[keep] = where[:n]
          if not keep.any():
            std = None
          elif keep.all():
            keep = None
          where = where[n:]
        newstdkeep.append(( std, keep ))
      assert not where.size
      stdmap[trans] = newstdkeep
    return Function( self.ndims, stdmap, self.igrad, indices.shape[0], side=self.side )

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

  def _derivative( self, var, axes, seen ):
    grads = [ derivative( choice, var, axes, seen ) for choice in self.choices ]
    if not any( grads ): # all-zero special case; better would be allow merging of intervals
      return zeros( self.shape + _taketuple(var.shape,axes) )
    return choose( self.level[(...,)+(_,)*len(axes)], grads )

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

  def _derivative( self, var, axes, seen ):
    G = derivative( self.func, var, axes, seen )
    n = len(axes)
    a = slice(None)
    return -sum( self[(...,a,a,_,_)+(_,)*n] * G[(...,_,a,a,_)+(a,)*n] * self[(...,_,_,a,a)+(_,)*n], [-2-n, -3-n] )

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
    if any( n == None for n in lengths ):
      assert all( n == None for n in lengths )
      sh = None
    else:
      sh = _sum( lengths )
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

  def _derivative( self, var, axes, seen ):
    funcs = [ derivative( func, var, axes, seen ) for func in self.funcs ]
    return concatenate( funcs, axis=self.axis )

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
        j = _min( N1[ifun1+1], N2[ifun2+1] )
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

  def _inflate( self, dofmap, length, axis ):
    assert not isinstance( self.shape[axis], int )
    return concatenate( [ inflate(func,dofmap,length,axis) for func in self.funcs ], self.axis )

  def _take( self, indices, axis ):
    if axis != self.axis:
      return concatenate( [ take(aslength(func,self.shape[axis],axis),indices,axis) for func in self.funcs ], self.axis )
    if not indices.isconstant:
      raise NotImplementedError
    indices, = indices.eval()
    ifuncs = numpy.hstack([ numpy.repeat(ifunc,func.shape[axis]) for ifunc, func in enumerate(self.funcs) ])[indices]
    splits, = numpy.nonzero( numpy.diff(ifuncs) != 0 )
    funcs = []
    for i, j in zip( numpy.hstack([ 0, splits+1 ]), numpy.hstack([ splits+1, len(indices) ]) ):
      ifunc = ifuncs[i]
      assert numpy.all( ifuncs[i:j] == ifunc )
      offset = _sum( func.shape[axis] for func in self.funcs[:ifunc] )
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
    return concatenate( funcs, self.axis - _sum( axis < self.axis for axis in axes ) )

  def _power( self, n ):
    if n.shape[self.axis] != 1:
      raise NotImplementedError
    return concatenate( [ power( func, n ) for func in self.funcs ], self.axis )

  def _diagonalize( self ):
    if self.axis < self.ndim-1:
      return concatenate( [ diagonalize(func) for func in self.funcs ], self.axis )

  def _revolved( self ):
    return concatenate( [ revolved(func) for func in self.funcs ], self.axis )

  def _edit( self, op ):
    return concatenate( [ op(func) for func in self.funcs ], self.axis )

  def _kronecker( self, axis, length, pos ):
    return concatenate( [ kronecker(func,axis,length,pos) for func in self.funcs ], self.axis+(axis<=self.axis) )

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
    dtype = _jointdtype( func1.dtype, func2.dtype )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    self.axis_shiftright = axis-len(shape)
    Array.__init__( self, args=(func1,func2), shape=shape, dtype=dtype )

  def evalf( self, a, b ):
    assert a.ndim == b.ndim == self.ndim+1
    return numeric.cross( a, b, self.axis_shiftright )

  def _derivative( self, var, axes, seen ):
    ext = (...,)+(_,)*len(axes)
    return cross( self.func1[ext], derivative(self.func2,var,axes,seen), axis=self.axis ) \
         - cross( self.func2[ext], derivative(self.func1,var,axes,seen), axis=self.axis )

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

  def _derivative( self, var, axes, seen ):
    Finv = swapaxes( inverse( self.func ) )
    G = derivative( self.func, var, axes, seen )
    ext = (...,)+(_,)*len(axes)
    return self[ext] * sum( Finv[ext] * G, axis=[-2-len(axes),-1-len(axes)] )

  def _edit( self, op ):
    return determinant( op(self.func) )

class Multiply( Array ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
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
    if self.funcs[0].shape[-2:] == (1,1):
      return determinant( self.funcs[1] ) * (self.funcs[0][...,0,0]**self.shape[-1])
    if self.funcs[1].shape[-2:] == (1,1):
      return determinant( self.funcs[0] ) * (self.funcs[1][...,0,0]**self.shape[-1])

  def _product( self, axis ):
    func1, func2 = self.funcs
    return product( func1, axis ) * product( func2, axis )

  def _multiply( self, other ):
    func1, func2 = self.funcs
    func1_other = _call( func1, '_multiply', other )
    if func1_other is not None:
      return multiply( func1_other, func2 )
    func2_other = _call( func2, '_multiply', other )
    if func2_other is not None:
      return multiply( func1, func2_other )

  def _derivative( self, var, axes, seen ):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*len(axes)
    return func1[ext] * derivative( func2, var, axes, seen ) \
         + func2[ext] * derivative( func1, var, axes, seen )

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) * takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return take( aslength(func1,self.shape[axis],axis), index, axis ) * take( aslength(func2,self.shape[axis],axis), index, axis )

  def _power( self, n ):
    func1, func2 = self.funcs
    func1pow = _call( func1, '_power', n )
    func2pow = _call( func2, '_power', n )
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

class Add( Array ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
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

  def _derivative( self, var, axes, seen ):
    func1, func2 = self.funcs
    return derivative( func1, var, axes, seen ) + derivative( func2, var, axes, seen )

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
    func1_other = _call( func1, '_add', other )
    if func1_other is not None:
      return add( func1_other, func2 )
    func2_other = _call( func2, '_add', other )
    if func2_other is not None:
      return add( func1, func2_other )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return add( op(func1), op(func2) )

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

  def _derivative( self, var, axes, seen ):
    return blockadd( *( derivative( func, var, axes, seen ) for func in self.funcs ) )

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

  @property
  def blocks( self ):
    for func in self.funcs:
      for ind, f in blocks( func ):
        yield ind, f

class Dot( Array ):
  'dot'

  def __init__( self, func1, func2, naxes ):
    'constructor'

    assert _issorted( func1, func2 )
    assert isinstance( func1, Array )
    assert naxes > 0
    self.naxes = naxes
    self.funcs = func1, func2
    args = self.funcs[:1+isinstance( func2, Array )]
    shape = _jointshape( func1.shape, func2.shape )[:-naxes]
    Array.__init__( self, args=args, shape=shape, dtype=_jointdtype(func1.dtype,func2.dtype) )

  def evalf( self, arr1, arr2=None ):
    assert arr1.ndim == self.ndim+1+self.naxes
    return numeric.contract_fast( arr1, arr2 if arr2 is not None else self.funcs[1], self.naxes )

  @property
  def axes( self ):
    return list( range( self.ndim, self.ndim + self.naxes ) )

  def _get( self, axis, item ):
    func1, func2 = self.funcs
    return dot( get( aslength(func1,self.shape[axis],axis), axis, item ),
                get( aslength(func2,self.shape[axis],axis), axis, item ), [ ax-1 for ax in self.axes ] )

  def _derivative( self, var, axes, seen ):
    func1, func2 = self.funcs
    ext = (...,)+(_,)*len(axes)
    return dot( derivative( func1, var, axes, seen ), func2[ext], self.axes ) \
         + dot( func1[ext], derivative( func2, var, axes, seen ), self.axes )

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
        f, g12 = common
        tryconcat = _call( g12, '_concatenate', axis )
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
    trysum = _call( self.func, '_sum', axis+(axis>=self.axis) )
    if trysum is not None:
      return sum( trysum, self.axis )

  def _derivative( self, var, axes, seen ):
    return sum( derivative( self.func, var, axes, seen ), self.axis )

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

  def _derivative( self, var, axes, seen ):
    return Debug( derivative( self.func, var, axes, seen ) )

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

  def _derivative( self, var, axes, seen ):
    fder = derivative( self.func, var, axes, seen )
    return transpose( takediag( fder, self.func.ndim-2, self.func.ndim-1 ), tuple(range(self.func.ndim-2))+(-1,)+tuple(range(self.func.ndim-2,fder.ndim-2)) )

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

  def _derivative( self, var, axes, seen ):
    return take( derivative( self.func, var, axes, seen ), self.indices, self.axis )

  def _take( self, index, axis ):
    if axis == self.axis:
      return take( self.func, self.indices[index], axis )
    trytake = _call( self.func, '_take', index, axis )
    if trytake is not None:
      return take( trytake, self.indices, self.axis )

  def _edit( self, op ):
    return take( op(self.func), self.indices, self.axis )

  def _opposite( self ):
    return Take( opposite(self.func), opposite(self.indices), self.axis )

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

  def _derivative( self, var, axes, seen ):
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    ext = (...,)+(_,)*len(axes)
    powerm1 = choose( equal( self.power, 0 ), [ self.power-1, 0 ] ) # avoid introducing negative powers where possible
    return ( self.power * power( self.func, powerm1 ) )[ext] * derivative( self.func, var, axes, seen ) \
         + ( ln( self.func ) * self )[ext] * derivative( self.power, var, axes )

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
    shape = args.shape[1:]
    self.args = args
    self.evalfun = evalfun
    self.deriv = deriv
    Array.__init__( self, args=[args], shape=shape, dtype=dtype )

  def evalf( self, args ):
    assert args.shape[1:] == self.args.shape
    return self.evalfun( *args.swapaxes(0,1) )

  def _derivative( self, var, axes, seen ):
    if self.deriv is None:
      raise NotImplementedError( 'derivative is not defined for this operator' )
    return ( self.deriv( self.args )[(...,)+(_,)*len(axes)] * derivative( self.args, var, axes, seen ) ).sum( 0 )

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

  def _derivative( self, var, axes, seen ):
    return zeros( self.shape + _taketuple(var.shape,axes) )

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

class Pointdata( Array ):
  'pointdata'

  def __init__ ( self, data, shape ):
    'constructor'

    warnings.warn( 'Pointdata is deprecated; use Topology.elem_eval( ..., asfunction=True ) instead', DeprecationWarning )
    assert isinstance(data,dict)
    self.data = data
    for trans in data:
      break
    Array.__init__( self, args=[TransformChain(0,trans.fromdims),POINTS], shape=shape, dtype=float )

  def evalf( self, trans, points ):
    head = trans.lookup( self.data )
    tail = trans.slicefrom( len(head) )
    evalpoints = tail.apply( points )
    myvals, mypoints = self.data[head]
    assert numpy.equal( mypoints, evalpoints ).all(), 'Illegal point set'
    return myvals

  def update_max( self, func ):
    func = asarray(func)
    assert func.shape == self.shape
    data = dict( (trans,(numpy.maximum(func.eval((trans,trans),points),values),points)) for trans,(values,points) in self.data.items() )

    return Pointdata( data, self.shape )

class Sampled( Array ):
  'sampled'

  def __init__ ( self, data ):
    assert isinstance(data,dict)
    self.data = data.copy()
    items = iter(self.data.items())
    trans, (values,points) = next(items)
    fromdims = trans.fromdims
    shape = values.shape[1:]
    assert all( trans.fromdims == fromdims and values.shape == points.shape[:1]+shape for trans, (values,points) in items )
    Array.__init__( self, args=[TransformChain(0,fromdims),POINTS], shape=shape, dtype=float )

  def evalf( self, trans, points ):
    head = trans.lookup( self.data )
    tail = trans.slicefrom( len(head) )
    evalpoints = tail.apply( points )
    myvals, mypoints = self.data[head]
    assert numpy.equal( mypoints, evalpoints ).all(), 'Illegal point set'
    return myvals

class Elemwise( Array ):
  'elementwise constant data'

  def __init__( self, fmap, shape, default=None, side=0 ):
    self.fmap = fmap
    self.default = default
    self.side = side
    for trans in fmap:
      break
    Array.__init__( self, args=[TransformChain(side,trans.fromdims)], shape=shape, dtype=float )

  def evalf( self, trans ):
    trans = trans.lookup( self.fmap )
    value = self.fmap.get( trans, self.default )
    assert value is not None, 'transformation not found: {}'.format( trans )
    value = numpy.asarray( value )
    assert value.shape == self.shape, 'wrong shape: {} != {}'.format( value.shape, self.shape )
    return value[_]

  def _derivative( self, var, axes, seen ):
    return zeros( self.shape+_taketuple(var.shape,axes) )

  def _opposite( self ):
    return Elemwise( self.fmap, self.shape, self.default, 1-self.side )

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

  def _derivative( self, var, axes, seen ):
    return zeros( self.shape+_taketuple(var.shape,axes), dtype=self.dtype )

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
    sh = _max( self.shape[-2], self.shape[-1] )
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

  def _revolved( self ):
    return self

  def _kronecker( self, axis, length, pos ):
    return zeros( self.shape[:axis]+(length,)+self.shape[axis:], dtype=self.dtype )

class Inflate( Array ):
  'inflate'

  def __init__( self, func, dofmap, length, axis ):
    'constructor'

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

  def _inflate( self, dofmap, length, axis ):
    assert axis != self.axis
    if axis > self.axis:
      return
    return inflate( inflate( self.func, dofmap, length, axis ), self.dofmap, self.length, self.axis )

  def _derivative( self, var, axes, seen ):
    return inflate( derivative(self.func,var,axes,seen), self.dofmap, self.length, self.axis )

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
    return inflate( arr, self.dofmap, self.length, self.axis - _sum( axis < self.axis for axis in axes ) )

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
    assert index.isconstant
    index, = index.eval()
    renumber = numpy.empty( self.shape[axis], dtype=int )
    renumber[:] = -1
    renumber[index] = numpy.arange( len(index) )
    select = take( renumber != -1, self.dofmap, axis=0 )
    dofmap = take( renumber, take( self.dofmap, select, axis=0 ), axis=0 )
    return inflate( take( self.func, select, axis ), dofmap, len(index), self.axis )

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

  def _revolved( self ):
    return inflate( revolved(self.func), self.dofmap, self.length, self.axis )

  def _edit( self, op ):
    return inflate( op(self.func), op(self.dofmap), self.length, self.axis )

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

  def _derivative( self, var, axes, seen ):
    result = derivative( self.func, var, axes, seen )
    # move axis `self.ndim-1` to the end
    result = transpose( result, [ i for i in range(result.ndim) if i != self.func.ndim-1 ] + [ self.func.ndim-1 ] )
    # diagonalize last axis
    result = diagonalize( result )
    # move diagonalized axes left of the derivatives axes
    return transpose( result, tuple( range(self.func.ndim-1) ) + (result.ndim-2,result.ndim-1) + tuple( range(self.func.ndim-1,result.ndim-2) ) )

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

  def _derivative( self, var, axes, seen ):
    return repeat( derivative( self.func, var, axes, seen ), self.length, self.axis )

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
    return aslength( func, self.length, self.axis - _sum( axis < self.axis for axis in axes ) )

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

  def _derivative( self, var, axes, seen ):
    return Guard( derivative(self.fun,var,axes,seen) )

class TrigNormal( Array ):
  'cos, sin'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    Array.__init__( self, args=[angle], shape=(2,), dtype=float )

  def _derivative( self, var, axes, seen ):
    return TrigTangent( self.angle )[(...,)+(_,)*len(axes)] * derivative( self.angle, var, axes, seen )

  def evalf( self, angle ):
    return numpy.array([ numpy.cos(angle), numpy.sin(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return numpy.array( 1 if isinstance(other,TrigNormal) else 0 )

  def _opposite( self ):
    return TrigNormal( opposite(self.angle) )

  def _edit( self, op ):
    return TrigNormal( edit(self.angle,op) )

class TrigTangent( Array ):
  '-sin, cos'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    Array.__init__( self, args=[angle], shape=(2,), dtype=float )

  def _derivative( self, var, axes, seen ):
    return -TrigNormal( self.angle )[(...,)+(_,)*len(axes)] * derivative( self.angle, var, axes, seen )

  def evalf( self, angle ):
    return numpy.array([ -numpy.sin(angle), numpy.cos(angle) ]).T

  def _dot( self, other, axes ):
    assert axes == (0,)
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return numpy.array( 1 if isinstance(other,TrigTangent) else 0 )

  def _opposite( self ):
    return TrigTangent( opposite(self.angle) )

  def _edit( self, op ):
    return TrigTangent( edit(self.angle,op) )

class Find( Array ):
  'indices of boolean index vector'

  def __init__( self, where ):
    assert isarray(where) and where.ndim == 1 and where.dtype == bool
    Array.__init__( self, args=[where], shape=['~{}'.format(where.shape[0])], dtype=int )

  def evalf( self, where ):
    assert where.shape[0] == 1
    where, = where
    index, = where.nonzero()
    return index[_]

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

  def _derivative( self, var, axes, seen ):
    return kronecker( derivative( self.func, var, axes, seen ), self.axis, self.length, self.pos )

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
    return kronecker( power(self.func,n), self.axis, self.length, self.pos )

  def _pointwise( self, evalf, deriv, dtype ):
    value = evalf( *numpy.zeros(self.shape[0]) )
    assert value.dtype == dtype
    if value == 0:
      return kronecker( pointwise( self.func, evalf, deriv, dtype ), self.axis, self.length, self.pos )

  def _edit( self, op ):
    return kronecker( op(self.func), self.axis, self.length, self.pos )

class DerivativeTargetBase( Array ):
  'base class for derivative targets'

  pass

class DerivativeTarget( DerivativeTargetBase ):
  'helper class for computing derivatives'

  def __init__( self, shape ):
    DerivativeTargetBase.__init__( self, args=[], shape=shape, dtype=float )

  def evalf( self ):
    raise ValueError( 'unwrap {!r} before evaluation'.format( self ) )

  def _edit( self, op ):
    return self

  def _derivative( self, var, axes, seen ):
    if var is self:
      result = numpy.array(1)
      for i, axis in enumerate( axes ):
        result = result * align( eye( self.shape[axis] ), ( axis, self.ndim+i ), self.ndim+len(axes) )
      return result
    else:
      return zeros( self.shape+_taketuple(var.shape,axes) )

class LocalCoords( DerivativeTargetBase ):
  'trivial func'

  def __init__( self, ndims, side=0 ):
    'constructor'

    self.side = side
    DerivativeTargetBase.__init__( self, args=[POINTS,TransformChain(side,ndims)], shape=[ndims], dtype=float )

  def evalf( self, points, trans ):
    'evaluate'

    ptrans = trans.split( self.shape[0] )[1]
    return ptrans.apply( points ).astype( float )

  def _derivative( self, var, axes, seen ):
    if isinstance( var, LocalCoords ):
      ndims, = var.shape
      return eye( ndims ) if self.shape[0] == ndims \
        else Transform( self.shape[0], ndims, self.side )
    else:
      return zeros( self.shape+_taketuple(var.shape,axes) )

  def _opposite( self ):
    ndims, = self.shape
    return LocalCoords( ndims, 1-self.side )


# CIRCULAR SYMMETRY

class RevolutionAngle( Array ):
  'scalar with a 2pi gradient in highest local dimension'

  def __init__( self ):
    Array.__init__( self, args=[], shape=(), dtype=float )

  def evalf( self ):
    return numpy.zeros( [1] )

  def _derivative( self, var, axes, seen ):
    if isinstance( var, LocalCoords ):
      ndims, = var.shape
      lgrad = numpy.zeros( ndims )
      lgrad[-1] = 2*numpy.pi
      return lgrad
    else:
      return zeros( _taketuple(var.shape,axes) )

class Revolved( Array ):
  'implement an extra local dimension with zero gradient'

  def __init__( self, func ):
    assert isarray( func )
    self.func = func
    Array.__init__( self, args=[func], shape=func.shape, dtype=func.dtype )

  @property
  def blocks( self ):
    return self.func.blocks

  def evalf( self, func ):
    return func

  def _derivative( self, var, axes, seen ):
    if isinstance( var, LocalCoords ):
      newvar = LocalCoords( var.shape[0]-1 )
      return revolved( concatenate( [ derivative(self.func,newvar,axes,seen), zeros(self.func.shape+(1,)) ], axis=-1 ) )
    else:
      result = derivative( self.func, var, axes, seen )
      assert iszero( result )
      return result

  def _edit( self, op ):
    return revolved( op(self.func) )


# AUXILIARY FUNCTIONS (FOR INTERNAL USE)

_max = max
_min = min
_sum = sum
_abs = abs
_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )
_taketuple = lambda values, index: tuple( values[i] for i in index )
_issorted = lambda a, b: not isevaluable(b) or isevaluable(a) and id(a) <= id(b)
_sorted = lambda a, b: (a,b) if _issorted(a,b) else (b,a)

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
  itype = _max( kind_order.index(dtype.kind) if isinstance(dtype,numpy.dtype)
           else type_order.index(dtype) for dtype in dtypes )
  return type_order[itype]

def _matchndim( *arrays ):
  'introduce singleton dimensions to match ndims'

  arrays = [ asarray(array) for array in arrays ]
  ndim = _max( array.ndim for array in arrays )
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

def _call( obj, attr, *args ):
  'call method if it exists, return None otherwise'

  f = getattr( obj, attr, None )
  return f and f( *args )

def _norm_and_sort( ndim, args ):
  'norm axes, sort, and assert unique'

  normargs = tuple( sorted( numeric.normdim( ndim, arg ) for arg in args ) )
  assert _ascending( normargs ) # strict
  return normargs

def _unpack( funcsp ):
  for axes, func in funcsp.blocks:
    dofax = axes[0]
    if isinstance( func, Align ):
      func = func.func
    for trans, std in func.stdmap.items():
      dofs, = dofax.eval( trans )
      yield trans, dofs, std

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
trignormal = lambda angle: TrigNormal( asarray(angle) )
trigtangent = lambda angle: TrigTangent( asarray(angle) )
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
eye = lambda n: diagonalize( expand( [1.], (n,) ) )
norm2 = lambda arg, axis=-1: sqrt( sum( multiply( arg, arg ), axis ) )
heaviside = lambda arg: choose( greater( arg, 0 ), [0.,1.] )
divide = lambda arg1, arg2: multiply( arg1, reciprocal(arg2) )
subtract = lambda arg1, arg2: add( arg1, negative(arg2) )
mean = lambda arg: .5 * ( arg + opposite(arg) )
jump = lambda arg: opposite(arg) - arg
add_T = lambda arg, axes=(-2,-1): swapaxes( arg, axes ) + arg
edit = lambda arg, f: arg._edit(f) if isevaluable(arg) else arg
blocks = lambda arg: asarray(arg).blocks

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
  offset = 0 # ndofs = _sum( f.shape[0] for f in funcs )

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

  retval = _call( arg, '_repeat', length, axis )
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

  retval = _call( arg, '_get', iax, item )
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

  retval = _call( arg, '_align', axes, ndim )
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

  if arg.shape[axis] == 1:
    return get( arg, axis, 0 )

  retval = _call( arg, '_sum', axis )
  if retval is not None:
    assert retval.shape == arg.shape[:axis] + arg.shape[axis+1:], 'bug in %s._sum' % arg
    return retval

  return Sum( arg, axis )

def dot( arg1, arg2, axes ):
  'dot product'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if not util.isiterable(axes):
    axes = axes,

  if len(axes) == 0:
    return arg1 * arg2

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
      axis -= _sum( ax<axis for ax in axes )
      return align( dotargs, [ ax + (ax>=axis) for ax in range(dotargs.ndim) ], dotargs.ndim+1 )

  retval = _call( arg1, '_dot', arg2, axes )
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg1
    return retval

  retval = _call( arg2, '_dot', arg1, axes )
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg2
    return retval

  shuffle = list( range( len(shape) ) )
  for ax in reversed( axes ):
    shuffle.append( shuffle.pop(ax) )

  a, b = _sorted( transpose(arg1,shuffle), transpose(arg2,shuffle) )
  return Dot( a, b, len(axes) )

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

  retval = _call( arg, '_determinant' )
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

  retval = _call( arg, '_inverse' )
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

  retval = _call( arg, '_takediag' )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._takediag' % arg
    return retval

  return TakeDiag( arg )

def partial_derivative( func, arg_key, arg_axes=None ):
  '''partial derivative of a function

  Compute the partial derivative of `func` with respect to argument `arg_key`,
  limited to the axes `arg_axes` of argument `arg_key`.

  Parameters
  ----------
  func : function
  arg_key : int or str
      Reference to an argument of `func`.  If `arg_key` is an `int`, `arg_key`
      is the index of a positional argument of `func`.  If `arg_key` is a
      `str`, `arg_key` is the name of an argument of `func`.
  arg_axes : iterable of int, default all axes
      List of axes, where each axis should be in `[0,arg.ndim)`, where `arg` is
      the argument refered to by `arg_key`.

  Returns
  -------
  function
      Partial derivative of `func`.  The shape of this function is the
      concatenation of the shape of `func` and the shape of the `arg_axes` of
      `arg`, where `arg` is the argument refered to by `arg_key`.
  '''

  if not isinstance( arg_key, (int, str) ):
    raise ValueError( 'arg_key: expected an int or str, got {!r}'.format( arg_key ) )
  if arg_axes is not None:
    arg_axes = tuple(arg_axes) # evaluate iterator

  sig = inspect.signature( func )
  if isinstance( arg_key, str ) and arg_key in sig.parameters:
    # convert `arg_key` to index if possible
    param = sig.parameters[arg_key]
    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
      for i, (n, p) in enumerate( self._signature.parameters.items() ):
        if p.kind not in (p.POSITIONAL, p.POSITIONAL_OR_KEYWORD):
          break
        if n == arg_key:
          arg_key = i
          break

  @functools.wraps( func )
  def wrapper( *args, **kwargs ):
    ba = sig.bind( *args, **kwargs )
    # add default arguments
    for param in sig.parameters.values():
      if (param.name not in ba.arguments and param.default is not param.empty):
        ba.arguments[param.name] = param.default

    # replace argument `arg_key` with a derivative helper
    args = list(ba.args)
    kwargs = dict(ba.kwargs)
    keyargs = args if isinstance(arg_key,int) else kwargs
    orig = keyargs[arg_key]
    orig_axes = arg_axes if arg_axes is not None else tuple(range(orig.ndim))
    var = DerivativeTarget( orig.shape )
    keyargs[arg_key] = var

    # compute derivative and replace derivative helper with original argument
    replace = lambda f: orig if f is var else edit( f, replace )
    return replace( derivative( func( *args, **kwargs ), var, orig_axes ) )

  return wrapper

def derivative( func, var, axes, seen=None ):
  'derivative'

  assert isinstance( var, DerivativeTargetBase ), 'invalid derivative target {!r}'.format(var)
  if seen is None:
    seen = {}
  func = asarray( func )
  shape = _taketuple( var.shape, axes )
  if func in seen:
    result = seen[func]
  else:
    result = func._derivative( var, axes, seen )
    seen[func] = result
  assert result.shape == func.shape+shape, 'bug in %s._derivative' % func
  return result

def localgradient( arg, ndims ):
  'local derivative'

  return derivative( arg, LocalCoords(ndims), axes=(0,) )

def dotnorm( arg, coords, ndims=0 ):
  'normal component'

  return sum( arg * coords.normal( ndims-1 ), -1 )

def kronecker( arg, axis, length, pos ):
  'kronecker'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim+1, axis )
  assert 0 <= pos < length
  if length == 1:
    return insert( arg, axis )
  retval = _call( arg, '_kronecker', axis, length, pos )
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

  retval = _call( arg, '_diagonalize' )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._diagonalize' % arg
    return retval

  return Diagonalize( arg )

def concatenate( args, axis=0 ):
  'concatenate'

  args = _matchndim( *args )
  axis = numeric.normdim( args[0].ndim, axis )
  i = 0

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

  while i+1 < len(args):
    arg1, arg2 = args[i:i+2]
    arg12 = _call( arg1, '_concatenate', arg2, axis )
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

  retval = _call( aligned_arg, '_product', arg.ndim-1 )
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
  retval = _call( level, '_choose', choices )
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

  retval = _call( arg1, '_cross', arg2, axis )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._cross' % arg1
    return retval

  retval = _call( arg2, '_cross', arg1, axis )
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
  retval = _call( args, '_pointwise', evalf, deriv, dtype )
  if retval is not None:
    return retval
  return Pointwise( args, evalf, deriv, dtype )

def multiply( arg1, arg2 ):
  'multiply'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if arg1 == arg2:
    return power( arg1, 2 )

  for idim, sh in enumerate( shape ):
    if sh == 1:
      return insert( multiply( get(arg1,idim,0), get(arg2,idim,0) ), idim )

  retval = _call( arg1, '_multiply', arg2 )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._multiply' % arg1
    return retval

  retval = _call( arg2, '_multiply', arg1 )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._multiply' % arg2
    return retval

  return Multiply( *_sorted(arg1,arg2) )

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

  retval = _call( arg1, '_add', arg2 )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._add' % arg1
    return retval

  retval = _call( arg2, '_add', arg1 )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._add' % arg2
    return retval

  return Add( *_sorted(arg1,arg2) )

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

  retval = _call( arg, '_power', n )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._power' % arg
    return retval

  return Power( arg, n )

def sign( arg ):
  'sign'

  arg = asarray( arg )

  if isinstance( arg, numpy.ndarray ):
    return numpy.sign( arg )

  retval = _call( arg, '_sign' )
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

  # Use _call to see if the object has its own _eig function
  ret = _call( aligned_arg, '_eig', symmetric )
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

def revolved( arg ):
  arg = asarray( arg )
  retval = _call( arg, '_revolved' )
  if retval is not None:
    return retval
  return Revolved( arg )

def swapaxes( arg, axes=(-2,-1) ):
  'swap axes'

  arg = asarray( arg )
  n1, n2 = axes
  trans = numpy.arange( arg.ndim )
  trans[n1] = numeric.normdim( arg.ndim, n2 )
  trans[n2] = numeric.normdim( arg.ndim, n1 )
  return align( arg, trans, arg.ndim )

def opposite( arg ):
  'evaluate jump over interface'

  if not isevaluable( arg ):
    return arg

  retval = _call( arg, '_opposite' )
  if retval is not None:
    return retval

  return arg._edit( opposite )

def function( fmap, nmap, ndofs, ndims ):
  'create function on ndims-element'

  length = '~%d' % ndofs
  func = Function( ndims, fmap, igrad=0, length=length )
  dofmap = DofMap( nmap, length=length )
  return Inflate( func, dofmap, ndofs, axis=0 )

def take( arg, index, axis ):
  'take index'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if isinstance( index, slice ):
    if numeric.isint( arg.shape[axis] ):
      index = numpy.arange(arg.shape[axis])[index]
    else:
      assert index.start == None or index.start >= 0
      assert index.step == None or index.step == 1
      assert index.stop != None and index.stop >= 0
      index = numpy.arange( index.start or 0, index.stop )

  index = asarray( index )
  assert index.ndim == 1

  if index.dtype == bool:
    assert index.shape[0] == arg.shape[axis]
    index = find( index )

  assert index.dtype == int

  if index.isconstant:
    index_, = index.eval()
    if len(index_) == 1:
      return insert( get( arg, axis, index_[0] ), axis )
    if len(index_) == arg.shape[axis] and numpy.all(numpy.diff(index_) == 1):
      return arg

  retval = _call( arg, '_take', index, axis )
  if retval is not None:
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
  axis = numeric.normdim( arg.ndim, axis )

  retval = _call( arg, '_inflate', dofmap, length, axis )
  if retval is not None:
    return retval

  return Inflate( arg, dofmap, length, axis )

def pointdata ( topo, ischeme, func=None, shape=None, value=None ):
  'point data'

  from . import topology
  assert isinstance(topo,topology.Topology)

  if func is not None:
    assert value is None
    assert shape is None
    shape = func.shape
  else: # func is None
    if value is not None:
      assert shape is None
      value = numpy.asarray( value )
    else: # value is None
      assert shape is not None
      value = numpy.zeros( shape )
    shape = value.shape

  data = {}
  for elem in topo:
    # TODO use cache for getischeme
    ipoints, iweights = elem.reference.getischeme( ischeme )
    values = numpy.empty( ipoints.shape[:-1]+shape, dtype=float )
    values[:] = func.eval(elem,ischeme) if func is not None else value
    data[ elem.transform ] = values, ipoints

  return Pointdata( data, shape )

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

  # FRAGILE! makes lots of assumptions on the nature of funcsp
  supp = []
  for trans, dofs, stds in _unpack( funcsp ):
    for std, keep in stds:
      nshapes = 0 if not std \
           else keep.sum() if keep is not None \
           else std.nshapes
      if numpy.intersect1d( dofs[:nshapes], indices, assume_unique=True ).size:
        supp.append( trans )
      dofs = dofs[nshapes:]
      trans = trans[:-1]
    assert not dofs.size
  return supp

def J( geometry, ndims=None ):
  if ndims is None:
    ndims = len(geometry)
  elif ndims < 0:
    ndims += len(geometry)
  return jacobian( geometry, ndims ) * Iwscale(ndims)


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
