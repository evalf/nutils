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
:mod:`nutils.topology` onto Python space. The notabe class of :class:`ArrayFunc`
objects map onto the space of Numpy arrays of predefined dimension and shape.
Most functions used in nutils applicatons are of this latter type, including the
geometry and function bases for analysis.

Nutils functions are essentially postponed python functions, stored in a tree
structure of input/output dependencies. Many :class:`ArrayFunc` objects have
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

from __future__ import print_function, division
from . import util, numpy, numeric, log, core, cache, transform, rational, _
import sys, warnings, itertools, functools, operator

CACHE = 'Cache'
TRANS = 'Trans'
POINTS = 'Points'

TOKENS = CACHE, TRANS, POINTS

class Evaluable( cache.Immutable ):
  'Base class'

  def __init__( self, args ):
    'constructor'

    assert all( isinstance(arg,Evaluable) or arg in TOKENS for arg in args )
    self.__args = tuple(args)

  def evalf( self, *args ):
    raise NotImplementedError( 'Evaluable derivatives should implement the evalf method' )

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
      asciitree += '\n' + select[n] + ( ('\n' + bridge[n]).join( arg.asciitree( seen ).splitlines() ) if isinstance(arg,Evaluable) else '<{}>'.format(arg) )
    index = len(seen)
    seen.append( self )
    return '%{} = {}'.format( index, asciitree )

  def __str__( self ):
    return self.__class__.__name__

  def eval( self, elem, ischeme, fcache=cache.WrapperDummyCache() ):
    'evaluate'
    
    if isinstance( elem, tuple ):
      assert isinstance( ischeme, numpy.ndarray )
      points = ischeme
      transform, opposite = elem
      assert points.shape[-1] == transform.fromdims == opposite.fromdims
      trans = elem
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
        try: # python2/3
          exec( 'raise EvaluationError, excargs, traceback' )
        except SyntaxError:
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

    try:
      dot = subprocess.Popen( [dotpath,'-T'+imgtype], stdin=subprocess.PIPE, stdout=open(imgpath,'w') )
    except OSError:
      log.error( 'error: failed to execute', dotpath )
      return False

    print >> dot.stdin, 'digraph {'
    print >> dot.stdin, 'graph [ dpi=72 ];'
    dot.stdin.writelines( '%d [label="%d. %s"];\n' % (i,i,name)
      for i, name in enumerate( TOKENS + ops + (self,) ) )
    dot.stdin.writelines( '%d -> %d;\n' % (j,i)
      for i, indices in enumerate( ([],)*len(TOKENS) + inds )
        for j in indices )
    print >> dot.stdin, '}'
    dot.stdin.close()

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
      if isinstance( item, Evaluable ):
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


# INDEXVECTOR
#
# 1D int vector, used for indexing

class IndexVector( Evaluable ):

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__( self, args, length ):
    self.shape = length,
    self.ndim = 1
    Evaluable.__init__( self, args=args )

  def __str__( self ):
    return '%s<%s>' % ( self.__class__.__name__, ','.join( str(n) for n in self.shape ) )

class DofMap( IndexVector ):
  'dof axis'

  def __init__( self, dofmap, axis, target, side=0, offset=0 ):
    'new'

    self.side = side
    self.dofmap = dofmap
    self.offset = offset
    self.target = target
    for trans in dofmap:
      break

    IndexVector.__init__( self, args=[TransformChain(side,trans.fromdims)], length=axis )

  def __add__( self, offset ):
    assert numeric.isint( offset )
    return DofMap( self.dofmap, self.shape[0], self.target, self.side, self.offset+offset )

  def evalf( self, trans ):
    'evaluate'

    return self.dofmap[ trans.lookup(self.dofmap) ] + self.offset

  def _opposite( self ):
    return DofMap( self.dofmap, self.shape[0], self.target, 1-self.side )


# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

class ArrayFunc( Evaluable ):
  'array function'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__( self, args, shape, dtype=float ):
    'constructor'

    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    assert dtype is int or dtype is float
    self.dtype = dtype
    Evaluable.__init__( self, args=args )

  # mathematical operators

  __mul__  = lambda self, other: multiply( self, other )
  __rmul__ = lambda self, other: multiply( other, self )
  __div__  = lambda self, other: divide( self, other )
  __truediv__ = __div__
  __rdiv__ = lambda self, other: divide( other, self )
  __rtruediv__ = __rdiv__
  __add__  = lambda self, other: add( self, other )
  __radd__ = lambda self, other: add( other, self )
  __sub__  = lambda self, other: subtract( self, other )
  __rsub__ = lambda self, other: subtract( other, self )
  __neg__  = lambda self: negative( self )
  __pow__  = lambda self, n: power( self, n )
  __abs__  = lambda self: abs( self )
  __len__  = lambda self: self.shape[0]
  sum      = lambda self, axis=None: sum( self, axis )

  @property
  def size( self ):
    return numpy.prod( self.shape, dtype=int )

  # standalone methods

  @property
  def blocks( self ):
    return [( Tuple([ numpy.arange(n) if numeric.isint(n) else None for n in self.shape ]), self )]

  def vector( self, ndims ):
    'vectorize'

    return vectorize( [self] * ndims )

  def dot( self, weights, axis=0 ):
    'array contraction'

    weights = numpy.asarray( weights, dtype=float )
    assert weights.ndim == 1
    s = [ numpy.newaxis ] * self.ndim
    s[axis] = slice(None)
    return dot( self, weights[tuple(s)], axes=axis )

  def __getitem__( self, item ):
    'get item, general function which can eliminate, add or modify axes.'

    myitem = list( item if isinstance( item, tuple ) else [item] )
    n = 0
    arr = self
    while myitem:
      it = myitem.pop(0)
      if isinstance(it,numpy.ndarray): # numpy first because of 'equals issues'
        arr = take( arr, it, n )
        n += 1
      elif numeric.isint(it): # retrieve one item from axis
        arr = get( arr, n, it )
      elif it == _: # insert a singleton axis
        arr = insert( arr, n )
        n += 1
      elif it == slice(None): # select entire axis
        n += 1
      elif it == Ellipsis: # skip to end
        remaining_items = len(myitem) - myitem.count(_)
        skip = arr.ndim - n - remaining_items
        assert skip >= 0, 'shape=%s, item=%s' % ( self.shape, _obj2str(item) )
        n += skip
      elif isinstance(it,slice) and it.step in (1,None) and it.stop == ( it.start or 0 ) + 1: # special case: unit length slice
        arr = insert( get( arr, n, it.start or 0 ), n )
        n += 1
      elif isinstance(it,(slice,list,tuple)): # modify axis (shorten, extend or renumber one axis)
        arr = take( arr, it, n )
        n += 1
      else:
        raise NotImplementedError
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

class ElementSize( ArrayFunc ):
  'dimension of hypercube with same volume as element'

  def __init__( self, geometry, ndims=None ):
    assert geometry.ndim == 1
    self.ndims = len(geometry) if ndims is None else len(geometry)+ndims if ndims < 0 else ndims
    iwscale = jacobian( geometry, self.ndims ) * Iwscale(self.ndims)
    ArrayFunc.__init__( self, args=[iwscale], shape=() )

  def evalf( self, iwscale ):
    volume = iwscale.sum()
    return numpy.power( volume, 1/self.ndims )[_]

class Orientation( ArrayFunc ):
  'sign'

  def __init__( self, ndims, side=0 ):
    'constructor'

    ArrayFunc.__init__( self, args=[TransformChain(side,ndims)], shape=() )
    self.side = side
    self.ndims = ndims

  def evalf( self, trans ):
    head, tail = trans.split( self.ndims )
    return numpy.array([ head.orientation ])

  def _opposite( self ):
    return Orientation( self.ndims, 1-self.side )

  def _localgradient( self, ndims ):
    return _zeros( (ndims,) )

class Align( ArrayFunc ):
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
    ArrayFunc.__init__( self, args=[func], shape=shape )

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

  def _localgradient( self, ndims ):
    return align( localgradient( self.func, ndims ), self.axes+(self.ndim,), self.ndim+1 )

  def _multiply( self, other ):
    if not _isfunc(other) and len(self.axes) == other.ndim:
      return align( self.func * transpose( other, self.axes ), self.axes, self.ndim )
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func * other.func, self.axes, self.ndim )

  def _add( self, other ):
    if not _isfunc(other) and len(self.axes) == self.ndim:
      return align( self.func + transpose( other, self.axes ), self.axes, self.ndim )
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func + other.func, self.axes, self.ndim )

  def _take( self, indices, axis ):
    try:
      n = self.axes.index( axis )
    except ValueError:
      assert isinstance( indices, DofMap )
      return self
    return align( take( self.func, indices, n ), self.axes, self.ndim )

  def _edit( self, op ):
    return align( op(self.func), self.axes, self.ndim )

class Get( ArrayFunc ):
  'get'

  def __init__( self, func, axis, item ):
    'constructor'

    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    assert 0 <= item < func.shape[axis], 'item is out of bounds'
    self.item_shiftright = (Ellipsis,item) + (slice(None),)*(func.ndim-axis-1)
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func], shape=shape )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return arr[ self.item_shiftright ]

  def _localgradient( self, ndims ):
    f = localgradient( self.func, ndims )
    return get( f, self.axis, self.item )

  def _get( self, i, item ):
    tryget = _call( self.func, '_get', i+(i>=self.axis), item )
    if tryget is not None:
      return get( tryget, self.axis, self.item )

  def _take( self, indices, axis ):
    return get( take( self.func, indices, axis+(axis>=self.axis) ), self.axis, self.item )

  def _edit( self, op ):
    return get( op(self.func), self.axis, self.item )

class Product( ArrayFunc ):
  'product'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] > 1
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-1] )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numpy.product( arr, axis=-1 )

  def _localgradient( self, ndims ):
    grad = localgradient( self.func, ndims )
    funcs = stack( [ util.product( self.func[...,j] for j in range(self.func.shape[-1]) if i != j ) for i in range( self.func.shape[-1] ) ], axis=-1 )
    return ( grad * funcs[...,_] ).sum( -2 )

    ## this is a cleaner form, but is invalid if self.func contains zero values:
    #return self[...,_] * ( localgradient(self.func,ndims) / self.func[...,_] ).sum(-2)

  def _get( self, i, item ):
    func = get( self.func, i, item )
    return product( func, -1 )

  def _edit( self, op ):
    return product( op(self.func), -1 )

class Iwscale( ArrayFunc ):
  'integration weights'

  def __init__( self, ndims ):
    'constructor'

    self.fromdims = ndims
    ArrayFunc.__init__( self, args=[TransformChain(0,ndims)], shape=() )

  def evalf( self, trans ):
    'evaluate'

    assert trans.fromdims == self.fromdims
    return abs( numpy.asarray( trans.split(self.fromdims)[1].det, dtype=float )[_] )

class Transform( ArrayFunc ):
  'transform'

  def __init__( self, todims, fromdims, side ):
    'constructor'

    assert fromdims != todims
    self.fromdims = fromdims
    self.todims = todims
    self.side = side
    ArrayFunc.__init__( self, args=[TransformChain(side,fromdims)], shape=(todims,fromdims) )

  def evalf( self, trans ):
    'transform'

    trans = trans.split(self.fromdims)[0].split(self.todims)[1]
    matrix = trans.linear
    assert matrix.shape == (self.todims,self.fromdims)
    return matrix.astype( float )[_]

  def _localgradient( self, ndims ):
    return _zeros( self.shape + (ndims,) )

  def _opposite( self ):
    return Transform( self.todims, self.fromdims, 1-self.side )

class Function( ArrayFunc ):
  'function'

  def __init__( self, ndims, stdmap, igrad, axis, side=0 ):
    'constructor'

    self.side = side
    self.ndims = ndims
    self.stdmap = stdmap
    self.igrad = igrad
    for trans in stdmap:
      break
    ArrayFunc.__init__( self, args=(CACHE,POINTS,TransformChain(side,trans.fromdims)), shape=(axis,)+(ndims,)*igrad )

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

  def _localgradient( self, ndims ):
    grad = Function( self.ndims, self.stdmap, self.igrad+1, self.shape[0], self.side )
    return grad if ndims == self.ndims \
      else dot( grad[...,_], Transform( self.ndims, ndims, self.side ), axes=-2 )

  def _take( self, indices, axis ):
    if axis != 0:
      return
    assert isinstance( indices, DofMap )
    assert indices.shape[0] == self.shape[0]
    stdmap = {}
    for trans, stdkeep in self.stdmap.items():
      ind = indices.dofmap[trans]
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
    return Function( self.ndims, stdmap, self.igrad, indices.target, side=self.side )

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, level, choices ):
    'constructor'

    self.level = level
    self.choices = tuple( choices )
    shape = _jointshape( level.shape, *[ choice.shape for choice in choices ] )
    assert level.ndim == len( shape )
    self.ivar = [ i for i, choice in enumerate(choices) if isinstance(choice,ArrayFunc) ]
    ArrayFunc.__init__( self, args=[ level ] + [ choices[i] for i in self.ivar ], shape=shape )

  def evalf( self, level, *varchoices ):
    'choose'

    choices = [ choice[_] for choice in self.choices ]
    for i, choice in zip( self.ivar, varchoices ):
      choices[i] = choice
    assert all( choice.ndim == self.ndim+1 for choice in choices )
    return numpy.choose( level, choices )

  def _localgradient( self, ndims ):
    grads = [ localgradient( choice, ndims ) for choice in self.choices ]
    if not any( grads ): # all-zero special case; better would be allow merging of intervals
      return _zeros( self.shape + (ndims,) )
    return Choose( self.level[...,_], grads )

  def _edit( self, op ):
    return choose( op(self.level), [ op(choice) for choice in self.choices ] )

class Choose2D( ArrayFunc ):
  'piecewise function'

  def __init__( self, coords, contour, fin, fout ):
    'constructor'

    shape = _jointshape( fin.shape, fout.shape )
    self.contour = contour
    ArrayFunc.__init__( self, args=(coords,contour,fin,fout), shape=shape )

  @staticmethod
  def evalf( self, xy, fin, fout ):
    'evaluate'

    from matplotlib import nxutils
    mask = nxutils.points_inside_poly( xy.T, self.contour )
    out = numpy.empty( fin.shape or fout.shape )
    out[...,mask] = fin[...,mask] if fin.shape else fin
    out[...,~mask] = fout[...,~mask] if fout.shape else fout
    return out

class Inverse( ArrayFunc ):
  'inverse'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

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

  def _localgradient( self, ndims ):
    G = localgradient( self.func, ndims )
    H = sum( self[...,_,:,:,_]
              * G[...,:,:,_,:], -3 )
    I = sum( self[...,:,:,_,_]
              * H[...,_,:,:,:], -3 )
    return -I

  def _edit( self, op ):
    return inverse( op(self.func) )

class Concatenate( ArrayFunc ):
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
    self.funcs = tuple(funcs)
    self.ivar = [ i for i, func in enumerate(funcs) if isinstance(func,Evaluable) ]
    self.axis = axis
    self.axis_shiftright = axis-ndim
    ArrayFunc.__init__( self, args=[ funcs[i] for i in self.ivar ], shape=shape )

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
    axis = self.axis - (self.axis > i)
    return concatenate( [ get( f, i, item ) for f in self.funcs ], axis=axis )

  def _localgradient( self, ndims ):
    funcs = [ localgradient( func, ndims ) for func in self.funcs ]
    return concatenate( funcs, axis=self.axis )

  def _multiply( self, other ):
    funcs = []
    n0 = 0
    for func in self.funcs:
      n1 = n0 + func.shape[ self.axis ]
      funcs.append( func * take( other, slice(n0,n1), self.axis ) )
      n0 = n1
    assert n0 == self.shape[ self.axis ]
    return concatenate( funcs, self.axis )

  def _cross( self, other, axis ):
    if axis == self.axis:
      n = 1, 2, 0
      m = 2, 0, 1
      return take(self,n,axis) * take(other,m,axis) - take(self,m,axis) * take(other,n,axis)
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
        j = min( N1[ifun1+1], N2[ifun2+1] )
        funcs.append( take( self.funcs[ifun1], slice(i-N1[ifun1],j-N1[ifun1]), self.axis )
                    + take( other.funcs[ifun2], slice(i-N2[ifun2],j-N2[ifun2]), self.axis ))
        i = j
        ifun1 += i >= N1[ifun1+1]
        ifun2 += i >= N2[ifun2+1]
      assert ifun1 == len(self.funcs)
      assert ifun2 == len(other.funcs)
      return concatenate( funcs, axis=self.axis )
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

  def _inflate( self, dofmap, axis ):
    assert not isinstance( self.shape[axis], int )
    return concatenate( [ inflate(func,dofmap,axis) for func in self.funcs ], self.axis )

  def _take( self, indices, axis ):
    if axis != self.axis:
      return concatenate( [ take(func,indices,axis) for func in self.funcs ], self.axis )
    funcs = []
    while len(indices):
      n = 0
      for func in self.funcs:
        if n <= indices[0] < n + func.shape[axis]:
          break
        n += func.shape[axis]
      else:
        raise Exception( 'index out of bounds' )
      length = 1
      while length < len(indices) and n <= indices[length] < n + func.shape[axis]:
        length += 1
      funcs.append( take( func, indices[:length]-n, axis ) )
      indices = indices[length:]
    assert funcs, 'empty slice'
    if len( funcs ) == 1:
      return funcs[0]
    return concatenate( funcs, axis=axis )

  def _dot( self, other, naxes ):
    axes = range( self.ndim-naxes, self.ndim )
    n0 = 0
    funcs = []
    for f in self.funcs:
      n1 = n0 + f.shape[self.axis]
      funcs.append( dot( f, take( other, slice(n0,n1), self.axis ), axes ) )
      n0 = n1
    if self.axis >= self.ndim - naxes:
      return util.sum( funcs )
    return concatenate( funcs, self.axis )

  def _power( self, n ):
    return concatenate( [ power( func, n ) for func in self.funcs ], self.axis )

  def _diagonalize( self ):
    if self.axis < self.ndim-1:
      return concatenate( [ diagonalize(func) for func in self.funcs ], self.axis )

  def _revolved( self ):
    return concatenate( [ revolved(func) for func in self.funcs ], self.axis )

  def _edit( self, op ):
    return concatenate( [ op(func) for func in self.funcs ], self.axis )

class Interpolate( ArrayFunc ):
  'interpolate uniformly spaced data; stepwise for now'

  def __init__( self, x, xp, fp, left=None, right=None ):
    'constructor'

    xp = numpy.array( xp )
    fp = numpy.array( fp )
    assert xp.ndim == fp.ndim == 1
    if not numpy.all( numpy.diff(xp) > 0 ):
      warnings.warn( 'supplied x-values are non-increasing' )

    assert x.ndim == 0
    ArrayFunc.__init__( self, args=[x], shape=() )
    self.xp = xp
    self.fp = fp
    self.left = left
    self.right = right

  def evalf( self, x ):
    return numpy.interp( x, self.xp, self.fp, self.left, self.right )

class Cross( ArrayFunc ):
  'cross product'

  def __init__( self, func1, func2, axis ):
    'contructor'

    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    shape = _jointshape( func1.shape, func2.shape )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    self.axis_shiftright = axis-len(shape)
    ArrayFunc.__init__( self, args=(func1,func2), shape=shape )

  def evalf( self, a, b ):
    assert a.ndim == b.ndim == self.ndim+1
    return numeric.cross( a, b, self.axis_shiftright )

  def _localgradient( self, ndims ):
    return cross( self.func1[...,_], localgradient(self.func2,ndims), axis=self.axis ) \
         - cross( self.func2[...,_], localgradient(self.func1,ndims), axis=self.axis )

  def _take( self, index, axis ):
    if axis != self.axis:
      return cross( take(self.func1,index,axis), take(self.func2,index,axis), self.axis )

  def _edit( self, op ):
    return cross( op(self.func1), op(self.func2), self.axis )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, func ):
    'contructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-2] )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+3
    return numpy.linalg.det( arr )

  def _localgradient( self, ndims ):
    Finv = swapaxes( inverse( self.func ) )
    G = localgradient( self.func, ndims )
    return self[...,_] * sum( Finv[...,_] * G, axis=[-3,-2] )

  def _edit( self, op ):
    return determinant( op(self.func) )

class DofIndex( ArrayFunc ):
  'element-based indexing'

  def __init__( self, array, iax, index ):
    'constructor'

    assert index.ndim >= 1
    assert isinstance( array, numpy.ndarray )
    self.array = array
    assert 0 <= iax < self.array.ndim
    self.iax = iax
    self.index = index
    shape = self.array.shape[:iax] + index.shape + self.array.shape[iax+1:]
    ArrayFunc.__init__( self, args=[index], shape=shape )

  def evalf( self, index ):
    'evaluate'

    item = [ slice(None) ] * self.array.ndim
    item[self.iax] = index
    return self.array[ tuple(item) ][_]

  def _get( self, i, item ):
    if self.iax <= i < self.iax + self.index.ndim:
      index = get( self.index, i - self.iax, item )
      return take( self.array, index, self.iax )
    return take( get( self.array, i, item ), self.index, self.iax if i > self.iax else self.iax-1 )

  def _add( self, other ):
    if isinstance( other, DofIndex ) and self.iax == other.iax and self.index == other.index:
      return take( self.array + other.array, self.index, self.iax )

  def _multiply( self, other ):
    if not _isfunc(other) and other.ndim == 0:
      return take( self.array * other, self.index, self.iax )

  def _localgradient( self, ndims ):
    return _zeros( self.shape + (ndims,) )

  def _concatenate( self, other, axis ):
    if isinstance( other, DofIndex ) and self.iax == other.iax and self.index == other.index:
      array = numpy.concatenate( [ self.array, other.array ], axis )
      return take( array, self.index, self.iax )

  def _edit( self, op ):
    return take( self.array, op(self.index), self.iax )

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
    assert isinstance( func1, ArrayFunc )
    self.funcs = func1, func2
    args = self.funcs[:1+isinstance( func2, ArrayFunc )]
    shape = _jointshape( func1.shape, func2.shape )
    ArrayFunc.__init__( self, args=args, shape=shape )

  def evalf( self, arr1, arr2=None ):
    assert arr1.ndim == self.ndim+1
    if arr2 is None:
      return arr1 * self.funcs[1]
    assert arr2.ndim == self.ndim+1
    return arr1 * arr2

  def _sum( self, axis ):
    func1, func2 = self.funcs
    return dot( func1, func2, [axis] )

  def _get( self, i, item ):
    func1, func2 = self.funcs
    return get( func1, i, item ) * get( func2, i, item )

  def _add( self, other ):
    func1, func2 = self.funcs
    if _equal( other, func1 ):
      return func1 * (func2+1)
    if _equal( other, func2 ):
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
    if not _isfunc( other ) and not _isfunc( func2 ):
      return multiply( func1, numpy.multiply( func2, other ) )
    func1_other = _call( func1, '_multiply', other )
    if func1_other is not None:
      return multiply( func1_other, func2 )
    func2_other = _call( func2, '_multiply', other )
    if func2_other is not None:
      return multiply( func1, func2_other )

  def _localgradient( self, ndims ):
    func1, func2 = self.funcs
    return func1[...,_] * localgradient( func2, ndims ) \
         + func2[...,_] * localgradient( func1, ndims )

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) * takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return take( func1, index, axis ) * take( func2, index, axis )

  def _power( self, n ):
    func1, func2 = self.funcs
    if not _isfunc( func2 ):
      return multiply( power(func1,n), numpy.power(func2,n) )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return multiply( op(func1), op(func2) )

  def _dot( self, other, naxes ):
    func1, func2 = self.funcs
    if all( sh == 1 for sh in func1.shape[-naxes:] ):
      return func1[(Ellipsis,)+(0,)*naxes] * dot( func2, other, list(range(self.ndim-naxes,self.ndim)) )
    if all( sh == 1 for sh in func2.shape[-naxes:] ):
      return func2[(Ellipsis,)+(0,)*naxes] * dot( func1, other, list(range(self.ndim-naxes,self.ndim)) )


class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
    assert isinstance( func1, ArrayFunc )
    self.funcs = func1, func2
    args = self.funcs[:1+isinstance( func2, ArrayFunc )]
    shape = _jointshape( func1.shape, func2.shape )
    ArrayFunc.__init__( self, args=args, shape=shape )

  def evalf( self, arr1, arr2=None ):
    assert arr1.ndim == self.ndim+1
    return arr1 + ( arr2 if arr2 is not None else self.funcs[1] )

  def _sum( self, axis ):
    return sum( self.funcs[0], axis ) + sum( self.funcs[1], axis )

  def _localgradient( self, ndims ):
    func1, func2 = self.funcs
    return localgradient( func1, ndims ) + localgradient( func2, ndims )

  def _get( self, i, item ):
    func1, func2 = self.funcs
    return get( func1, i, item ) + get( func2, i, item )

  def _takediag( self ):
    func1, func2 = self.funcs
    return takediag( func1 ) + takediag( func2 )

  def _take( self, index, axis ):
    func1, func2 = self.funcs
    return take( func1, index, axis ) + take( func2, index, axis )

  def _add( self, other ):
    func1, func2 = self.funcs
    if not _isfunc( other ) and not _isfunc( func2 ):
      return add( func1, numpy.add( func2, other ) )
    func1_other = _call( func1, '_add', other )
    if func1_other is not None:
      return add( func1_other, func2 )
    func2_other = _call( func2, '_add', other )
    if func2_other is not None:
      return add( func1, func2_other )

  def _edit( self, op ):
    func1, func2 = self.funcs
    return add( op(func1), op(func2) )

class BlockAdd( ArrayFunc ):
  'block addition (used for DG)'

  def __init__( self, funcs ):
    'constructor'

    self.funcs = tuple( funcs )
    shape = _jointshape( *( func.shape for func in self.funcs ) )
    if not isinstance( funcs[-1], Evaluable ):
      self.const = funcs[-1]
      funcs = funcs[:-1]
    else:
      self.const = 0
    ArrayFunc.__init__( self, args=funcs, shape=shape )

  def evalf( self, *args ):
    assert all( arg.ndim == self.ndim+1 for arg in args )
    return functools.reduce( operator.add, args ) + self.const

  def _add( self, other ):
    return blockadd( self, other )

  def _dot( self, other, naxes ):
    n = numpy.arange( self.ndim-naxes, self.ndim )
    return blockadd( *( dot( func, other, n ) for func in self.funcs ) )

  def _edit( self, op ):
    return blockadd( *map( op, self.funcs ) )

  def _sum( self, axis ):
    return blockadd( *( sum( func, axis ) for func in self.funcs ) )

  def _localgradient( self, ndims ):
    return blockadd( *( localgradient( func, ndims ) for func in self.funcs ) )

  def _get( self, i, item ):
    return blockadd( *( get( func, i, item ) for func in self.funcs ) )

  def _takediag( self ):
    return blockadd( *( takediag( func ) for func in self.funcs ) )

  def _take( self, indices, axis ):
    return blockadd( *( take( func, indices, axis ) for func in self.funcs ) )

  def _align( self, axes, ndim ):
    return blockadd( *( align( func, axes, ndim ) for func in self.funcs ) )

  def _multiply( self, other ):
    return blockadd( *( multiply( func, other ) for func in self.funcs ) )

  def _inflate( self, dofmap, axis ):
    return blockadd( *( inflate( func, dofmap, axis ) for func in self.funcs ) )

  @property
  def blocks( self ):
    for func in self.funcs:
      for ind, f in blocks( func ):
        yield ind, f

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, naxes ):
    'constructor'

    assert _issorted( func1, func2 )
    assert isinstance( func1, ArrayFunc )
    assert naxes > 0
    self.naxes = naxes
    self.funcs = func1, func2
    args = self.funcs[:1+isinstance( func2, ArrayFunc )]
    shape = _jointshape( func1.shape, func2.shape )[:-naxes]
    ArrayFunc.__init__( self, args=args, shape=shape )

  def evalf( self, arr1, arr2=None ):
    assert arr1.ndim == self.ndim+1+self.naxes
    return numeric.contract_fast( arr1, arr2 if arr2 is not None else self.funcs[1], self.naxes )

  @property
  def axes( self ):
    return list( range( self.ndim, self.ndim + self.naxes ) )

  def _get( self, i, item ):
    func1, func2 = self.funcs
    return dot( get( func1, i, item ), get( func2, i, item ), [ ax-1 for ax in self.axes ] )

  def _localgradient( self, ndims ):
    func1, func2 = self.funcs
    return dot( localgradient( func1, ndims ), func2[...,_], self.axes ) \
         + dot( func1[...,_], localgradient( func2, ndims ), self.axes )

  def _multiply( self, other ):
    func1, func2 = self.funcs
    for ax in self.axes:
      other = insert( other, ax )
    assert other.ndim == func1.ndim == func2.ndim
    if not _isfunc( other ) and not _isfunc( func2 ):
      return dot( func1, numpy.multiply( func2, other ), self.axes )
    func1_other = _call( func1, '_multiply', other )
    if func1_other is not None:
      return dot( func1_other, func2, self.axes )
    func2_other = _call( func2, '_multiply', other )
    if func2_other is not None:
      return dot( func1, func2_other, self.axes )

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
    return dot( take(func1,index,axis), take(func2,index,axis), self.axes )

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

class Sum( ArrayFunc ):
  'sum'

  def __init__( self, func, axis ):
    'constructor'

    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    self.axis_shiftright = axis-func.ndim
    ArrayFunc.__init__( self, args=[func], shape=shape )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numpy.sum( arr, self.axis_shiftright )

  def _sum( self, axis ):
    trysum = _call( self.func, '_sum', axis+(axis>=self.axis) )
    if trysum is not None:
      return sum( trysum, self.axis )

  def _localgradient( self, ndims ):
    return sum( localgradient( self.func, ndims ), self.axis )

  def _edit( self, op ):
    return sum( op(self.func), axis=self.axis )

class Debug( ArrayFunc ):
  'debug'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  def evalf( self, arr ):
    'debug'

    assert arr.ndim == self.ndim+1
    log.debug( 'debug output:\n%s' % arr )
    return arr

  def __str__( self ):
    'string representation'

    return '{DEBUG}'

  def _localgradient( self, ndims ):
    return Debug( localgradient( self.func, ndims ) )

  def _edit( self, op ):
    return Debug( op(self.func) )

class TakeDiag( ArrayFunc ):
  'extract diagonal'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-1] )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+2
    return numeric.takediag( arr )

  def _localgradient( self, ndims ):
    return swapaxes( takediag( localgradient( self.func, ndims ), -3, -2 ) )

  def _sum( self, axis ):
    if axis != self.ndim-1:
      return takediag( sum( self.func, axis ) )

  def _edit( self, op ):
    return takediag( op(self.func) )

class Take( ArrayFunc ):
  'generalization of numpy.take(), to accept lists, slices, arrays'

  def __init__( self, func, indices, axis ):
    'constructor'

    assert func.shape[axis] != 1
    self.func = func
    self.axis = axis
    self.indices = indices

    args = [func]
    s = [ slice(None) ] * func.ndim

    if _isevaluable(indices):
      s[axis] = indices
      newlen, = indices.shape
      args.append( Tuple((Ellipsis,)+tuple(s)) )
    else:
      # try for regular slice
      start = indices[0]
      step = indices[1] - start
      stop = start + step * len(indices)
      s[axis] = slice( start, stop, step ) if numpy.all( numpy.diff(indices) == step ) else indices
      newlen, = numpy.empty( func.shape[axis] )[ indices ].shape
      assert newlen > 0
      self.item = (Ellipsis,)+tuple(s)

    shape = func.shape[:axis] + (newlen,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=args, shape=shape )

  def evalf( self, arr, item=None ):
    assert arr.ndim == self.ndim+1
    return arr[ item or self.item ]

  def _localgradient( self, ndims ):
    return take( localgradient( self.func, ndims ), self.indices, self.axis )

  def _take( self, index, axis ):
    if axis == self.axis:
      if numpy.all( numpy.diff( self.indices ) == 1 ):
        indices = index + self.indices[0]
      else:
        indices = self.indices[index]
      return take( self.func, indices, axis )
    trytake = _call( self.func, '_take', index, axis )
    if trytake is not None:
      return take( trytake, self.indices, self.axis )

  def _edit( self, op ):
    return take( op(self.func), self.indices, self.axis )

class Power( ArrayFunc ):
  'power'

  def __init__( self, func, power ):
    'constructor'

    self.func = func
    self.power = power
    shape = _jointshape( func.shape, power.shape )
    self.varbase = isinstance( func, ArrayFunc )
    self.varexp = isinstance( power, ArrayFunc )
    assert self.varbase or self.varexp
    args = ([func] if self.varbase else []) + ([power] if self.varexp else [])
    ArrayFunc.__init__( self, args=args, shape=shape )

  def evalf( self, *args ):
    return numpy.power( args[0] if self.varbase else self.func,
                        args[-1] if self.varexp else self.power )

  def _localgradient( self, ndims ):
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    powerm1 = self.power-1 if _isfunc(self.power) else numpy.choose( self.power==0, [self.power-1,0] ) # avoid introducing negative powers where possible
    return ( self.power * power( self.func, powerm1 ) )[...,_] * localgradient( self.func, ndims ) \
         + ( ln( self.func ) * self )[...,_] * localgradient( self.power, ndims )

  def _power( self, n ):
    func = self.func
    newpower = n * self.power
    if _iszero( self.power % 2 ) and not _iszero( newpower % 2 ):
      func = abs( func )
    return power( func, newpower )

  def _get( self, i, item ):
    return get( self.func, i, item )**get( self.power, i, item )

  def _sum( self, axis ):
    if not _isfunc(self.power) and numpy.all( self.power == 2 ):
      return dot( self.func, self.func, axis )

  def _takediag( self ):
    return power( takediag( self.func ), takediag( self.power ) )

  def _take( self, index, axis ):
    return power( take( self.func, index, axis ), take( self.power, index, axis ) )

  def _multiply( self, other ):
    if isinstance( other, Power ) and self.func == other.func:
      return power( self.func, self.power + other.power )
    if other == self.func:
      return power( self.func, self.power + 1 )

  def _sign( self ):
    if _iszero( self.power % 2 ):
      return expand( 1., self.shape )

  def _edit( self, op ):
    return power( op(self.func), op(self.power) )

class ElemFunc( ArrayFunc ):
  'trivial func'

  def __init__( self, ndims, side=0 ):
    'constructor'

    self.side = side
    ArrayFunc.__init__( self, args=[POINTS,TransformChain(side,ndims)], shape=[ndims] )

  def evalf( self, points, trans ):
    'evaluate'

    ptrans = trans.split( self.shape[0] )[1]
    return ptrans.apply( points ).astype( float )

  def _localgradient( self, ndims ):
    return eye( ndims ) if self.shape[0] == ndims \
      else Transform( self.shape[0], ndims, self.side )

  def _opposite( self ):
    ndims, = self.shape
    return ElemFunc( ndims, 1-self.side )

class Pointwise( ArrayFunc ):
  'pointwise transformation'

  def __init__( self, args, evalfun, deriv ):
    'constructor'

    assert _isfunc( args )
    shape = args.shape[1:]
    self.args = args
    self.evalfun = evalfun
    self.deriv = deriv
    ArrayFunc.__init__( self, args=[args], shape=shape )

  def evalf( self, args ):
    assert args.shape[1:] == self.args.shape
    return self.evalfun( *args.swapaxes(0,1) )

  def _localgradient( self, ndims ):
    return ( self.deriv( self.args )[...,_] * localgradient( self.args, ndims ) ).sum( 0 )

  def _takediag( self ):
    return pointwise( takediag(self.args), self.evalfun, self.deriv )

  def _get( self, axis, item ):
    return pointwise( get( self.args, axis+1, item ), self.evalfun, self.deriv )

  def _take( self, index, axis ):
    return pointwise( take( self.args, index, axis+1 ), self.evalfun, self.deriv )

  def _edit( self, op ):
    return pointwise( op(self.args), self.evalfun, self.deriv )

class Sign( ArrayFunc ):
  'sign'

  def __init__( self, func ):
    'constructor'

    assert _isfunc( func )
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  def evalf( self, arr ):
    assert arr.ndim == self.ndim+1
    return numpy.sign( arr )

  def _localgradient( self, ndims ):
    return _zeros( self.shape + (ndims,) )

  def _takediag( self ):
    return sign( takediag(self.func) )

  def _get( self, axis, item ):
    return sign( get( self.func, axis, item ) )

  def _take( self, index, axis ):
    return sign( take( self.func, index, axis ) )

  def _sign( self ):
    return self

  def _power( self, n ):
    if _iszero( n % 2 ):
      return expand( 1., self.shape )

  def _edit( self, op ):
    return sign( op(self.func) )

class Pointdata( ArrayFunc ):
  'pointdata'

  def __init__ ( self, data, shape ):
    'constructor'

    warnings.warn( 'Pointdata is deprecated; use Topology.elem_eval( ..., asfunction=True ) instead', DeprecationWarning )
    assert isinstance(data,dict)
    self.data = data
    for trans in data:
      break
    ArrayFunc.__init__( self, args=[TransformChain(0,trans.fromdims),POINTS], shape=shape )

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

class Sampled( ArrayFunc ):
  'sampled'

  def __init__ ( self, data ):
    assert isinstance(data,dict)
    self.data = data.copy()
    items = iter(self.data.items())
    trans, (values,points) = next(items)
    fromdims = trans.fromdims
    shape = values.shape[1:]
    assert all( trans.fromdims == fromdims and values.shape == points.shape[:1]+shape for trans, (values,points) in items )
    ArrayFunc.__init__( self, args=[TransformChain(0,fromdims),POINTS], shape=shape )

  def evalf( self, trans, points ):
    head = trans.lookup( self.data )
    tail = trans.slicefrom( len(head) )
    evalpoints = tail.apply( points )
    myvals, mypoints = self.data[head]
    assert numpy.equal( mypoints, evalpoints ).all(), 'Illegal point set'
    return myvals

class Elemwise( ArrayFunc ):
  'elementwise constant data'

  def __init__( self, fmap, shape, default=None, side=0 ):
    self.fmap = fmap
    self.default = default
    self.side = side
    for trans in fmap:
      break
    ArrayFunc.__init__( self, args=[TransformChain(side,trans.fromdims)], shape=shape )

  def evalf( self, trans ):
    trans = trans.lookup( self.fmap )
    value = self.fmap.get( trans, self.default )
    assert value is not None, 'transformation not found: {}'.format( trans )
    value = numpy.asarray( value )
    assert value.shape == self.shape, 'wrong shape: {} != {}'.format( value.shape, self.shape )
    return value[_]

  def _localgradient( self, ndims ):
    return _zeros( self.shape+(ndims,) )

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

class ArrayFromTuple( ArrayFunc ):
  'array from tuple'

  def __init__( self, arrays, index, shape ):
    self.arrays = arrays
    self.index = index
    ArrayFunc.__init__( self, args=[arrays], shape=shape )

  def evalf( self, arrays ):
    return arrays[ self.index ]

  def _edit( self, op ):
    return array_from_tuple( op(self.arrays), self.index, self.shape )

class Zeros( ArrayFunc ):
  'zero'

  def __init__( self, shape ):
    'constructor'

    shape = tuple( shape )
    ArrayFunc.__init__( self, args=[], shape=shape )

  def evalf( self ):
    'prepend point axes'

    assert not any( sh is None for sh in self.shape ), 'cannot evaluate zeros for shape %s' % (self.shape,)
    return numpy.zeros( (1,) + self.shape )

  @property
  def blocks( self ):
    return ()

  def _repeat( self, length, axis ):
    assert self.shape[axis] == 1
    return _zeros( self.shape[:axis] + (length,) + self.shape[axis+1:] )

  def _localgradient( self, ndims ):
    return _zeros( self.shape+(ndims,) )

  def _add( self, other ):
    shape = _jointshape( self.shape, other.shape )
    return expand( other, shape )

  def _multiply( self, other ):
    shape = _jointshape( self.shape, other.shape )
    return _zeros( shape )

  def _dot( self, other, naxes ):
    shape = _jointshape( self.shape, other.shape )
    return _zeros( shape[:-naxes] )

  def _cross( self, other, axis ):
    shape = _jointshape( self.shape, other.shape )
    return _zeros( shape )

  def _diagonalize( self ):
    return _zeros( self.shape + (self.shape[-1],) )

  def _sum( self, axis ):
    return _zeros( self.shape[:axis] + self.shape[axis+1:] )

  def _align( self, axes, ndim ):
    shape = [1] * ndim
    for ax, sh in zip( axes, self.shape ):
      shape[ax] = sh
    return _zeros( shape )

  def _get( self, i, item ):
    return _zeros( self.shape[:i] + self.shape[i+1:] )

  def _takediag( self ):
    sh = max( self.shape[-2], self.shape[-1] )
    return _zeros( self.shape[:-2] + (sh,) )

  def _take( self, index, axis ):
    return _zeros( self.shape[:axis] + index.shape + self.shape[axis+1:] )

  def _inflate( self, dofmap, axis ):
    assert not isinstance( self.shape[axis], int )
    return _zeros( self.shape[:axis] + (dofmap.target,) + self.shape[axis+1:] )

  def _power( self, n ):
    return self

  def _pointwise( self, evalf, deriv ):
    value = evalf( *numpy.zeros(self.shape[0]) )
    if value == 0:
      return _zeros( self.shape[1:] )
    return expand( numpy.array(value)[(_,)*(self.ndim-1)], self.shape[1:] )

class Inflate( ArrayFunc ):
  'inflate'

  def __init__( self, func, dofmap, axis ):
    'constructor'

    self.func = func
    self.dofmap = dofmap
    self.axis = axis
    shape = func.shape[:axis] + (dofmap.target,) + func.shape[axis+1:]
    self.axis_shiftright = axis-func.ndim
    ArrayFunc.__init__( self, args=[func,dofmap], shape=shape )

  def evalf( self, array, indices ):
    'inflate'

    assert array.ndim == self.ndim+1
    warnings.warn( 'using explicit inflation; this is usually a bug.' )
    shape = list( array.shape )
    shape[self.axis_shiftright] = self.dofmap.target
    inflated = numpy.zeros( shape )
    inflated[(Ellipsis,indices)+(slice(None),)*(-self.axis_shiftright-1)] = array
    return inflated

  @property
  def blocks( self ):
    for ind, f in blocks( self.func ):
      assert ind[self.axis] == None
      yield Tuple( ind[:self.axis] + (self.dofmap,) + ind[self.axis+1:] ), f

  def _inflate( self, dofmap, axis ):
    assert axis != self.axis
    if axis > self.axis:
      return
    return inflate( inflate( self.func, dofmap, axis ), self.dofmap, self.axis )

  def _localgradient( self, ndims ):
    return inflate( localgradient(self.func,ndims), self.dofmap, self.axis )

  def _align( self, shuffle, ndims ):
    return inflate( align(self.func,shuffle,ndims), self.dofmap, shuffle[self.axis] )

  def _get( self, axis, item ):
    assert axis != self.axis
    return inflate( get(self.func,axis,item), self.dofmap, self.axis-(axis<self.axis) )

  def _dot( self, other, naxes ):
    axes = range( self.ndim-naxes, self.ndim )
    if isinstance( other, Inflate ) and other.axis == self.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    arr = dot( self.func, other, axes )
    if self.axis >= self.ndim - naxes:
      return arr
    return inflate( arr, self.dofmap, self.axis )

  def _multiply( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    return inflate( multiply(self.func,other), self.dofmap, self.axis )

  def _add( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis and self.dofmap == other.dofmap:
      return inflate( add(self.func,other.func), self.dofmap, self.axis )
    return blockadd( self, other )

  def _cross( self, other, axis ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    return inflate( cross(self.func,other,axis), self.dofmap, self.axis )

  def _power( self, n ):
    return inflate( power(self.func,n), self.dofmap, self.axis )

  def _takediag( self ):
    assert self.axis < self.ndim-2
    return inflate( takediag(self.func), self.dofmap, self.axis )

  def _take( self, index, axis ):
    if axis == self.axis:
      if index == self.dofmap:
        return self.func
      assert numeric.isintarray(index) and index.ndim == 1
      if self.dofmap.offset != 0:
        raise NotImplementedError
      reverse_index = numpy.empty( self.shape[axis], dtype=int )
      reverse_index[:] = -1
      reverse_index[index] = numpy.arange( len(index) )
      globaldofs = {}
      localdofs = {}
      for trans, dofs in self.dofmap.dofmap.items():
        newdofs = reverse_index[dofs]
        keep = newdofs != -1
        globaldofs[trans] = newdofs[keep]
        localdofs[trans], = numpy.where(keep)
      strlen = '~%d'%len(index)
      dofmap = DofMap( globaldofs, axis=strlen, target=len(index), side=self.dofmap.side )
      index = DofMap( localdofs, axis=self.dofmap.shape[0], target=strlen, side=self.dofmap.side )
    else:
      dofmap = self.dofmap
    return inflate( take( self.func, index, axis ), dofmap, self.axis )

  def _diagonalize( self ):
    assert self.axis < self.ndim-1
    return inflate( diagonalize(self.func), self.dofmap, self.axis )

  def _sum( self, axis ):
    arr = sum( self.func, axis )
    if axis == self.axis:
      return arr
    return inflate( arr, self.dofmap, self.axis-(axis<self.axis) )

  def _repeat( self, length, axis ):
    if axis != self.axis:
      return inflate( repeat(self.func,length,axis), self.dofmap, self.axis )

  def _revolved( self ):
    return inflate( revolved(self.func), self.dofmap, self.axis )

  def _edit( self, op ):
    return inflate( op(self.func), op(self.dofmap), self.axis )

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  def __init__( self, func ):
    'constructor'

    n = func.shape[-1]
    assert n != 1
    shape = func.shape + (n,)
    self.func = func
    ArrayFunc.__init__( self, args=[func] if isinstance(func,ArrayFunc) else [], shape=shape )

  def evalf( self, arr=None ):
    assert arr is None or arr.ndim == self.ndim
    return numeric.diagonalize( arr if arr is not None else self.func[_] )

  def _localgradient( self, ndims ):
    return swapaxes( diagonalize( swapaxes( localgradient( self.func, ndims ), (-2,-1) ) ), (-3,-1) )

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

  def _add( self, other ):
    if isinstance( other, Diagonalize ):
      return diagonalize( self.func + other.func )

  def _sum( self, axis ):
    if axis >= self.ndim-2:
      return self.func
    return diagonalize( sum( self.func, axis ) )

  def _align( self, axes, ndim ):
    if axes[-2:] in [ (ndim-2,ndim-1), (ndim-1,ndim-2) ]:
      return diagonalize( align( self.func, axes[:-2] + (ndim-2,), ndim-1 ) )

  def _edit( self, op ):
    return diagonalize( op(self.func) )

  def _takediag( self ):
    return self.func

class Repeat( ArrayFunc ):
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
    ArrayFunc.__init__( self, args=[func] if isinstance(func,ArrayFunc) else [], shape=shape )

  def evalf( self, arr=None ):
    assert arr is None or arr.ndim == self.ndim+1
    return numeric.fastrepeat( arr if arr is not None else self.func[_], self.length, self.axis_shiftright )

  def _localgradient( self, ndims ):
    return repeat( localgradient( self.func, ndims ), self.length, self.axis )

  def _get( self, axis, item ):
    if axis == self.axis:
      assert 0 <= item < self.length
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

  def _dot( self, other, naxes ):
    axes = range( self.ndim-naxes, self.ndim )
    func = dot( self.func, other, axes )
    if other.shape[self.axis] != 1:
      assert other.shape[self.axis] == self.length
      return func
    if self.axis >= self.ndim - naxes:
      return func * self.length
    return aslength( func, self.length, self.axis )

  def _edit( self, op ):
    return repeat( op(self.func), self.length, self.axis )

  def _concatenate( self, other, axis ):
    if axis == self.axis:
      return
    if isinstance( other, Repeat ):
      return aslength( aslength( concatenate( [self.func,other.func], axis ), self.length, self.axis ), other.length, other.axis )
    return aslength( concatenate( [self.func,other], axis ), self.length, self.axis )

class Guard( ArrayFunc ):
  'bar all simplifications'

  def __init__( self, fun ):
    self.fun = fun
    ArrayFunc.__init__( self, args=[fun], shape=fun.shape )

  @staticmethod
  def evalf( dat ):
    return dat

  def _edit( self, op ):
    return Guard( op(self.fun) )

  def _localgradient( self, ndims ):
    return Guard( localgradient(self.fun,ndims) )

class TrigNormal( ArrayFunc ):
  'cos, sin'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    ArrayFunc.__init__( self, args=[angle], shape=(2,) )

  def _localgradient( self, ndims ):
    return TrigTangent( self.angle )[:,_] * localgradient( self.angle, ndims )

  def evalf( self, angle ):
    return numpy.array([ numpy.cos(angle), numpy.sin(angle) ]).T

  def _dot( self, other, naxes ):
    assert naxes == 1
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return numpy.array( 1 if isinstance(other,TrigNormal) else 0 )

  def _opposite( self ):
    return TrigNormal( opposite(self.angle) )

  def _edit( self, op ):
    return TrigNormal( edit(self.angle,op) )

class TrigTangent( ArrayFunc ):
  '-sin, cos'

  def __init__( self, angle ):
    assert angle.ndim == 0
    self.angle = angle
    ArrayFunc.__init__( self, args=[angle], shape=(2,) )

  def _localgradient( self, ndims ):
    return -TrigNormal( self.angle )[:,_] * localgradient( self.angle, ndims )

  def evalf( self, angle ):
    return numpy.array([ -numpy.sin(angle), numpy.cos(angle) ]).T

  def _dot( self, other, naxes ):
    assert naxes == 1
    if isinstance( other, (TrigTangent,TrigNormal) ) and self.angle == other.angle:
      return numpy.array( 1 if isinstance(other,TrigTangent) else 0 )

  def _opposite( self ):
    return TrigTangent( opposite(self.angle) )

  def _edit( self, op ):
    return TrigTangent( edit(self.angle,op) )

# CIRCULAR SYMMETRY

class RevolutionAngle( ArrayFunc ):
  'scalar with a 2pi gradient in highest local dimension'

  def __init__( self ):
    ArrayFunc.__init__( self, args=[], shape=() )

  def evalf( self ):
    return numpy.zeros( [1] )

  def _localgradient( self, ndims ):
    lgrad = numpy.zeros( ndims )
    lgrad[-1] = 2*numpy.pi
    return lgrad

class Revolved( ArrayFunc ):
  'implement an extra local dimension with zero gradient'

  def __init__( self, func ):
    assert _isfunc( func )
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  @property
  def blocks( self ):
    return self.func.blocks

  def evalf( self, func ):
    return func

  def _localgradient( self, ndims ):
    return revolved( concatenate( [ localgradient(self.func,ndims-1), _zeros(self.func.shape+(1,)) ], axis=-1 ) )

  def _edit( self, op ):
    return revolved( op(self.func) )


# AUXILIARY FUNCTIONS

def _jointshape( *shapes ):
  'determine shape after singleton expansion'

  ndim = len(shapes[0])
  combshape = [1] * ndim
  for shape in shapes:
    assert len(shape) == ndim
    for i, sh in enumerate(shape):
      if combshape[i] == 1:
        combshape[i] = sh
      else:
        assert sh in ( combshape[i], 1 ), 'incompatible shapes: %s' % ', '.join( str(sh) for sh in shapes )
  return tuple(combshape)

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
  if _equal( a1, b1 ):
    return a1, (a2,b2)
  if _equal( a1, b2 ):
    return a1, (a2,b1)
  if _equal( a2, b1 ):
    return a2, (a1,b2)
  if _equal( a2, b2 ):
    return a2, (a1,b1)

_max = max
_min = min
_sum = sum
_isfunc = lambda arg: isinstance( arg, ArrayFunc )
_isevaluable = lambda arg: isinstance( arg, Evaluable )
_isscalar = lambda arg: asarray(arg).ndim == 0
_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_iszero = lambda arg: isinstance( arg, Zeros ) or isinstance( arg, numpy.ndarray ) and numpy.all( arg == 0 )
_isunit = lambda arg: not _isfunc(arg) and ( numpy.asarray(arg) == 1 ).all()
_subsnonesh = lambda shape: tuple( 1 if sh is None else sh for sh in shape )
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )
_zeros = lambda shape: Zeros( shape )
_zeros_like = lambda arr: _zeros( arr.shape )

# for consistency in Add and Multiply arguments: the smallest Evaluable first
_issorted = lambda a, b: not isinstance(b,Evaluable) or isinstance(a,Evaluable) and id(a) <= id(b)
_sorted = lambda a, b: (a,b) if _issorted(a,b) else (b,a)

def _call( obj, attr, *args ):
  'call method if it exists, return None otherwise'

  f = getattr( obj, attr, None )
  return f and f( *args )

def _norm_and_sort( ndim, args ):
  'norm axes, sort, and assert unique'

  normargs = tuple( sorted( numeric.normdim( ndim, arg ) for arg in args ) )
  assert _ascending( normargs ) # strict
  return normargs

def _jointdtype( *args ):
  'determine joint dtype'

  if any( asarray(arg).dtype == float for arg in args ):
    return float
  return int

def _dtypestr( arg ):
  if arg.dtype == int:
    return 'int'
  if arg.dtype == float:
    return 'double'
  raise Exception( 'unknown dtype %s' % arg.dtype )

def _equal( arg1, arg2 ):
  'compare two objects'

  if arg1 is arg2:
    return True
  if isinstance( arg1, dict ) or isinstance( arg2, dict ):
    return False
  if isinstance( arg1, (list,tuple) ):
    if not isinstance( arg2, (list,tuple) ) or len(arg1) != len(arg2):
      return False
    return all( _equal(v1,v2) for v1, v2 in zip( arg1, arg2 ) )
  if not isinstance( arg1, numpy.ndarray ) and not isinstance( arg2, numpy.ndarray ):
    return arg1 == arg2
  elif isinstance( arg1, numpy.ndarray ) and isinstance( arg2, numpy.ndarray ):
    return arg1.shape == arg2.shape and numpy.all( arg1 == arg2 )
  else:
    return False

def asarray( arg ):
  'convert to ArrayFunc or numpy.ndarray'

  if _isfunc(arg):
    return arg

  if isinstance( arg, numpy.ndarray ) or not util.isiterable( arg ):
    array = numpy.asarray( arg )
    assert array.dtype != object
    if numpy.all( array == 0 ):
      return _zeros( array.shape )
    return array

  assert isinstance( arg, (list,tuple) ) # be strict to avoid infinite loops

  args = [ asarray(a) for a in arg ]
  ndim = _max( arg.ndim for arg in args )
  args = [ arg[(_,)*(ndim-arg.ndim)] for arg in args ]

  if all( isinstance( arg, numpy.ndarray ) for arg in args ):
    array = numpy.array( args )
    assert array.dtype != object
    if numpy.all( array == 0 ):
      return _zeros( array.shape )
    return array

  return stack( args, axis=0 )

def asfunc( obj ):
  'convert to Evaluable'
  return obj if isinstance( obj, Evaluable ) \
    else asarray( obj ) # TODO make asarray return ArrayFunc always

def _asarray( arg ):
  warnings.warn( '_asarray is deprecated, use asarray instead', DeprecationWarning )
  return asarray( arg )


# FUNCTIONS

def insert( arg, n ):
  'insert axis'

  arg = asarray( arg )
  n = numeric.normdim( arg.ndim+1, n )
  I = numpy.arange( arg.ndim )
  return align( arg, I + (I>=n), arg.ndim+1 )

def stack( args, axis=0 ):
  'stack functions along new axis'

  args = [ insert( arg, axis ) for arg in args ]
  ndim = args[0].ndim
  assert all( arg.ndim == ndim for arg in args[1:] ), 'arguments have non-matching shapes'
  return concatenate( args, axis )

def chain( funcs ):
  'chain'

  funcs = [ asarray(func) for func in funcs ]
  shapes = [ func.shape[0] for func in funcs ]
  return [ concatenate( [ func if i==j else _zeros( (sh,) + func.shape[1:] )
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
    item = 0 if sh == 1 \
      else numeric.normdim( sh, item )

  if not _isfunc( arg ):
    return numeric.get( arg, iax, item )

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

  if not _isfunc( arg ):
    return numeric.align( arg, axes, ndim )

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

def grad( arg, coords, ndims=0 ):
  'local derivative'

  arg = asarray( arg )
  if _isfunc( arg ):
    return arg.grad( coords, ndims )
  return _zeros( arg.shape + coords.shape )

def symgrad( arg, coords, ndims=0 ):
  'symmetric gradient'

  if _isfunc( arg ):
    return arg.symgrad( coords, ndims )
  return _zeros( arg.shape + coords.shape )

def div( arg, coords, ndims=0 ):
  'gradient'

  if _isfunc( arg ):
    return arg.div( coords, ndims )
  assert arg.shape[-1:] == coords.shape
  return _zeros( arg.shape[:-1] )

def sum( arg, axis=None, axes=None ):
  'sum over one or multiply axes'

  if axes is not None:
    assert axis is None, 'axes and axis cannot be simultaneously specified'
    warnings.warn( 'The axes argument is deprecated; please use axis instead', DeprecationWarning, stacklevel=2 )
    axis = axes

  if axis is None:
    warnings.warn( '''Please specify sum(...,-1) explicitly for summing over
  the last axis. Summation without an axis argument will be deprecated in
  nutils 3.0, to transition to numpy consistent behaviour in nutils 4.0''', DeprecationWarning, stacklevel=2 )
    axis = -1

  arg = asarray( arg )

  if util.isiterable(axis):
    if len(axis) == 0:
      return arg
    axis = _norm_and_sort( arg.ndim, axis )
    assert numpy.all( numpy.diff(axis) > 0 ), 'duplicate axes in sum'
    arg = sum( arg, axis[1:] )
    axis = axis[0]
  else:
    axis = numeric.normdim( arg.ndim, axis )

  if arg.shape[axis] == 1:
    return get( arg, axis, 0 )

  if not _isfunc( arg ):
    return arg.sum( axis )

  retval = _call( arg, '_sum', axis )
  if retval is not None:
    assert retval.shape == arg.shape[:axis] + arg.shape[axis+1:], 'bug in %s._sum' % arg
    return retval

  return Sum( arg, axis )

def dot( arg1, arg2, axes ):
  'dot product'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if util.isiterable(axes):
    if len(axes) == 0:
      return arg1 * arg2
    axes = _norm_and_sort( len(shape), axes )
    assert numpy.all( numpy.diff(axes) > 0 ), 'duplicate axes in sum'
  else:
    axes = numeric.normdim( len(shape), axes ),

  if _iszero( arg1 ) or _iszero( arg2 ):
    return _zeros([ s for i, s in enumerate(shape) if i not in axes ])

  if _isunit( arg1 ):
    return sum( expand( arg2, shape ), axes )

  if _isunit( arg2 ):
    return sum( expand( arg1, shape ), axes )

  if not _isfunc(arg1) and not _isfunc(arg2):
    return numeric.contract( arg1, arg2, axes )

  for i, axis in enumerate( axes ):
    if arg1.shape[axis] == 1 or arg2.shape[axis] == 1:
      arg1 = sum( arg1, axis )
      arg2 = sum( arg2, axis )
      axes = axes[:i] + tuple( axis-1 for axis in axes[i+1:] )
      return dot( arg1, arg2, axes )

  shuffle = list( range( len(shape) ) )
  for ax in reversed( axes ):
    shuffle.append( shuffle.pop(ax) )

  arg1 = transpose( arg1, shuffle )
  arg2 = transpose( arg2, shuffle )

  naxes = len( axes )
  dotshape = tuple( shape[i] for i in shuffle[:-naxes] )

  retval = _call( arg1, '_dot', arg2, naxes )
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg1
    return retval

  retval = _call( arg2, '_dot', arg1, naxes )
  if retval is not None:
    assert retval.shape == dotshape, 'bug in %s._dot' % arg2
    return retval

  a, b = _sorted( arg1, arg2 )
  return Dot( a, b, naxes )

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

  if not _isfunc( arg ):
    return numpy.linalg.det( arg )

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

  if not _isfunc( arg ):
    return numpy.linalg.inv( arg ).transpose( trans )

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

  if not _isfunc( arg ):
    return numeric.takediag( arg )

  retval = _call( arg, '_takediag' )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._takediag' % arg
    return retval

  return TakeDiag( arg )

def localgradient( arg, ndims ):
  'local derivative'

  arg = asarray( arg )
  shape = arg.shape + (ndims,)

  if not _isfunc( arg ):
    return _zeros( shape )

  lgrad = arg._localgradient( ndims )
  assert lgrad.shape == shape, 'bug in %s._localgradient' % arg

  return lgrad

def dotnorm( arg, coords, ndims=0 ):
  'normal component'

  return sum( arg * coords.normal( ndims-1 ), -1 )

def kronecker( arg, axis, length, pos ):
  'kronecker'

  axis = numeric.normdim( arg.ndim+1, axis )
  arg = insert( arg, axis )
  args = [ _zeros_like(arg) ] * length
  args[pos] = arg
  return concatenate( args, axis=axis )

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

  if all( _iszero(arg) for arg in args ):
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
    return _zeros( shape )

  while i+1 < len(args):
    arg1, arg2 = args[i:i+2]
    if not _isfunc(arg1) and not _isfunc(arg2):
      arg12 = numpy.concatenate( [ arg1, arg2 ], axis )
    else:
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

  if trans is None:
    invtrans = range( arg.ndim-1, -1, -1 )
  else:
    trans = _normdims( arg.ndim, trans )
    assert util.allequal( sorted(trans), range(arg.ndim) )
    invtrans = numpy.empty( arg.ndim, dtype=int )
    invtrans[ numpy.asarray(trans) ] = numpy.arange( arg.ndim )

  return align( arg, invtrans, arg.ndim )

def product( arg, axis ):
  'product'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  shape = arg.shape[:axis] + arg.shape[axis+1:]

  if arg.shape[axis] == 1:
    return get( arg, axis, 0 )

  if not _isfunc( arg ):
    return numpy.product( arg, axis )

  trans = list(range(axis)) + [-1] + list(range(axis,arg.ndim-1))
  aligned_arg = align( arg, trans, arg.ndim )

  retval = _call( aligned_arg, '_product', arg.ndim-1 )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._product' % aligned_arg
    return retval

  return Product( aligned_arg )

def choose( level, choices ):
  'choose'

  if not _isfunc(level) and not any( _isfunc(choice) for choice in choices ):
    return numpy.choose( level, choices )
  level_choices = _matchndim( level, *choices )
  if all( map( _iszero, level_choices[1:] ) ):
    shape = _jointshape( *( a.shape for a in level_choices ) )
    return _zeros( shape )
  return Choose( level_choices[0], level_choices[1:] )

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

  if not any(map(_isfunc, itertools.chain( condlist, choicelist, [default] ))):
    return numpy.select( condlist, choicelist, default=default )
  level = pointwise( condlist, _condlist_to_level, None )
  return choose( level, (default,)+tuple(choicelist) )

def cross( arg1, arg2, axis ):
  'cross product'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )
  axis = numeric.normdim( len(shape), axis )

  if not _isfunc(arg1) and not _isfunc(arg2):
    return numeric.cross(arg1,arg2,axis)

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

def pointwise( args, evalf, deriv ):
  'general pointwise operation'

  args = asarray( _matchndim(*args) )
  if _isfunc(args):
    retval = _call( args, '_pointwise', evalf, deriv )
    if retval is not None:
      return retval
    return Pointwise( args, evalf, deriv )
  return evalf( *args )

def multiply( arg1, arg2 ):
  'multiply'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if _isunit( arg1 ):
    return expand( arg2, shape )

  if _isunit( arg2 ):
    return expand( arg1, shape )

  if not _isfunc(arg1) and not _isfunc(arg2):
    return numpy.multiply( arg1, arg2 )

  if _equal( arg1, arg2 ):
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

  if _iszero( arg1 ):
    return expand( arg2, shape )

  if _iszero( arg2 ):
    return expand( arg1, shape )

  if not _isfunc(arg1) and not _isfunc(arg2):
    return numpy.add( arg1, arg2 )

  if _equal( arg1, arg2 ):
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
      key.append( ( arg.dofmap, arg.axis ) )
      arg = arg.func
    inflates.setdefault( tuple(key), [] ).append( arg )
  # add inflate args with the same axis and dofmap, blockadd the remainder
  args = []
  for key, values in inflates.items():
    if key is ():
      continue
    arg = functools.reduce( operator.add, values )
    for dofmap, axis in reversed( key ):
      arg = inflate( arg, dofmap, axis )
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

  if not _isfunc( n ) and numpy.all( n == 1 ):
    return expand( arg, shape )

  if _iszero( n ):
    return numpy.ones( shape )

  if not _isfunc( arg ) and not _isfunc( n ):
    return numpy.power( arg.astype(float), n )

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

  # When it's an array calculate directly
  if not _isfunc(aligned_arg):
    eigval, eigvec = numpy.linalg.eigh( aligned_arg ) if symmetric else numpy.linalg.eig( aligned_arg )
  else:
    # Use _call to see if the object has its own _eig function
    ret = _call( aligned_arg, '_eig' )
    if ret is not None:
      # Check the shapes
      eigval, eigvec = ret
    else:
      eig = Eig( aligned_arg, symmetric=symmetric )
      eigval = array_from_tuple( eig, index=0, shape=aligned_arg.shape[:-1] )
      eigvec = array_from_tuple( eig, index=1, shape=aligned_arg.shape )

  # Return the evaluable function objects in a tuple like numpy
  eigval = transpose( diagonalize( eigval ), trans )
  eigvec = transpose( eigvec, trans )
  assert eigvec.shape == arg.shape
  assert eigval.shape == arg.shape
  return eigval, eigvec

def array_from_tuple( arrays, index, shape ):
  if isinstance( arrays, Tuple ):
    array = arrays.items[index]
    assert array.shape == shape
    return array
  else:
    return ArrayFromTuple( arrays, index, shape )

def revolved( arg ):
  arg = asarray( arg )
  if not _isfunc( arg ) or _iszero( arg ):
    return arg
  retval = _call( arg, '_revolved' )
  if retval is not None:
    return retval
  return Revolved( arg )

negative = lambda arg: multiply( arg, -1 )
nsymgrad = lambda arg, coords: ( symgrad(arg,coords) * coords.normal() ).sum(-1)
ngrad = lambda arg, coords: ( grad(arg,coords) * coords.normal() ).sum(-1)
sin = lambda arg: pointwise( [arg], numpy.sin, cos )
cos = lambda arg: pointwise( [arg], numpy.cos, lambda x: -sin(x) )
trignormal = lambda angle: TrigNormal( angle )
trigtangent = lambda angle: TrigTangent( angle )
rotmat = lambda arg: asarray([ trignormal(arg), trigtangent(arg) ])
tan = lambda arg: pointwise( [arg], numpy.tan, lambda x: cos(x)**-2 )
arcsin = lambda arg: pointwise( [arg], numpy.arcsin, lambda x: reciprocal(sqrt(1-x**2)) )
arccos = lambda arg: pointwise( [arg], numpy.arccos, lambda x: -reciprocal(sqrt(1-x**2)) )
exp = lambda arg: pointwise( [arg], numpy.exp, exp )
ln = lambda arg: pointwise( [arg], numpy.log, reciprocal )
log2 = lambda arg: ln(arg) / ln(2)
log10 = lambda arg: ln(arg) / ln(10)
sqrt = lambda arg: power( arg, .5 )
reciprocal = lambda arg: power( arg, -1 )
argmin = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmin(numeric.stack(x),axis=0), _zeros_like )
argmax = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmax(numeric.stack(x),axis=0), _zeros_like )
arctan2 = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.arctan2, lambda x: stack([x[1],-x[0]]) / sum(power(x,2),0) )
greater = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.greater, _zeros_like )
less = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.less, _zeros_like )
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
edit = lambda arg, f: arg._edit(f) if _isevaluable(arg) else arg

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

  if not isinstance( arg, Evaluable ):
    return arg

  retval = _call( arg, '_opposite' )
  if retval is not None:
    return retval

  return arg._edit( opposite )

def function( fmap, nmap, ndofs, ndims ):
  'create function on ndims-element'

  axis = '~%d' % ndofs
  func = Function( ndims, fmap, igrad=0, axis=axis )
  dofmap = DofMap( nmap, axis=axis, target=ndofs )
  return Inflate( func, dofmap, axis=0 )

def take( arg, index, axis ):
  'take index'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if isinstance( index, IndexVector ):
    if index.shape[0] == 1:
      return insert( get( arg, axis, index[0] ), axis )
    retval = _call( arg, '_take', index, axis )
    if retval is not None:
      return retval
    if arg.shape[axis] == 1:
      return arg
    return DofIndex( arg, axis, index ) if isinstance(arg,numpy.ndarray) else Take( arg, index, axis )

  if isinstance( index, slice ):
    n = arg.shape[axis]
    if n == 1:
      assert index.stop != None and index.stop > 0
      n = index.stop
    index = numpy.arange( *index.indices(n) )
  else:
    index = numpy.asarray( index )
    assert numpy.all( index >= 0 )

  if numeric.isboolarray(index) and index.ndim == 1 and len(index) == arg.shape[axis]:
    index, = numpy.where( index )

  assert numeric.isintarray(index) and index.ndim == 1 and len(index) > 0

  if len(index) == arg.shape[axis] and all( index == numpy.arange(arg.shape[axis]) ):
    return arg

  if arg.shape[axis] == 1:
    return repeat( arg, index.shape[0], axis )

  if len(index) == 1:
    return insert( get( arg, axis, index[0] ), axis )

  retval = _call( arg, '_take', index, axis )
  if retval is not None:
    return retval

  if not _isfunc( arg ):
    return numpy.take( arg, index, axis )

  return Take( arg, index, axis )

def inflate( arg, dofmap, axis ):
  'inflate'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  assert not isinstance( arg.shape[axis], int )

  retval = _call( arg, '_inflate', dofmap, axis )
  if retval is not None:
    return retval

  return Inflate( arg, dofmap, axis )

def blocks( arg ):
  arg = asarray( arg )
  return arg.blocks if _isfunc( arg ) \
    else [] if numpy.all( arg == 0 ) \
    else [( Tuple( numpy.arange(n) for n in arg.shape ), arg )]

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

def _unpack( funcsp ):
  for axes, func in funcsp.blocks:
    dofax = axes[0]
    assert isinstance( dofax, DofMap )
    dofmap = dofax.dofmap
    if isinstance( func, Align ):
      func = func.func
    stdmap = func.stdmap
    for trans, dofs in dofmap.items():
      yield trans, dofs + dofax.offset, stdmap[trans]
  
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
