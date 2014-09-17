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
for a higher lever style programming. It also allows for automatic
differentiation and code optimization.

It is important to realize that nutils functions do not map for a physical
xy-domain but from a topology, where a point is characterized by the combination
of an element and its local coordinate. This is a natural fit for typical finite
element operations such as quadrature. Evaluation from physical coordinates is
possible only via inverting of the geometry function, which is a fundamentally
expensive and currently unsupported operation.
"""

from . import util, numpy, numeric, log, core, cache, transform, rational, _
import sys, warnings

CACHE   = object()
TRANS   = object()
POINTS  = object()
WEIGHTS = object()

class Evaluable( object ):
  'Base class'

  __metaclass__ = cache.Meta

  def __init__( self, args ):
    'constructor'

    self.__args = tuple(args)

  def evalf( self ):
    raise NotImplementedError, 'Evaluable derivatives should implement the evalf method'

  @cache.property
  def serialized( self ):
    '''returns (data,ops,inds), where len(ops) = len(inds)-1'''

    mydata = [ CACHE, TRANS, POINTS, WEIGHTS ]
    myops = []
    myinds = []

    indices = []
    for arg in self.__args:

      if not isinstance( arg, Evaluable ):
        index = _isindex( arg, mydata )
        if index == -1:
          ren = numpy.hstack([ numpy.arange(len(mydata)), len(mydata)+1+numpy.arange(len(myops)) ])
          myinds = map( ren.__getitem__, myinds )
          indices = map( ren.__getitem__, indices )
          index = len(mydata)
          mydata.append( arg )
        indices.append( index )
        continue

      index = _isindex( arg, myops )
      if index != -1:
        indices.append( len(mydata)+index )
        continue
        
      argdata, argops, arginds = arg.serialized

      renumber = numpy.empty( len(argdata)+len(argops), dtype=int )
      ndata = len(mydata)
      for i, obj in enumerate( argdata ):
        index = _isindex( obj, mydata )
        if index == -1:
          index = len(mydata)
          mydata.append( obj )
        renumber[i] = index
      if len(mydata) > ndata:
        ren = numpy.hstack([ numpy.arange(ndata), len(mydata)+numpy.arange(len(myops)) ])
        myinds = map( ren.__getitem__, myinds )
        indices = map( ren.__getitem__, indices )
      for i, (op,ind) in enumerate( zip(argops,arginds) ):
        index = _isindex( op, myops )
        if index == -1:
          index = len(myops)
          myops.append( op )
          myinds.append( renumber[ind] )
        renumber[len(argdata)+i] = len(mydata)+index

      indices.append( len(mydata)+len(myops) )
      myops.append( arg )
      myinds.append( renumber[arginds[-1]] )

    myinds.append( indices )
    for op, ind in zip( myops, myinds ) + [ (self,indices) ]:
      for i, arg in zip( ind, op.__args ):
        assert arg is ( mydata[i] if i < len(mydata) else myops[i-len(mydata)] )

    return tuple(mydata), tuple(myops), tuple(myinds)

  def asciitree( self ):
    'string representation'

    key = self.evalf.__name__
    lines = []
    indent = '\n' + ' ' + ' ' * len(key)
    for it in reversed( self.args ):
      s = it.asciitree() if isinstance(it,Evaluable) else _obj2str(it)
      lines.append( indent.join( s.splitlines() ) )
      indent = '\n' + '|' + ' ' * len(key)
    indent = '\n' + '+' + '-' * (len(key)-1) + ' '
    return key + ' ' + indent.join( reversed( lines ) )

  def __str__( self ):
    return self.__class__.__name__

  def eval( self, elem, ischeme, fcache=lambda f, *args: f(*args) ):
    'evaluate'
    
    if isinstance( elem, tuple ):
      assert isinstance( ischeme, numpy.ndarray )
      points = ischeme
      weights = None
      transform, opposite = elem
      assert points.shape[-1] == transform.fromdims == opposite.fromdims
      trans = elem
    else:
      trans = elem.transform, elem.opposite
      if isinstance( ischeme, dict ):
        ischeme = ischeme[elem]
      if isinstance( ischeme, str ):
        points, weights = fcache( elem.reference.getischeme, ischeme )
      elif isinstance( ischeme, tuple ):
        points, weights = ischeme
        assert points.shape[-1] == elem.ndims
        assert points.shape[:-1] == weights.shape, 'non matching shapes: points.shape=%s, weights.shape=%s' % ( points.shape, weights.shape )
      elif isinstance( ischeme, numpy.ndarray ):
        points = ischeme.astype( float )
        weights = None
        assert points.shape[-1] == elem.ndims
      elif ischeme is None:
        points = weights = None
      else:
        raise Exception, 'invalid integration scheme of type %r' % type(ischeme)

    data, ops, inds = self.serialized
    assert data[:4] == ( CACHE, TRANS, POINTS, WEIGHTS )
    values = [ fcache, trans, points, weights ] + list( data[4:] )
    for op, indices in zip( ops, inds ) + [(self,inds[-1])]:
      args = [ values[i] for i in indices ]
      try:
        retval = op.evalf( *args )
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        raise EvaluationError, ( etype, evalue, self, values ), traceback
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

    try:
      dot = subprocess.Popen( [dotpath,'-T'+imgtype], stdin=subprocess.PIPE, stdout=open(imgpath,'w') )
    except OSError:
      log.error( 'error: failed to execute', dotpath )
      return False

    print >> dot.stdin, 'digraph {'
    print >> dot.stdin, 'graph [ dpi=72 ];'

    data = self.data + [ '<cache>', '<trans>', '<points>', '<weights>' ]
    for i, (op,indices) in enumerate( self.operations ):
      args = [ '%%%d=%s' % ( iarg, _obj2str( data[idx] ) ) for iarg, idx in enumerate( indices ) if idx < 0 ]
      vertex = 'label="%s"' % r'\n'.join( [ '%d. %s' % ( i, op ) ] + args )
      print >> dot.stdin, '%d [%s]' % ( i, vertex )
      for iarg, idx in enumerate( indices ):
        if idx >= 0:
          print >> dot.stdin, '%d -> %d [label="%%%d"];' % ( idx, i, iarg );

    print >> dot.stdin, '}'
    dot.stdin.close()

    log.path( os.path.basename(imgpath) )

  def stackstr( self, values=None ):
    'print stack'

    data, ops, inds = self.serialized
    if values is None:
      values = [ '<cache>', '<trans>', '<points>', '<weights>' ] + data[4:]

    lines = []
    for i, (op,indices) in enumerate( zip(ops,inds) + [(self,inds[-1])] ):
      line = '  %%%d =' % i
      args = [ '%%%d' % (idx-len(data)) if idx >= len(data) else _obj2str(values[idx]) for idx in indices ]
      try:
        code = op.evalf.func_code
        names = code.co_varnames[ :code.co_argcount ]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(indices) - len(names) ) )
        args = [ '%s=%s' % item for item in zip( names, args ) ]
      except:
        pass
      line += ' %s( %s )' % ( op, ', '.join( args ) )
      lines.append( line )
      if i == len(values):
        break
    return '\n'.join( lines )

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

    return '\n%s --> %s: %s' % ( self.evaluable.stackstr( self.values ), self.etype.__name__, self.evalue )

class Tuple( Evaluable ):
  'combine'

  def __init__( self, items ):
    'constructor'

    self.items = tuple( items )
    Evaluable.__init__( self, args=self.items )

  @staticmethod
  def evalf( *f ):
    'evaluate'

    return f

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

  @staticmethod
  def evalf( points ):
    'evaluate'

    return points.shape[:-1]

class Elemtrans( Evaluable ):
  'transform'

  def __init__( self, side ):
    Evaluable.__init__( self, args=[TRANS,side] )
    self.side = side

  @staticmethod
  def evalf( trans, side ):
    return trans[side]

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
  __rdiv__ = lambda self, other: divide( other, self )
  __add__  = lambda self, other: add( self, other )
  __radd__ = lambda self, other: add( other, self )
  __sub__  = lambda self, other: subtract( self, other )
  __rsub__ = lambda self, other: subtract( other, self )
  __neg__  = lambda self: negative( self )
  __pow__  = lambda self, n: power( self, n )
  __abs__  = lambda self: abs( self )
  __len__  = lambda self: self.shape[0]
  sum      = lambda self, axes=-1: sum( self, axes )

  # standalone methods

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
      if numeric.isint(it): # retrieve one item from axis
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
      elif isinstance(it,(slice,list,tuple,numpy.ndarray)): # modify axis (shorten, extend or renumber one axis)
        arr = take( arr, it, n )
        n += 1
      else:
        raise NotImplementedError
      assert n <= arr.ndim
    return arr

  def __iter__( self ):
    'split first axis'

    if not self.shape:
      raise TypeError, 'scalar function is not iterable'

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
        raise Exception, 'failed to converge in %d iterations' % maxiter
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
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims )
    return normal * ElemSign( ndims )

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
      Jinv = ( J[_,:,:] * Ginv[:,_,:] ).sum()
    else:
      raise Exception, 'cannot invert %sx%s jacobian' % J.shape
    return sum( localgradient( self, ndims )[...,_] * Jinv, axes=-2 )

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

    return '%s<%s>' % ( self.__class__.__name__, ','.join(map(str,self.shape)) )

  __repr__ = __str__

class ElemSign( ArrayFunc ):
  'sign'

  def __init__( self, ndims, side=0 ):
    'constructor'

    ArrayFunc.__init__( self, args=[Elemtrans(side),ndims], shape=() )
    self.side = side
    self.ndims = ndims

  @staticmethod
  def evalf( trans, ndims ):
    try:
      ntrans = trans.slice( todims=ndims+1, fromdims=ndims )
    except: # possibly ndim topo, n+1dim geom
      return numpy.array( 1 )
    return numpy.array( -1 if ntrans.isflipped else 1 )

  def _opposite( self ):
    return ElemSign( self.ndims, 1-self.side )

  def _localgradient( self, ndims ):
    return _zeros( (ndims,) )

class ElemArea( ArrayFunc ):
  'element area'

  def __init__( self, weights ):
    'constructor'

    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights], shape=weights.shape )

  @staticmethod
  def evalf( weights ):
    'evaluate'

    return numpy.sum( weights )

class ElemInt( ArrayFunc ):
  'elementwise integration'

  def __init__( self, func, weights ):
    'constructor'

    assert _isfunc( func ) and _isfunc( weights )
    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights,func,func.ndim], shape=func.shape )

  @staticmethod
  def evalf( w, f, ndim ):
    'evaluate'

    if f.ndim == ndim: # the missing point axis problem
      return f * w.sum()
    return numeric.dot( w, f ) if w.size else numpy.zeros( f.shape[1:] )

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
    negaxes = [ ax-ndim for ax in self.axes ]
    ArrayFunc.__init__( self, args=[func,negaxes,ndim], shape=shape )

  @staticmethod
  def evalf( arr, trans, ndim ):
    'align'

    extra = arr.ndim - len(trans)
    return numeric.align( arr, range(extra)+trans, ndim+extra )

  def _elemint( self, weights ):
    return align( elemint( self.func, weights ), self.axes, self.ndim )

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
    n = self.axes.index( axis )
    return align( take( self.func, indices, n ), self.axes, self.ndim )

  def _opposite( self ):
    return align( opposite(self.func), self.axes, self.ndim )

class Get( ArrayFunc ):
  'get'

  def __init__( self, func, axis, item ):
    'constructor'

    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    assert 0 <= item < func.shape[axis], 'item is out of bounds'
    s = (Ellipsis,item) + (slice(None),)*(func.ndim-axis-1)
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=(func,s), shape=shape )

  evalf = staticmethod(numpy.ndarray.__getitem__)

  def _localgradient( self, ndims ):
    f = localgradient( self.func, ndims )
    return get( f, self.axis, self.item )

  def _get( self, i, item ):
    tryget = get( self.func, i+(i>=self.axis), item )
    if not isinstance( tryget, Get ): # avoid inf recursion
      return get( tryget, self.axis, self.item )

  def _take( self, indices, axis ):
    return get( take( self.func, indices, axis+(axis>=self.axis) ), self.axis, self.item )

  def _opposite( self ):
    return get( opposite(self.func), self.axis, self.item )

class Product( ArrayFunc ):
  'product'

  def __init__( self, func, axis ):
    'constructor'

    self.func = func
    self.axis = axis
    assert 0 <= axis < func.ndim
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], shape=shape )

  evalf = staticmethod(numpy.product)

  def _localgradient( self, ndims ):
    return self[...,_] * ( localgradient(self.func,ndims) / self.func[...,_] ).sum( self.axis )

  def _get( self, i, item ):
    func = get( self.func, i+(i>=self.axis), item )
    return product( func, self.axis-(i<self.axis) )

  def _opposite( self ):
    return product( opposite(self.func), self.axis )

class IWeights( ArrayFunc ):
  'integration weights'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[Elemtrans(0),WEIGHTS], shape=() )

  @staticmethod
  def evalf( trans, weights ):
    'evaluate'

    return float( trans.slice(todims=trans.fromdims).det ) * weights

class Transform( ArrayFunc ):
  'transform'

  def __init__( self, todims, fromdims, side ):
    'constructor'

    assert fromdims != todims
    self.fromdims = fromdims
    self.todims = todims
    self.side = side
    ArrayFunc.__init__( self, args=[Elemtrans(side),todims,fromdims], shape=(todims,fromdims) )

  @staticmethod
  def evalf( trans, todims, fromdims ):
    'transform'

    matrix = trans.slice(fromdims=fromdims,todims=todims).linear
    assert matrix.ndim == 2
    return matrix.astype( float )

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
    ArrayFunc.__init__( self, args=(CACHE,POINTS,Elemtrans(side),stdmap,igrad), shape=(axis,)+(ndims,)*igrad )

  @staticmethod
  def evalf( cache, points, trans, stdmap, igrad ):
    'evaluate'

    fvals = []
    head = trans.lookup( stdmap )
    for std, keep in stdmap[head]:
      if std:
        transpoints = cache( trans[len(head):].apply, points )
        F = cache( std.eval, transpoints, igrad )
        if keep is not None:
          F = F[(Ellipsis,keep)+(slice(None),)*igrad]
        if igrad:
          invlinear = head.slice(todims=head.fromdims).invlinear.astype( float )
          if invlinear.ndim:
            for axis in range(-igrad,0):
              F = numeric.dot( F, invlinear, axis )
          elif invlinear != 1:
            F = F * (invlinear**igrad)
        fvals.append( F )
      head = head[:-1]
    return fvals[0] if len(fvals) == 1 else numpy.concatenate( fvals, axis=-1-igrad )

  def _opposite( self ):
    return Function( self.ndims, self.stdmap, self.igrad, self.shape[0], 1-self.side )

  def _localgradient( self, ndims ):
    grad = Function( self.ndims, self.stdmap, self.igrad+1, self.shape[0], self.side )
    return grad if ndims == self.ndims \
      else dot( grad[...,_], Transform( self.ndims, ndims, self.side ), axes=-2 )

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, level, choices ):
    'constructor'

    self.level = level
    self.choices = tuple(choices)
    shape = _jointshape( *[ choice.shape for choice in choices ] )
    level = level[ (_,)*(len(shape)-level.ndim) ]
    assert level.ndim == len( shape )
    ArrayFunc.__init__( self, args=(level,)+self.choices, shape=shape )

  @staticmethod
  def evalf( level, *choices ):
    'choose'

    return numpy.choose( level, choices )

  def _localgradient( self, ndims ):
    grads = [ localgradient( choice, ndims ) for choice in self.choices ]
    if not any( grads ): # all-zero special case; better would be allow merging of intervals
      return _zeros( self.shape + (ndims,) )
    return Choose( self.level[...,_], grads )

  def _opposite( self ):
    return choose( opposite(self.level), tuple(opposite(c) for c in self.choices) )

class Choose2D( ArrayFunc ):
  'piecewise function'

  def __init__( self, coords, contour, fin, fout ):
    'constructor'

    shape = _jointshape( fin.shape, fout.shape )
    ArrayFunc.__init__( self, args=(coords,contour,fin,fout), shape=shape )

  @staticmethod
  def evalf( xy, contour, fin, fout ):
    'evaluate'

    from matplotlib import nxutils
    mask = nxutils.points_inside_poly( xy.T, contour )
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

  evalf = staticmethod(numpy.linalg.inv)

  def _localgradient( self, ndims ):
    G = localgradient( self.func, ndims )
    H = sum( self[...,_,:,:,_]
              * G[...,:,:,_,:], -3 )
    I = sum( self[...,:,:,_,_]
              * H[...,_,:,:,:], -3 )
    return -I

  def _opposite( self ):
    return Inverse( opposite(self.func) )

class DofMap( ArrayFunc ):
  'dof axis'

  def __init__( self, dofmap, axis, side=0 ):
    'new'

    self.side = side
    self.dofmap = dofmap
    ArrayFunc.__init__( self, args=(Elemtrans(side),dofmap), shape=[axis], dtype=int )

  @staticmethod
  def evalf( trans, dofmap ):
    'evaluate'

    return dofmap[ trans.lookup(dofmap) ]

  def _opposite( self ):
    return DofMap( self.dofmap, self.shape[0], 1-self.side )

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
      sh = sum( lengths )
    shape = _jointshape( *[ func.shape[:axis] + (sh,) + func.shape[axis+1:] for func in funcs ] )
    self.funcs = tuple(funcs)
    self.axis = axis
    ArrayFunc.__init__( self, args=(axis-ndim,)+self.funcs, shape=shape )

  @staticmethod
  def evalf( iax, *arrays ):
    'evaluate'

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
      for f, ind in blocks( func ):
        yield f, ind[:self.axis] + (ind[self.axis]+n,) + ind[self.axis+1:]
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

  def _inflate( self, dofmap, length, axis ):
    assert not isinstance( self.shape[axis], int )
    return concatenate( [ inflate(func,dofmap,length,axis) for func in self.funcs ], self.axis )

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
        raise Exception, 'index out of bounds'
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

  def _negative( self ):
    return concatenate( [ -func for func in self.funcs ], self.axis )

  def _opposite( self ):
    return concatenate( [ opposite(func) for func in self.funcs ], self.axis )

  def _power( self, n ):
    return concatenate( [ power( func, n ) for func in self.funcs ], self.axis )

  def _repeat( self, length, axis ):
    if axis != self.axis:
      return concatenate( [ repeat( func, length, axis ) for func in self.funcs ], self.axis )

  def _diagonalize( self ):
    if self.axis < self.ndim-1:
      return concatenate( [ diagonalize(func) for func in self.funcs ], self.axis )

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
    ArrayFunc.__init__( self, args=[x,xp,fp,left,right], shape=() )

  evalf = staticmethod(numpy.interp)

class Cross( ArrayFunc ):
  'cross product'

  def __init__( self, func1, func2, axis ):
    'contructor'

    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    shape = _jointshape( func1.shape, func2.shape )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    ArrayFunc.__init__( self, args=(func1,func2,axis-len(shape)), shape=shape )

  evalf = staticmethod(numeric.cross)

  def _localgradient( self, ndims ):
    return cross( self.func1[...,_], localgradient(self.func2,ndims), axis=self.axis ) \
         - cross( self.func2[...,_], localgradient(self.func1,ndims), axis=self.axis )

  def _take( self, index, axis ):
    if axis != self.axis:
      return cross( take(self.func1,index,axis), take(self.func2,index,axis), self.axis )

  def _opposite( self ):
    return cross( opposite(self.func1), opposite(self.func2), self.axis )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, func ):
    'contructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-2] )

  evalf = staticmethod(numpy.linalg.det)

  def _localgradient( self, ndims ):
    Finv = swapaxes( inverse( self.func ) )
    G = localgradient( self.func, ndims )
    return self[...,_] * sum( Finv[...,_] * G, axes=[-3,-2] )

  def _opposite( self ):
    return determinant( opposite(self.func) )

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
    item = [ slice(None) ] * self.array.ndim
    item[iax] = index
    ArrayFunc.__init__( self, args=[self.array]+item, shape=shape )

  @staticmethod
  def evalf( arr, *item ):
    'evaluate'

    return arr[item]

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

  def _opposite( self ):
    return take( self.array, opposite(self.index), self.iax )

  def _negative( self ):
    return take( -self.array, self.index, self.iax )

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, shape=shape )

  evalf = staticmethod(numpy.multiply)

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
      return determinant( self.funcs[1] ) * self.funcs[0][...,0,0]
    if self.funcs[1].shape[-2:] == (1,1):
      return determinant( self.funcs[0] ) * self.funcs[1][...,0,0]

  def _product( self, axis ):
    func1, func2 = self.funcs
    return product( func1, axis ) * product( func2, axis )

  def _multiply( self, other ):
    func1, func2 = self.funcs
    func1_other = multiply( func1, other )
    if Multiply( *_sorted(func1,other) ) != func1_other:
      return multiply( func1_other, func2 )
    func2_other = multiply( func2, other )
    if Multiply( *_sorted(func2,other) ) != func2_other:
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

  def _opposite( self ):
    func1, func2 = self.funcs
    return opposite(func1) * opposite(func2)

  def _negative( self ):
    func1, func2 = self.funcs
    negfunc1 = -func1
    if not isinstance( negfunc1, Negative ):
      return multiply( negfunc1, func2 )
    negfunc2 = -func2
    if not isinstance( negfunc2, Negative ):
      return multiply( func1, negfunc2 )

class Negative( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  evalf = staticmethod(numpy.negative)

  @property
  def blocks( self ):
    for f, ind in blocks( self.func ):
      yield negative(f), ind

  def _add( self, other ):
    if isinstance( other, Negative ):
      return negative( self.func + other.func )

  def _multiply( self, other ):
    if isinstance( other, Negative ):
      return self.func * other.func
    return negative( self.func * other )

  def _negative( self ):
    return self.func

  def _elemint( self, weights ):
    return -elemint( self.func, weights )

  def _align( self, axes, ndim ):
    return -align( self.func, axes, ndim )

  def _get( self, i, item ):
    return -get( self.func, i, item )

  def _sum( self, axis ):
    return -sum( self.func, axis )

  def _localgradient( self, ndims ):
    return -localgradient( self.func, ndims )

  def _takediag( self ):
    return -takediag( self.func )

  def _take( self, index, axis ):
    return -take( self.func, index, axis )

  def _opposite( self ):
    return -opposite(self.func)

  def _power( self, n ):
    if n%2 == 0:
      return power( self.func, n )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    assert _issorted( func1, func2 )
    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    dtype = _jointdtype(func1,func2)
    ArrayFunc.__init__( self, args=self.funcs, shape=shape, dtype=dtype )

  evalf = staticmethod(numpy.add)

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

  def _opposite( self ):
    func1, func2 = self.funcs
    return opposite(func1) + opposite(func2)

  def _add( self, other ):
    func1, func2 = self.funcs
    func1_other = add( func1, other )
    if Add( *_sorted(func1,other) ) != func1_other:
      return add( func1_other, func2 )
    func2_other = add( func2, other )
    if Add( *_sorted(func2,other) ) != func2_other:
      return add( func1, func2_other )

class BlockAdd( Add ):
  'block addition (used for DG)'

  def _multiply( self, other ):
    func1, func2 = self.funcs
    return BlockAdd( *_sorted( func1 * other, func2 * other ) )

  def _inflate( self, dofmap, length, axis ):
    func1, func2 = self.funcs
    return BlockAdd( *_sorted( inflate( func1, dofmap, length, axis ),
                               inflate( func2, dofmap, length, axis ) ) )

  def _align( self, axes, ndim ):
    func1, func2 = self.funcs
    return BlockAdd( *_sorted( align(func1,axes,ndim), align(func2,axes,ndim) ) )

  def _negative( self ):
    func1, func2 = self.funcs
    return BlockAdd( *_sorted( negative(func1), negative(func2) ) )

  def _add( self, other ):
    func1, func2 = self.funcs
    try1 = func1 + other
    if try1 != BlockAdd( *_sorted( func1, other ) ):
      return try1 + func2
    try2 = func2 + other
    if try2 != BlockAdd( *_sorted( func2, other ) ):
      return try2 + func1
    return BlockAdd( *_sorted( self, other ) )

  @property
  def blocks( self ):
    func1, func2 = self.funcs
    for f, ind in blocks( func1 ):
      yield f, ind
    for f, ind in blocks( func2 ):
      yield f, ind

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, naxes ):
    'constructor'

    assert naxes > 0
    self.func1 = func1
    self.func2 = func2
    self.naxes = naxes
    shape = _jointshape( func1.shape, func2.shape )[:-naxes]
    ArrayFunc.__init__( self, args=(func1,func2,naxes), shape=shape )

  evalf = staticmethod(numeric.contract_fast)

  @property
  def axes( self ):
    return range( self.ndim, self.ndim + self.naxes )

  def _get( self, i, item ):
    return dot( get( self.func1, i, item ), get( self.func2, i, item ), [ ax-1 for ax in self.axes ] )

  def _localgradient( self, ndims ):
    return dot( localgradient( self.func1, ndims ), self.func2[...,_], self.axes ) \
         + dot( self.func1[...,_], localgradient( self.func2, ndims ), self.axes )

  def _multiply( self, other ):
    for ax in self.axes:
      other = insert( other, ax )
    assert other.ndim == self.func1.ndim == self.func2.ndim
    func1_other = multiply( self.func1, other )
    if Multiply( *_sorted(self.func1,other) ) != func1_other:
      return dot( func1_other, self.func2, self.axes )
    func2_other = multiply( self.func2, other )
    if Multiply( *_sorted(self.func2,other) ) != func2_other:
      return dot( self.func1, func2_other, self.axes )

  def _add( self, other ):
    if isinstance( other, Dot ) and self.axes == other.axes:
      common = _findcommon( (self.func1,self.func2), (other.func1,other.func2) )
      if common:
        f, (g1,g2) = common
        return dot( f, g1 + g2, self.axes )

  def _takediag( self ):
    n1, n2 = self.ndim-2, self.ndim-1
    return dot( takediag( self.func1, n1, n2 ), takediag( self.func2, n1, n2 ), [ ax-2 for ax in self.axes ] )

  def _sum( self, axis ):
    return dot( self.func1, self.func2, self.axes + [axis] )

  def _take( self, index, axis ):
    return dot( take(self.func1,index,axis), take(self.func2,index,axis), self.axes )

  def _concatenate( self, other, axis ):
    if isinstance( other, Dot ) and other.axes == self.axes:
      common = _findcommon( (self.func1,self.func2), (other.func1,other.func2) )
      if common:
        f, g12 = common
        tryconcat = concatenate( g12, axis )
        if not isinstance( tryconcat, Concatenate ): # avoid inf recursion
          return dot( f, tryconcat, self.axes )

  def _opposite( self ):
    return dot( opposite(self.func1), opposite(self.func2), self.axes )

  def _negative( self ):
    negfunc1 = -self.func1
    if not isinstance( negfunc1, Negative ):
      return dot( negfunc1, self.func2, self.axes )
    negfunc2 = -self.func2
    if not isinstance( negfunc2, Negative ):
      return dot( self.func1, negfunc2, self.axes )

class Sum( ArrayFunc ):
  'sum'

  def __init__( self, func, axis ):
    'constructor'

    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], shape=shape )

  evalf = staticmethod(numpy.sum)

  def _sum( self, axis ):
    trysum = sum( self.func, axis+(axis>=self.axis) )
    if not isinstance( trysum, Sum ): # avoid inf recursion
      return sum( trysum, self.axis )

  def _localgradient( self, ndims ):
    return sum( localgradient( self.func, ndims ), self.axis )

  def _opposite( self ):
    return sum( opposite(self.func), axes=self.axis )

class Debug( ArrayFunc ):
  'debug'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  @staticmethod
  def evalf( arr ):
    'debug'

    log.debug( 'debug output:\n%s' % arr )
    return arr

  def __str__( self ):
    'string representation'

    return '{DEBUG}'

  def _localgradient( self, ndims ):
    return Debug( localgradient( self.func, ndims ) )

class TakeDiag( ArrayFunc ):
  'extract diagonal'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-1] )

  evalf = staticmethod(numeric.takediag)

  def _localgradient( self, ndims ):
    return swapaxes( takediag( localgradient( self.func, ndims ), -3, -2 ) )

  def _sum( self, axis ):
    if axis != self.ndim-1:
      return takediag( sum( self.func, axis ) )

  def _opposite( self ):
    return takediag( opposite(self.func) )

class Take( ArrayFunc ):
  'generalization of numpy.take(), to accept lists, slices, arrays'

  def __init__( self, func, indices, axis ):
    'constructor'

    assert func.shape[axis] != 1
    self.func = func
    self.axis = axis
    self.indices = indices

    # try for regular slice
    start = indices[0]
    step = indices[1] - start
    stop = start + step * len(indices)

    s = [ slice(None) ] * func.ndim
    s[axis] = slice( start, stop, step ) if numpy.all( numpy.diff(indices) == step ) \
         else indices

    newlen, = numpy.empty( func.shape[axis] )[ indices ].shape
    assert newlen > 0
    shape = func.shape[:axis] + (newlen,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+tuple(s)), shape=shape )

  evalf = staticmethod(numpy.ndarray.__getitem__)

  def _localgradient( self, ndims ):
    return take( localgradient( self.func, ndims ), self.indices, self.axis )

  def _opposite( self ):
    return take( opposite(self.func), self.indices, self.axis )

  def _take( self, index, axis ):
    if axis == self.axis:
      if numpy.all( numpy.diff( self.indices ) == 1 ):
        indices = index + self.indices[0]
      else:
        indices = self.indices[index]
      return take( self.func, indices, axis )
    trytake = take( self.func, index, axis )
    if not isinstance( trytake, Take ): # avoid inf recursion
      return take( trytake, self.indices, self.axis )

class Power( ArrayFunc ):
  'power'

  def __init__( self, func, power ):
    'constructor'

    assert _isfunc( func ) or _isfunc( power )
    assert _isscalar( power )
    self.func = func
    self.power = power
    ArrayFunc.__init__( self, args=[func,power], shape=func.shape )

  evalf = staticmethod(numpy.power)

  def _localgradient( self, ndims ):
    # self = func**power
    # ln self = power * ln func
    # self` / self = power` * ln func + power * func` / func
    # self` = power` * ln func * self + power * func` * func**(power-1)
    return self.power * power( self.func, self.power-1 )[...,_] * localgradient( self.func, ndims ) \
         + ( ln( self.func ) * self )[...,_] * localgradient( self.power, ndims )

  def _power( self, n ):
    func = self.func
    newpower = n * self.power
    if self.power % 2 == 0 and newpower % 2 != 0:
      func = abs( func )
    return power( func, newpower )

  def _get( self, i, item ):
    return get( self.func, i, item )**self.power

  def _sum( self, axis ):
    if self.power == 2:
      return dot( self.func, self.func, axis )

  def _takediag( self ):
    return takediag( self.func )**self.power

  def _take( self, index, axis ):
    return power( take( self.func, index, axis ), self.power )

  def _opposite( self ):
    return power( opposite(self.func), self.power )

  def _multiply( self, other ):
    if isinstance( other, Power ) and self.func == other.func:
      return power( self.func, self.power + other.power )
    if other == self.func:
      return power( self.func, self.power + 1 )

  def _sign( self ):
    if self.power % 2 == 0:
      return expand( 1., self.shape )

class ElemFunc( ArrayFunc ):
  'trivial func'

  def __init__( self, ndims, side=0 ):
    'constructor'

    self.side = side
    ArrayFunc.__init__( self, args=[POINTS,Elemtrans(side),ndims], shape=[ndims] )

  @staticmethod
  def evalf( points, trans, ndims ):
    'evaluate'

    return trans.slice(todims=ndims).apply( points ).astype( float )

  def _localgradient( self, ndims ):
    return eye( ndims ) if self.shape[0] == ndims \
      else Transform( self.shape[0], ndims, self.side )

  def _opposite( self ):
    ndims, = self.shape
    return ElemFunc( ndims, 1-self.side )

class Pointwise( ArrayFunc ):
  'pointwise transformation'

  def __init__( self, arr, evalf, deriv ):
    'constructor'

    assert _isfunc( arr )
    shape = arr.shape[1:]
    self.arr = arr
    self.evalf = evalf
    self.deriv = deriv
    ArrayFunc.__init__( self, args=tuple(arr), shape=shape )

  def _localgradient( self, ndims ):
    return ( self.deriv( self.arr )[...,_] * localgradient( self.arr, ndims ) ).sum( 0 )

  def _takediag( self ):
    return pointwise( takediag(self.arr), self.evalf, self.deriv )

  def _get( self, axis, item ):
    return pointwise( get( self.arr, axis+1, item ), self.evalf, self.deriv )

  def _take( self, index, axis ):
    return pointwise( take( self.arr, index, axis+1 ), self.evalf, self.deriv )

  def _opposite( self ):
    opp_arr = [ opposite(f) for f in self.arr ]
    return pointwise( opp_arr, self.evalf, self.deriv )

class Sign( ArrayFunc ):
  'sign'

  def __init__( self, func ):
    'constructor'

    assert _isfunc( func )
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  evalf = staticmethod(numpy.sign)

  def _localgradient( self, ndims ):
    return _zeros( self.shape + (ndims,) )

  def _takediag( self ):
    return sign( takediag(self.func) )

  def _get( self, axis, item ):
    return sign( get( self.func, axis, item ) )

  def _take( self, index, axis ):
    return sign( take( self.func, index, axis ) )

  def _opposite( self ):
    return sign( opposite( self.func ) )

  def _sign( self ):
    return self

  def _power( self, n ):
    if n % 2 == 0:
      return expand( 1., self.shape )

class Pointdata( ArrayFunc ):

  def __init__ ( self, data, shape ):
    'constructor'

    assert isinstance(data,dict)
    self.data = data
    ArrayFunc.__init__( self, args=[Elemtrans(0),POINTS,self.data], shape=shape )

  @staticmethod
  def evalf( trans, points, data ):
    myvals,mypoint = data[trans]
    assert numpy.equal( mypoint, points ).all(), 'Illegal point set'
    return myvals

  def update_max( self, func ):
    func = asarray(func)
    assert func.shape == self.shape
    data = dict( (trans,(numpy.maximum(func.eval((trans,trans),points),values),points)) for trans,(values,points) in self.data.iteritems() )

    return Pointdata( data, self.shape )


class Eig( Evaluable ):
  'Eig'

  def __init__( self, func, symmetric=False, sort=False ):
    'contructor'

    Evaluable.__init__( self, args=[func] )
    self.symmetric = symmetric
    self.func = func
    self.shape = func.shape
    self.evalf = numpy.linalg.eigh if symmetric else numpy.linalg.eig 

  def _opposite( self ):
    return Eig( opposite(self.func), self.symmetric )

class ArrayFromTuple( ArrayFunc ):

  def __init__( self, arrays, index, shape ):
    self.arrays = arrays
    self.index = index
    ArrayFunc.__init__( self, args=[arrays,index], shape=shape )

  @staticmethod
  def evalf( arrays, index ):
    return arrays[ index ]

  def _opposite( self ):
    return ArrayFromTuple( opposite(self.arrays), self.index, self.shape )

# PRIORITY OBJECTS
#
# Prority objects get priority in situations like A + B, which can be evaluated
# as A.__add__(B) and B.__radd__(A), such that they get to decide on how the
# operation is performed. The reason is that these objects are wasteful,
# generally introducing a lot of zeros, and we would like to see them disappear
# by the act of subsequent operations. For this annihilation to work well
# priority objects keep themselves at the surface where magic happens.
#
# Update: "priority objects" as such do not exist anymore, might be
# reintroduced later on.

class Zeros( ArrayFunc ):
  'zero'

  def __init__( self, shape ):
    'constructor'

    shape = tuple( shape )
    ArrayFunc.__init__( self, args=[POINTS,shape], shape=shape )

  @staticmethod
  def evalf( points, shape ):
    'prepend point axes'

    assert not any( sh is None for sh in shape ), 'cannot evaluate zeros for shape %s' % (shape,)
    shape = points.shape[:-1] + shape
    strides = [0] * len(shape)
    return numpy.lib.stride_tricks.as_strided( numpy.array(0.), shape, strides )

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

  def _negative( self ):
    return self

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

  def _inflate( self, dofmap, length, axis ):
    assert not isinstance( self.shape[axis], int )
    return _zeros( self.shape[:axis] + (length,) + self.shape[axis+1:] )

  def _elemint( self, weights ):
    return numpy.zeros( [1]*self.ndim )

  def _power( self, n ):
    return self

  def _opposite( self ):
    return self

class Inflate( ArrayFunc ):
  'inflate'

  def __init__( self, func, dofmap, length, axis ):
    'constructor'

    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.axis = axis
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,dofmap,length,axis-func.ndim], shape=shape )

  @staticmethod
  def evalf( array, indices, length, axis ):
    'inflate'

    warnings.warn( 'using explicit inflation; this is usually a bug.' )
    shape = list( array.shape )
    shape[axis] = length
    inflated = numpy.zeros( shape )
    inflated[(Ellipsis,indices)+(slice(None),)*(-axis-1)] = array
    return inflated

  @property
  def blocks( self ):
    for f, ind in blocks( self.func ):
      assert ind[self.axis] == None
      yield f, ind[:self.axis] + (self.dofmap,) + ind[self.axis+1:]

  def _inflate( self, dofmap, length, axis ):
    assert axis != self.axis
    if axis > self.axis:
      return
    return inflate( inflate( self.func, dofmap, length, axis ), self.dofmap, self.length, self.axis )

  def _localgradient( self, ndims ):
    return inflate( localgradient(self.func,ndims), self.dofmap, self.length, self.axis )

  def _align( self, shuffle, ndims ):
    return inflate( align(self.func,shuffle,ndims), self.dofmap, self.length, shuffle[self.axis] )

  def _get( self, axis, item ):
    assert axis != self.axis
    return inflate( get(self.func,axis,item), self.dofmap, self.length, self.axis-(axis<self.axis) )

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
    return inflate( arr, self.dofmap, self.length, self.axis )

  def _multiply( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    return inflate( multiply(self.func,other), self.dofmap, self.length, self.axis )

  def _add( self, other ):
    if isinstance( other, Inflate ) and self.axis == other.axis and self.dofmap == other.dofmap:
      return inflate( add(self.func,other.func), self.dofmap, self.length, self.axis )
    return BlockAdd( *_sorted( self, other ) )

  def _cross( self, other, axis ):
    if isinstance( other, Inflate ) and self.axis == other.axis:
      assert self.dofmap == other.dofmap
      other = other.func
    elif other.shape[self.axis] != 1:
      other = take( other, self.dofmap, self.axis )
    return inflate( cross(self.func,other,axis), self.dofmap, self.length, self.axis )

  def _negative( self ):
    return inflate( negative(self.func), self.dofmap, self.length, self.axis )

  def _power( self, n ):
    return inflate( power(self.func,n), self.dofmap, self.length, self.axis )

  def _takediag( self ):
    assert self.axis < self.ndim-2
    return inflate( takediag(self.func), self.dofmap, self.length, self.axis )

  def _take( self, index, axis ):
    if axis == self.axis:
      assert index == self.dofmap
      return self.func
    return inflate( take( self.func, index, axis ), self.dofmap, self.length, self.axis )

  def _diagonalize( self ):
    assert self.axis < self.ndim-1
    return inflate( diagonalize(self.func), self.dofmap, self.length, self.axis )

  def _sum( self, axis ):
    arr = sum( self.func, axis )
    if axis == self.axis:
      return arr
    return inflate( arr, self.dofmap, self.length, self.axis-(axis<self.axis) )

  def _opposite( self ):
    return inflate( opposite(self.func), opposite(self.dofmap), self.length, self.axis )

  def _repeat( self, length, axis ):
    if axis != self.axis:
      return inflate( repeat(self.func,length,axis), self.dofmap, self.length, self.axis )

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  def __init__( self, func ):
    'constructor'

    n = func.shape[-1]
    assert n != 1
    shape = func.shape + (n,)
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=shape )

  evalf = staticmethod(numeric.diagonalize)

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

  def _negative( self ):
    return diagonalize( -self.func )

  def _sum( self, axis ):
    if axis >= self.ndim-2:
      return self.func
    return diagonalize( sum( self.func, axis ) )

  def _align( self, axes, ndim ):
    if axes[-2:] in [ (ndim-2,ndim-1), (ndim-1,ndim-2) ]:
      return diagonalize( align( self.func, axes[:-2] + (ndim-2,), ndim-1 ) )

  def _opposite( self ):
    return diagonalize( opposite(self.func) )

class Repeat( ArrayFunc ):
  'repeat singleton axis'

  def __init__( self, func, length, axis ):
    'constructor'

    assert func.shape[axis] == 1
    self.func = func
    self.axis = axis
    self.length = length
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,length,axis-func.ndim], shape=shape )

  evalf = staticmethod(numeric.fastrepeat)

  def _negative( self ):
    return repeat( -self.func, self.length, self.axis )

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
    return repeat( power( self.func, n ), self.length, self.axis )

  def _add( self, other ):
    return repeat( self.func + other, self.length, self.axis )

  def _multiply( self, other ):
    return repeat( self.func * other, self.length, self.axis )

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
      return repeat( cross( self.func, other, axis ), self.length, self.axis )

  def _dot( self, other, naxes ):
    axes = range( self.ndim-naxes, self.ndim )
    func = dot( self.func, other, axes )
    if other.shape[self.axis] != 1:
      assert other.shape[self.axis] == self.length
      return func
    if self.axis >= self.ndim - naxes:
      return func * self.length
    return repeat( func, self.length, self.axis )

  def _opposite( self ):
    return repeat( opposite(self.func), self.length, self.axis )

class Const( ArrayFunc ):
  'pointwise transformation'

  def __init__( self, func ):
    'constructor'

    func = numpy.asarray( func )
    ArrayFunc.__init__( self, args=(POINTS,func), shape=func.shape )

  @staticmethod
  def evalf( points, arr ):
    'prepend point axes'

    shape = points.shape[:-1] + arr.shape
    strides = (0,) * (points.ndim-1) + arr.strides
    return numpy.lib.stride_tricks.as_strided( arr, shape, strides )

  def _localgradient( self, ndims ):
    return _zeros( self.shape+(ndims,) )

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

def _isiterable( obj ):
  'check for iterability'

  try:
    iter(obj)
  except TypeError:
    return False
  return True

def _obj2str( obj ):
  'convert object to string'

  if isinstance( obj, numpy.ndarray ):
    if obj.size < 6:
      return _obj2str(obj.tolist())
    return 'array<%s>' % 'x'.join( map( str, obj.shape ) )
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
  if obj is POINTS:
    return '<points>'
  if obj is WEIGHTS:
    return '<weights>'
  if obj is TRANS:
    return '<trans>'
  return str(obj)

def _findcommon( (a1,a2), (b1,b2) ):
  'find common item in 2x2 data'

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
_isscalar = lambda arg: asarray(arg).ndim == 0
_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_iszero = lambda arg: isinstance( arg, Zeros ) or isinstance( arg, numpy.ndarray ) and numpy.all( arg == 0 )
_isunit = lambda arg: not _isfunc(arg) and ( numpy.asarray(arg) == 1 ).all()
_subsnonesh = lambda shape: tuple( 1 if sh is None else sh for sh in shape )
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )
_zeros = lambda shape: Zeros( shape )
_zeros_like = lambda arr: _zeros( arr.shape )

# for consistency in Add and Multiply arguments: the smallest Evaluable first
_issorted = lambda a, b: not isinstance(b,Evaluable) or isinstance(a,Evaluable) and a <= b
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
  raise Exception, 'unknown dtype %s' % arg.dtype

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
  arg = numpy.asarray( arg )
  if arg.dtype == object:
    return stack( arg, axis=0 )
  elif numpy.all( arg == 0 ):
    return _zeros( arg.shape )
  else:
    return arg

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

  funcs = map( asarray, funcs )
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
      nmap.update( dict( (key, val+offset) for key, val in dofmap.dofmap.iteritems() ) )
      assert len( nmap ) == targetlen, 'Don`t allow overlap.'

    if cascade is None:
      cascade = func.cascade
    else:
      assert func.cascade == cascade, 'Functions have to be defined on domains of same dimension.'

    offset += inflated_func.shape[0]

  return function( fmap, nmap, offset, cascade.ndims )

def vectorize( args ):
  'vectorize'

  return util.sum( kronecker( func, axis=1, length=len(args), pos=ifun ) for ifun, func in enumerate( chain( args ) ) )

def expand( arg, shape ):
  'expand'

  arg = asarray( arg )
  shape = tuple(shape)
  assert len(shape) == arg.ndim

  for i, sh in enumerate( shape ):
    arg = repeat( arg, sh, i )
  assert arg.shape == shape

  return arg

def repeat( arg, length, axis ):
  'repeat'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if arg.shape[axis] == length:
    return arg

  assert arg.shape[axis] == 1

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
  assert numeric.isint( sh ), 'cannot get item %r from axis %r' % ( item, sh )

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

  if list(axes) == range(ndim):
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

def elemint( arg, weights ):
  'elementwise integration'

  arg = asarray( arg )

  if not _isfunc( arg ):
    return arg * ElemArea( weights )

  retval = _call( arg, '_elemint', weights )
  if retval is not None:
    return retval

  return ElemInt( arg, weights )

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

def sum( arg, axes=-1 ):
  'sum over multiply axes'

  arg = asarray( arg )

  if _isiterable(axes):
    if len(axes) == 0:
      return arg
    axes = _norm_and_sort( arg.ndim, axes )
    assert numpy.all( numpy.diff(axes) > 0 ), 'duplicate axes in sum'
    arg = sum( arg, axes[1:] )
    axis = axes[0]
  else:
    axis = numeric.normdim( arg.ndim, axes )

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

  if _isiterable(axes):
    if len(axes) == 0:
      return arg1 * arg2
    axes = _norm_and_sort( len(shape), axes )
    assert numpy.all( numpy.diff(axes) > 0 ), 'duplicate axes in sum'
  else:
    axes = numeric.normdim( len(shape), axes ),

  for i, axis in enumerate( axes ):
    if arg1.shape[axis] == 1 or arg2.shape[axis] == 1:
      arg1 = sum( arg1, axis )
      arg2 = sum( arg2, axis )
      axes = axes[:i] + tuple( axis-1 for axis in axes[i+1:] )
      return dot( arg1, arg2, axes )

  if _isunit( arg1 ):
    return sum( expand( arg2, shape ), axes )

  if _isunit( arg2 ):
    return sum( expand( arg1, shape ), axes )

  if not _isfunc(arg1) and not _isfunc(arg2):
    return numeric.contract( arg1, arg2, axes )

  shuffle = range( len(shape) )
  for ax in reversed( axes ):
    shuffle.append( shuffle.pop(ax) )

  arg1 = transpose( arg1, shuffle )
  arg2 = transpose( arg2, shuffle )

  naxes = len( axes )
  shape = tuple( shape[i] for i in shuffle[:-naxes] )

  retval = _call( arg1, '_dot', arg2, naxes )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._dot' % arg1
    return retval

  retval = _call( arg2, '_dot', arg1, naxes )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._dot' % arg2
    return retval

  return Dot( arg1, arg2, naxes )

def determinant( arg, axes=(-2,-1) ):
  'determinant'

  arg = asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict

  n = arg.shape[ax1]
  assert n == arg.shape[ax2]
  if n == 1:
    return get( get( arg, ax2, 0 ), ax1, 0 )

  trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
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

  trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
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

  axes = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
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

  return sum( arg * coords.normal( ndims-1 ) )

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
  if trans is None:
    invtrans = range( arg.ndim-1, -1, -1 )
  else:
    trans = _normdims( arg.ndim, trans )
    assert sorted(trans) == range(arg.ndim)
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

  retval = _call( arg, '_product', axis )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._product' % arg
    return retval

  return Product( arg, axis )

def choose( level, choices ):
  'choose'

  choices = _matchndim( *choices )
  if _isfunc(level) or any( _isfunc(choice) for choice in choices ):
    return Choose( level, choices )
  return numpy.choose( level, choices )

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

  arg1, arg2 = _matchndim( arg1, arg2 if arg2 is not None else arg1 )
  axis = numeric.normdim( arg1.ndim, axis )
  return insert(arg1,axis+1) * insert(arg2,axis)

def pointwise( args, evalf, deriv ):
  'general pointwise operation'

  args = asarray( _matchndim(*args) )
  if _isfunc(args):
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

def negative( arg ):
  'make negative'

  arg = asarray(arg)

  if not _isfunc( arg ):
    return numpy.negative( arg )

  retval = _call( arg, '_negative' )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._negative' % arg
    return retval

  return Negative( arg )

def power( arg, n ):
  'power'

  arg = asarray( arg )
  assert _isscalar( n )

  if n == 1:
    return arg

  if n == 0:
    return numpy.ones( arg.shape )

  if not _isfunc( arg ) and not _isfunc( n ):
    return numpy.power( arg, n )

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
  trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
  aligned_arg = align( arg, trans, arg.ndim )

  # When it's an array calculate directly
  if not _isfunc(aligned_arg):
    eigval, eigvec = numeric.eigh( aligned_arg ) if symmetric else numeric.eig( aligned_arg )
  else:
    # Use _call to see if the object has its own _eig function
    ret = _call( aligned_arg, '_eig' )
    if ret is not None:
      # Check the shapes
      eigval, eigvec = ret
    else:
      eig = Eig( aligned_arg, symmetric=symmetric )
      eigval = ArrayFromTuple( eig, index=0, shape=aligned_arg.shape[:-1] )
      eigvec = ArrayFromTuple( eig, index=1, shape=aligned_arg.shape )

  # Return the evaluable function objects in a tuple like numpy
  eigval = transpose( diagonalize( eigval ), trans )
  eigvec = transpose( eigvec, trans )
  assert eigvec.shape == arg.shape
  assert eigval.shape == arg.shape
  return eigval, eigvec

nsymgrad = lambda arg, coords: ( symgrad(arg,coords) * coords.normal() ).sum()
ngrad = lambda arg, coords: ( grad(arg,coords) * coords.normal() ).sum()
sin = lambda arg: pointwise( [arg], numpy.sin, cos )
cos = lambda arg: pointwise( [arg], numpy.cos, lambda x: -sin(x) )
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
piecewise = lambda level, intervals, *funcs: choose( sum( greater( insert(level,-1), intervals ) ), funcs )
trace = lambda arg, n1=-2, n2=-1: sum( takediag( arg, n1, n2 ) )
eye = lambda n: diagonalize( expand( [1.], (n,) ) )
norm2 = lambda arg, axis=-1: sqrt( sum( arg * arg, axis ) )
heaviside = lambda arg: choose( greater( arg, 0 ), [0.,1.] )
divide = lambda arg1, arg2: multiply( arg1, reciprocal(arg2) )
subtract = lambda arg1, arg2: add( arg1, negative(arg2) )
mean = lambda arg: .5 * ( arg + opposite(arg) )
jump = lambda arg: arg - opposite(arg)
add_T = lambda arg, axes=(-2,-1): swapaxes( arg, axes ) + arg

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

  return arg._opposite()

def function( fmap, nmap, ndofs, ndims ):
  'create function on ndims-element'

  axis = '~%d' % ndofs
  func = Function( ndims, fmap, igrad=0, axis=axis )
  dofmap = DofMap( nmap, axis=axis )
  return Inflate( func, dofmap, length=ndofs, axis=0 )

def take( arg, index, axis ):
  'take index'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if isinstance( index, slice ):
    assert index.start == None or index.start >= 0
    assert index.stop != None and index.stop > 0
    index = numpy.arange( index.start or 0, index.stop, index.step )
    assert index.size > 0
  elif not _isfunc( index ):
    index = numpy.asarray( index, dtype=int )
    assert numpy.all( index >= 0 )
    assert index.size > 0
  assert index.ndim == 1

  if arg.shape[axis] == 1:
    return repeat( arg, index.shape[0], axis )

  if not _isfunc( index ):
    allindices = numpy.arange( arg.shape[axis] )
    index = allindices[index]
    if numpy.all( index == allindices ):
      return arg

  if index.shape[0] == 1:
    return insert( get( arg, axis, index[0] ), axis )

  retval = _call( arg, '_take', index, axis )
  if retval is not None:
    return retval

  if _isfunc( index ):
    return DofIndex( arg, axis, index )

  if not _isfunc( arg ):
    return numpy.take( arg, index, axis )

  return Take( arg, index, axis )

def inflate( arg, dofmap, length, axis ):
  'inflate'

  arg = asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  assert not isinstance( arg.shape[axis], int )

  retval = _call( arg, '_inflate', dofmap, length, axis )
  if retval is not None:
    return retval

  return Inflate( arg, dofmap, length, axis )

def blocks( arg ):
  arg = asarray( arg )
  try:
    blocks = arg.blocks
  except AttributeError:
    blocks = [( arg, tuple( numpy.arange(n) if numeric.isint(n) else None for n in arg.shape ) )]
  return blocks

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
  __log__ = log.range( 'dof', ndofs )
  for i in __log__:
    pert = dofs.copy()
    pert[i] += step
    x1 = tuple( wi.dot( pert ) for wi in w )
    dfunc_fd.append( (func( *x1 ) - func( *x0 ))/step )
  return dfunc_fd

def iweights( coords, ndims ):
  'integration weights scale'

  assert coords.ndim == 1
  J = localgradient( coords, ndims )
  cndims, = coords.shape
  assert J.shape == (cndims,ndims), 'wrong jacobian shape: got %s, expected %s' % ( J.shape, (cndims, ndims) )
  assert cndims >= ndims, 'geometry dimension < topology dimension'
  detJ = abs( determinant( J ) ) if cndims == ndims \
    else 1. if ndims == 0 \
    else determinant( ( J[:,:,_] * J[:,_,:] ).sum(0) )**.5
  return detJ * IWeights()

def _unpack( funcsp ):
  for func, axes in funcsp.blocks:
    dofax = axes[0]
    if isinstance( dofax, Add ):
      dofax, dof0 = dofax.funcs
    else:
      dof0 = 0
    dofmap = dofax.dofmap
    stdmap = func.stdmap
    for trans, dofs in dofmap.items():
      yield trans, dofs + dof0, stdmap[trans]
  
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


## INTERNAL HELPER FUNCTIONS

def _isindex( item, iterable ):
  for index, obj in enumerate(iterable):
    if obj is item:
      return index
  return -1


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
