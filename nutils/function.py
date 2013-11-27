from . import util, numpy, numeric, log, prop, core, _
import sys, warnings

ELEM    = object()
POINTS  = object()
WEIGHTS = object()

# EVALUABLE
#
# Base class for everything that is evaluable. Evaluables hold an argument list
# and a callable, which will be called by the __call__ method. Function
# arguments that are themselves Evaluables are traced, and an optimal call
# ordering is determined such that any unique argument is evaluated only once.

class EvaluationError( Exception ):
  'evaluation error'

  def __init__( self, etype, evalue, evaluable, values ):
    'constructor'

    self.etype = etype
    self.evalue = evalue
    self.evaluable = evaluable
    self.values = values

  def __str__( self ):
    'string representation'

    return '\n%s --> %s: %s' % ( self.evaluable.stackstr( self.values ), self.etype.__name__, self.evalue )

class Evaluable( object ):
  'evaluable base classs'

  __slots__ = 'operations', 'data', '__args', '__evalf'

  def __init__( self, args, evalf ):
    'constructor'

    self.__args = tuple(args)
    self.__evalf = evalf
    self.operations = None

  def recurse_index( self, data, operations, cbuild ):
    'compile'

    indices = numpy.empty( len(self.__args), dtype=int )
    for iarg, arg in enumerate( self.__args ):
      if isinstance(arg,Evaluable):
        for idx, (op,evalf,idcs) in enumerate( operations ):
          if op == arg:
            break
        else:
          idx = arg.recurse_index(data,operations,cbuild)
      elif arg is ELEM:
        idx = -3
      elif arg is POINTS:
        idx = -2
      elif arg is WEIGHTS:
        idx = -1
      else:
        data.insert( 0, arg )
        idx = -len(data)-3
      indices[iarg] = idx
    if cbuild and self.__cdef:
      evalf = cbuild[ self.__cdef() ]
      if prop.use_c_funcs == 'debug':
        evalf = numeric.check_equal_wrapper( evalf, self.__evalf )
    else:
      evalf = self.__evalf
    operations.append( (self,evalf,indices) )
    return len(operations)-1

  def compile( self ):
    'compile'

    log.context( 'compiling' )

    if self.operations is None:
      cbuild = getattr( prop, 'use_c_funcs', False ) and CBuilder()
      self.data = []
      operations = []
      self.recurse_index( self.data, operations, cbuild ) # compile expressions
      self.operations = [ (evalf,idcs) for (op,evalf,idcs) in operations ] # break cycles!
      if getattr( prop, 'dot', False ):
        self.graphviz()
      if cbuild:
        cbuild.compile()

  def __call__( self, elem, ischeme ):
    'evaluate'

    if isinstance( ischeme, dict ):
      ischeme = ischeme[elem]

    if isinstance( ischeme, str ):
      points, weights = elem.eval( ischeme )
    elif isinstance( ischeme, tuple ):
      points, weights = ischeme
      assert points.shape[-1] == elem.ndims
      assert points.shape[:-1] == weights.shape, 'non matching shapes: points.shape=%s, weights.shape=%s' % ( points.shape, weights.shape )
    elif isinstance( ischeme, numpy.ndarray ):
      points = ischeme
      weights = None
      assert points.shape[-1] == elem.ndims
    elif ischeme is None:
      points = weights = None
    else:
      raise Exception, 'invalid integration scheme of type %r' % type(ischeme)

    self.compile()
    N = len(self.data) + 3
    values = self.data + [ elem, points, weights ]
    for evalf, indices in self.operations:
      args = [ values[N+i] for i in indices ]
      try:
        retval = evalf( *args )
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        raise EvaluationError, ( etype, evalue, self, values ), traceback
      values.append( retval )
    return values[-1]

  def graphviz( self, title='graphviz' ):
    'create function graph'

    log.context( title )
    self.compile()

    import os, subprocess

    dotpath = getattr( prop, 'dot', True )
    if not isinstance( dotpath, str ):
      dotpath = 'dot'

    imgtype = getattr( prop, 'imagetype', 'png' )
    imgpath = util.getpath( 'dot{0:03x}.' + imgtype )

    try:
      dot = subprocess.Popen( [dotpath,'-Tjpg'], stdin=subprocess.PIPE, stdout=open(imgpath,'w') )
    except OSError:
      log.error( 'error: failed to execute', dotpath )
      return False

    print >> dot.stdin, 'digraph {'
    print >> dot.stdin, 'graph [ dpi=72 ];'

    data = self.data + [ '<elem>', '<points>', '<weights>' ]
    for i, (evalf,indices) in enumerate( self.operations ):
      args = [ '%%%d=%s' % ( iarg, _obj2str( data[idx] ) ) for iarg, idx in enumerate( indices ) if idx < 0 ]
      vertex = 'label="%s"' % r'\n'.join( [ '%d. %s' % ( i, evalf.__name__ ) ] + args )
      #if evalf == Inflate.inflate:
      #  vertex += ', style=filled, fillcolor=black, fontcolor=white'
      print >> dot.stdin, '%d [%s]' % ( i, vertex )
      for iarg, idx in enumerate( indices ):
        if idx >= 0:
          print >> dot.stdin, '%d -> %d [label="%%%d"];' % ( idx, i, iarg );

    print >> dot.stdin, '}'
    dot.stdin.close()

    log.path( os.path.basename(imgpath) )

  def stackstr( self, values=None ):
    'print stack'

    self.compile()
    if values is None:
      values = self.data + [ '<elem>', '<points>', '<weights>' ]

    N = len(self.data) + 3

    lines = []
    for i, (evalf,indices) in enumerate( self.operations ):
      line = '  %%%d =' % i
      args = [ '%%%d' % idx if idx >= 0 else _obj2str(values[N+idx]) for idx in indices ]
      try:
        code = evalf.func_code
        names = code.co_varnames[ :code.co_argcount ]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(indices) - len(names) ) )
        args = [ '%s=%s' % item for item in zip( names, args ) ]
      except:
        pass
      line += ' %s( %s )' % ( evalf.__name__, ', '.join( args ) )
      lines.append( line )
      if N+i == len(values):
        break
    return '\n'.join( lines )

  def __eq__( self, other ):
    'compare'

    return self is other or (
          self.__class__ == other.__class__
      and self.__evalf == other.__evalf
      and len( self.__args ) == len( other.__args )
      and all( _equal(arg1,arg2) for arg1, arg2 in zip( self.__args, other.__args ) ) )

  def __ne__( self, other ):
    'not equal'

    return not self == other

  def asciitree( self ):
    'string representation'

    key = self.__evalf.__name__
    lines = []
    indent = '\n' + ' ' + ' ' * len(key)
    for it in reversed( self.__args ):
      s = it.asciitree() if isinstance(it,Evaluable) else _obj2str(it)
      lines.append( indent.join( s.splitlines() ) )
      indent = '\n' + '|' + ' ' * len(key)
    indent = '\n' + '+' + '-' * (len(key)-1) + ' '
    return key + ' ' + indent.join( reversed( lines ) )

class Tuple( Evaluable ):
  'combine'

  __slots__ = 'items',

  def __init__( self, items ):
    'constructor'

    self.items = tuple( items )
    Evaluable.__init__( self, args=self.items, evalf=self.vartuple )

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

  @staticmethod
  def vartuple( *f ):
    'evaluate'

    return f

class PointShape( Evaluable ):
  'shape of integration points'

  __slots__ = ()

  def __init__( self ):
    'constructor'

    return Evaluable.__init__( self, args=[POINTS], evalf=self.pointshape )

  @staticmethod
  def pointshape( points ):
    'evaluate'

    return points.shape[:-1]

class Cascade( Evaluable ):
  'point cascade: list of (elem,points) tuples'

  __slots__ = 'ndims', 'side'

  def __init__( self, ndims, side=0 ):
    'constructor'

    self.ndims = ndims
    self.side = side
    Evaluable.__init__( self, args=[ELEM,POINTS,ndims,side], evalf=self.cascade )

  def transform( self, ndims ):
    if ndims == self.ndims:
      return eye( ndims )
    assert self.ndims > ndims
    return Transform( self, Cascade(ndims,self.side) )

  @staticmethod
  def cascade( elem, points, ndims, side ):
    'evaluate'

    while elem.ndims != ndims \
        or elem.interface and elem.interface[side][0].ndims < elem.ndims: # TODO make less dirty
      elem, transform = elem.interface[side] if elem.interface \
                   else elem.context or elem.parent
      points = transform.eval( points )

    cascade = [ (elem,points) ]
    while elem.parent:
      elem, transform = elem.parent
      points = transform.eval( points )
      cascade.append( (elem,points) )

    return cascade

  @property
  def inv( self ):
    return Cascade( self.ndims, 1-self.side )

# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

class ArrayFunc( Evaluable ):
  'array function'

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530
  __slots__ = 'shape', 'ndim', 'dtype'

  def __init__( self, evalf, args, shape, dtype=float ):
    'constructor'

    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    assert dtype is int or dtype is float
    self.dtype = dtype
    Evaluable.__init__( self, evalf=evalf, args=args )

  # mathematical operators

  def __mul__( self, other ): return multiply( self, other )
  def __rmul__( self, other ): return multiply( other, self )
  def __div__( self, other ): return divide( self, other )
  def __rdiv__( self, other ): return divide( other, self )
  def __add__( self, other ): return add( self, other )
  def __radd__( self, other ): return add( other, self )
  def __sub__( self, other ): return subtract( self, other )
  def __rsub__( self, other ): return subtract( other, self )
  def __neg__( self ): return negative( self )
  def __pow__( self, n ): return power( self, n )
  def sum( self, axes=-1 ): return sum( self, axes )

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

  def __kronecker__( self, axis, length, pos ):
    'kronecker'

    assert self.shape[axis] == 1
    funcs = [ _zeros_like(self) ] * length
    funcs[pos] = self
    return Concatenate( funcs, axis=axis )

  def __getitem__( self, item ):
    'get item, general function which can eliminate, add or modify axes.'
  
    myitem = list( item if isinstance( item, tuple ) else [item] )
    n = 0
    arr = self
    while myitem:
      it = myitem.pop(0)
      if isinstance(it,int): # retrieve one item from axis
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
      normal = OrientationHack()[_]
    else:
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims )
    return normal

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

  def dotnorm( self, coords, ndims=0 ):
    'normal component'

    return sum( self * coords.normal( ndims-1 ) )

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

class ElemArea( ArrayFunc ):
  'element area'

  __slots__ = ()

  def __init__( self, weights ):
    'constructor'

    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights], evalf=self.elemarea, shape=weights.shape )

  @staticmethod
  def elemarea( weights ):
    'evaluate'

    return numpy.sum( weights )

class ElemInt( ArrayFunc ):
  'elementwise integration'

  __slots__ = ()

  def __init__( self, func, weights ):
    'constructor'

    assert _isfunc( func ) and _isfunc( weights )
    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights,func,func.ndim], evalf=self.elemint, shape=func.shape )

  @staticmethod
  def elemint( w, f, ndim ):
    'evaluate'

    if f.ndim == ndim: # the missing point axis problem
      return f * w.sum()
    return numeric.dot( w, f ) if w.size else numpy.zeros( f.shape[1:] )

class Align( ArrayFunc ):
  'align axes'

  __slots__ = 'func', 'axes'

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
    ArrayFunc.__init__( self, args=[func,negaxes,ndim], evalf=self.align, shape=shape )

  @staticmethod
  def align( arr, trans, ndim ):
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

  __slots__ = 'func', 'axis', 'item'

  def __init__( self, func, axis, item ):
    'constructor'

    self.func = func
    self.axis = axis
    self.item = item
    assert 0 <= axis < func.ndim, 'axis is out of bounds'
    assert 0 <= item < func.shape[axis], 'item is out of bounds'
    s = (Ellipsis,item) + (slice(None),)*(func.ndim-axis-1)
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=(func,s), evalf=numpy.ndarray.__getitem__, shape=shape )

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

  __slots__ = 'func', 'axis'

  def __init__( self, func, axis ):
    'constructor'

    self.func = func
    self.axis = axis
    assert 0 <= axis < func.ndim
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], evalf=numpy.product, shape=shape )

  def _localgradient( self, ndims ):
    return self[...,_] * ( localgradient(self.func,ndims) / self.func[...,_] ).sum( self.axis )

  def _get( self, i, item ):
    func = get( self.func, i+(i>=self.axis), item )
    return product( func, self.axis-(i<self.axis) )

  def _opposite( self ):
    return product( opposite(self.func), self.axis )

class IWeights( ArrayFunc ):
  'integration weights'

  __slots__ = ()

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[ELEM,WEIGHTS], evalf=self.iweights, shape=() )

  @staticmethod
  def iweights( elem, weights ):
    'evaluate'

    return elem.root_det * weights

class OrientationHack( ArrayFunc ):
  'orientation hack for 1d elements; VERY dirty'

  __slots__ = 'side',

  def __init__( self, side=0 ):
    'constructor'

    self.side = side
    ArrayFunc.__init__( self, args=[ELEM,side], evalf=self.orientation, shape=[] )

  @staticmethod
  def orientation( elem, side ):
    'evaluate'

    pelem, trans = elem.interface[side] if elem.interface else elem.context
    offset, = trans.offset
    return numpy.sign( offset - .5 )

  def _opposite( self ):
    return OrientationHack( 1-self.side )

class Transform( ArrayFunc ):
  'transform'

  __slots__ = 'fromcascade', 'tocascade', 'side'

  def __init__( self, fromcascade, tocascade, side=0 ):
    'constructor'

    assert fromcascade.ndims > tocascade.ndims
    self.fromcascade = fromcascade
    self.tocascade = tocascade
    self.side = side
    ArrayFunc.__init__( self, args=[fromcascade,tocascade,side], evalf=self.transform, shape=(fromcascade.ndims,tocascade.ndims) )

  @staticmethod
  def transform( fromcascade, tocascade, side ):
    'transform'

    fromelem = fromcascade[0][0]
    toelem = tocascade[0][0]

    elem = toelem
    T = elem.inv_root_transform
    while elem is not fromelem:
      elem, transform = elem.interface[side] if elem.interface \
                   else elem.context or elem.parent
      T = numpy.dot( transform.transform, T )
    T = numpy.dot( elem.root_transform, T )

    return T

  def _localgradient( self, ndims ):
    return _zeros( self.shape + (ndims,) )

  def _opposite( self ):
    return Transform( self.fromcascade, self.tocascade, 1-self.side )

class Function( ArrayFunc ):
  'function'

  __slots__ = 'cascade', 'stdmap', 'igrad'

  def __init__( self, cascade, stdmap, igrad, axis ):
    'constructor'

    self.cascade = cascade
    self.stdmap = stdmap
    self.igrad = igrad
    ArrayFunc.__init__( self, args=(cascade,stdmap,igrad), evalf=self.function, shape=(axis,)+(cascade.ndims,)*igrad )

  @staticmethod
  def function( cascade, stdmap, igrad ):
    'evaluate'

    fvals = []
    for elem, points in cascade:
      std = stdmap.get(elem)
      if not std:
        continue
      if isinstance( std, tuple ):
        std, keep = std
        F = std.eval(points,grad=igrad)[(Ellipsis,keep)+(slice(None),)*igrad]
      else:
        F = std.eval(points,grad=igrad)
      for axis in range(-igrad,0):
        F = numeric.dot( F, elem.inv_root_transform, axis )
      fvals.append( F )
    assert fvals, 'no function values encountered'
    return fvals[0] if len(fvals) == 1 else numpy.concatenate( fvals, axis=-1-igrad )

  def _opposite( self ):
    return Function( self.cascade.inv, self.stdmap, self.igrad, self.shape[0] )

  def _localgradient( self, ndims ):
    assert ndims <= self.cascade.ndims
    grad = Function( self.cascade, self.stdmap, self.igrad+1, self.shape[0] )
    return grad if ndims == self.cascade.ndims \
      else dot( grad[...,_], self.cascade.transform( ndims ), axes=-2 )

class Choose( ArrayFunc ):
  'piecewise function'

  __slots__ = 'level', 'choices'

  def __init__( self, level, choices ):
    'constructor'

    self.level = level
    self.choices = tuple(choices)
    shape = _jointshape( *[ choice.shape for choice in choices ] )
    level = level[ (_,)*(len(shape)-level.ndim) ]
    assert level.ndim == len( shape )
    ArrayFunc.__init__( self, args=(level,)+self.choices, evalf=self.choose, shape=shape )

  @staticmethod
  def choose( level, *choices ):
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

  __slots__ = ()

  def __init__( self, coords, contour, fin, fout ):
    'constructor'

    shape = _jointshape( fin.shape, fout.shape )
    ArrayFunc.__init__( self, args=(coords,contour,fin,fout), evalf=self.choose2d, shape=shape )

  @staticmethod
  def choose2d( xy, contour, fin, fout ):
    'evaluate'

    from matplotlib import nxutils
    mask = nxutils.points_inside_poly( xy.T, contour )
    out = numpy.empty( fin.shape or fout.shape )
    out[...,mask] = fin[...,mask] if fin.shape else fin
    out[...,~mask] = fout[...,~mask] if fout.shape else fout
    return out

class Inverse( ArrayFunc ):
  'inverse'

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numeric.inverse, shape=func.shape )

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

  __slots__ = 'cascade', 'dofmap'

  def __init__( self, cascade, dofmap, axis ):
    'new'

    self.cascade = cascade
    self.dofmap = dofmap
    ArrayFunc.__init__( self, args=(cascade,dofmap), evalf=self.evalmap, shape=[axis], dtype=int )

  @staticmethod
  def evalmap( cascade, dofmap ):
    'evaluate'

    alldofs = []
    for elem, points in cascade:
      dofs = dofmap.get( elem )
      if dofs is not None:
        alldofs.append( dofs )
    assert alldofs, 'no dofs encountered'
    return alldofs[0] if len(alldofs) == 1 else numpy.concatenate( alldofs )

  def _opposite( self ):
    return DofMap( self.cascade.inv, self.dofmap, self.shape[0] )

class Concatenate( ArrayFunc ):
  'concatenate'

  __slots__ = 'funcs', 'axis'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    funcs = [ _asarray(func) for func in funcs ]
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
    ArrayFunc.__init__( self, args=(axis-ndim,)+self.funcs, evalf=self.concatenate, shape=shape )

  @staticmethod
  def concatenate( iax, *arrays ):
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

class Interpolate( ArrayFunc ):
  'interpolate uniformly spaced data; stepwise for now'

  __slots__ = ()

  def __init__( self, x, xp, fp, left=None, right=None ):
    'constructor'

    xp = numpy.array( xp )
    fp = numpy.array( fp )
    assert xp.ndim == fp.ndim == 1
    if not numpy.all( numpy.diff(xp) > 0 ):
      warnings.warn( 'supplied x-values are non-increasing' )

    assert x.ndim == 0
    ArrayFunc.__init__( self, args=[x,xp,fp,left,right], evalf=numpy.interp, shape=() )

class Cross( ArrayFunc ):
  'cross product'

  __slots__ = 'func1', 'func2', 'axis'

  def __init__( self, func1, func2, axis ):
    'contructor'

    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    shape = _jointshape( func1.shape, func2.shape )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    ArrayFunc.__init__( self, args=(func1,func2,axis-len(shape)), evalf=numeric.cross, shape=shape )

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

  __slots__ = 'func',

  def __init__( self, func ):
    'contructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numeric.determinant, shape=func.shape[:-2] )

  def _localgradient( self, ndims ):
    Finv = swapaxes( inverse( self.func ) )
    G = localgradient( self.func, ndims )
    return self[...,_] * sum( Finv[...,_] * G, axes=[-3,-2] )

  def _opposite( self ):
    return determinant( opposite(self.func) )

class DofIndex( ArrayFunc ):
  'element-based indexing'

  __slots__ = 'array', 'iax', 'index'

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
    ArrayFunc.__init__( self, args=[self.array]+item, evalf=self.dofindex, shape=shape )

  @staticmethod
  def dofindex( arr, *item ):
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

  __slots__ = 'funcs',

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.multiply, shape=shape )

  def cdef( self ):
    'generate C code'

    arg1 = 'arg1[%s]' % _cindex( self.funcs[0].shape, self.shape )
    arg2 = 'arg2[%s]' % _cindex( self.funcs[1].shape, self.shape )
    body = 'retval[0] = %s * %s;\nretval++;' % ( arg1, arg2 )
    shape = _cshape( self.shape )
    for i in reversed( range(self.ndim) ):
      body = _cfor( varname='i%d'%i, count=shape[i], do=body )
    if self.ndim:
      body = 'int %s;\n' % ', '.join( 'i%d' % idim for idim in range(self.ndim) ) + body
    args = [ '%s* retval' % _dtypestr(self),
             '%s* arg1' % _dtypestr(self.funcs[0]),
             '%s* arg2' % _dtypestr(self.funcs[1]) ] \
         + [ 'int %s' % s for s in shape if isinstance(s,str) ]
    return _cfunc( args, body ), init_args_for( self, *self.funcs )

  def __eq__( self, other ):
    'compare'

    return self is other or (
          isinstance( other, Multiply )
      and _matchpairs( self.funcs, other.funcs ) )

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
    if Multiply( func1, other ) != func1_other:
      return multiply( func1_other, func2 )
    func2_other = multiply( func2, other )
    if Multiply( func2, other ) != func2_other:
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

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.negative, shape=func.shape )

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

  __slots__ = 'funcs',

  def __init__( self, func1, func2 ):
    'constructor'

    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    dtype = _jointdtype(func1,func2)
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.add, shape=shape, dtype=dtype )

  def cdef( self ):
    'generate C code'

    arg1 = 'arg1[%s]' % _cindex( self.funcs[0].shape, self.shape )
    arg2 = 'arg2[%s]' % _cindex( self.funcs[1].shape, self.shape )
    body = 'retval[0] = %s + %s;\nretval++;' % ( arg1, arg2 )
    shape = _cshape( self.shape )
    for i in reversed( range(self.ndim) ):
      body = _cfor( varname='i%d'%i, count=shape[i], do=body )
    if self.ndim:
      body = 'int %s;\n' % ', '.join( 'i%d' % idim for idim in range(self.ndim) ) + body
    args = [ '%s* retval' % _dtypestr(self),
             '%s* arg1' % _dtypestr(self.funcs[0]),
             '%s* arg2' % _dtypestr(self.funcs[1]) ] \
         + [ 'int %s' % s for s in shape if isinstance(s,str) ]
    return _cfunc( args, body ), init_args_for( self, *self.funcs )

  def __eq__( self, other ):
    'compare'

    return self is other or (
          isinstance( other, Add )
      and _matchpairs( self.funcs, other.funcs ) )

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
    if Add( func1, other ) != func1_other:
      return add( func1_other, func2 )
    func2_other = add( func2, other )
    if Add( func2, other ) != func2_other:
      return add( func1, func2_other )

class BlockAdd( Add ):
  'block addition (used for DG)'

  __slots__ = ()

  def _multiply( self, other ):
    func1, func2 = self.funcs
    return BlockAdd( func1 * other, func2 * other )

  def _inflate( self, dofmap, length, axis ):
    func1, func2 = self.funcs
    return BlockAdd( inflate( func1, dofmap, length, axis ),
                     inflate( func2, dofmap, length, axis ) )

  def _align( self, axes, ndim ):
    func1, func2 = self.funcs
    return BlockAdd( align(func1,axes,ndim), align(func2,axes,ndim) )

  def _negative( self ):
    func1, func2 = self.funcs
    return BlockAdd( negative(func1), negative(func2) )

  def _add( self, other ):
    func1, func2 = self.funcs
    try1 = func1 + other
    if try1 != BlockAdd( func1, other ):
      return try1 + func2
    try2 = func2 + other
    if try2 != BlockAdd( func2, other ):
      return try2 + func1
    return BlockAdd( self, other )

  @property
  def blocks( self ):
    func1, func2 = self.funcs
    for f, ind in blocks( func1 ):
      yield f, ind
    for f, ind in blocks( func2 ):
      yield f, ind

class Dot( ArrayFunc ):
  'dot'

  __slots__ = 'func1', 'func2', 'naxes'

  def __init__( self, func1, func2, naxes ):
    'constructor'

    assert naxes > 0
    self.func1 = func1
    self.func2 = func2
    self.naxes = naxes
    shape = _jointshape( func1.shape, func2.shape )[:-naxes]
    ArrayFunc.__init__( self, args=(func1,func2,naxes), evalf=numeric.contract_fast, shape=shape )

  def cdef( self ):
    'generate C code'

    shape = _jointshape( self.func1.shape, self.func2.shape )
    arg1 = 'arg1[%s]' % _cindex( self.func1.shape, shape )
    arg2 = 'arg2[%s]' % _cindex( self.func2.shape, shape )
    body = 'sum += %s * %s;\n' % ( arg1, arg2 )
    cshape = _cshape( shape )
    for i in reversed( range(self.ndim+self.naxes) ):
      body = _cfor( varname='i%d'%i, count=cshape[i], do=body )
      if i == self.ndim:
        body = 'sum = 0;\n%s\nretval[0] = sum;\nretval++;\n' % body
    body = '%s sum;\n' % _dtypestr(self) + body
    body = 'int %s;\n' % ', '.join( 'i%d' % idim for idim in range(self.ndim+self.naxes) ) + body
    args = [ '%s* retval' % _dtypestr(self),
             '%s* arg1' % _dtypestr(self.func1),
             '%s* arg2' % _dtypestr(self.func2) ] \
         + [ 'int %s' % s for s in cshape if isinstance(s,str) ]
    return _cfunc( args, body ), init_args_for( self, self.func1, self.func2, None )

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
    if Multiply( self.func1, other ) != func1_other:
      return dot( func1_other, self.func2, self.axes )
    func2_other = multiply( self.func2, other )
    if Multiply( self.func2, other ) != func2_other:
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

  __slots__ = 'axis', 'func'

  def __init__( self, func, axis ):
    'constructor'

    self.axis = axis
    self.func = func
    assert 0 <= axis < func.ndim, 'axis out of bounds'
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], evalf=numpy.sum, shape=shape )

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

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=self.debug, shape=func.shape )

  @staticmethod
  def debug( arr ):
    'debug'

    print arr
    return arr

  def __str__( self ):
    'string representation'

    return '{DEBUG}'

  def _localgradient( self, ndims ):
    return Debug( localgradient( self.func, ndims ) )

class TakeDiag( ArrayFunc ):
  'extract diagonal'

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numeric.takediag, shape=func.shape[:-1] )

  def _localgradient( self, ndims ):
    return swapaxes( takediag( localgradient( self.func, ndims ), -3, -2 ) )

  def _sum( self, axis ):
    if axis != self.ndim-1:
      return takediag( sum( self.func, axis ) )

  def _opposite( self ):
    return takediag( opposite(self.func) )

class Take( ArrayFunc ):
  'generalization of numpy.take(), to accept lists, slices, arrays'

  __slots__ = 'func', 'axis', 'indices'

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
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+tuple(s)), evalf=numpy.ndarray.__getitem__, shape=shape )

  def _localgradient( self, ndims ):
    return take( localgradient( self.func, ndims ), self.indices, self.axis )

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

  __slots__ = 'func', 'power'

  def __init__( self, func, power ):
    'constructor'

    assert _isfunc( func )
    assert _isscalar( power )
    self.func = func
    self.power = power
    ArrayFunc.__init__( self, args=[func,power], evalf=numpy.power, shape=func.shape )

  def _localgradient( self, ndims ):
    return self.power * ( self.func**(self.power-1) )[...,_] * localgradient( self.func, ndims )

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

  __slots__ = 'domainelem', 'side', 'cascade'

  def __init__( self, domainelem, side=0 ):
    'constructor'

    self.domainelem = domainelem
    self.side = side
    self.cascade = Cascade( domainelem.ndims, side )
    ArrayFunc.__init__( self, args=[self.cascade,domainelem], evalf=self.elemfunc, shape=[domainelem.ndims] )

  @staticmethod
  def elemfunc( cascade, domainelem ):
    'evaluate'

    for elem, points in cascade:
      if elem is domainelem:
        return points
    raise Exception, '%r not found' % domainelem

  def _localgradient( self, ndims ):
    return self.cascade.transform( ndims )

  def _opposite( self ):
    return ElemFunc( self.domainelem, 1-self.side )

  def find( self, elem, C ):
    'find coordinates'

    assert C.ndim == 2 and C.shape[1] == self.domainelem.ndims
    assert elem.ndims == self.domainelem.ndims # for now
    pelem, transform = elem.parent
    offset = transform.offset
    Tinv = transform.invtrans
    while pelem is not self.domainelem:
      pelem, newtransform = pelem.parent
      transform = transform.nest( newtransform )
    return elem.select_contained( transform.invapply( C ), eps=1e-10 )

class Pointwise( ArrayFunc ):
  'pointwise transformation'

  __slots__ = 'args', 'evalf', 'deriv'

  def __init__( self, args, evalf, deriv ):
    'constructor'

    assert _isfunc( args )
    shape = args.shape[1:]
    self.args = args
    self.evalf = evalf
    self.deriv = deriv
    ArrayFunc.__init__( self, args=tuple(args), evalf=evalf, shape=shape )

  def _localgradient( self, ndims ):
    return ( self.deriv( self.args )[...,_] * localgradient( self.args, ndims ) ).sum( 0 )

  def _takediag( self ):
    return pointwise( takediag(self.args), self.evalf, self.deriv )

  def _get( self, axis, item ):
    return pointwise( get( self.args, axis+1, item ), self.evalf, self.deriv )

  def _take( self, index, axis ):
    return pointwise( take( self.args, index, axis+1 ), self.evalf, self.deriv )

  def _opposite( self ):
    # args = [ opposite(arg,side) for arg in self.args ] # @TO
    args = [ opposite(arg) for arg in self.args ]
    return pointwise( args, self.evalf, self.deriv )

class Sign( ArrayFunc ):
  'sign'

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    assert _isfunc( func )
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.sign, shape=func.shape )

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

  __slots__ = 'data',

  def __init__ ( self, data, shape ):
    'constructor'

    assert isinstance(data,dict)
    self.data = data
    ArrayFunc.__init__( self, args=[ELEM,POINTS,self.data], evalf=self.pointdata, shape=shape )
    
  @staticmethod  
  def pointdata( elem, points, data ):
    myvals,mypoint = data[elem]
    assert mypoint is points, 'Illegal point set'
    return myvals

  def update_max( self, func ):
    func = _asarray(func)
    assert func.shape == self.shape
    data = dict( (elem,(numpy.maximum(func(elem,points),values),points)) for elem,(values,points) in self.data.iteritems() )

    return Pointdata( data, self.shape )

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

  __slots__ = ()

  def __init__( self, shape ):
    'constructor'

    shape = tuple( shape )
    ArrayFunc.__init__( self, args=[POINTS,shape], evalf=self.zeros, shape=shape )

  @property
  def blocks( self ):
    return ()

  @staticmethod
  def zeros( points, shape ):
    'prepend point axes'

    assert not any( sh is None for sh in shape ), 'cannot evaluate zeros for shape %s' % (shape,)
    shape = points.shape[:-1] + shape
    strides = [0] * len(shape)
    return numpy.lib.stride_tricks.as_strided( numpy.array(0.), shape, strides )

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

  __slots__ = 'func', 'dofmap', 'length', 'axis'

  def __init__( self, func, dofmap, length, axis ):
    'constructor'

    self.func = func
    self.dofmap = dofmap
    self.length = length
    self.axis = axis
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,dofmap,length,axis-func.ndim], evalf=self.inflate, shape=shape )

  @property
  def blocks( self ):
    for f, ind in blocks( self.func ):
      assert ind[self.axis] == None
      yield f, ind[:self.axis] + (self.dofmap,) + ind[self.axis+1:]

  @staticmethod
  def inflate( array, indices, length, axis ):
    'inflate'

    warnings.warn( 'using explicit inflation; this is usually a bug.' )
    shape = list( array.shape )
    shape[axis] = length
    inflated = numpy.zeros( shape )
    inflated[(Ellipsis,indices)+(slice(None),)*(-axis-1)] = array
    return inflated

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
    return BlockAdd( self, other )

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

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  __slots__ = 'func',

  def __init__( self, func ):
    'constructor'

    n = func.shape[-1]
    assert n != 1
    shape = func.shape + (n,)
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numeric.diagonalize, shape=shape )

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

  __slots__ = 'func', 'axis', 'length'

  def __init__( self, func, length, axis ):
    'constructor'

    assert func.shape[axis] == 1
    self.func = func
    self.axis = axis
    self.length = length
    shape = func.shape[:axis] + (length,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,length,axis-func.ndim], evalf=numeric.fastrepeat, shape=shape )

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
    assert self.axis < self.ndim-2
    return repeat( takediag( self.func ), self.length, self.axis )

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

  __slots__ = ()

  def __init__( self, func ):
    'constructor'

    func = numpy.asarray( func )
    ArrayFunc.__init__( self, args=(POINTS,func), evalf=self.const, shape=func.shape )

  @staticmethod
  def const( points, arr ):
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

  arrays = [ _asarray(array) for array in arrays ]
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
  if obj is ELEM:
    return '<elem>'
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

_matchpairs = lambda (a1,a2), (b1,b2): _equal(a1,b1) and _equal(a2,b2) or _equal(a1,b2) and _equal(a2,b1)
_max = max
_min = min
_sum = sum
_isfunc = lambda arg: isinstance( arg, ArrayFunc )
_isscalar = lambda arg: _asarray(arg).ndim == 0
_isint = lambda arg: numpy.asarray( arg ).dtype == int
_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_iszero = lambda arg: isinstance( arg, Zeros ) or isinstance( arg, numpy.ndarray ) and numpy.all( arg == 0 )
_isunit = lambda arg: not _isfunc(arg) and ( numpy.asarray(arg) == 1 ).all()
_subsnonesh = lambda shape: tuple( 1 if sh is None else sh for sh in shape )
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )
_zeros = lambda shape: Zeros( shape )
_zeros_like = lambda arr: _zeros( arr.shape )

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

  if any( _asarray(arg).dtype == float for arg in args ):
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

def _asarray( arg ):
  'convert to ArrayFunc or numpy.ndarray'
  
  if _isfunc(arg):
    return arg
  if isinstance( arg, (list,tuple) ) and any( _isfunc(f) for f in arg ):
    return stack( arg, axis=0 )
  arg = numpy.asarray( arg )
  if numpy.all( arg == 0 ):
    return _zeros( arg.shape )
  assert arg.dtype != object
  return arg

# FUNCTIONS

def insert( arg, n ):
  'insert axis'

  arg = _asarray( arg )
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

  funcs = map( _asarray, funcs )
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

  arg = _asarray( arg )
  shape = tuple(shape)
  assert len(shape) == arg.ndim

  for i, sh in enumerate( shape ):
    arg = repeat( arg, sh, i )
  assert arg.shape == shape

  return arg

def repeat( arg, length, axis ):
  'repeat'

  arg = _asarray( arg )
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

  assert _isint( item ) and _isscalar( item )

  arg = _asarray( arg )
  iax = numeric.normdim( arg.ndim, iax )
  sh = arg.shape[iax]
  assert isinstance(sh,int), 'cannot get item %r from axis %r' % ( item, sh )

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

  arg = _asarray( arg )

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

  arg = _asarray(arg)
  axis = numeric.normdim(arg.ndim,axis)
  if axis == 0:
    return arg
  return transpose( args, [axis] + range(axis) + range(axis+1,args.ndim) )

def elemint( arg, weights ):
  'elementwise integration'

  arg = _asarray( arg )

  if not _isfunc( arg ):
    return arg * ElemArea( weights )

  retval = _call( arg, '_elemint', weights )
  if retval is not None:
    return retval

  return ElemInt( arg, weights )

def grad( arg, coords, ndims=0 ):
  'local derivative'

  arg = _asarray( arg )
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

  arg = _asarray( arg )

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

  arg = _asarray( arg )
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
    return numeric.determinant( arg )

  retval = _call( arg, '_determinant' )
  if retval is not None:
    assert retval.shape == shape, 'bug in %s._determinant' % arg
    return retval

  return Determinant( arg )

def inverse( arg, axes=(-2,-1) ):
  'inverse'

  arg = _asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )
  assert ax2 > ax1 # strict

  n = arg.shape[ax1]
  assert arg.shape[ax2] == n
  if n == 1:
    return reciprocal( arg )

  trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
  arg = align( arg, trans, arg.ndim )

  if not _isfunc( arg ):
    return numeric.inverse( arg ).transpose( trans )

  retval = _call( arg, '_inverse' )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._inverse' % arg
    return transpose( retval, trans )

  return transpose( Inverse(arg), trans )

def takediag( arg, ax1=-2, ax2=-1 ):
  'takediag'

  arg = _asarray( arg )
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

  arg = _asarray( arg )
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

  arg = _asarray( arg )
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
  
  arg = _asarray( arg )
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

  arg = _asarray( arg )
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

  args = _asarray( args )
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

  return Multiply( arg1, arg2 )

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

  return Add( arg1, arg2 )

def negative( arg ):
  'make negative'

  arg = _asarray(arg)

  if not _isfunc( arg ):
    return numpy.negative( arg )

  retval = _call( arg, '_negative' )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._negative' % arg
    return retval

  return Negative( arg )

def power( arg, n ):
  'power'

  arg = _asarray( arg )
  assert _isscalar( n )

  if n == 1:
    return arg

  if n == 0:
    return numpy.ones( arg.shape )

  if isinstance( arg, numpy.ndarray ):
    return numpy.power( arg, n )

  retval = _call( arg, '_power', n )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._power' % arg
    return retval

  return Power( arg, n )

def sign( arg ):
  'sign'

  arg = _asarray( arg )

  if isinstance( arg, numpy.ndarray ):
    return numpy.sign( arg )

  retval = _call( arg, '_sign' )
  if retval is not None:
    assert retval.shape == arg.shape, 'bug in %s._sign' % arg
    return retval

  return Sign( arg )

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
  
  arg = _asarray( arg )
  n1, n2 = axes
  trans = numpy.arange( arg.ndim )
  trans[n1] = numeric.normdim( arg.ndim, n2 )
  trans[n2] = numeric.normdim( arg.ndim, n1 )
  return align( arg, trans, arg.ndim )

def opposite( arg ):
  'evaluate jump over interface'

  arg = _asarray( arg )

  if not _isfunc( arg ):
    return arg
    
  return arg._opposite()

def function( fmap, nmap, ndofs, ndims ):
  'create function on ndims-element'

  axis = '~%d' % ndofs
  cascade = Cascade(ndims)
  func = Function( cascade, fmap, igrad=0, axis=axis )
  dofmap = DofMap( cascade, nmap, axis=axis )
  return Inflate( func, dofmap, length=ndofs, axis=0 )

def take( arg, index, axis ):
  'take index'

  arg = _asarray( arg )
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

  arg = _asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )
  assert not isinstance( arg.shape[axis], int )

  retval = _call( arg, '_inflate', dofmap, length, axis )
  if retval is not None:
    return retval

  return Inflate( arg, dofmap, length, axis )
  
def blocks( arg ):
  arg = _asarray( arg )
  try:
    blocks = arg.blocks
  except AttributeError:
    blocks = [( arg, tuple( numpy.arange(n) if isinstance(n,int) else None for n in arg.shape ) )]
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
    ipoints, iweights = elem.eval( ischeme )
    values = numpy.empty( ipoints.shape[:-1]+shape, dtype=float )
    values[:] = func(elem,ischeme) if func is not None else value
    data[ elem ] = values, ipoints

  return Pointdata( data, shape )

def fdapprox( func, w, dofs, delta=1.e-5 ):
  '''Finite difference approximation of the variation of func in directions w around dofs
  I: func,  the functional to differentiate
     dofs,  DOF vector of linearization point
     w,     the function space or a tuple of chained spaces
     delta, finite difference step scaling of ||dofs||_inf
  O: dfunc, approximate variation of func'''
  log.context( 'FD approx' )
  if not isinstance( w, tuple ): w = w,
  x0 = tuple( wi.dot( dofs ) for wi in w )
  step = numpy.linalg.norm( dofs, numpy.inf )*delta
  ndofs = len( dofs )
  dfunc_fd = []
  for i in log.iterate( 'dof', range(ndofs) ):
    pert = dofs.copy()
    pert[i] += step
    x1 = tuple( wi.dot( pert ) for wi in w )
    dfunc_fd.append( (func( *x1 ) - func( *x0 ))/step )
  return dfunc_fd

def iwscale( coords, ndims ):
  'integration weights scale'

  assert coords.ndim == 1
  J = localgradient( coords, ndims )
  cndims, = coords.shape
  assert J.shape == (cndims,ndims), 'wrong jacobian shape: got %s, expected %s' % ( J.shape, (cndims, ndims) )
  if cndims == ndims:
    detJ = determinant( J )
  elif ndims == 1:
    detJ = norm2( J[:,0], axis=0 )
  elif cndims == 3 and ndims == 2:
    detJ = norm2( cross( J[:,0], J[:,1], axis=0 ), axis=0 )
  elif ndims == 0:
    detJ = 1.
  else:
    raise NotImplementedError, 'cannot compute determinant for %dx%d jacobian' % J.shape[:2]
  return detJ

def supp( funcsp, indices ):
  'find support of selection of basis functions'

  # FRAGILE! makes lots of assumptions on the nature of funcsp
  supp = []
  for func, axes in funcsp.blocks:
    dofmap = axes[0].dofmap
    stdmap = func.stdmap
    for elem, dofs in dofmap.items():
      while True:
        std = stdmap.get( elem )
        nshapes = 0 if not std \
           else std[1].sum() if isinstance( std, tuple ) \
           else std.nshapes
        if numpy.intersect1d( dofs[:nshapes], indices, assume_unique=True ).size:
          supp.append( elem )
        dofs = dofs[nshapes:]
        if not elem.parent:
          break
        elem, trans = elem.parent
      assert not dofs.size
  return supp

# CBUILDER
#
# Set of tools for just-in-time compilation of evaluables. Prototyping stage.

def _cshape( shape ):
  return [ sh if isinstance(sh,int) else 'n%d' % i for i, sh in enumerate(shape) ]

def _cstrides( shape ):
  if not shape:
    return []
  strides = [1]
  for sh in _cshape(shape)[:0:-1]:
    strides.append( '%s*%s' % ( strides[-1], sh ) )
  return strides[::-1]

def _cindex( myshape, shape ):
  assert len(shape) == len(myshape)
  index = []
  for i, stride in enumerate( _cstrides(myshape) ):
    if myshape[i] == shape[i]:
      index.append( '%s*i%d' % (stride,i) )
    else:
      assert myshape[i] == 1
  return '+'.join( index ) or '0'

def _cfor( varname, count, do ):
  return 'for ( %s=0; %s<%s; %s++ ) { %s }' % ( varname, varname, count, varname, do )

def _cfunc( args, body ):
  return '( %s )\n{ %s }\n' % ( ', '.join( args ), body )

class CBuilder( object ):
  'cbuilder'

  def __init__( self, cachedir='/tmp/nutils' ):
    from cffi import FFI
    self.ffi = FFI()
    self.codebase = {}
    self.allcode = []
    self.cachedir = cachedir

  def compile( self ):
    log.context( 'c-builder' )
    if not self.codebase:
      return
    import os, hashlib
    code = '\n'.join( self.allcode )
    fname = os.path.join( self.cachedir, hashlib.md5( code ).hexdigest() )
    if os.path.isfile( fname+'.so' ):
      log.info( 'in cache' )
    else:
      log.info( 'compiling %d c-functions' % len(self.codebase) )
      if not os.path.isdir( self.cachedir ):
        os.mkdir( self.cachedir )
      open( fname+'.c', 'w' ).write( code )
      assert os.system( 'gcc -fpic -g -c -o %s.o -O3 -march=native -mtune=native %s.c' % ((fname,)*2) ) == 0
      assert os.system( 'gcc -shared -Wl,-soname,%s.so -o %s.so %s.o -lc' % ((fname,)*3) ) == 0
      os.unlink( fname+'.c' )
      os.unlink( fname+'.o' )
    self.so = self.ffi.dlopen( fname+'.so' )

  def cast( self, args, ctypes ):
    assert len(args) == len(ctypes)
    for arg, ctype in zip( args, ctypes ):
      if ctype[-1] == '*':
        assert isinstance( arg, numpy.ndarray )
        # assert arg.flags.c_contiguous
        # TODO check dtype
        arg = arg.ctypes.data
      yield self.ffi.cast( ctype, arg )

  def __getitem__( self, (code,init) ):
    evalf = self.codebase.get(code)
    if evalf:
      return evalf
    arglist, body = code.split( '\n', 1 )
    assert arglist[0] == '(' and arglist[-1] == ')'
    ctypes = [ item.strip().rsplit(' ',1)[0] for item in arglist[1:-1].split( ',' ) ]
    handle = 'func%d' % len(self.codebase)
    head = 'void ' + handle
    self.ffi.cdef( head + arglist + ';' )
    def evalf( *args ):
      cfunc = getattr( self.so, handle )
      retval, iterargs = init( args )
      for args in iterargs:
        cfunc( *self.cast( args, ctypes ) )
      return retval
    self.codebase[code] = evalf
    self.allcode.append( head + code )
    return evalf

def init_args_for( RETVAL, *ARGS ):
  shape = RETVAL.shape
  ndim = _max( ARG.ndim for ARG in ARGS if ARG is not None )
  jointshape = _jointshape( *[ ARG.shape for ARG in ARGS if ARG is not None ] )
  def init( args ):
    assert len(args) == len(ARGS)
    # 'None' ARGS are filtered out; values are built into the c code
    args = [ numpy.ascontiguousarray( arg, dtype=ARG.dtype ) if arg.ndim else arg for arg, ARG in zip( args, ARGS ) if ARG is not None ]
    haspoints = _max( [ arg.ndim for arg in args ] ) - ndim
    assert haspoints in (0,1)
    if haspoints:
      neval = _max( [ arg.shape[0] for arg in args if arg.ndim > ndim ] )
      retshape = [ neval ]
    else:
      neval = 1
      retshape = []
    assert all( arg.shape[0] == neval for arg in args if arg.ndim > ndim )
    varsizes = []
    for idim, sh in enumerate( jointshape ):
      if not isinstance(sh,int):
        for arg in args:
          sh = arg.shape[idim-len(jointshape)]
          if sh != 1:
            break
        varsizes.append( sh )
      if idim < len(shape):
        retshape.append( sh )
    retval = numpy.empty( retshape, dtype=RETVAL.dtype )
    args = [ retval[...,_] if haspoints else [retval[...,_]] * neval ] \
         + [ [arg[...,_]] * neval if arg.ndim == ndim else arg[...,_] for arg in args ]
    args = [ arg + tuple(varsizes) for arg in zip( *args ) ]
    return retval, args
  return init

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
