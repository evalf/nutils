from . import util, numpy, numeric, log, prop, core, _
import sys

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

    return '%s: %s\n%s' % ( self.etype.__name__, self.evalue, self.evaluable.stackstr( self.values ) )

class Evaluable( object ):
  'evaluable base classs'

  operations = None

  def __init__( self, args, evalf ):
    'constructor'

    self.__args = tuple(args)
    self.__evalf = evalf

  def verify( self, value ):
    'check result'

    return '= %s %s' % ( _obj2str(value), type(value) )

  def recurse_index( self, data, operations ):
    'compile'

    indices = numpy.empty( len(self.__args), dtype=int )
    for iarg, arg in enumerate( self.__args ):
      if isinstance(arg,Evaluable):
        for idx, (op,idcs) in enumerate( operations ):
          if op == arg:
            break
        else:
          idx = arg.recurse_index(data,operations)
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
    operations.append( (self,indices) )
    return len(operations)-1

  def compile( self ):
    'compile'

    if self.operations is None:
      self.data = []
      self.operations = []
      self.recurse_index( self.data, self.operations ) # compile expressions

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
    values = list( self.data ) + [ elem, points, weights ]
    for op, indices in self.operations:
      args = [ values[N+i] for i in indices ]
      try:
        retval = op.__evalf( *args )
      except KeyboardInterrupt:
        raise
      except:
        etype, evalue, traceback = sys.exc_info()
        raise EvaluationError, ( etype, evalue, self, values ), traceback
      values.append( retval )
    return values[-1]

  def argnames( self ):
    'function argument names'
  
    import inspect
    try:
      argnames, varargs, keywords, defaults = inspect.getargspec( self.__evalf )
    except:
      argnames = [ '%%%d' % n for n in range(len(self.__args)) ]
    else:
      for n in range( len(self.__args) - len(argnames) ):
        argnames.append( '%s[%d]' % (varargs,n) )
    return argnames

  def __graphviz__( self ):
    'graphviz representation'

    args = [ '%s=%s' % ( argname, _obj2str(arg) ) for argname, arg in zip( self.argnames(), self.__args ) if not isinstance(arg,Evaluable) ]
    label = self.__class__.__name__
    return { 'label': r'\n'.join( [ label ] + args ) }

  def graphviz( self ):
    'create function graph'

    log.context( 'creating graphviz' )

    import os, subprocess

    dotpath = getattr( prop, 'dot', False )
    if not dotpath:
      return False
    if dotpath is True:
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

    self.compile()
    for i, (op,indices) in enumerate( self.operations ):

      node = op.__graphviz__()
      node['label'] = '%d. %s' % ( i, node.get('label','') )
      print >> dot.stdin, '%d [%s]' % ( i, ' '.join( '%s="%s"' % item for item in node.iteritems() ) )
      argnames = op.argnames()
      for n, idx in enumerate( indices ):
        if idx >= 0:
          print >> dot.stdin, '%d -> %d [label="%s"];' % ( idx, i, argnames[n] );

    print >> dot.stdin, '}'
    dot.stdin.close()

    log.path( os.path.basename(imgpath) )

  def stackstr( self, values=None ):
    'print stack'

    self.compile()
    if values is None:
      values = self.data + ( '<elem>', '<points>', '<weights>' )

    N = len(self.data) + 3

    lines = []
    for i, (op,indices) in enumerate( self.operations ):
      line = '  %%%d =' % i
      args = [ '%%%d' % idx if idx >= 0 else _obj2str(values[N+idx]) for idx in indices ]
      try:
        code = op.__evalf.func_code
        names = code.co_varnames[ :code.co_argcount ]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(indices) - len(names) ) )
        args = [ '%s=%s' % item for item in zip( names, args ) ]
      except:
        pass
      line += ' %s( %s )' % ( op.__evalf.__name__, ', '.join( args ) )
      if N+i < len(values):
        line += ' ' + op.verify( values[N+i] )
      lines.append( line )
      if N+i == len(values):
        break
    return '\n'.join( lines )

  def __eq__( self, other ):
    'compare'

    if self is other:
      return True
      
    if self.__class__ != other.__class__ or self.__evalf != other.__evalf or len( self.__args ) != len( other.__args ):
      return False

    return all( _equal(arg1,arg2) for arg1, arg2 in zip( self.__args, other.__args ) )

  def __ne__( self, other ):
    'not equal'

    return not self == other

  def __str__( self ):
    'string representation'

    key = self.__evalf.__name__
    lines = []
    indent = '\n' + ' ' + ' ' * len(key)
    for it in reversed( self.__args ):
      lines.append( indent.join( _obj2str(it).splitlines() ) )
      indent = '\n' + '|' + ' ' * len(key)
    indent = '\n' + '+' + '-' * (len(key)-1) + ' '
    return key + ' ' + indent.join( reversed( lines ) )

class Tuple( Evaluable ):
  'combine'

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

# ARRAYFUNC
#
# The main evaluable. Closely mimics a numpy array.

class ArrayFunc( Evaluable ):
  'array function'

  __array_priority__ = 1. # fix numpy's idiotic default behaviour
  __priority__ = False

  @classmethod
  def stack( cls, funcs, axis ):
    'stack'

    return stack( funcs, axis )

  def verify( self, value ):
    'check result'

    s = '=> ' + _obj2str(value)
    s += ' \ (%s)' % ','.join(map(str,self.shape))
    return s

  def find( self, elem, target, start, tol=1e-10, maxiter=999 ):
    'iteratively find x for f(x) = target, starting at x=start'

    points = start
    Jinv = inv( localgradient( self, elem.ndims ), 0, 1 )
    r = target - self( elem, points )
    niter = 0
    while numpy.any( numeric.contract( r, r, axis=-1 ) > tol ):
      niter += 1
      if niter >= maxiter:
        raise Exception, 'failed to converge in %d iterations' % maxiter
      points = points.offset( numeric.contract( Jinv( elem, points ), r[:,_,:], axis=-1 ) )
      r = target - self( elem, points )
    return points

  def sum( self, axes=-1 ):
    'sum'

    axes = list( _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] ) )
    if not axes:
      return self

    ax = axes.pop()
    return sum( get( self, ax, 0 ) if self.shape[ax] == 1 else Sum( self, [ax] ), axes )

  def normalized( self ):
    'normalize last axis'

    return self / norm2( self, axis=-1 )

  def normal( self, ndims=-1 ):
    'normal'

    assert len(self.shape) == 1
    if ndims <= 0:
      ndims += self.shape[0]

    if self.shape[0] == 2 and ndims == 1:
      grad = localgradient( self, ndims=1 )
      normal = concatenate([ grad[1,:], -grad[0,:] ])
    elif self.shape[0] == 3 and ndims == 2:
      grad = localgradient( self, ndims=2 )
      normal = cross( grad[:,0], grad[:,1], axis=0 )
    elif self.shape[0] == 2 and ndims == 0:
      grad = localgradient( self, ndims=1 )
      normal = grad[:,0] * Orientation()
    elif self.shape[0] == 3 and ndims == 1:
      grad = localgradient( self, ndims=1 )
      normal = cross( grad[:,0], self.normal(), axis=0 )
    elif self.shape[0] == 1 and ndims == 0:
      return numpy.array( 1 ) # TODO fix direction!!!!
    else:
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims )
    return normal.normalized()

  def iweights( self, ndims ):
    'integration weights for [ndims] topology'

    J = localgradient( self, ndims )
    cndims, = self.shape
    assert J.shape == (cndims,ndims), 'wrong jacobian shape: got %s, expected %s' % ( J.shape, (cndims, ndims) )
    if cndims == ndims:
      detJ = det( J, 0, 1 )
    elif ndims == 1:
      detJ = norm2( J[:,0], axis=0 )
    elif cndims == 3 and ndims == 2:
      detJ = norm2( cross( J[:,0], J[:,1], axis=0 ), axis=0 )
    elif ndims == 0:
      detJ = 1
    else:
      raise NotImplementedError, 'cannot compute determinant for %dx%d jacobian' % J.shape[:2]
    return detJ * IWeights()

  def curvature( self, ndims=-1 ):
    'curvature'

    if ndims <= 0:
      ndims += self.shape[0]
    assert ndims == 1 and self.shape == (2,)
    J = localgradient( self, ndims )
    H = localgradient( J, ndims )
    dx, dy = J[:,0]
    ddx, ddy = H[:,0,0]
    return ( dy * ddx - dx * ddy ) / norm2( J[:,0], axis=0 )**3

  def swapaxes( self, n1, n2 ):
    'swap axes'

    trans = numpy.arange( self.ndim )
    trans[n1] = _normdim( self.ndim, n2 )
    trans[n2] = _normdim( self.ndim, n1 )
    return align( self, trans, self.ndim )

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
      Jinv = inv( J, 0, 1 )
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      Jinv = inv( concatenate( [ J, coords.normal()[:,_] ], axis=1 ), 0, 1 )[:-1,:]
    else:
      raise Exception, 'cannot invert jacobian'
    return sum( localgradient( self, ndims )[...,_] * Jinv, axes=-2 )

  def symgrad( self, coords, ndims=0 ):
    'gradient'

    g = self.grad( coords, ndims )
    return .5 * ( g + g.swapaxes(-2,-1) )

  def div( self, coords, ndims=0 ):
    'gradient'

    return trace( self.grad( coords, ndims ), -1, -2 )

  def dotnorm( self, coords, ndims=0 ):
    'normal component'

    return sum( self * coords.normal( ndims-1 ) )

  def ngrad( self, coords, ndims=0 ):
    'normal gradient'

    return self.grad( coords, ndims ).dotnorm( coords, ndims )

  def nsymgrad( self, coords, ndims=0 ):
    'normal gradient'

    return self.symgrad( coords, ndims ).dotnorm( coords, ndims )

  def __cross__( self, other, axis ):
    'cross product'

    arg1, arg2 = _matchndim( self, other )
    axis = _normdim( arg1.ndim, axis )
    return Cross( arg1, arg2, axis )

  @property
  def T( self ):
    'transpose'

    return transpose( self )

  def __nonzero__( self ):
    'nonzero'

    return True

  def __take__( self, indices, axis ):
    'take'

    axis = _normdim( self.ndim, axis )
    if self.shape[ axis ] == 1:
      if isinstance( indices, slice ):
        assert indices.start == None or indices.start >= 0
        assert indices.stop != None and indices.stop > 0
        n = numpy.arange( indices.stop )[indices]
      else:
        n = numpy.array( indices, dtype=int )
        assert numpy.all( n >= 0 )
      return expand( self, self.shape[:axis] + (len(n),) + self.shape[axis+1:] )
    n = numpy.arange( self.shape[axis] )[indices]
    assert len(n) > 0
    if len(n) == 1:
      return insert( get( self, axis, n[0] ), axis )
    return Take( self, indices, axis )

  def __takediag__( self, ax1, ax2 ):
    'takediag'

    ax1, ax2 = _norm_and_sort( self.ndim, (ax1,ax2) )

    axes = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,self.ndim-2)
    func = align( self, axes, self.ndim )

    if func.shape[-1] == 1:
      return get( func, -1, 0 )

    if func.shape[-2] == 1:
      return get( func, -2, 0 )

    return TakeDiag( func )

  def __init__( self, evalf, args, shape ):
    'constructor'

    self.evalf = evalf
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    Evaluable.__init__( self, evalf=evalf, args=args )

  def __kronecker__( self, axis, length, pos ):
    'kronecker'

    funcs = [ None ] * length
    funcs[pos] = self
    return Kronecker( funcs, axis=axis )

  def __elemint__( self, weights ):
    'elementwise integration'
  
    return ElemInt( self, weights )

  def __getitem__( self, item ):
    'get item, general function which can eliminate, add or modify axes.'
  
    tmp = item
    item = list( item if isinstance( item, tuple ) else [item] )
    n = 0
    arr = self
    while item:
      it = item.pop(0)
      if isinstance(it,int): # retrieve one item from axis
        arr = get( arr, n, it )
      elif it == _: # insert a singleton axis
        arr = insert( arr, n )
        n += 1
      elif it == slice(None): # select entire axis
        n += 1
      elif it == Ellipsis: # skip to end
        remaining_items = len(item) - item.count(_)
        skip = arr.ndim - n - remaining_items
        assert skip >= 0
        n += skip
      elif isinstance(it,slice) and it.step in (1,None) and it.stop == ( it.start or 0 ) + 1: # special case: unit length slice
        arr = insert( get( arr, n, it.start ), n )
        n += 1
      elif isinstance(it,(slice,list,tuple,numpy.ndarray)): # modify axis (shorten, extend or renumber one axis)
        arr = take( arr, it, n )
        n += 1
      else:
        raise NotImplementedError
      assert n <= arr.ndim
    return arr

  def __reciprocal__( self ):
    'reciprocal'

    return Reciprocal( self )

  def __expand__( self, shape ):
    'expand'

    shape = tuple(shape)
    if shape == self.shape:
      return self

    return Expand( self, shape )

  def __get__( self, i, item ):
    'get item'

    i = _normdim( self.ndim, i )
    shape = list(self.shape)
    sh = shape.pop(i)
    assert isinstance(sh,int), 'cannot get item %r from axis %r' % ( item, sh )
    item = _normdim( sh, item )
    return Get( self, (i,item) )

  def __align__( self, axes, ndim ):
    'insert singleton dimension'

    assert all( 0 <= ax < ndim for ax in axes )
    if list(axes) == range(ndim):
      return self
    return Align( self, axes, ndim )

  def __iter__( self ):
    'split first axis'

    if not self.shape:
      raise TypeError, 'scalar function is not iterable'

    return ( self[i,...] for i in range(self.shape[0]) )

  def __inv__( self, ax1, ax2 ):
    'inverse'

    n = self.shape[ax1]
    assert self.shape[ax2] == n
    if n == 1:
      return reciprocal( self )

    return Inverse( self, ax1, ax2 )

  def __norm2__( self, axis=-1 ):
    'norm2'

    return sum( self * self, axis )**.5

  def __det__( self, ax1, ax2 ):
    'determinant'

    ax1, ax2 = _norm_and_sort( self.ndim, [ax1,ax2] )

    n = self.shape[ax1]
    assert n == self.shape[ax2]
    if n == 1:
      return get( get( self, ax2, 0 ), ax1, 0 )

    return Determinant( self, ax1, ax2 )

  def __prod__( self, axis ):
    'product'

    axis = _normdim( self.ndim, axis )
    if self.shape[axis] == 1:
      return get( self, axis, 0 )

    func = align( self, range(axis) + [-1] + range(axis,self.ndim-1), self.ndim )
    return Product( func )

  def __mul__( self, other ):
    'right multiplication'

    if _haspriority(other) and not _haspriority(self):
      return other * self # prefer specific over general

    func1, func2 = _matchndim( self, other )
    shape = _jointshape( func1.shape, func2.shape )

    if _iszero( func2 ):
      return _const( 0., shape )

    if _isunit( func2 ):
      return expand( func1, shape )

    for iax in range( len(shape) ):
      if func1.shape[iax] == func2.shape[iax] == 1:
        return insert( get( func1, iax, 0 ) * get( func2, iax, 0 ), iax )

    return Multiply( func1, func2 )

  def __rmul__( self, other ):
    'right multiply'

    return self * other

  def __div__( self, other ):
    'divide'

    if not _isfunc( other ):
      return self * reciprocal( other ) # faster

    func1, func2 = _matchndim( self, other )
    shape = _jointshape( func1.shape, func2.shape )

    for iax in range( len(shape) ):
      if func1.shape[iax] == func2.shape[iax] == 1:
        return insert( get( func1, iax, 0 ) / get( func2, iax, 0 ), iax )

    return Divide( func1, func2 )

  def __rdiv__( self, other ):
    'right divide'

    func1, func2 = _matchndim( self, other )

    if _iszero( func2 ):
      return _const( 0., _jointshape(func1.shape,func2.shape) )
    
    return Divide( func2, func1 )

  def __add__( self, other ):
    'add'

    if _haspriority(other) and not _haspriority(self):
      return other + self # prefer specific over general

    func1, func2 = _matchndim( self, other )

    if _iszero(func2):
      return expand( func1, _jointshape( func1.shape, func2.shape ) )

    if func1 == func2:
      return func1 * 2

    if not _haspriority(func1) and not _haspriority(func2): # don't cover the black balloons (race cond with Add.align)
      for iax in range( func1.ndim ):
        if func1.shape[iax] == func2.shape[iax] == 1:
          return insert( get( func1, iax, 0 ) + get( func2, iax, 0 ), iax )

    return Add( func1, func2 )

  def __radd__( self, other ):
    'right addition'

    return self + other

  def __sub__( self, other ):
    'subtract'

    func1, func2 = _matchndim( self, other )

    if _iszero( func2 ):
      return expand( func1, _jointshape( func1.shape, func2.shape ) )

    for iax in range( func1.ndim ):
      if func1.shape[iax] == func2.shape[iax] == 1:
        return insert( get( func1, iax, 0 ) - get( func2, iax, 0 ), iax )

    return Subtract( func1, func2 )

  def __rsub__( self, other ):
    'right subtract'

    if _isfunc( other ):
      return other - self

    func1, func2 = _matchndim( self, other )
    shape = _jointshape( func1.shape, func2.shape )

    if _iszero(func2):
      return -expand( func1, shape )

    for iax in range( len(shape) ):
      if func1.shape[iax] == func2.shape[iax] == 1:
        return insert( get( func2, iax, 0 ) - get( func1, iax, 0 ), iax )

    return Subtract( func2, func1 )

  def __neg__( self ):
    'negate'

    return Negate( self )

  def __pow__( self, n ):
    'power'

    assert _isscalar( n )
    if n == 1:
      return self
    if n == 0:
      return numpy.ones( self.shape )
    if n < 0:
      return reciprocal( self**-n )
    return Power( self, n )

  def __graphviz__( self ):
    'graphviz representation'

    args = Evaluable.__graphviz__( self )
    args['label'] += r'\n[%s]' % ','.join( map(str,self.shape) )
    if self.__priority__:
      args['fontcolor'] = 'white'
      args['fillcolor'] = 'black'
      args['style'] = 'filled'
    return args

  @core.deprecated( old='f.norm2(...)', new='function.norm2(f,...)' )
  def norm2( self, axis=-1 ):
    return norm2( self, axis )

  @core.deprecated( old='f.localgradient(...)', new='function.localgradient(f,...)' )
  def localgradient( self, ndims ):
    return localgradient( self, ndims )

  @core.deprecated( old='f.trace(...)', new='function.trace(f,...)' )
  def trace( self, n1=-2, n2=-1 ):
    return trace( self, n1=-2, n2=-1 )

class ElemArea( ArrayFunc ):
  'element area'

  def __init__( self, weights ):
    'constructor'

    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights], evalf=self.elemarea, shape=weights.shape )

  @staticmethod
  def elemarea( weights ):
    'evaluate'

    return sum( weights )

class ElemInt( ArrayFunc ):
  'elementwise integration'

  def __init__( self, func, weights ):
    'constructor'

    assert _isfunc( func ) and _isfunc( weights )
    assert weights.ndim == 0
    ArrayFunc.__init__( self, args=[weights,func], evalf=self.elemint, shape=func.shape )

  @staticmethod
  def elemint( w, f ):
    'evaluate'

    return numeric.dot( w, f ) if w.size else numpy.zeros( f.shape[1:] )

class Align( ArrayFunc ):
  'align axes'

  @classmethod
  def stack( cls, funcs, axis ):
    'stack funcs along new axis'

    axes = funcs[0].axes
    ndim = funcs[0].ndim
    if not all( isinstance(func,cls) and func.axes == axes and func.ndim == ndim for func in funcs ):
      return ArrayFunc.stack( funcs, axis )

    newaxes = [ ax if ax < axis else ax+1 for ax in axes ] + [ axis ]
    return Align( funcs[0].stack( funcs, len(axes) ), newaxes )

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

  def __elemint__( self, weights ):
    'elementwise integration'
  
    return align( ElemInt( self.func, weights ), self.axes, self.ndim )

  def __align__( self, axes, ndim ):
    'align'

    assert len(axes) == self.ndim
    assert all( 0 <= ax < ndim for ax in axes )
    newaxes = [ axes[i] for i in self.axes ]
    return align( self.func, newaxes, ndim )

  def __takediag__( self, ax1, ax2 ):
    'take diag'

    func = self.func
    ax1, ax2 = _norm_and_sort( self.ndim, [ax1,ax2] )
    if ax1 not in self.axes and ax2 not in self.axes:
      axes = [ ax - (ax>ax1) - (ax>ax2) for ax in self.axes ]
    elif ax2 not in self.axes:
      axes = [ ax - (ax>ax1) - (ax>ax2) if ax != ax1 else -1 for ax in self.axes ]
    elif ax1 not in self.axes:
      axes = [ ax - (ax>ax1) - (ax>ax2) if ax != ax2 else -1 for ax in self.axes ]
    else:
      func = takediag( func, self.axes.index(ax1), self.axes.index(ax2) )
      axes = [ ax - (ax>ax1) - (ax>ax2) for ax in self.axes if ax not in (ax1,ax2) ] + [ -1 ]
    return align( func, axes, self.ndim-1 )

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    axes = [ ax - (ax>i) for ax in self.axes if ax != i ]
    if len(axes) == len(self.axes):
      return align( self.func, axes, self.ndim-1 )

    n = self.axes.index( i )
    return align( get( self.func, n, item ), axes, self.ndim-1 )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    sumaxes = []
    for ax in axes:
      try:
        idx = self.axes.index( ax )
        sumaxes.append( idx )
      except:
        pass # trivial summation over singleton axis

    trans = [ ax - _sum(i<ax for i in axes) for ax in self.axes if ax not in axes ]
    return align( sum( self.func, sumaxes ), trans, self.ndim-len(axes) )

  def __localgradient__( self, ndims ):
    'local gradient'

    return align( localgradient( self.func, ndims ), self.axes+(self.ndim,), self.ndim+1 )

  def __mul__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Align )

    #TODO make less restrictive:
    if isinstance( func2, Align ) and func1.axes == func2.axes:
      return Align( func1.func * func2.func, func1.axes, func1.ndim )

    if not _isfunc(func2) and len(func1.axes) == func2.ndim:
      return align( func1.func * transpose( func2, func1.axes ), func1.axes, func1.ndim )

    return ArrayFunc.__mul__( func1, func2 )

  def __add__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Align )

    #TODO make less restrictive:
    if isinstance( func2, Align ) and func1.axes == func2.axes:
      return Align( func1.func + func2.func, func1.axes, func1.ndim )

    if not _isfunc(func2) and len(func1.axes) == func1.ndim:
      return align( func1.func + transform( func2, func1.axes ), func1.axes, func1.ndim )

    return ArrayFunc.__add__( func1, func2 )

  def __div__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Align )

    #TODO make less restrictive:
    if isinstance( func2, Align ) and func1.axes == func2.axes:
      return Align( func1.func / func2.func, func1.axes, func1.ndim )

    if not _isfunc(func2) and len(func1.axes) == func1.ndim:
      return align( func1.func / transform( func2, func1.axes ), func1.axes, func1.ndim )

    return ArrayFunc.__div__( self, other )

  def __graphviz__( self ):
    'graphviz representation'

    newsh = [ '?' ] * self.ndim
    for src, dst in enumerate( self.axes ):
      newsh[dst] = str(src)
    return { 'shape': 'trapezium',
             'label': ','.join(newsh) }

class Get( ArrayFunc ):
  'get'

  def __init__( self, func, *items ):
    'constructor'

    self.func = func
    self.items = sorted( items )
    last_ax = -1
    for iax, item in self.items:
      assert iax > last_ax, 'axis %d is repeated or out of bounds' % iax
      last_ax = iax
    assert last_ax < func.ndim
    s = [ slice(None) ] * func.ndim
    shape = list( func.shape )
    for i, item in reversed( self.items ):
      sh = shape.pop( i )
      if sh == 1:
        s[i] = 0
      else:
        assert 0 <= item < sh
        s[i] = item
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+tuple(s)), evalf=numpy.ndarray.__getitem__, shape=shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    f = localgradient( self.func, ndims )
    for i, item in reversed( self.items ):
      f = get( f, i, item )
    return f

  def __get__( self, i, item ):
    'get item'

    i = _normdim( self.ndim, i )
    n = _sum( iax <= i for iax, it in self.items )
    items = self.items[:n] + [(i+n,item)] + self.items[n:]
    return Get( self.func, *items )

  def __graphviz__( self ):
    'graphviz representation'

    getitem = [ ':' ] * self.func.ndim
    for src, item in self.items:
      getitem[src] = str(item)
    return { 'shape': 'invtrapezium',
             'label': ','.join(getitem) }

class Reciprocal( ArrayFunc ):
  'reciprocal'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.reciprocal, shape=func.shape )

  def __get__( self, i, item ):
    'get item'

    return reciprocal( get( self.func, i, item ) )

  def __align__( self, axes, ndim ):
    'align'

    return reciprocal( align( self.func, axes, ndim ) )

  def __mul__( self, other ):
    'multiply'

    return other / self.func

class Product( ArrayFunc ):
  'product'

  def __init__( self, func ):
    'constructor'

    ArrayFunc.__init__( self, args=[func,-1], evalf=numpy.prod, shape=func.shape[:-1] )

class IWeights( ArrayFunc ):
  'integration weights'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[ELEM,WEIGHTS], evalf=self.iweights, shape=() )

  @staticmethod
  def iweights( elem, weights ):
    'evaluate'

    det = 1
    while elem.parent:
      elem, transform = elem.parent
      det *= transform.det
    return weights * det

class Orientation( ArrayFunc ):
  'point orientation'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[ELEM], evalf=self.orientation, shape=() )

  @staticmethod
  def orientation( elem ):
    'evaluate'

    # VERY TEMPORARY
    elem, transform = elem.parent
    return ( transform.offset > .5 ) * 2 - 1

class Transform( ArrayFunc ):
  'transform'

  def __init__( self, fromdims, todims ):
    'constructor'

    assert fromdims > todims
    ArrayFunc.__init__( self, args=[ELEM,fromdims,todims], evalf=self.transform, shape=(fromdims,todims) )

  @staticmethod
  def transform( elem, fromdims, todims ):
    'transform'

    assert elem.ndims <= todims
    while elem.ndims < todims:
      elem, transform = elem.context or elem.parent

    fromelem = elem
    toelem, transform = fromelem.context or fromelem.parent

    T = transform.get_transform()
    while toelem.ndims < fromdims:
      toelem, transform = toelem.context or toelem.parent
      T = transform.transform_from( T, axis=0 )

    while fromelem.parent:
      fromelem, transform = fromelem.parent
      T = transform.transform_to( T, axis=1 )
    
    while toelem.parent:
      toelem, transform = toelem.parent
      T = transform.transform_from( T, axis=0 )

    return T

  def __localgradient__( self, ndims ):
    'local gradient'

    return _const( 0., self.shape + (ndims,) )

class Function( ArrayFunc ):
  'local gradient'

  def __init__( self, stdmap, igrad ):
    'constructor'

    self.stdmap = stdmap
    self.igrad = igrad
    ArrayFunc.__init__( self, args=(ELEM,POINTS,stdmap,igrad), evalf=self.function, shape=(None,)+(stdmap.ndims,)*igrad )

  @staticmethod
  def function( elem, points, stdmap, igrad ):
    'evaluate'

    while elem.ndims < stdmap.ndims:
      elem, transform = elem.context or elem.parent
      points = transform.eval( points )

    fvals = None
    while True:
      std = stdmap.get(elem)
      if std:
        if isinstance( std, tuple ):
          std, keep = std
          F = std.eval(points,grad=igrad)[(Ellipsis,keep)+(slice(None),)*igrad]
        else:
          F = std.eval(points,grad=igrad)
        fvals = F if fvals is None else numpy.concatenate( [ fvals, F ], axis=1 )
      if not elem.parent:
        break
      elem, transform = elem.parent
      points = transform.eval( points )
      if fvals is not None:
        for axis in range(2,2+igrad):
          fvals = numeric.dot( fvals, transform.invtrans, axis )

    assert fvals is not None, 'no function values encountered'
    return fvals

  def __localgradient__( self, ndims ):
    'local gradient'

    assert ndims <= self.stdmap.ndims
    grad = Function( self.stdmap, self.igrad+1 )
    return grad if ndims == self.stdmap.ndims \
      else sum( grad[...,_] * Transform( self.stdmap.ndims, ndims ), axes=-2 )

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, level, choices, *warnargs ):
    'constructor'

    assert not warnargs, 'ERROR: the Choose object has changed. Please use piecewise instead.'

    self.level = level
    self.choices = tuple(choices)
    shape = _jointshape( *[ choice.shape for choice in choices ] )
    assert level.ndim == 0
    ArrayFunc.__init__( self, args=(level,)+self.choices, evalf=self.choose, shape=shape )

  @staticmethod
  def choose( level, *choices ):
    'choose'

    return numpy.choose( level, [ c.T for c in choices ] ).T

  def __localgradient__( self, ndims ):
    'gradient'

    grads = [ localgradient( choice, ndims ) for choice in self.choices ]
    if not any( grads ): # all-zero special case; better would be allow merging of intervals
      return _const( 0., self.shape + (ndims,) )
    return Choose( self.level[...,_], grads )

class Choose2D( ArrayFunc ):
  'piecewise function'

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

  def __init__( self, func, ax1, ax2 ):
    'constructor'

    ax1, ax2 = _norm_and_sort( func.ndim, (ax1,ax2) )
    assert func.shape[ax1] == func.shape[ax2]
    self.func = func
    self.axes = ax1, ax2
    ArrayFunc.__init__( self, args=(func,(ax1-func.ndim,ax2-func.ndim)), evalf=numeric.inv, shape=func.shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    ax1, ax2 = self.axes
    G = localgradient( self.func, ndims )
    H = sum( self[...,_,_].swapaxes(ax1,-1) * G[...,_].swapaxes(ax2,-1) )
    I = sum( self[...,_,_].swapaxes(ax2,-1) * H[...,_].swapaxes(ax1,-1) )
    return -I

class DofMap( ArrayFunc ):
  'dof axis'

  def __init__( self, dofmap ):
    'new'

    self.dofmap = dofmap
    ArrayFunc.__init__( self, args=(ELEM,dofmap), evalf=self.evalmap, shape=[None] )

  @staticmethod
  def evalmap( elem, dofmap ):
    'evaluate'

    while elem.ndims < dofmap.ndims:
      elem, dummy = elem.context or elem.parent

    dofs = dofmap.get( elem )
    while dofs is None:
      elem, transform = elem.parent
      dofs = dofmap.get( elem )
    return dofs

class Concatenate( ArrayFunc ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    funcs = [ _asarray(func) for func in funcs ]
    ndim = funcs[0].ndim
    assert all( func.ndim == ndim for func in funcs )
    axis = _normdim( ndim, axis )
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
    retval = numpy.empty( shape, dtype=_jointdtype(*arrays) )
    n0 = 0
    for array in arrays:
      n1 = n0 + array.shape[iax]
      retval[(slice(None),)*( iax if iax >= 0 else iax + ndim )+(slice(n0,n1),)] = array
      n0 = n1
    assert n0 == axlen
    return retval

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    if i == self.axis:
      assert self.shape[i] is not None
      for f in self.funcs:
        if item < f.shape[i]:
          return get( f, i, item )
        item -= f.shape[0]

    axis = self.axis - (self.axis > i)
    return concatenate( [ get( f, i, item ) for f in self.funcs ], axis=axis )

  def __localgradient__( self, ndims ):
    'gradient'

    funcs = [ localgradient( func, ndims ) for func in self.funcs ]
    return concatenate( funcs, axis=self.axis )

  def __mul__( self, other ):
    'multiply'

    if not isinstance( other, Concatenate ) or self.ndim != other.ndim or self.axis != other.axis or [ f.shape[self.axis] for f in self.funcs ] != [ g.shape[other.axis] for g in other.funcs ]:
      return ArrayFunc.__mul__( self, other )

    return concatenate( [ f * g for f, g in zip(self.funcs,other.funcs) ], self.axis )

  def __add__( self, other ):
    'addition'

    if not isinstance( other, Concatenate ) or self.axis != other.axis:
      return ArrayFunc.__add__( self, other )

    fg = zip( self.funcs, other.funcs )
    if any( f.shape != g.shape for (f,g) in fg ):
      return ArrayFunc.__add__( self, other )

    return concatenate( [ f+g for (f,g) in fg ], axis=self.axis )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    if self.axis not in axes:
      return ArrayFunc.sum( self, axes )

    axes = [ ax if ax < self.axis else ax-1 for ax in axes if ax != self.axis ]
    return sum( util.sum( sum( f, self.axis ) if f.shape[self.axis] != 1 else get( f, self.axis, 0 ) for f in self.funcs ), axes )

  def __align__( self, axes, ndim ):
    'align'

    assert all( 0 <= ax < ndim for ax in axes )
    assert len(axes) == self.ndim
    funcs = [ align( func, axes, ndim ) for func in self.funcs ]
    axis = axes[ self.axis ]
    return concatenate( funcs, axis )

  def __takediag__( self, ax1, ax2 ):
    'take diagonal'

    ax1, ax2 = _norm_and_sort( self.ndim, [ax1,ax2] )
    if ax1 == self.axis:
      axis = ax2
    elif ax2 == self.axis:
      axis = ax1
    else:
      axis = self.axis - (self.axis>ax1) - (self.axis>ax2)
      return concatenate( [ takediag( f, ax1, ax2 ) for f in self.funcs ], axis=axis )

    n0 = 0
    funcs = []
    for func in self.funcs:
      n1 = n0 + func.shape[self.axis]
      funcs.append( takediag( take( func, slice(n0,n1), axis ), axis, self.axis ) )
      n0 = n1
    assert n0 == self.shape[self.axis]
    return concatenate( funcs, axis=-1 )

class Heaviside( ArrayFunc ):
  'heaviside function'

  def __init__( self, levelset ):
    'constructor'

    self.levelset = levelset
    ArrayFunc.__init__( self, args=[levelset], evalf=self.heaviside, shape=levelset.shape )

  @staticmethod
  def heaviside( f ):
    'evaluate'

    return ( f > 0 ).astype( float )

class Interp1D( ArrayFunc ):
  'interpolate data'

  def __init__( self, x, xp, yp, left=None, right=None ):
    'constructor'

    assert _isfunc( x )
    xp = UseableArray( xp )
    yp = UseableArray( yp )
    assert ( numpy.diff(xp) > 0 ).all()
    assert xp.ndim == yp.ndim == 1
    assert xp.shape == yp.shape
    ArrayFunc.__init__( self, args=(x,xp,yp,left,right), evalf=numpy.interp, shape=() )

class Cross( ArrayFunc ):
  'cross product'

  def __init__( self, f1, f2, axis ):
    'contructor'

    shape = _jointshape( f1.shape, f2.shape )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    ArrayFunc.__init__( self, args=(f1,f2,axis-len(shape)), evalf=numeric.cross, shape=shape )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, func, ax1, ax2 ):
    'contructor'

    assert 0 <= ax1 < ax2 < func.ndim
    shape = func.shape[:ax1] + func.shape[ax1+1:ax2] + func.shape[ax2+1:]
    self.axes = ax1, ax2
    self.func = func
    ArrayFunc.__init__( self, args=(func,ax1-func.ndim,ax2-func.ndim), evalf=numeric.det, shape=shape )

  def __localgradient__( self, ndims ):
    'local gradient; jacobi formula'

    ax1, ax2 = self.axes
    return self * sum( inv( self.func, ax1, ax2 ).swapaxes(ax1,ax2)[...,_] * localgradient( self.func, ndims ), axes=[ax1,ax2] )

class DofIndex( ArrayFunc ):
  'element-based indexing'

  @classmethod
  def stack( cls, funcs, axis ):
    'stack'

    array = funcs[0].array
    iax = funcs[0].iax
    if not all( isinstance(func,cls) and ( func.array == array ).all() and func.iax == iax for func in funcs ):
      return ArrayFunc.stack( funcs, axis )

    index = funcs[0].index.stack( [ func.index for func in funcs ], axis )
    return DofIndex( array, iax, index )

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

  def __get__( self, i, item ):
    'get item'

    i = _normdim( self.ndim, i )
    if self.iax <= i < self.iax + self.index.ndim:
      index = get( self.index, i - self.iax, item )
      return DofIndex( self.array, self.iax, index )

    return DofIndex( get( self.array, i, item ), self.iax if i > self.iax else self.iax-1, self.index )

  def __add__( self, other ):
    'add'

    if not isinstance( other, DofIndex ) or self.iax != other.iax or self.index != other.index:
      return ArrayFunc.__add__( self, other )

    n = _min( self.array.shape[0], other.array.shape[0] )
    return DofIndex( self.array[:n] + other.array[:n], self.iax, self.index )

  def __sub__( self, other ):
    'add'

    if not isinstance( other, DofIndex ) or self.iax != other.iax or self.index != other.index:
      return ArrayFunc.__sub__( self, other )

    n = _min( self.array.shape[0], other.array.shape[0] )
    return DofIndex( self.array[:n] - other.array[:n], self.iax, self.index )

  def __mul__( self, other ):
    'multiply'

    other = _asarray(other)
    if not _isfunc(other) and other.ndim == 0:
      return DofIndex( self.array * other, self.iax, self.index )

    return ArrayFunc.__mul__( self, other )

  def __localgradient__( self, ndims ):
    'local gradient'

    return _const( 0., self.shape + (ndims,) )

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.multiply, shape=shape )

  def __eq__( self, other ):
    'compare'

    if self is other:
      return True
      
    if not isinstance(other,Multiply):
      return False

    return _equal( self.funcs[0], other.funcs[0] ) and _equal( self.funcs[1], other.funcs[1] ) \
        or _equal( self.funcs[0], other.funcs[1] ) and _equal( self.funcs[1], other.funcs[0] )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )

    func1, func2 = self.funcs
    dotaxes = []
    shift = 0
    for ax in axes:
      myax = ax - shift
      if func1.shape[myax] == 1 or func2.shape[myax] == 1:
        func1 = sum( func1, ax )
        func2 = sum( func2, ax )
        shift += 1
      else:
        dotaxes.append( myax )
    return Dot( func1, func2, dotaxes ) if dotaxes else func1 * func2

  def __get__( self, i, item ):
    'get'

    func1, func2 = self.funcs
    return get( func1, i, item ) * get( func2, i, item )

  def __det__( self, ax1, ax2 ):
    'determinant'

    if self.funcs[0].shape == ():
      return det( self.funcs[1], ax1, ax2 ) * self.funcs[0]

    if self.funcs[1].shape == ():
      return det( self.funcs[0], ax1, ax2 ) * self.funcs[1]

    return ArrayFunc.__det__( self, ax1, ax2 )

  def __prod__( self, axis ):
    'product'

    axis = _normdim( self.ndim, axis )
    func1, func2 = self.funcs
    n = self.shape[-1]
    return ( get( func1, -1, 0 )**n if func1.shape[-1] == 1 else prod( func1, -1 ) ) \
         * ( get( func2, -1, 0 )**n if func2.shape[-1] == 1 else prod( func2, -1 ) )

  def __mul__( self, other ):
    'multiply'

    if not _isfunc( other ) and not _isfunc( self.funcs[1] ):
      return self.funcs[0] * ( self.funcs[1] * other )

    return ArrayFunc.__mul__( self, other )

  def __div__( self, other ):
    'multiply'

    if not _isfunc( other ) and not _isfunc( self.funcs[1] ):
      return self.funcs[0] * ( self.funcs[1] / other )

    return ArrayFunc.__div__( self, other )

  def __localgradient__( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1[...,_] * localgradient( func2, ndims ) \
         + func2[...,_] * localgradient( func1, ndims )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1, n1, n2 ) * takediag( func2, n1, n2 )

class Divide( ArrayFunc ):
  'divide'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.divide, shape=shape )

  def __get__( self, i, item ):
    'get item'

    func1, func2 = self.funcs
    return get( func1, i, item ) / get( func2, i, item )

  def __localgradient__( self, ndims ):
    'local gradient'

    func1, func2 = self.funcs
    grad1 = localgradient( func1, ndims )
    grad2 = localgradient( func2, ndims )
    return ( grad1 - func1[...,_] * grad2 / func2[...,_] ) / func2[...,_]

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1, n1, n2 ) / takediag( func2, n1, n2 )

class Negate( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.negative, shape=func.shape )

  def __add__( self, other ):
    'addition'

    return other - self.func

  def __sub__( self, other ):
    'subtract'

    return -( other + self.func )

  def __mul__( self, other ):
    'multiply'

    if isinstance( other, Negate ):
      return self.func * other.func

    return -( self.func * other )

  def __div__( self, other ):
    'divide'

    if isinstance( other, Negate ):
      return self.func / other.func

    return -( self.func / other )

  def __neg__( self ):
    'negate'

    return self.func

  def __elemint__( self, weights ):
    'elementwise integration'
  
    return -ElemInt( self.func, weights )

  def __align__( self, axes, ndim ):
    'align'

    return -align( self.func, axes, ndim )

  def __get__( self, i, item ):
    'get'

    return -get( self.func, i, item )

  def sum( self, axes=-1 ):
    'sum'

    return -sum( self.func, axes )

  def __localgradient__( self, ndims ):
    'local gradient'

    return -localgradient( self.func, ndims )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    return -takediag( self.func, n1, n2 )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    self.__priority__ = _haspriority(func1) or _haspriority(func2)
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.add, shape=shape )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    if not axes:
      return self

    return sum( self.funcs[0], axes ) + sum( self.funcs[1], axes )

  def __mul__( self, other ):
    'multiply'

    if _haspriority(self) or _haspriority(other):
      return self.funcs[0] * other + self.funcs[1] * other

    return ArrayFunc.__mul__( self, other )

  def __align__( self, axes, ndim ):
    'multiply'

    if _haspriority(self):
      return align( self.funcs[0], axes, ndim ) + align( self.funcs[1], axes, ndim )

    return ArrayFunc.__align__( self, axes, ndim )

  def __eq__( self, other ):
    'compare'

    return self is other or ( isinstance(other,Add) and ( self.funcs == other.funcs or self.funcs == other.funcs[::-1] ) )

  def __localgradient__( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return localgradient( func1, ndims ) + localgradient( func2, ndims )

  def __get__( self, i, item ):
    'get'

    func1, func2 = self.funcs
    return get( func1, i, item ) + get( func2, i, item )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1, n1, n2 ) + takediag( func2, n1, n2 )

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.subtract, shape=shape )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    if not axes:
      return self

    return sum( self.funcs[0], axes ) - sum( self.funcs[1], axes )

  def __neg__( self ):
    'negate'

    func1, func2 = self.funcs
    return func2 - func1

  def __localgradient__( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return localgradient( func1, ndims ) - localgradient( func2, ndims )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1, n1, n2 ) - takediag( func2, n1, n2 )

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, axes ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    axes = _norm_and_sort( len(shape), axes )
    shape = list(shape)
    for axis in reversed(axes):
      shape.pop( axis )

    self.func1 = func1
    self.func2 = func2
    self.axes = axes
    ArrayFunc.__init__( self, args=(func1,func2,tuple( ax-func1.ndim for ax in axes )), evalf=numeric.contract, shape=shape )

  @classmethod
  def stack( cls, funcs, axis ):
    'stack funcs along new axis'

    axes = funcs[0].axes
    if not all( isinstance(func,cls) and func.axes == axes for func in funcs ):
      return ArrayFunc.stack( funcs, axis )

    func1 = funcs[0].func1
    if all( func.func1 == func1 for func in funcs ):
      func2 = funcs[0].func2
      newaxes = [ ax if ax < axis else ax + 1 for ax in axes ]
      return Dot( insert( func1, axis ), func2.stack( [ func.func2 for func in funcs ], axis ), newaxes )

    return ArrayFunc.stack( funcs, axis )

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    axes = []
    for ax in reversed(self.axes): # TODO check if we want reversed here
      if ax <= i:
        i += 1
        axes.append( ax )
      else:
        axes.append( ax-1 )
    return sum( get( self.func1, i, item ) * get( self.func2, i, item ), axes )

  def __localgradient__( self, ndims ):
    'local gradient'

    return sum( localgradient( self.func1, ndims ) * self.func2[...,_], self.axes ) \
         + sum( self.func1[...,_] * localgradient( self.func2, ndims ), self.axes )

  def __mul__( self, other ):
    'multiply'

    other = _asarray(other)
    if not _isfunc(other) and other.ndim == 0 and isinstance( self.func2, DofIndex ):
      return sum( self.func1 * ( self.func2 * other ), self.axes )

    return ArrayFunc.__mul__( self, other )

  def __add__( self, other ):
    'add'

    #TODO check for other combinations
    if isinstance( other, Dot ) and self.func1 == other.func1 and self.axes == other.axes and self.shape == other.shape:
      return sum( self.func1 * ( self.func2 + other.func2 ), self.axes )

    return ArrayFunc.__add__( self, other )

  def __sub__( self, other ):
    'add'

    #TODO check for other combinations
    if isinstance( other, Dot ) and self.func1 == other.func1 and self.axes == other.axes and self.shape == other.shape:
      return sum( self.func1 * ( self.func2 - other.func2 ), self.axes )

    return ArrayFunc.__sub__( self, other )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    n1, n2 = _norm_and_sort( len(self.shape), (n1,n2) )
    assert n1 < n2 # strict
    for ax in self.axes: # shift n1 & n2 to original axes
      if ax <= n1:
        n1 += 1
      if ax <= n2:
        n2 += 1
    axes = [ ax-(n1<ax)-(n2<ax) for ax in self.axes ]
    return sum( takediag( self.func1, n1, n2 ) * takediag( self.func2, n1, n2 ), axes )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    for ax1 in self.axes:
      axes = [ ax2+(ax2>=ax1) for ax2 in axes ]
    axes.extend( self.axes )
    return sum( self.func1 * self.func2, axes )

class Sum( ArrayFunc ):
  'sum'

  def __init__( self, func, axes ):
    'constructor'

    self.func = func
    self.axes = _norm_and_sort( func.ndim, axes )
    negaxes = [ ax-func.ndim for ax in reversed(self.axes) ]
    shape = list(func.shape)
    for ax in reversed(self.axes):
      shape.pop(ax)
    ArrayFunc.__init__( self, args=[func,negaxes], evalf=self.dosum, shape=shape )

  @staticmethod
  def dosum( arr, axes ):
    'sum'

    for ax in axes:
      arr = sum( arr, ax )
    return arr

class Debug( ArrayFunc ):
  'debug'

  def __init__( self, func, show=False ):
    'constructor'

    self.func = func
    self.show = show
    ArrayFunc.__init__( self, args=[func,show], evalf=self.debug, shape=func.shape )

  @staticmethod
  def debug( arr, show ):
    'debug'

    if show:
      print arr
    return arr

  def __localgradient__( self, ndims ):
    'local gradient'

    return Debug( localgradient( self.func, ndims ) )

  def __str__( self ):
    'string representation'

    return '{DEBUG}'

class TakeDiag( ArrayFunc ):
  'extract diagonal'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func,-2,-1], evalf=numeric.takediag, shape=func.shape[:-1] )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    if not axes:
      return self

    if axes[-1] == self.ndim-1:
      return ArrayFunc.sum( takediag( sum( self.func, axes[:-1] ) ) )

    return takediag( sum( self.func, axes ) )

class Take( ArrayFunc ):
  'generalization of numpy.take(), to accept lists, slices, arrays'

  def __init__( self, func, indices, axis ):
    'constructor'

    self.func = func
    self.axis = axis
    self.indices = indices
    s = [ slice(None) ] * func.ndim
    s[axis] = indices
    newlen, = numpy.empty( func.shape[axis] )[ indices ].shape
    assert newlen > 0
    shape = func.shape[:axis] + (newlen,) + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+tuple(s)), evalf=numpy.ndarray.__getitem__, shape=shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    return take( localgradient( self.func, ndims ), self.indices, self.axis )

class Power( ArrayFunc ):
  'power'

  def __init__( self, func, power ):
    'constructor'

    assert _isfunc( func )
    assert _isscalar( power )
    self.func = func
    self.power = power
    ArrayFunc.__init__( self, args=[func,power], evalf=numpy.power, shape=func.shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    return self.power * ( self.func**(self.power-1) )[...,_] * localgradient( self.func, ndims )

  def __get__( self, i, item ):
    'get'

    return get( self.func, i, item )**self.power

  def sum( self, axes=-1 ):
    'sum'

    if self.power == 2:
      return sum( self.func * self.func, axes )

    return ArrayFunc.sum( self, axes )

  def __takediag__( self, n1=-2, n2=-1 ):
    'take diagonal'

    return takediag( self.func, n1, n2 )**self.power

class Pointwise( ArrayFunc ):
  'pointwise transformation'

  def __init__( self, args, evalf, deriv ):
    'constructor'

    assert _isfunc( args )
    shape = args.shape[1:]
    self.args = args
    self.deriv = deriv
    ArrayFunc.__init__( self, args=tuple(args), evalf=evalf, shape=shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    return ( self.deriv( self.args )[...,_] * localgradient( self.args, ndims ) ).sum( 0 )

# PRIORITY OBJECTS
#
# Prority objects get priority in situations like A + B, which can be evaluated
# as A.__add__(B) and B.__radd__(A), such that they get to decide on how the
# operation is performed. The reason is that these objects are wasteful,
# generally introducing a lot of zeros, and we would like to see them disappear
# by the act of subsequent operations. For this annihilation to work well
# priority objects will keep themselves at the surface where magic happens.

class Inflate( ArrayFunc ):
  'expand locally supported functions'

  __priority__ = True

  def __init__( self, shape, blocks ):
    'constructor'

    self.blocks = blocks
    arrays_indices = []
    for func, indices in blocks:
      assert func
      assert isinstance(indices,tuple) # required for comparison later
      assert func.ndim == len(shape), 'wrongly shaped block: func.shape=%s, shape=%s' % ( func.shape, shape )
      assert len(indices) == len(shape)
      for n, ind in enumerate( indices ):
        if ind == slice(None):
          assert shape[n] == func.shape[n]
        else:
          assert shape[n] > 1 # we need this to idenify singleton dimensions with static axes
          assert ind.ndim == 1 # TODO check dtype
      arrays_indices.append( Tuple([ func, Tuple(indices) ]) )
    ArrayFunc.__init__( self, args=[tuple(shape)]+arrays_indices, evalf=self.inflate, shape=shape )

  @staticmethod
  def inflate( shape, *arrays_indices ):
    'evaluate'

    pointsh = ()
    for array, index in arrays_indices:
      if array.ndim > len(shape):
        pointsh = array.shape[:-len(shape)]
    retval = numpy.zeros( pointsh + shape )
    for array, index in arrays_indices:
      count = _sum( isinstance(ind,numpy.ndarray) for ind in index )
      iarr = 0
      for i in range( len(index) ):
        if isinstance(index[i],numpy.ndarray):
          s = [ numpy.newaxis ] * count
          s[iarr] = slice(None)
          index = index[:i] + (index[i][tuple(s)],) + index[i+1:]
          iarr += 1
      assert iarr == count
      retval[ (Ellipsis,)+index ] += array
    return retval

  def get_func_ind( self, iblk=0 ):
    'get function object and placement index'

    return self.blocks[iblk]

  def __kronecker__( self, axis, length, pos ):
    'kronecker'

    blocks = [ ( kronecker( func, axis, length, pos ),
                 ind[:axis] + (slice(None),) + ind[axis:] ) for func, ind in self.blocks ]
    return Inflate( self.shape[:axis] + (length,) + self.shape[axis:], blocks )

  def __norm2__( self, axis=-1 ):
    'norm2'

    axis = _normdim( self.ndim, axis )
    blocks = [ ( norm2( func, axis ), ind[:axis]+ind[axis+1:] ) for func, ind in self.blocks ]
    shape = self.shape[:axis] + self.shape[axis+1:]
    return Inflate( shape, blocks )

  def __getitem__( self, item ):
    'get item'

    if item == ():
      return self
    origitem = item # for debug msg
    nnew = _sum( it == numpy.newaxis for it in item )
    if Ellipsis in item:
      n = item.index( Ellipsis )
      item = item[:n] + (slice(None),) * (self.ndim-(len(item)-1-nnew)) + item[n+1:]
    # assert len(item) - nnew == self.ndim, 'invalid item: shape=%s, item=(%s)' % ( self.shape, ','.join(map(_obj2str,origitem)) )
    assert len(item) <= self.ndim + nnew, 'invalid item: shape=%s, item=(%s)' % ( self.shape, ','.join(map(_obj2str,origitem)) )
    item += (slice(None),) * ( self.ndim + nnew - len(item) )
    shape = self.shape
    blocks = self.blocks
    i = 0
    for it in item:
      if it == numpy.newaxis:
        shape = shape[:i] + (1,) + shape[i:]
        blocks = [ ( insert( func, i ), ind[:i]+(slice(None),)+ind[i:] ) for func, ind in blocks ]
        i += 1
      elif isinstance(it,int):
        blocks = [ ( get( func, i, it ), ind[:i]+ind[i+1:] ) for func, ind in blocks if get( func, i, it ) ]
        shape = shape[:i] + shape[i+1:]
      elif isinstance(it,list):
        blocks = [ ( take( func, it, i ), ind[:i]+(slice(None),)+ind[i+1:] ) for func, ind in blocks ]
        shape = shape[:i] + (len(it),) + shape[i+1:]
        i += 1
      else:
        assert it == slice(None), 'invalid item in getitem: %r' % it
        i += 1
    assert i == len(shape)
    return Inflate( shape, blocks )

  def __localgradient__( self, ndims ):
    'local gradient'

    blocks = [ (localgradient( func, ndims ),ind+(slice(None),)) for func, ind in self.blocks ]
    return Inflate( self.shape+(ndims,), blocks )

  def match_blocks( self, other ):
    'match blocks for multiplication/division'

    func1, func2 = _matchndim( self, other )
    shape = _jointshape( func1.shape, func2.shape )
    assert isinstance( func1, Inflate )
    if isinstance( func2, Inflate ):
      func2_blocks = func2.blocks
    else:
      func2_blocks = ( func2, (slice(None),) * len(shape) ),

    for f1, ind1 in func1.blocks:
      for f2, ind2 in func2_blocks:
        assert len(ind1) == len(ind2) == len(shape)
        ind = []
        for i in range( len(shape) ):
          if func1.shape[i] == 1 and ind1[i] == slice(None):
            ind.append( ind2[i] )
          elif func2.shape[i] == 1 and ind2[i] == slice(None):
            ind.append( ind1[i] )
          elif ind1[i] == ind2[i]:
            assert func1.shape[i] == func2.shape[i]
            ind.append( ind1[i] )
          elif ind1[i] == slice(None): # ind2[i] != slice(None):
            assert func1.shape[i] == func2.shape[i]
            f1 = takeindex( f1, i, ind2[i] )
            ind.append( ind2[i] )
          elif ind2[i] == slice(None): # ind1[i] != slice(None):
            assert func1.shape[i] == func2.shape[i]
            f2 = takeindex( f2, i, ind1[i] )
            ind.append( ind1[i] )
          else: # ind1[i] != slice(None) and ind2[i] != slice(None)
            break
        else:
          yield f1, f2, tuple(ind)

  def __mul__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Inflate )
    shape = _jointshape( func1.shape, func2.shape )
    blocks = []
    for f1, f2, ind in func1.match_blocks( func2 ):
      f12 = f1 * f2
      if not _iszero(f12):
        blocks.append( (f12,ind) )
    if not blocks:
      return _const( 0., shape )
    return Inflate( shape, blocks )

  def __div__( self, other ):
    'divide'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Inflate )
    shape = _jointshape( func1.shape, func2.shape )
    blocks = []
    for f1, f2, ind in func1.match_blocks( func2 ):
      f12 = f1 / f2
      blocks.append( (f12,ind) )
    return Inflate( shape, blocks )

  def __cross__( self, other, axis ):
    'cross product'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Inflate )
    shape = _jointshape( func1.shape, func2.shape )
    blocks = []
    for f1, f2, ind in func1.match_blocks( func2 ):
      f12 = cross( f1, f2, axis )
      if not _iszero(f12):
        blocks.append( (f12,ind) )
    if not blocks:
      return _const( 0., shape )
    return Inflate( shape, blocks )

  def __add__( self, other ):
    'add'

    other = _asarray( other )
    shape = _jointshape( self.shape, other.shape ) # check matching shapes
    if _iszero( other ):
      return expand( self, shape )

    assert isinstance( other, Inflate )
    blocks = []
    other_blocks = other.blocks
    for func1, ind1 in self.blocks:
      for i, (func2,ind2) in enumerate( other_blocks ):
        if ind1 == ind2:
          func12 = func1 + func2
          if func12:
            blocks.append(( func12, ind1 ))
          other_blocks = other_blocks[:i] + other_blocks[i+1:]
          break
      else:
        blocks.append(( func1, ind1 ))
    blocks.extend( other_blocks )
    return Inflate( self.shape, blocks )

  def __sub__( self, other ):
    'subtract'

    return self + (-other)

  def __defunct_outer__( self, axis=0 ):
    'outer product'

    #return self.insert(axis) * self.insert(axis+1)
    blocks = []
    for iblk in range( len(self.blocks) ):
      func, ind1 = self.blocks[iblk]
      func1 = insert( func, axis )
      func2 = insert( func, axis+1 )
      product = func1 * func2
      blocks.append(( product, ind1[:axis+1] + ind1[axis:] ))
      for jblk in range( iblk+1, len(self.blocks) ):
        func, ind2 = self.blocks[jblk]
        assert ind1[:axis] == ind2[:axis] and ind1[axis+1:] == ind2[axis+1:]
        func2 = insert( func, axis+1 )
        product = func1 * func2
        if not _iszero(product):
          blocks.append(( product, ind1[:axis+1] + ind2[axis:] ))
          blocks.append(( product.swapaxes(axis,axis+1), ind2[:axis+1] + ind1[axis:] ))
    return Inflate( self.shape[:axis+1] + self.shape[axis:], blocks )

  def sum( self, axes=-1 ):
    'sum'

    keep = numpy.ones( self.ndim, dtype=bool )
    keep[axes] = False
    if numpy.all( keep ):
      return self

    shape = tuple( sh for n,sh in enumerate(self.shape) if keep[n] )
    blocks = []
    dense = _const( 0., shape )
    indall = ( slice(None), ) * len(shape)
    for func, ind in self.blocks:
      ind = tuple( i for n,i in enumerate(ind) if keep[n] )
      func = sum( func, axes )
      if ind == indall:
        dense += func
      else:
        blocks.append(( func, ind ))

    if not blocks:
      return dense

    if not _iszero( dense ):
      blocks.append(( dense, indall ))

    return Inflate( shape, blocks )

  def __neg__( self ):
    'negate'

    blocks = [ (-func,ind) for func, ind in self.blocks ]
    return Inflate( self.shape, blocks )

  def dot( self, weights ):
    'array contraction'

    assert weights.ndim == 1
    s = (slice(None),)+(numpy.newaxis,)*(self.ndim-1)
    return sum( self * weights[s], axes=0 )

  def vector( self, ndims ):
    'vectorize'

    return vectorize( [self] * ndims )

  def __align__( self, axes, ndim ):
    'align'

    assert len(axes) == self.ndim
    shape = [1] * ndim
    for n, sh in zip( axes, self.shape ):
      shape[n] = sh
    blocks = []
    for func, ind in self.blocks:
      transind = [ slice(None) ] * ndim
      for n, i in zip( axes, ind ):
        transind[n] = i
      blocks.append(( align( func, axes, ndim ), tuple(transind) ))
    return Inflate( shape, blocks )

  def __takediag__( self, n1=-2, n2=-1 ):
    'trace'

    n1, n2 = _norm_and_sort( len(self.shape), (n1,n2) )
    assert n1 < n2 # strict
    sh = self.shape[n1]
    assert self.shape[n2] == sh
    shape = self.shape[:n1] + self.shape[n1+1:n2] + self.shape[n2+1:] + (sh,)
    blocks = []
    for func, ind in self.blocks:
      traceind = ind[:n1] + ind[n1+1:n2] + ind[n2+1:] + (slice(None),)
      blocks.append(( takediag( func, n1, n2 ), tuple(traceind) ))
    return Inflate( shape, blocks )

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  __priority__ = True

  def __init__( self, func, ax1, ax2 ):
    'constructor'

    assert 0 <= ax1 < ax2 < func.ndim+1
    self.toaxes = ax1, ax2
    n = func.shape[-1],
    shape = func.shape[:ax1] + n + func.shape[ax1:ax2-1] + n + func.shape[ax2-1:-1]
    self.func = func
    ArrayFunc.__init__( self, args=[func,ax1-(func.ndim+1),ax2-(func.ndim+1)], evalf=self.diagonalize, shape=shape )

  @staticmethod
  def diagonalize( data, ax1, ax2 ):
    'evaluate'

    shape = list(data.shape)
    n = shape.pop()
    shape.insert( ax2+len(shape)+1, n )
    shape.insert( ax1+len(shape)+1, n )
    assert shape[ax1] == shape[ax2] == n
    diagonalized = numpy.zeros( shape )
    numeric.takediag( diagonalized, ax1, ax2 )[:] = data
    return diagonalized

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    if i in self.toaxes:
      return kronecker( get( self.func, -1, item ),
        axis=self.toaxes[0] if i == self.toaxes[1] else self.toaxes[1]-1,
        pos=item, length=self.func.shape[-1] )

    ax1, ax2 = self.toaxes
    if i < ax1:
      ax1 -= 1
      ax2 -= 1
    elif i < ax2:
      ax2 -= 1
      i -= 1
    else:
      i -= 2

    return diagonalize( get( self.func, i, item ), ax1, ax2 )

  def __inv__( self, *axes ):
    'inverse'

    if _norm_and_sort( self.ndim, axes ) == self.toaxes:
      return diagonalize( reciprocal( self.func ), *self.toaxes )

    return ArrayFunc.__inv__( self, *axes )

  def __det__( self, *axes ):
    'determinant'

    axes = _norm_and_sort( self.ndim, axes )
    if axes == self.toaxes:
      return prod( self.func, -1 )

    return ArrayFunc.__det__( self, *axes )

  def __mul__( self, other ):
    'multiply'
 
    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Diagonalize )
    if _iszero( func2 ):
      return _const( 0., _jointshape(func1.shape,func2.shape) )
    return diagonalize( func1.func * takediag( func2, *func1.toaxes ), *func1.toaxes )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    if self.toaxes[0] in axes:
      sumax = self.toaxes[0]
      otherax = self.toaxes[1] - 1
    elif self.toaxes[1] in axes:
      sumax = self.toaxes[1]
      otherax = self.toaxes[0]
    else:
      return ArrayFunc.sum( self, axes )

    trans = range(otherax) + [-1] + range(otherax,self.ndim-2)
    remaining = [ ax if ax < sumax else ax-1 for ax in axes if ax != sumax ]
    return sum( align( self.func, trans, self.ndim-len(axes) ), remaining )

  def __align__( self, axes, ndim ):
    'align'

    assert all( 0 <= ax < ndim for ax in axes )
    assert len(axes) == self.ndim
    toaxes = [ axes[ax] for ax in self.toaxes ]
    axes = [ ax - (ax>toaxes[0]) - (ax>toaxes[1]) for ax in axes if ax not in toaxes ] + [ -1 ]
    return diagonalize( align( self.func, axes, ndim-1 ), *toaxes )

class Kronecker( ArrayFunc ):
  'kronecker'

  __priority__ = True

  def __init__( self, funcs, axis ):
    'constructor'

    shape = _jointshape( *[ func.shape for func in funcs if func is not None ] )
    axis = _normdim( len(shape)+1, axis )
    shape = shape[:axis] + (len(funcs),) + shape[axis:]
    self.funcs = tuple(funcs)
    self.axis = axis
    ArrayFunc.__init__( self, args=[self.axis-len(shape)]+list(funcs), evalf=self.kronecker, shape=shape )

  @staticmethod
  def kronecker( axis, *funcs ):
    'kronecker'

    shape, = set( func.shape for func in funcs if func is not None )
    axis += len(shape)+1
    shape = shape[:axis] + (len(funcs),) + shape[axis:]
    array = numpy.zeros( shape, dtype=float )
    for ifun, func in enumerate( funcs ):
      if func is not None:
        s = [slice(None)]*array.ndim
        s[axis] = ifun
        array[tuple(s)] = func
    return array

  def __takediag__( self, n1=-2, n2=-1 ):
    'trace'

    n1, n2 = _norm_and_sort( len(self.shape), (n1,n2) )

    if n1 == self.axis:
      n = n2-1
    elif n2 == self.axis:
      n = n1
    else:
      return ArrayFunc.__takediag__( self, n1, n2 )

    return Kronecker( [ func and get(func,n,ifun) for ifun, func in enumerate(self.funcs) ], axis=-1 )

  def __localgradient__( self, ndims ):
    'local gradient'

    funcs = [ func and localgradient( func, ndims ) for func in self.funcs ]
    return Kronecker( funcs, self.axis )

  def __add__( self, other ):
    'add'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Kronecker )

    if _iszero( func2 ):
      return func1

    if isinstance( func2, Kronecker ) and func1.axis == func2.axis:
      funcs = [ f1 if not f2 else f2 if not f1 else f1 + f2 for f1, f2 in zip( func1.funcs, func2.funcs ) ]
      if all( funcs ):
        return funcs[0].stack( funcs, func1.axis )
      return Kronecker( funcs, func1.axis )

    return ArrayFunc.__add__( self, other )

  def __mul__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Kronecker )
    funcs = [ func and func * get( func2, func1.axis, ifun ) for ifun, func in enumerate(func1.funcs) ]
    funcs = [ None if _iszero(func) else func for func in funcs ]
    if not any( funcs ):
      return _const( 0., _jointshape(func1.shape,func2.shape) )
    return Kronecker( funcs, self.axis )

  def __div__( self, other ):
    'multiply'

    func1, func2 = _matchndim( self, other )
    assert isinstance( func1, Kronecker )
    funcs = [ func / get( other, self.axis, ifun ) for ifun, func in enumerate(self.funcs) ]
    return Kronecker( funcs, self.axis )

  def __align__( self, trans, ndim ):
    'align'

    newaxis = trans[ self.axis ]
    trans = [ tr if tr < newaxis else tr-1 for tr in trans if tr != newaxis ]
    funcs = [ func and align( func, trans, ndim-1 ) for func in self.funcs ]
    return Kronecker( funcs, newaxis )

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    if i == self.axis:
      func = self.funcs[ item ]
      if not func:
        return _const( 0., self.shape[:i] + self.shape[i+1:] )
      return func

    if i > self.axis:
      i -= 1
      newaxis = self.axis
    else:
      newaxis = self.axis-1

    funcs = [ func and get( func, i, item ) for func in self.funcs ]
    return Kronecker( funcs, newaxis )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )

    if self.axis not in axes:
      newaxes = [ ax if ax < self.axis else ax-1 for ax in axes ]
      newaxis = self.axis - _sum( ax < self.axis for ax in axes )
      funcs = [ func and sum( func, newaxes ) for func in self.funcs ]
      return Kronecker( funcs, newaxis )

    newaxes = [ ax if ax < self.axis else ax-1 for ax in axes if ax != self.axis ]
    retval = 0
    for func in self.funcs:
      if func:
        retval += sum( func, newaxes )
    return retval

class Expand( ArrayFunc ):
  'singleton expand'

  __priority__ = True

  def __init__( self, func, shape ):
    'constructor'

    assert shape
    assert func.ndim == len(shape), 'non matching dimensions %d -> %d' % ( func.ndim, len(shape) )
    shape = tuple(shape)
    assert shape != func.shape, 'expanded shape matches original shape (=useless)'
    assert all( sh1 in (1,sh2) for sh1, sh2 in zip( func.shape, shape ) ), 'conflicting shapes %s->%s' % ( func.shape, shape )
    self.func = func
    for sh1, sh2 in zip( func.shape, shape ):
      assert sh1 in (sh2,1)
    ArrayFunc.__init__( self, args=(func,)+shape, evalf=numeric.expand, shape=shape )

  def __nonzero__( self ):
    'nonzero'

    return not _iszero( self.func )

  def __neg__( self ):
    'negate'

    return Expand( -self.func, self.shape )

  def __localgradient__( self, ndims ):
    'local gradient'

    return expand( localgradient( self.func, ndims ), self.shape+(ndims,) )

  def __get__( self, i, item ):
    'get'

    i = _normdim( self.ndim, i )
    shape = list(self.shape)
    sh = shape.pop(i)
    if sh == 1:
      assert isinstance( sh, int ) and 0 <= item < sh, 'item out of range'
      item = 0
    return expand( get( self.func, i, item ), shape )

  def sum( self, axes=-1 ):
    'sum'

    axes = _norm_and_sort( self.ndim, axes if _isiterable(axes) else [axes] )
    func = self.func
    if not func:
      return sum( func, axes )
    factor = 1
    for ax in reversed(axes):
      if func.shape[ax] == 1:
        func = get( func, ax, 0 )
        factor *= self.shape[ax]
      else:
        func = sum( func, ax )
    return func * factor

  def __prod__( self, axis ):
    'prod'

    axis = _normdim( self.ndim, axis )

    if self.func.shape[axis] == 1:
      return get( self.func, axis, 0 )**self.shape[axis]

    return expand( prod( self.func, axis ), self.shape[:axis] + self.shape[axis+1:] )

  def __reciprocal__( self ):
    'reciprocal'

    return expand( reciprocal( self.func ), self.shape )

  def __add__( self, other ):
    'multiply'

    add = self.func + other
    shape = add.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( add.shape[-self.ndim:], self.shape ) )
    return expand( add, shape )

  def __mul__( self, other ):
    'multiply'

    mul = self.func * other
    shape = mul.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( mul.shape[-self.ndim:], self.shape ) )
    return expand( mul, shape )

  def __div__( self, other ):
    'divide'

    div = self.func / other
    shape = div.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( div.shape[-self.ndim:], self.shape ) )
    return expand( div, shape )

  def __align__( self, axes, ndim ):
    'align'

    assert len(axes) == self.ndim
    assert all( 0 <= ax < ndim for ax in axes )
    shape = [ 1 ] * ndim
    for ax, sh in zip( axes, self.shape ):
      shape[ax] = sh
    return expand( align( self.func, axes, ndim ), shape )

# AUXILIARY FUNCTIONS

def _normdim( ndim, n ):
  'check bounds and make positive'

  assert _isint(ndim) and _isscalar(ndim) and ndim >= 0, 'ndim must be positive integer, got %s' % ndim
  if n < 0:
    n += ndim
  assert 0 <= n < ndim, 'argument out of bounds: %s not in [0,%s)' % (n,ndim)
  return n

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
        assert sh in ( combshape[i], 1 )
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
  if obj.__class__ == dict:
    return '{#%d}@%x' % ( len(obj), id(obj) )
  if isinstance( obj, slice ):
    I = ''
    if obj.start is not None:
      I += str(obj.start)
    I += ':'
    if obj.stop is not None:
      I += str(obj.stop)
    if obj.step is not None:
      I += ':' + str(obj.step)
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

_max = max
_min = min
_sum = sum
_isfunc = lambda arg: isinstance( arg, ArrayFunc )
_isscalar = lambda arg: _asarray(arg).ndim == 0
_isint = lambda arg: numpy.asarray( arg ).dtype == int
_ascending = lambda arg: ( numpy.diff(arg) > 0 ).all()
_iszero = lambda arg: not arg if _isfunc(arg) else ( numpy.asarray(arg) == 0 ).all()
_isunit = lambda arg: not _isfunc(arg) and ( numpy.asarray(arg) == 1 ).all()
_haspriority = lambda arg: _isfunc(arg) and arg.__priority__
_subsnonesh = lambda shape: tuple( 1 if sh is None else sh for sh in shape )
_const = lambda val, shape: expand( numpy.array(val).reshape((1,)*len(shape)), shape )

def _norm_and_sort( ndim, args ):
  'norm axes, sort, and assert unique'

  normargs = tuple( sorted( _normdim( ndim, arg ) for arg in args ) )
  assert _ascending( normargs ) # strict
  return normargs

def _jointdtype( *args ):
  'determine joint dtype'

  if any( _asarray(arg).dtype == float for arg in args ):
    return float
  return int

def _equal( arg1, arg2 ):
  'compare two objects'

  if arg1 is arg2:
    return True
  equals = arg1 == arg2
  if isinstance( equals, numpy.ndarray ):
    equals = equals.all()
  return equals

def _asarray( arg ):
  'convert to ArrayFunc or numpy.ndarray'
  
  if _isfunc(arg):
    return arg
  if isinstance( arg, (list,tuple) ) and any( _isfunc(f) for f in arg ):
    return stack( arg, axis=0 )
  arg = numpy.asarray( arg )
  assert arg.dtype != object
  return arg

# FUNCTIONS

def insert( arg, n ):
  'insert axis'

  arg = _asarray( arg )
  n = _normdim( arg.ndim+1, n )
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

  ndofs = 0
  allblocks = []
  funcs = map( _asarray, funcs )
  for func in funcs:
    if isinstance( func, Inflate ):
      func_blocks = func.blocks
    else:
      ind = (slice(None),)*func.ndim
      func_blocks = (func,ind),
    blocks = []
    for blockfunc, blockind in func_blocks:
      ind0 = blockind[0]
      if ind0 == slice(None):
        ind0 = numpy.arange( func.shape[0] )
      blocks.append(( blockfunc, (ndofs+ind0,)+blockind[1:] ))
    allblocks.append( blocks )
    ndofs += func.shape[0]
  return [ Inflate( (ndofs,)+func.shape[1:], blocks ) for func, blocks in zip( funcs, allblocks ) ]

def vectorize( funcs ):
  'vectorize'

  return util.sum( kronecker( func, axis=1, length=len(funcs), pos=ifun ) for ifun, func in enumerate( chain( funcs ) ) )

def expand( arg, shape ):
  'expand'

  arg = _asarray( arg )
  shape = tuple(shape)
  if shape == arg.shape:
    return arg
  if _isfunc( arg ):
    return arg.__expand__( shape )
  return Expand( arg, shape )

  #assert len(shape) == arg.ndim
  #if all( sh1 == sh2 or sh2 == None for sh1, sh2 in zip( arg.shape, shape ) ):
  #  return arg
  #return Expand( arg, shape )

def get( arg, i, item ):
  'get item'

  i = _normdim( arg.ndim, i )
  if _isfunc( arg ):
    return arg.__get__( i, item )
  arg = numpy.asarray( arg )
  assert _isint( item ) and _isscalar( item )
  return arg[ (slice(None),) * i + (item if arg.shape[i] > 1 else 0,) ]

def align( arg, axes, ndim ):
  'align'

  assert arg.ndim == len(axes)
  axes = [ _normdim(ndim,ax) for ax in axes ]
  if _isfunc( arg ):
    return arg.__align__( axes, ndim )
  return numeric.align( arg, axes, ndim )

def reciprocal( arg ):
  'reciprocal'

  arg = _asarray( arg )
  if _isfunc( arg ):
    return arg.__reciprocal__()
  return numpy.reciprocal( arg )

def elemint( arg, weights ):
  'elementwise integration'

  if _isfunc( arg ):
    return arg.__elemint__( weights )
  return arg * ElemArea( weights )

def takeindex( arg, iax, index ):
  'take index'

  iax = _normdim( arg.ndim, iax )
  return DofIndex( arg, iax, index )

def grad( arg, coords, ndims=0 ):
  'local derivative'

  if _isfunc( arg ):
    return arg.grad( coords, ndims )
  return _const( 0., arg.shape + coords.shape )

def symgrad( arg, coords, ndims=0 ):
  'gradient'

  if _isfunc( arg ):
    return arg.symgrad( coords, ndims )
  return _const( 0., arg.shape + coords.shape )

def div( arg, coords, ndims=0 ):
  'gradient'

  if _isfunc( arg ):
    return arg.div( coords, ndims )
  assert arg.shape[-1:] == coords.shape
  return _const( 0., arg.shape[:-1] )

def sum( arg, axes=-1 ):
  'sum over multiply axes'

  if _isfunc( arg ):
    return arg.sum( axes )
  arg = _asarray(arg)
  for ax in sorted( [ _normdim(arg.ndim,ax) for ax in axes ], reverse=True ) if _isiterable(axes) else [ axes ]:
    arg = arg.sum( ax )
  return arg

def det( arg, ax1, ax2 ):
  'determinant'

  if _isfunc( arg ):
    return arg.__det__( ax1, ax2 )
  return numeric.det( arg, ax1, ax2 )

def inv( arg, ax1, ax2 ):
  'inverse'

  if _isfunc( arg ):
    return arg.__inv__( ax1, ax2 )
  return numeric.inv( arg, (ax1,ax2) )

def norm2( arg, axis=-1 ):
  'norm2'

  arg = _asarray( arg )
  if _isfunc( arg ):
    return arg.__norm2__( axis )
  return numeric.norm2( arg, axis )

def takediag( arg, ax1=-2, ax2=-1 ):
  'takediag'

  if _isfunc( arg ):
    return arg.__takediag__( ax1, ax2 )
  return numeric.takediag( arg, ax1, ax2 )

def localgradient( arg, ndims ):
  'local derivative'

  if _isfunc( arg ):
    lgrad = arg.__localgradient__( ndims )
  else:
    arg = _asarray( arg )
    lgrad = _const( 0., arg.shape + (ndims,) )
  assert lgrad.ndim == arg.ndim + 1 and lgrad.shape[-1] == ndims \
     and all( sh2 == sh1 or sh1 is None and sh2 == 1 for sh1, sh2 in zip( arg.shape, lgrad.shape[:-1] ) ), \
      'bug found in localgradient(%d): %s -> %s' % ( ndims, arg.shape, lgrad.shape )
  return lgrad

def dotnorm( arg, coords, ndims=0 ):
  'normal component'

  return sum( arg * coords.normal( ndims-1 ) )

def kronecker( arg, axis, length, pos ):
  'kronecker'

  arg = _asarray( arg )
  if _isfunc( arg ):
    return arg.__kronecker__( axis, length, pos )
  newarr = numpy.zeros( arg.shape[:axis] + (length,) + arg.shape[axis:] )
  s = (slice(None),)*axis + (pos,)
  newarr[s] = arg
  return newarr

def diagonalize( arg, n1, n2 ):
  'diagonalize'

  arg = _asarray( arg )
  n1, n2 = _norm_and_sort( arg.ndim+1, [n1,n2] )
  return Diagonalize( arg, n1, n2 )

def concatenate( args, axis=0 ):
  'concatenate'

  concat = Concatenate if any( _isfunc(arg) for arg in args ) else numpy.concatenate
  return concat( _matchndim( *args ), axis )

def transpose( arg, trans=None ):
  'transpose'
  
  arg = _asarray( arg )
  if trans is None:
    invtrans = range( arg.ndim-1, -1, -1 )
  else:
    trans = numpy.asarray(trans)
    assert sorted(trans) == range(arg.ndim)
    invtrans = numpy.empty( arg.ndim, dtype=int )
    invtrans[ trans ] = numpy.arange( arg.ndim )
  return align( arg, invtrans, arg.ndim )

def prod( arg, axis ):
  'product'

  if _isfunc( arg ):
    return arg.__prod__( axis )
  return numpy.prod( arg, axis )

def choose( level, choices ):
  'choose'
  
  choices = _matchndim( *choices )
  if _isfunc(level) or any( _isfunc(choice) for choice in choices ):
    return Choose( level, choices )
  return numpy.choose( level, choices )

def cross( arg1, arg2, axis ):
  'cross product'

  if _isfunc(arg1) and not _haspriority(arg2):
    return arg1.__cross__(arg2,axis)
  if _isfunc(arg2):
    return -arg2.__cross__(arg1,axis)
  return numeric.cross(arg1,arg2,axis)

def outer( arg1, arg2=None, axis=0 ):
  'outer product'

  arg1, arg2 = _matchndim( arg1, arg2 if arg2 is not None else arg1 )
  axis = _normdim( arg1.ndim, axis )
  return insert(arg1,axis+1) * insert(arg1,axis)

def pointwise( args, evalf, deriv ):
  'general pointwise operation'

  if any( _isfunc(arg) for arg in args ):
    return Pointwise( _asarray(args), evalf, deriv )
  return evalf( *args )

sin = lambda arg: pointwise( [arg], numpy.sin, cos )
cos = lambda arg: pointwise( [arg], numpy.cos, lambda x: -sin(x) )
tan = lambda arg: pointwise( [arg], numpy.tan, lambda x: cos(x)**-2 )
exp = lambda arg: pointwise( [arg], numpy.exp, exp )
ln = lambda arg: pointwise( [arg], numpy.log, reciprocal )
log2 = lambda arg: ln(arg) / ln(2)
log10 = lambda arg: ln(arg) / ln(10)
power = lambda arg, power: _asarray( arg )**power
arctan2 = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.arctan2, lambda x: stack([x[1],-x[0]]) / sum(power(x,2),0) )
greater = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.greater, lambda x: numpy.zeros_like(x) )
less = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.less, lambda x: numpy.zeros_like(x) )
min = lambda arg1, arg2: choose( greater( arg1, arg2 ), [ arg1, arg2 ] )
max = lambda arg1, arg2: choose( greater( arg1, arg2 ), [ arg2, arg1 ] )
abs = lambda arg: choose( greater( arg, 0 ), -arg, arg )
sinh = lambda arg: .5 * ( exp(arg) - exp(-arg) )
cosh = lambda arg: .5 * ( exp(arg) + exp(-arg) )
tanh = lambda arg: 1 - 2. / ( exp(2*arg) + 1 )
arctanh = lambda arg: .5 * ( log(1+arg) - log(1-arg) )
piecewise = lambda level, intervals, *funcs: choose( sum( greater( insert(level,-1), intervals ) ), funcs )
trace = lambda arg, n1=-2, n2=-1: sum( takediag( arg, n1, n2 ) )

def take( arg, indices, axis ):
  if _isfunc( arg ):
    return arg.__take__( indices, axis )
  if isinstance( indices, slice ):
    s = [ slice(None) ] * arg.ndim
    s[axis] = indices
    return arg[ tuple(s) ]
  return numpy.take( arg, indices, axis )

@core.deprecated( old='Chain', new='chain' )
def Chain( funcs ):
  return chain( funcs )

@core.deprecated( old='Vectorize', new='vectorize' )
def Vectorize( funcs ):
  return vectorize( funcs )

@core.deprecated( old='Tan', new='tan' )
def Tan( func ):
  return tan( func )

@core.deprecated( old='Sinh', new='sinh' )
def Sinh( func ):
  return sinh( func )

@core.deprecated( old='Cosh', new='cosh' )
def Cosh( func ):
  return cosh( func )

@core.deprecated( old='Tanh', new='tanh' )
def Tanh( func ):
  return tanh( func )

@core.deprecated( old='Arctanh', new='arctanh' )
def Arctanh( func ):
  return arctanh( func )

@core.deprecated( old='StaticArray(arg)', new='arg' )
def StaticArray( arg ):
  return arg

@core.deprecated( old='Stack', new='stack' )
def Stack( arg ):
  return stack( arg )

@core.deprecated( old='Log', new='ln' )
def Log( arg ):
  return ln( arg )

@core.deprecated( old='Arctan2', new='arctan2' )
def Arctan2( arg1, arg2 ):
  return arctan2( arg1, arg2 )

@core.deprecated( old='Min', new='min' )
def Min( arg1, arg2 ):
  return min( arg1, arg2 )

@core.deprecated( old='Max', new='max' )
def Max( arg1, arg2 ):
  return max( arg1, arg2 )

@core.deprecated( old='Log10', new='log10' )
def Log10( arg ):
  return log10( arg )

@core.deprecated( old='Log2', new='log2' )
def Log2( arg ):
  return log2( arg )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
