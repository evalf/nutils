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

  __priority__ = False
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

    log.context( 'compiling' )

    if self.operations is None:
      self.data = []
      self.operations = []
      self.recurse_index( self.data, self.operations ) # compile expressions

      priority = set( op.__class__.__name__ for op, args in self.operations if op.__priority__ )
      if priority:
        log.warning( 'possible suboptimality:', ', '.join( priority ) )
        self.graphviz()

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

  def graphviz( self, title='graphviz' ):
    'create function graph'

    log.context( title )

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

  def asciitree( self ):
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

  __array_priority__ = 1. # http://stackoverflow.com/questions/7042496/numpy-coercion-problem-for-left-sided-binary-operator/7057530#7057530

  def __init__( self, evalf, args, shape ):
    'constructor'

    self.evalf = evalf
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
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

  # numpy array methods

  def sum( self, axes=-1 ): return sum( self, axes )

  # overloadable helper functions, should be called via corresponding module method

  def _multiply( self, other ): return Multiply( self, other )
  def _divide( self, other ): return Divide( self, other )
  def _add( self, other ): return Add( self, other )
  def _subtract( self, other ): return Subtract( self, other )
  def _take( self, indices, axis ): return Take( self, indices, axis )
  def _takediag( self ): return TakeDiag( self )
  def _cross( self, other, axis ): return Cross( self, other, axis )
  def _reciprocal( self ): return Reciprocal( self )
  def _negative( self ): return Negative( self )
  def _get( self, i, item ): return Get( self, (i,item) )
  def _repeat( self, shape ): return Repeat( self, shape )
  def _diagonalize( self ): return Diagonalize( self )
  def _align( self, axes, ndim ): return Align( self, axes, ndim )
  def _inverse( self ): return Inverse( self )
  def _determinant( self ): return Determinant( self )
  def _product( self, axis ): return Product( self, axis )
  def _localgradient( self, ndims ): raise NotImplementedError
  def _elemint( self, weights ): return ElemInt( self, weights )
  def _sum( self, axes=[] ): return Sum( self, axes )


  def __kronecker__( self, axis, length, pos ):
    'kronecker'

    funcs = [ _zeros_like(self) ] * length
    funcs[pos] = self
    return Kronecker( funcs, axis=axis )


  # standalone methods

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
        arr = insert( get( arr, n, it.start ), n )
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

  @classmethod
  def stack( cls, funcs, axis ):
    'stack'

    return stack( funcs, axis )

  def verify( self, value ):
    'check result'

    s = '=> ' + _obj2str(value)
    s += ' \ (%s)' % ','.join(map(str,self.shape))
    return s

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
      detJ = determinant( J, 0, 1 )
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

    trans = numpy.arange( self.ndim )
    trans[n1] = numeric.normdim( self.ndim, n2 )
    trans[n2] = numeric.normdim( self.ndim, n1 )
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
      Jinv = inverse( J )
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      G = ( J[:,:,_] * J[:,_,:] ).sum( 0 )
      Ginv = inverse( G )
      Jinv = ( J[_,:,:] * Ginv[:,_,:] ).sum()
    else:
      raise Exception, 'cannot invert %sx%s jacobian' % J.shape
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

  @property
  def T( self ):
    'transpose'

    return transpose( self )

  def __graphviz__( self ):
    'graphviz representation'

    args = Evaluable.__graphviz__( self )
    args['label'] += r'\n[%s]' % ','.join( map(str,self.shape) )
    if self.__priority__:
      args['fontcolor'] = 'white'
      args['fillcolor'] = 'black'
      args['style'] = 'filled'
    return args

  def __str__( self ):
    'string representation'

    return '%s<%s>' % ( self.__class__.__name__, ','.join(map(str,self.shape)) )

  __repr__ = __str__

  def norm2( self, axis=-1 ):
    warnings.warn( '''f.norm2(...) will be removed in future
  Please use function.norm2(f,...) instead.''', DeprecationWarning, stacklevel=2 )
    return norm2( self, axis )

  def localgradient( self, ndims ):
    warnings.warn( '''f.localgradient(...) will be removed in future
  Please use function.localgradient(f,...) instead.''', DeprecationWarning, stacklevel=2 )
    return localgradient( self, ndims )

  def trace( self, n1=-2, n2=-1 ):
    warnings.warn( '''f.trace(...) will be removed in future
  Please use function.trace(f,...) instead.''', DeprecationWarning, stacklevel=2 )
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
    self.__priority__ = func.__priority__
    ArrayFunc.__init__( self, args=[func,negaxes,ndim], evalf=self.align, shape=shape )

  @staticmethod
  def align( arr, trans, ndim ):
    'align'

    extra = arr.ndim - len(trans)
    return numeric.align( arr, range(extra)+trans, ndim+extra )

  def _elemint( self, weights ):
    'elementwise integration'
  
    return align( elemint( self.func, weights ), self.axes, self.ndim )

  def _align( self, axes, ndim ):
    'align'

    newaxes = [ axes[i] for i in self.axes ]
    return align( self.func, newaxes, ndim )

  def _takediag( self ):
    'take diag'

    if self.ndim-1 not in self.axes:
      return align( self.func, self.axes, self.ndim-1 )

    if self.ndim-2 not in self.axes:
      axes = [ ax if ax != self.ndim-1 else self.ndim-2 for ax in self.axes ]
      return align( self.func, axes, self.ndim-1 )

    if self.axes[-2:] in [ (self.ndim-2,self.ndim-1), (self.ndim-1,self.ndim-2) ]:
      axes = self.axes[:-2] + (self.ndim-2,)
      return align( takediag( self.func ), axes, self.ndim-1 )

    return ArrayFunc._takediag( self )

  def _get( self, i, item ):
    'get'

    axes = [ ax - (ax>i) for ax in self.axes if ax != i ]
    if len(axes) == len(self.axes):
      return align( self.func, axes, self.ndim-1 )

    n = self.axes.index( i )
    return align( get( self.func, n, item ), axes, self.ndim-1 )

  def _sum( self, axes ):
    'sum'

    sumaxes = []
    for ax in axes:
      try:
        idx = self.axes.index( ax )
      except ValueError:
        pass # trivial summation over singleton axis
      else:
        sumaxes.append( idx )

    trans = [ ax - _sum(i<ax for i in axes) for ax in self.axes if ax not in axes ]
    return align( sum( self.func, sumaxes ), trans, self.ndim-len(axes) )

  def _localgradient( self, ndims ):
    'local gradient'

    return align( localgradient( self.func, ndims ), self.axes+(self.ndim,), self.ndim+1 )

  def _multiply( self, other ):
    'multiply'

    if not _isfunc(other) and len(self.axes) == other.ndim:
      return align( self.func * transpose( other, self.axes ), self.axes, self.ndim )

    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func * other.func, self.axes, self.ndim )

    return ArrayFunc._multiply( self, other )

  def _add( self, other ):
    'add'

    #TODO make less restrictive:
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func + other.func, self.axes, self.ndim )

    if not _isfunc(other) and len(self.axes) == self.ndim:
      return align( self.func + transform( other, self.axes ), self.axes, self.ndim )

    return ArrayFunc._add( self, other )

  def _divide( self, other ):
    'multiply'

    #TODO make less restrictive:
    if isinstance( other, Align ) and self.axes == other.axes:
      return align( self.func / other.func, self.axes, self.ndim )

    if not _isfunc(other) and len(self.axes) == self.ndim:
      return align( self.func / transform( other, self.axes ), self.axes, self.ndim )

    return ArrayFunc._divide( self, other )

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
      assert 0 <= item < sh
      s[i] = item
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+tuple(s)), evalf=numpy.ndarray.__getitem__, shape=shape )

  def _localgradient( self, ndims ):
    'local gradient'

    f = localgradient( self.func, ndims )
    for i, item in reversed( self.items ):
      f = get( f, i, item )
    return f

  def _get( self, i, item ):
    'get item'

    selected = numpy.zeros( self.func.ndim, dtype=bool )
    for iax, it in self.items:
      selected[iax] = True
    renumber, = numpy.where( ~selected )
    assert len(renumber) == self.ndim
    return Get( self.func, (renumber[i],item), *self.items )

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

  def _localgradient( self, ndims ):
    'local gradient'

    return -localgradient( self.func, ndims ) / self.func[...,_]**2

  def _get( self, i, item ):
    'get item'

    return reciprocal( get( self.func, i, item ) )

  def _align( self, axes, ndim ):
    'align'

    return reciprocal( align( self.func, axes, ndim ) )

  def _multiply( self, other ):
    'multiply'

    return other / self.func

class Product( ArrayFunc ):
  'product'

  def __init__( self, func, axis ):
    'constructor'

    self.func = func
    self.axis = axis
    assert 0 <= axis < func.ndim
    shape = func.shape[:axis] + func.shape[axis+1:]
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], evalf=numpy.product, shape=shape )

  def _localgradient( self, ndims ):
    'local gradient'

    return self[...,_] * ( localgradient(self.func,ndims) / self.func[...,_] ).sum( self.axis )

  def _get( self, i, item ):
    'get item'

    func = get( self.func, i+(i>=self.axis), item )
    return product( func, self.axis-(i<self.axis) )

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

  def _localgradient( self, ndims ):
    'local gradient'

    return _zeros( self.shape + (ndims,) )

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

  def _localgradient( self, ndims ):
    'local gradient'

    assert ndims <= self.stdmap.ndims
    grad = Function( self.stdmap, self.igrad+1 )
    return grad if ndims == self.stdmap.ndims \
      else sum( grad[...,_] * transform( self.stdmap.ndims, ndims ), axes=-2 )

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, level, choices, *warnargs ):
    'constructor'

    assert not warnargs, 'ERROR: the Choose object has changed. Please use piecewise instead.'

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
    'gradient'

    grads = [ localgradient( choice, ndims ) for choice in self.choices ]
    if not any( grads ): # all-zero special case; better would be allow merging of intervals
      return _zeros( self.shape + (ndims,) )
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

  def __init__( self, func ):
    'constructor'

    assert func.shape[-1] == func.shape[-2]
    self.func = func
    ArrayFunc.__init__( self, args=[func,(-2,-1)], evalf=numeric.inv, shape=func.shape )

  def _localgradient( self, ndims ):
    'local gradient'

    G = localgradient( self.func, ndims )
    H = sum( self[...,_,:,:,_]
              * G[...,:,:,_,:], -3 )
    I = sum( self[...,:,:,_,_]
              * H[...,_,:,:,:], -3 )
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
    retval = numpy.empty( shape, dtype=_jointdtype(*arrays) )
    n0 = 0
    for array in arrays:
      n1 = n0 + array.shape[iax]
      retval[(slice(None),)*( iax if iax >= 0 else iax + ndim )+(slice(n0,n1),)] = array
      n0 = n1
    assert n0 == axlen
    return retval

  def _get( self, i, item ):
    'get'

    if i == self.axis:
      for f in self.funcs:
        if item < f.shape[i]:
          return get( f, i, item )
        item -= f.shape[i]

    axis = self.axis - (self.axis > i)
    return concatenate( [ get( f, i, item ) for f in self.funcs ], axis=axis )

  def _localgradient( self, ndims ):
    'gradient'

    funcs = [ localgradient( func, ndims ) for func in self.funcs ]
    return concatenate( funcs, axis=self.axis )

  def _multiply( self, other ):
    'multiply'

    if isinstance( other, Concatenate ) and self.axis == other.axis and [ f.shape[self.axis] for f in self.funcs ] == [ g.shape[other.axis] for g in other.funcs ]:
      return concatenate( [ f * g for f, g in zip(self.funcs,other.funcs) ], self.axis )

    return ArrayFunc._multiply( self, other )

  def _add( self, other ):
    'addition'

    if isinstance( other, Concatenate ) and self.axis == other.axis:
      fg = zip( self.funcs, other.funcs )
      if all( f.shape == g.shape for (f,g) in fg ):
        return concatenate( [ f+g for (f,g) in fg ], axis=self.axis )

    return ArrayFunc._add( self, other )

  def _sum( self, axes ):
    'sum'

    if self.axis in axes:
      axes = [ ax if ax < self.axis else ax-1 for ax in axes if ax != self.axis ]
      return sum( util.sum( sum( f, self.axis ) for f in self.funcs ), axes )

    return ArrayFunc._sum( self, axes )

  def _align( self, axes, ndim ):
    'align'

    funcs = [ align( func, axes, ndim ) for func in self.funcs ]
    axis = axes[ self.axis ]
    return concatenate( funcs, axis )

  def _takediag( self ):
    'take diagonal'

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
    'take'

    if axis == self.axis:
      n0 = 0
      funcs = []
      for func in self.funcs:
        n1 = n0 + func.shape[axis]
        ind = indices[n0:n1]-n0
        if len(ind) > 0:
          funcs.append( take( func, ind, axis ) )
        n0 = n1
      assert n0 == self.shape[axis]
      assert funcs, 'empty slice'
      if len( funcs ) == 1:
        return funcs[0]
      return concatenate( funcs, axis=axis )

    return ArrayFunc._take( self, indices, axis )

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

  def __init__( self, func1, func2, axis ):
    'contructor'

    self.func1 = func1
    self.func2 = func2
    self.axis = axis
    shape = _jointshape( func1.shape, func2.shape )
    assert 0 <= axis < len(shape), 'axis out of bounds: axis={0}, len(shape)={1}'.format( axis, len(shape) )
    ArrayFunc.__init__( self, args=(func1,func2,axis-len(shape)), evalf=numeric.cross, shape=shape )

  def _localgradient( self, ndims ):
    'local gradient'

    return cross( self.func1[...,_], localgradient(self.func2,ndims), axis=self.axis ) \
         - cross( self.func2[...,_], localgradient(self.func1,ndims), axis=self.axis )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, func ):
    'contructor'

    self.func = func
    ArrayFunc.__init__( self, args=(func,-2,-1), evalf=numeric.det, shape=func.shape[:-2] )

  def _localgradient( self, ndims ):
    'local gradient; jacobi formula'

    Finv = inverse( self.func ).swapaxes(-2,-1)
    G = localgradient( self.func, ndims )
    return self[...,_] * sum( Finv[...,_] * G, axes=[-3,-2] )

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

  def _get( self, i, item ):
    'get item'

    if self.iax <= i < self.iax + self.index.ndim:
      index = get( self.index, i - self.iax, item )
      return DofIndex( self.array, self.iax, index )

    return DofIndex( get( self.array, i, item ), self.iax if i > self.iax else self.iax-1, self.index )

  def _add( self, other ):
    'add'

    if isinstance( other, DofIndex ) and self.iax == other.iax and self.index == other.index:
      n = _min( self.array.shape[0], other.array.shape[0] )
      return DofIndex( self.array[:n] + other.array[:n], self.iax, self.index )

    return ArrayFunc._add( self, other )

  def _subtract( self, other ):
    'add'

    if isinstance( other, DofIndex ) and self.iax == other.iax and self.index == other.index:
      n = _min( self.array.shape[0], other.array.shape[0] )
      return DofIndex( self.array[:n] - other.array[:n], self.iax, self.index )

    return ArrayFunc._subtract( self, other )

  def _multiply( self, other ):
    'multiply'

    if not _isfunc(other) and other.ndim == 0:
      return DofIndex( self.array * other, self.iax, self.index )

    return ArrayFunc._multiply( self, other )

  def _localgradient( self, ndims ):
    'local gradient'

    return _zeros( self.shape + (ndims,) )

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

  def _sum( self, axes ):
    'sum'

    func1, func2 = self.funcs
    dotaxes = []
    shift = 0
    for ax in axes:
      myax = ax - shift
      if func1.shape[myax] == 1 or func2.shape[myax] == 1:
        func1 = sum( func1, myax )
        func2 = sum( func2, myax )
        shift += 1
      else:
        dotaxes.append( myax )
    return Dot( func1, func2, dotaxes ) if dotaxes else func1 * func2

  def _get( self, i, item ):
    'get'

    func1, func2 = self.funcs
    return get( func1, i, item ) * get( func2, i, item )

  def _determinant( self ):
    'determinant'

    if self.funcs[0].shape[-2:] == (1,1):
      return determinant( self.funcs[1] ) * self.funcs[0][...,0,0]

    if self.funcs[1].shape[-2:] == (1,1):
      return determinant( self.funcs[0] ) * self.funcs[1][...,0,0]

    return ArrayFunc._determinant( self )

  def _product( self, axis ):
    'product'

    func1, func2 = self.funcs
    return product( func1, axis ) * product( func2, axis )

  def _multiply( self, other ):
    'multiply'

    if not _isfunc( other ):
      if not _isfunc( self.funcs[1] ):
        return self.funcs[0] * ( self.funcs[1] * other )
      if not _isfunc( self.funcs[0] ):
        return self.funcs[1] * ( self.funcs[0] * other )

    return ArrayFunc._multiply( self, other )

  def _divide( self, other ):
    'multiply'

    if not _isfunc( other ) and not _isfunc( self.funcs[1] ):
      return self.funcs[0] * ( self.funcs[1] / other )

    return ArrayFunc._divide( self, other )

  def _localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1[...,_] * localgradient( func2, ndims ) \
         + func2[...,_] * localgradient( func1, ndims )

  def _takediag( self ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1 ) * takediag( func2 )

class Divide( ArrayFunc ):
  'divide'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.divide, shape=shape )

  def _get( self, i, item ):
    'get item'

    func1, func2 = self.funcs
    return get( func1, i, item ) / get( func2, i, item )

  def _localgradient( self, ndims ):
    'local gradient'

    func1, func2 = self.funcs
    grad1 = localgradient( func1, ndims )
    grad2 = localgradient( func2, ndims )
    return ( grad1 - func1[...,_] * grad2 / func2[...,_] ) / func2[...,_]

  def _takediag( self ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1 ) / takediag( func2 )

class Negative( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.negative, shape=func.shape )

  def _add( self, other ):
    'addition'

    return other - self.func

  def _subtract( self, other ):
    'subtract'

    return -( other + self.func )

  def _multiply( self, other ):
    'multiply'

    if isinstance( other, Negative ):
      return self.func * other.func

    return -( self.func * other )

  def _divide( self, other ):
    'divide'

    if isinstance( other, Negative ):
      return self.func / other.func

    return -( self.func / other )

  def _negative( self ):
    'negate'

    return self.func

  def _elemint( self, weights ):
    'elementwise integration'
  
    return -elemint( self.func, weights )

  def _align( self, axes, ndim ):
    'align'

    return -align( self.func, axes, ndim )

  def _get( self, i, item ):
    'get'

    return -get( self.func, i, item )

  def _sum( self, axes ):
    'sum'

    return -sum( self.func, axes )

  def _localgradient( self, ndims ):
    'local gradient'

    return -localgradient( self.func, ndims )

  def _takediag( self ):
    'take diagonal'

    return -takediag( self.func )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    self.funcs = func1, func2
    shape = _jointshape( func1.shape, func2.shape )
    self.__priority__ = _haspriority(func1) or _haspriority(func2)
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.add, shape=shape )

  def _sum( self, axes ):
    'sum'

    return sum( self.funcs[0], axes ) + sum( self.funcs[1], axes )

  def _multiply( self, other ):
    'multiply'

    if _haspriority(self) or _haspriority(other):
      return self.funcs[0] * other + self.funcs[1] * other

    return ArrayFunc._multiply( self, other )

  def _align( self, axes, ndim ):
    'align'

    if _haspriority(self):
      return align( self.funcs[0], axes, ndim ) + align( self.funcs[1], axes, ndim )

    return ArrayFunc._align( self, axes, ndim )

  def __eq__( self, other ):
    'compare'

    return self is other or isinstance(other,Add) and (
        _equal( self.funcs[0], other.funcs[0] ) and _equal( self.funcs[1], other.funcs[1] )
     or _equal( self.funcs[0], other.funcs[1] ) and _equal( self.funcs[1], other.funcs[0] ) )

  def _localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return localgradient( func1, ndims ) + localgradient( func2, ndims )

  def _get( self, i, item ):
    'get'

    func1, func2 = self.funcs
    return get( func1, i, item ) + get( func2, i, item )

  def _takediag( self ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1 ) + takediag( func2 )

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.subtract, shape=shape )

  def _sum( self, axes ):
    'sum'

    return sum( self.funcs[0], axes ) - sum( self.funcs[1], axes )

  def _negative( self ):
    'negate'

    func1, func2 = self.funcs
    return func2 - func1

  def _localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return localgradient( func1, ndims ) - localgradient( func2, ndims )

  def _takediag( self ):
    'take diagonal'

    func1, func2 = self.funcs
    return takediag( func1 ) - takediag( func2 )

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, axes ):
    'constructor'

    shape = _jointshape( func1.shape, func2.shape )
    axes = _norm_and_sort( len(shape), axes )
    shape = list(shape)
    orig = range( len(shape) )
    for axis in reversed(axes):
      shape.pop( axis )
      assert orig.pop( axis ) == axis

    self.func1 = func1
    self.func2 = func2
    self.axes = axes
    self.orig = orig # new -> original axis
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

  def _get( self, i, item ):
    'get'

    getax = self.orig[i]
    axes = [ ax - (ax>getax) for ax in self.axes ]
    return sum( get( self.func1, getax, item ) * get( self.func2, getax, item ), axes )

  def _localgradient( self, ndims ):
    'local gradient'

    return sum( localgradient( self.func1, ndims ) * self.func2[...,_], self.axes ) \
         + sum( self.func1[...,_] * localgradient( self.func2, ndims ), self.axes )

  def _multiply( self, other ):
    'multiply'

    if not _isfunc(other) and isinstance( self.func2, DofIndex ):
      return sum( self.func1 * ( self.func2 * other ), self.axes )

    return ArrayFunc._multiply( self, other )

  def _add( self, other ):
    'add'

    #TODO check for other combinations
    if isinstance( other, Dot ) and self.func1 == other.func1 and self.axes == other.axes and self.shape == other.shape:
      return sum( self.func1 * ( self.func2 + other.func2 ), self.axes )

    return ArrayFunc._add( self, other )

  def _subtract( self, other ):
    'add'

    #TODO check for other combinations
    if isinstance( other, Dot ) and self.func1 == other.func1 and self.axes == other.axes and self.shape == other.shape:
      return sum( self.func1 * ( self.func2 - other.func2 ), self.axes )

    return ArrayFunc._subtract( self, other )

  def _takediag( self ):
    'take diagonal'

    n1, n2 = self.orig[-2:]
    axes = [ ax-(n1<ax)-(n2<ax) for ax in self.axes ]
    return sum( takediag( self.func1, n1, n2 ) * takediag( self.func2, n1, n2 ), axes )

  def _sum( self, axes ):
    'sum'

    axes = self.axes + tuple( self.orig[ax] for ax in axes )
    return sum( self.func1 * self.func2, axes )

class Sum( ArrayFunc ):
  'sum'

  def __init__( self, func, axes ):
    'constructor'

    self.axes = tuple(axes)
    self.func = func
    assert all( 0 <= ax < func.ndim for ax in axes ), 'axes out of bounds'
    assert numpy.all( numpy.diff(axes) > 0 ), 'axes not sorted'
    negaxes = [ ax-func.ndim for ax in reversed(self.axes) ]
    shape = list(func.shape)
    for ax in reversed(self.axes):
      shape.pop(ax)
    ArrayFunc.__init__( self, args=[func,negaxes], evalf=self.dosum, shape=shape )

  def _localgradient( self, ndims ):
    'local gradient'

    return localgradient( self.func, ndims ).sum( self.axes )

  @staticmethod
  def dosum( arr, axes ):
    'sum'

    for ax in axes:
      arr = sum( arr, ax )
    return arr

class Debug( ArrayFunc ):
  'debug'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=self.debug, shape=func.shape )

  @staticmethod
  def debug( arr ):
    'debug'

    print arr
    return arr

  def _localgradient( self, ndims ):
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
    ArrayFunc.__init__( self, args=[func], evalf=numeric.takediag, shape=func.shape[:-1] )

  def _localgradient( self, ndims ):
    'local gradient'

    return takediag( localgradient( self.func, ndims ), -3, -2 ).swapaxes( -2, -1 )

  def _sum( self, axes ):
    'sum'

    if axes[-1] == self.ndim-1:
      return sum( takediag( sum( self.func, axes[:-1] ) ), -1 )

    return takediag( sum( self.func, axes ) )

class Take( ArrayFunc ):
  'generalization of numpy.take(), to accept lists, slices, arrays'

  def __init__( self, func, indices, axis ):
    'constructor'

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

  def _localgradient( self, ndims ):
    'local gradient'

    return self.power * ( self.func**(self.power-1) )[...,_] * localgradient( self.func, ndims )

  def _get( self, i, item ):
    'get'

    return get( self.func, i, item )**self.power

  def _sum( self, axes ):
    'sum'

    if self.power == 2:
      return sum( self.func * self.func, axes )

    return ArrayFunc._sum( self, axes )

  def _takediag( self ):
    'take diagonal'

    return takediag( self.func )**self.power

class Pointwise( ArrayFunc ):
  'pointwise transformation'

  def __init__( self, args, evalf, deriv ):
    'constructor'

    assert _isfunc( args )
    shape = args.shape[1:]
    self.args = args
    self.deriv = deriv
    ArrayFunc.__init__( self, args=tuple(args), evalf=evalf, shape=shape )

  def _localgradient( self, ndims ):
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

class Zeros( ArrayFunc ):
  'zero'

  __priority__ = True

  def __init__( self, shape ):
    'constructor'

    arrayshape = tuple( shape )
    ArrayFunc.__init__( self, args=[POINTS,arrayshape], evalf=self.zeros, shape=shape )

  @staticmethod
  def zeros( points, shape ):
    'prepend point axes'

    assert not any( sh is None for sh in shape ), 'cannot evaluate zeros for shape %s' % shape
    shape = points.shape[:-1] + shape
    strides = [0] * len(shape)
    return numpy.lib.stride_tricks.as_strided( numpy.array(0.), shape, strides )

  def _repeat( self, shape ):
    'repeat'

    return Zeros( shape )

  def _localgradient( self, ndims ):
    'local gradient'

    return _zeros( self.shape+(ndims,) )

  def _add( self, other ):
    'add'

    shape = _jointshape( self.shape, other.shape )
    return repeat( other, shape )

  def _subtract( self, other ):
    'subtract'

    shape = _jointshape( self.shape, other.shape )
    return -repeat( other, shape )

  def _multiply( self, other ):
    'multiply'

    shape = _jointshape( self.shape, other.shape )
    return _zeros( shape )

  def _divide( self, other ):
    'multiply'

    shape = _jointshape( self.shape, other.shape )
    return _zeros( shape )

  def _cross( self, other, axis ):
    'cross product'

    return self

  def _negative( self ):
    'negate'

    return self

  def _diagonalize( self ):
    'diagonalize'

    return _zeros( self.shape + (self.shape[-1],) )

  def _sum( self, axes ):
    'sum'

    shape = list( self.shape )
    for i in reversed( axes ):
      shape.pop( i )
    return _zeros( shape )

  def _align( self, axes, ndim ):
    'align'

    shape = [1] * ndim
    for ax, sh in zip( axes, self.shape ):
      shape[ax] = sh
    return _zeros( shape )

  def _get( self, i, item ):
    'get'

    return _zeros( self.shape[:i] + self.shape[i+1:] )

  def _takediag( self ):
    'trace'

    sh = max( self.shape[-2], self.shape[-1] )
    return _zeros( self.shape[:-2] + (sh,) )

  def _elemint( self, weights ):
    'elementwise integration'
  
    return numpy.zeros( [1]*self.ndim )


class Inflate( ArrayFunc ):
  'expand locally supported functions'

  __priority__ = True

  def __init__( self, shape, blocks ):
    'constructor'

    assert all( isinstance( sh, int ) for sh in shape ), 'Invalid shape: %s'%(shape,)
    assert blocks
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

  def _localgradient( self, ndims ):
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
            f1 = take( f1, ind2[i], i )
            ind.append( ind2[i] )
          elif ind2[i] == slice(None): # ind1[i] != slice(None):
            assert func1.shape[i] == func2.shape[i]
            f2 = take( f2, ind1[i], i )
            ind.append( ind1[i] )
          else: # ind1[i] != slice(None) and ind2[i] != slice(None)
            break
        else:
          yield f1, f2, tuple(ind)

  def _multiply( self, other ):
    'multiply'

    shape = _jointshape( self.shape, other.shape )
    blocks = []
    for f1, f2, ind in self.match_blocks( other ):
      f12 = f1 * f2
      if not _iszero(f12):
        blocks.append( (f12,ind) )
    if not blocks:
      return _zeros( shape )
    return Inflate( shape, blocks )

  def _divide( self, other ):
    'divide'

    shape = _jointshape( self.shape, other.shape )
    blocks = []
    for f1, f2, ind in self.match_blocks( other ):
      f12 = f1 / f2
      blocks.append( (f12,ind) )
    return Inflate( shape, blocks )

  def _cross( self, other, axis ):
    'cross product'

    shape = _jointshape( self.shape, other.shape )
    blocks = []
    for f1, f2, ind in self.match_blocks( other ):
      f12 = cross( f1, f2, axis )
      if not _iszero(f12):
        blocks.append( (f12,ind) )
    if not blocks:
      return _zeros( shape )
    return Inflate( shape, blocks )

  def _add( self, other ):
    'add'

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

  def _subtract( self, other ):
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

  def _sum( self, axes ):
    'sum'

    keep = numpy.ones( self.ndim, dtype=bool )
    keep[axes] = False

    shape = tuple( sh for n,sh in enumerate(self.shape) if keep[n] )
    blocks = []
    dense = _zeros( shape )
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

  def _negative( self ):
    'negate'

    blocks = [ (-func,ind) for func, ind in self.blocks ]
    return Inflate( self.shape, blocks )

  def __mod__( self, weights ):
    'dot shorthand'

    warnings.warn( '''array%w will be removed in future
  Please use array.dot(w) instead.''', DeprecationWarning, stacklevel=2 )
    return self.dot( weights )

  def dot( self, weights ):
    'array contraction'

    weights = numpy.asarray( weights, dtype=float )
    assert weights.ndim == 1
    s = (slice(None),)+(numpy.newaxis,)*(self.ndim-1)
    return sum( self * weights[s], axes=0 )

  def vector( self, ndims ):
    'vectorize'

    return vectorize( [self] * ndims )

  def _align( self, axes, ndim ):
    'align'

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

  def _takediag( self ):
    'trace'

    sh = max( self.shape[-2], self.shape[-1] )
    shape = self.shape[:-2] + (sh,)
    blocks = []
    for func, ind in self.blocks:
      assert ind[-2] == ind[-1] == slice(None)
      blocks.append(( takediag( func ), ind[:-1] ))
    return Inflate( shape, blocks )

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  __priority__ = True

  def __init__( self, func ):
    'constructor'

    n = func.shape[-1]
    assert n != 1
    shape = func.shape + (n,)
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=self.diagonalize, shape=shape )

  @staticmethod
  def diagonalize( data ):
    'evaluate'

    diagonalized = numpy.zeros( data.shape + (data.shape[-1],) )
    numeric.takediag( diagonalized )[:] = data
    return diagonalized

  def _get( self, i, item ):
    'get'

    if i >= self.ndim-2:
      return kronecker( get( self.func, -1, item ), axis=-1, pos=item, length=self.func.shape[-1] )

    return diagonalize( get( self.func, i, item ) )

  def _inverse( self ):
    'inverse'

    return diagonalize( reciprocal( self.func ) )

  def _determinant( self ):
    'determinant'

    return product( self.func, -1 )

  def _multiply( self, other ):
    'multiply'
 
    return diagonalize( self.func * takediag( other ) )

  def _negative( self ):
    'negate'

    return diagonalize( -self.func )

  def _sum( self, axes ):
    'sum'

    if axes[-1] >= self.ndim-2:
      return sum( self.func, axes[:-1] )

    return diagonalize( sum( self.func, axes ) )

  def _align( self, axes, ndim ):
    'align'

    if axes[-2:] in [ (ndim-2,ndim-1), (ndim-1,ndim-2) ]:
      return diagonalize( align( self.func, axes[:-2] + (ndim-2,), ndim-1 ) )

    return ArrayFunc._align( self, axes, ndim )

class Kronecker( ArrayFunc ):
  'kronecker'

  __priority__ = True

  def __init__( self, funcs, axis ):
    'constructor'

    shape = _jointshape( *[ func.shape for func in funcs ] )
    axis = numeric.normdim( len(shape)+1, axis )
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

  def _takediag( self ):
    'trace'

    if self.axis >= self.ndim-2:
      return Kronecker( [ get(func,-1,ifun) for ifun, func in enumerate(self.funcs) ], axis=-1 )

    return ArrayFunc._takediag( self )

  def _localgradient( self, ndims ):
    'local gradient'

    funcs = [ localgradient( func, ndims ) for func in self.funcs ]
    return Kronecker( funcs, self.axis )

  def _negative( self ):
    'negate'

    funcs = [ -func for func in self.funcs ]
    return Kronecker( funcs, self.axis )

  def _add( self, other ):
    'add'

    if isinstance( other, Kronecker ) and self.axis == other.axis:
      funcs = [ f1 + f2 for f1, f2 in zip( self.funcs, other.funcs ) ]
      return funcs[0].stack( funcs, self.axis )
      #return Kronecker( funcs, self.axis )

    return ArrayFunc._add( self, other )

  def _multiply( self, other ):
    'multiply'

    funcs = [ func * get( other, self.axis, ifun ) for ifun, func in enumerate(self.funcs) ]
    if all( _iszero(func) for func in funcs ):
      return _zeros( _jointshape(self.shape,other.shape) )

    return Kronecker( funcs, self.axis )

  def _divide( self, other ):
    'multiply'

    funcs = [ func / get( other, self.axis, ifun ) for ifun, func in enumerate(self.funcs) ]
    return Kronecker( funcs, self.axis )

  def _align( self, axes, ndim ):
    'align'

    newaxis = axes[ self.axis ]
    axes = [ tr if tr < newaxis else tr-1 for tr in axes if tr != newaxis ]
    funcs = [ align( func, axes, ndim-1 ) for func in self.funcs ]
    return Kronecker( funcs, newaxis )

  def _get( self, i, item ):
    'get'

    if i == self.axis:
      return self.funcs[ item ]

    if i > self.axis:
      i -= 1
      newaxis = self.axis
    else:
      newaxis = self.axis-1

    funcs = [ get( func, i, item ) for func in self.funcs ]
    return Kronecker( funcs, newaxis )

  def _sum( self, axes ):
    'sum'

    if self.axis not in axes:
      newaxes = [ ax if ax < self.axis else ax-1 for ax in axes ]
      newaxis = self.axis - _sum( ax < self.axis for ax in axes )
      funcs = [ sum( func, newaxes ) for func in self.funcs ]
      return Kronecker( funcs, newaxis )

    newaxes = [ ax if ax < self.axis else ax-1 for ax in axes if ax != self.axis ]
    retval = 0
    for func in self.funcs:
      retval += sum( func, newaxes )
    return retval

class Repeat( ArrayFunc ):
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

  def _negative( self ):
    'negate'

    return Repeat( -self.func, self.shape )

  def _localgradient( self, ndims ):
    'local gradient'

    return repeat( localgradient( self.func, ndims ), self.shape+(ndims,) )

  def _get( self, i, item ):
    'get'

    shape = self.shape[:i] + self.shape[i+1:]
    return repeat( get( self.func, i, item ), shape )

  def _sum( self, axes ):
    'sum'

    func = self.func
    factor = 1
    shape = list( self.shape )
    for ax in reversed(axes):
      sh = shape.pop( ax )
      if func.shape[ax] == 1:
        factor *= sh
      func = sum( func, ax )
    return repeat( func * factor, shape )

  def _product( self, axis ):
    'product'

    return repeat( product( self.func, axis ), self.shape[:axis] + self.shape[axis+1:] )

  def _reciprocal( self ):
    'reciprocal'

    return repeat( reciprocal( self.func ), self.shape )

  def _add( self, other ):
    'add'

    add = self.func + other
    shape = add.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( add.shape[-self.ndim:], self.shape ) )
    return repeat( add, shape )

  def _subtract( self, other ):
    'multiply'

    sub = self.func - other
    shape = sub.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( sub.shape[-self.ndim:], self.shape ) )
    return repeat( sub, shape )

  def _multiply( self, other ):
    'multiply'

    mul = self.func * other
    shape = mul.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( mul.shape[-self.ndim:], self.shape ) )
    return repeat( mul, shape )

  def _divide( self, other ):
    'divide'

    div = self.func / other
    shape = div.shape[:-self.ndim] + tuple( sh2 if sh1 == 1 else sh1 for sh1, sh2 in zip( div.shape[-self.ndim:], self.shape ) )
    return repeat( div, shape )

  def _align( self, axes, ndim ):
    'align'

    shape = [ 1 ] * ndim
    for ax, sh in zip( axes, self.shape ):
      shape[ax] = sh
    return repeat( align( self.func, axes, ndim ), shape )

class Const( ArrayFunc ):
  'pointwise transformation'

  __priority__ = True

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
    'local gradient'

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
_iszero = lambda arg: isinstance( arg, Zeros )
_isunit = lambda arg: not _isfunc(arg) and ( numpy.asarray(arg) == 1 ).all()
_haspriority = lambda arg: _isfunc(arg) and arg.__priority__
_subsnonesh = lambda shape: tuple( 1 if sh is None else sh for sh in shape )
_normdims = lambda ndim, shapes: tuple( numeric.normdim(ndim,sh) for sh in shapes )
_zeros = lambda shape: Zeros( shape )
_zeros_like = lambda arr: _zeros( arr.shape )

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

def _equal( arg1, arg2 ):
  'compare two objects'

  if arg1 is arg2:
    return True
  if not isinstance( arg1, numpy.ndarray ) and not isinstance( arg2, numpy.ndarray ):
    return arg1 == arg2
  arg1 = numpy.asarray( arg1 )
  arg2 = numpy.asarray( arg2 )
  if arg1.shape != arg2.shape:
    return False
  return numpy.all( arg1 == arg2 )

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

def repeat( arg, shape ):
  'repeat'

  arg = _asarray( arg )
  shape = tuple(shape)

  if shape == arg.shape:
    return arg

  assert shape == _jointshape( arg.shape, shape )

  if _isfunc( arg ):
    return arg._repeat( shape )

  return Repeat( arg, shape )

  #assert len(shape) == arg.ndim
  #if all( sh1 == sh2 or sh2 == None for sh1, sh2 in zip( arg.shape, shape ) ):
  #  return arg
  #return Repeat( arg, shape )

def get( arg, iax, item ):
  'get item'

  assert _isint( item ) and _isscalar( item )

  arg = _asarray( arg )
  iax = numeric.normdim( arg.ndim, iax )
  sh = arg.shape[iax]
  assert isinstance(sh,int), 'cannot get item %r from axis %r' % ( item, sh )

  item = 0 if sh == 1 \
    else numeric.normdim( sh, item )

  if _isfunc( arg ):
    return arg._get( iax, item )

  return arg[ (slice(None),) * iax + (item,) ]

def align( arg, axes, ndim ):
  'align'

  arg = _asarray( arg )

  assert ndim >= len(axes)
  assert len(axes) == arg.ndim
  axes = _normdims( ndim, axes )
  assert len(set(axes)) == len(axes), 'duplicate axes in align'

  if list(axes) == range(ndim):
    return arg

  if _isfunc( arg ):
    return arg._align( axes, ndim )

  return numeric.align( arg, axes, ndim )

def bringforward( arg, axis ):
  'bring axis forward'

  arg = _asarray(arg)
  axis = numeric.normdim(arg.ndim,axis)
  if axis == 0:
    return arg
  return transpose( args, [axis] + range(axis) + range(axis+1,args.ndim) )

def reciprocal( arg ):
  'reciprocal'

  arg = _asarray( arg )
  if _isfunc( arg ):
    return arg._reciprocal()

  return numpy.reciprocal( arg )

def elemint( arg, weights ):
  'elementwise integration'

  arg = _asarray( arg )

  if _isfunc( arg ):
    return arg._elemint( weights )

  return arg * ElemArea( weights )

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

  axes = list( _norm_and_sort( arg.ndim, axes if _isiterable(axes) else [axes] ) )
  assert numpy.all( numpy.diff(axes) > 0 ), 'duplicate axes in sum'

  shift = 0
  sumaxes = []
  for ax in axes:
    if arg.shape[ax] == 1:
      arg = get( arg, ax, 0 )
      shift += 1
    else:
      sumaxes.append( ax-shift )
      
  if not sumaxes:
    return arg

  if _isfunc( arg ):
    return arg._sum( sumaxes )

  for ax in reversed( sumaxes ):
    arg = arg.sum( ax )

  return arg

def determinant( arg, ax1, ax2 ):
  'determinant'

  arg = _asarray( arg )

  ax1, ax2 = _norm_and_sort( arg.ndim, [ax1,ax2] )
  assert ax2 > ax1 # strict

  n = arg.shape[ax1]
  assert n == arg.shape[ax2]
  if n == 1:
    return get( get( arg, ax2, 0 ), ax1, 0 )

  if _isfunc( arg ):
    trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
    return align( arg, trans, arg.ndim )._determinant()

  return numeric.det( arg, ax1, ax2 )

def inverse( arg, axes=(-2,-1) ):
  'inverse'

  arg = _asarray( arg )
  ax1, ax2 = _norm_and_sort( arg.ndim, axes )

  n = arg.shape[ax1]
  assert arg.shape[ax2] == n
  if n == 1:
    return reciprocal( arg )

  if _isfunc( arg ):
    trans = range(ax1) + [-2] + range(ax1,ax2-1) + [-1] + range(ax2-1,arg.ndim-2)
    return transpose( align( arg, trans, arg.ndim )._inverse(), trans )

  return numeric.inv( arg, (ax1,ax2) )

def inv( arg, ax1=-2, ax2=-1 ):
  warnings.warn( '''inv(array,i,j) will be removed in future
  Please use inverse(array,(i,j)) instead.''', DeprecationWarning, stacklevel=2 )
  return inverse( arg, (ax1,ax2) )

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

  if _isfunc( arg ):
    return arg._takediag()

  return numeric.takediag( arg )

def localgradient( arg, ndims ):
  'local derivative'

  arg = _asarray( arg )

  if _isfunc( arg ):
    lgrad = arg._localgradient( ndims )
  else:
    lgrad = _zeros( arg.shape + (ndims,) )

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

def diagonalize( arg ):
  'diagonalize'

  arg = _asarray( arg )

  if arg.shape[-1] == 1:
    return arg[...,_]

  if _isfunc( arg ):
    return arg._diagonalize()

  return Diagonalize( arg )

def concatenate( args, axis=0 ):
  'concatenate'

  args = _matchndim( *args )

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

  if any( _isfunc(arg) for arg in args ):
    return Concatenate( args, axis )

  return numpy.concatenate( args, axis )

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

  if arg.shape[axis] == 1:
    return get( arg, axis, 0 )

  if _isfunc( arg ):
    return arg._product( axis )

  return numpy.product( arg, axis )

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

  if _isfunc(arg1) and ( not _isfunc(arg2) or _haspriority(arg1) and not _haspriority(arg2) ):
    return arg1._cross(arg2,axis)

  if _isfunc(arg2):
    return -arg2._cross(arg1,axis)

  return numeric.cross(arg1,arg2,axis)

def outer( arg1, arg2=None, axis=0 ):
  'outer product'

  arg1, arg2 = _matchndim( arg1, arg2 if arg2 is not None else arg1 )
  axis = numeric.normdim( arg1.ndim, axis )
  return insert(arg1,axis+1) * insert(arg1,axis)

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
    return repeat( arg2, shape )

  if _isunit( arg2 ):
    return repeat( arg1, shape )

# for iax in range( len(shape) ):
#   if arg1.shape[iax] == arg2.shape[iax] == 1:
#     return insert( multiply( get(arg1,iax,0), get(arg2,iax,0) ), iax )

  if _isfunc( arg2 ) and ( not _isfunc( arg1 ) or _haspriority(arg2) and not _haspriority(arg1) ):
    return arg2._multiply( arg1 )

  if _isfunc( arg1 ):
    return arg1._multiply( arg2 )

  return numpy.multiply( arg1, arg2 )

def divide( arg1, arg2 ):
  'divide'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if _isunit( arg1 ):
    return repeat( reciprocal(arg2), shape )

  if _isunit( arg2 ):
    return repeat( arg1, shape )

# for iax in range( len(shape) ):
#   if arg1.shape[iax] == arg2.shape[iax] == 1:
#     return insert( divide( get(arg1,iax,0), get(arg2,iax,0) ), iax )

  if _isfunc( arg1 ):
    if _isfunc( arg2 ):
      return arg1._divide( arg2 )
    return arg1._multiply( numpy.reciprocal(arg2) )

  if _isfunc( arg2 ):
    return Divide( arg1, arg2 )

  return numpy.divide( arg1, arg2 )

def add( arg1, arg2 ):
  'add'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if _iszero( arg1 ):
    return repeat( arg2, shape )

  if _iszero( arg2 ):
    return repeat( arg1, shape )

# if not _haspriority(arg1) and not _haspriority(arg2): # don't cover the black balloons (race cond with Add.align)
#   for iax in range( arg1.ndim ):
#     if arg1.shape[iax] == arg2.shape[iax] == 1:
#       return insert( get(arg1,iax,0) + get(arg2,iax,0), iax )

  if _isfunc( arg2 ) and ( not _isfunc( arg1 ) or _haspriority(arg2) and not _haspriority(arg1) ):
    return arg2._add( arg1 )

  if _isfunc( arg1 ):
    return arg1._add( arg2 )

  return numpy.add( arg1, arg2 )

def subtract( arg1, arg2 ):
  'subtract'

  arg1, arg2 = _matchndim( arg1, arg2 )
  shape = _jointshape( arg1.shape, arg2.shape )

  if _iszero( arg1 ):
    return repeat( -arg2, shape )

  if _iszero( arg2 ):
    return repeat( arg1, shape )

# if not _haspriority(arg1) and not _haspriority(arg2): # don't cover the black balloons (race cond with Add.align)
#   for iax in range( arg1.ndim ):
#     if arg1.shape[iax] == arg2.shape[iax] == 1:
#       return insert( get(arg1,iax,0) + get(arg2,iax,0), iax )

  if _isfunc( arg2 ) and _haspriority(arg2) and not _haspriority(arg1):
    return (-arg2)._add( arg1 )

  if _isfunc( arg2 ) and not _isfunc( arg1 ):
    return Subtract( arg1, arg2 )

  if _isfunc( arg1 ):
    return arg1._subtract( arg2 )

  return numpy.subtract( arg1, arg2 )

def negative( arg ):
  'make negative'

  arg = _asarray(arg)
  if _isfunc( arg ):
    return arg._negative()

  return numpy.negative( arg )

def power( arg, n ):
  'power'

  arg = _asarray( arg )
  assert _isscalar( n )

  if n == 1:
    return arg

  if n == 0:
    return numpy.ones( arg.shape )

  if n < 0:
    return reciprocal( power( arg, -n ) )

  return Power( arg, n )

nsymgrad = lambda arg, coords: ( symgrad(arg,coords) * coords.normal() ).sum()
ngrad = lambda arg, coords: ( grad(arg,coords) * coords.normal() ).sum()
sin = lambda arg: pointwise( [arg], numpy.sin, cos )
cos = lambda arg: pointwise( [arg], numpy.cos, lambda x: -sin(x) )
tan = lambda arg: pointwise( [arg], numpy.tan, lambda x: cos(x)**-2 )
exp = lambda arg: pointwise( [arg], numpy.exp, exp )
ln = lambda arg: pointwise( [arg], numpy.log, reciprocal )
log2 = lambda arg: ln(arg) / ln(2)
log10 = lambda arg: ln(arg) / ln(10)
sqrt = lambda arg: arg**.5
argmin = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmin(numeric.stack(x),axis=0), _zeros_like )
argmax = lambda arg, axis: pointwise( bringforward(arg,axis), lambda *x: numpy.argmax(numeric.stack(x),axis=0), _zeros_like )
arctan2 = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.arctan2, lambda x: stack([x[1],-x[0]]) / sum(power(x,2),0) )
greater = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.greater, _zeros_like )
less = lambda arg1, arg2=None: pointwise( arg1 if arg2 is None else [arg1,arg2], numpy.less, _zeros_like )
min = lambda arg1, *args: choose( argmin( arg1 if not args else (arg1,)+args, axis=0 ), arg1 if not args else (arg1,)+args )
max = lambda arg1, *args: choose( argmax( arg1 if not args else (arg1,)+args, axis=0 ), arg1 if not args else (arg1,)+args )
sign = lambda arg: pointwise( [arg], numpy.sign, _zeros_like )
abs = lambda arg: arg * sign(arg)
sinh = lambda arg: .5 * ( exp(arg) - exp(-arg) )
cosh = lambda arg: .5 * ( exp(arg) + exp(-arg) )
tanh = lambda arg: 1 - 2. / ( exp(2*arg) + 1 )
arctanh = lambda arg: .5 * ( ln(1+arg) - ln(1-arg) )
piecewise = lambda level, intervals, *funcs: choose( sum( greater( insert(level,-1), intervals ) ), funcs )
trace = lambda arg, n1=-2, n2=-1: sum( takediag( arg, n1, n2 ) )
eye = lambda n: diagonalize( repeat( [1.], (n,) ) )
norm2 = lambda arg, axis=-1: sqrt( sum( arg * arg, axis ) )


def transform( fromdims, todims ):
  'transform to lower-dimensional space'

  if fromdims == todims:
    return eye( fromdims )
  assert fromdims > todims
  return Transform( fromdims, todims )

def take( arg, index, axis ):
  'take index'

  arg = _asarray( arg )
  axis = numeric.normdim( arg.ndim, axis )

  if _isfunc( index ):
    return DofIndex( arg, axis, index )

  if arg.shape[axis] == 1:
    if isinstance( index, slice ):
      assert index.start == None or index.start >= 0
      assert index.stop != None and index.stop > 0
      n = numpy.arange( index.start or 0, index.stop, index.step )
    else:
      n = numpy.asarray( index, dtype=int )
      assert numpy.all( n >= 0 )
    assert len(n) > 0, 'taking empty slice'
    return repeat( arg, arg.shape[:axis] + (len(n),) + arg.shape[axis+1:] )

  allindices = numpy.arange( arg.shape[axis] )
  index = allindices[index]
  assert len(index) > 0

  if numpy.all( index == allindices ):
    return arg

  if len(index) == 1:
    return insert( get( arg, axis, index[0] ), axis )

  if _isfunc( arg ):
    return arg._take( index, axis )

  return numpy.take( arg, index, axis )

def Chain( funcs ):
  warnings.warn( '''Chain will be removed in future
  Please use chain instead.''', DeprecationWarning, stacklevel=2 )
  return chain( funcs )

def Vectorize( funcs ):
  warnings.warn( '''Vectorize will be removed in future
  Please use vectorize instead.''', DeprecationWarning, stacklevel=2 )
  return vectorize( funcs )

def Tan( func ):
  warnings.warn( '''Tan will be removed in future
  Please use tan instead.''', DeprecationWarning, stacklevel=2 )
  return tan( func )

def Sin( func ):
  warnings.warn( '''Sin will be removed in future
  Please use sin instead.''', DeprecationWarning, stacklevel=2 )
  return sin( func )

def Cos( func ):
  warnings.warn( '''Cos will be removed in future
  Please use cos instead.''', DeprecationWarning, stacklevel=2 )
  return cos( func )

def Sinh( func ):
  warnings.warn( '''Sinh will be removed in future
  Please use sinh instead.''', DeprecationWarning, stacklevel=2 )
  return sinh( func )

def Cosh( func ):
  warnings.warn( '''Cosh will be removed in future
  Please use cosh instead.''', DeprecationWarning, stacklevel=2 )
  return cosh( func )

def Tanh( func ):
  warnings.warn( '''Tanh will be removed in future
  Please use tanh instead.''', DeprecationWarning, stacklevel=2 )
  return tanh( func )

def Arctanh( func ):
  warnings.warn( '''Arctanh will be removed in future
  Please use arctanh instead.''', DeprecationWarning, stacklevel=2 )
  return arctanh( func )

def StaticArray( arg ):
  warnings.warn( '''StaticArray will be removed in future
  Things should just work without it now.''', DeprecationWarning, stacklevel=2 )
  return arg

def Stack( arg ):
  warnings.warn( '''Stack will be removed in future
  Please use stack instead.''', DeprecationWarning, stacklevel=2 )
  return stack( arg )

def Log( arg ):
  warnings.warn( '''Log will be removed in future
  Please use ln instead.''', DeprecationWarning, stacklevel=2 )
  return ln( arg )

def Arctan2( arg1, arg2 ):
  warnings.warn( '''Arctan2 will be removed in future
  Please use arctan2 instead.''', DeprecationWarning, stacklevel=2 )
  return arctan2( arg1, arg2 )

def Min( arg1, arg2 ):
  warnings.warn( '''Min will be removed in future
  Please use min instead.''', DeprecationWarning, stacklevel=2 )
  return min( arg1, arg2 )

def Max( arg1, arg2 ):
  warnings.warn( '''Max will be removed in future
  Please use max instead.''', DeprecationWarning, stacklevel=2 )
  return max( arg1, arg2 )

def Log10( arg ):
  warnings.warn( '''Log10 will be removed in future
  Please use log10 instead.''', DeprecationWarning, stacklevel=2 )
  return log10( arg )

def Log2( arg ):
  warnings.warn( '''Log2 will be removed in future
  Please use log2 instead.''', DeprecationWarning, stacklevel=2 )
  return log2( arg )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
