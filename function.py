from . import util, element, numpy, _

def obj2str( obj ):
  'convert object to string'

  if isinstance( obj, numpy.ndarray ) and obj.ndim > 0:
    return 'array[%s]' % 'x'.join( map( str, obj.shape ) )
  if isinstance( obj, list ):
    return '[#%d]' % len(obj)
  if isinstance( obj, tuple ):
    if len(obj) < 10:
      return '(%s)' % ','.join( obj2str(o) for o in obj )
    return '(#%d)' % len(obj)
  if isinstance( obj, dict ):
    return '{#%d}' % len(obj)
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
  if isinstance( obj, DofAxis ):
    return '$'
  if obj is Ellipsis:
    return '...'
  if obj is None:
    return '_'
  return str(obj)

class StaticArray( numpy.ndarray ):
  'array wrapper'

  def __new__( cls, array ):
    'new'

    return numpy.asarray( array ).view( cls )

  def __eq__( self, other ):
    'compare'

    if self is other:
      return True
    equal = numpy.ndarray.__eq__( self, other )
    if isinstance( equal, numpy.ndarray ):
      equal = equal.view( numpy.ndarray ).all()
    return equal

  def __ne__( self, other ):
    'not equal'

    return not self == other

  def __nonzero__( self ):
    'nonzero'

    return self != 0

  def indices( self ):
    'get indices for numpy array'

    return ()

  def grad( self, coords ):
    'local derivative'

    return ZERO

  def sum( self, ax1=-1, *axes ):
    'sum over multiply axes'

    if self.ndim == 0:
      assert self == 0
      return ZERO

    assert not axes # for now
    return self.view( numpy.ndarray ).sum( ax1 ).view( StaticArray ) # NOTE cast does not work for scalars

  def localgradient( self, ndims ):
    'local derivative'

    return ZERO

  def __str__( self ):
    'string representation'

    if all( sh==1 for sh in self.shape ):
      return numpy.ndarray.__str__( self )
    return 'array[%s]' % 'x'.join( str(i) for i in self.shape )

ZERO = StaticArray( 0 )

def normdim( length, n ):
  'sort and make positive'

  return sorted( ni + length if ni < 0 else ni for ni in n )

def indent( key, *items ):
  'indent string  by two spaces'

  #return key + ''.join( '\n. ' + '\n  '.join( str(s).splitlines() ) for s in items )
  #indent = '\n' + '.' + ' ' * len(key)
  #indent1 = '\n' + ' ' + ' ' * len(key)
  #return key + ' ' + indent2.join( [ indent.join( str(s).splitlines() ) for s in items ] )

  lines = []
  indent = '\n' + ' ' + ' ' * len(key)
  for it in reversed( items ):
    lines.append( indent.join( str(it).splitlines() ) )
    indent = '\n' + '|' + ' ' * len(key)

  indent = '\n' + '+' + '-' * (len(key)-1) + ' '
  return key + ' ' + indent.join( reversed( lines ) )
  
class StackIndex( int ):
  'stack index'

  def __str__( self ):
    'string representation'

    return '%%%d' % self

class Evaluable( object ):
  'evaluable base classs'

  operations = None

  def __init__( self, args=None, clone=None ):
    'constructor'

    if clone:
      assert isinstance( clone, Evaluable )
      assert args is None
      self.__args = clone.__args
    else:
      assert clone is None
      self.__args = tuple(args)

  def recurse_index( self, operations ):
    'compile'

    for i, (op,idx) in enumerate( operations ):
      if op == self:
        return StackIndex( i )

    indices = [ arg.recurse_index( operations ) if isinstance( arg, Evaluable ) else arg for arg in self.__args ]
    operations.append(( self, indices ))
    return StackIndex( len(operations)-1 )

  def compile( self ):
    'compile'

    if self.operations is None:
      operations = [ (ELEM,None), (POINTS,None) ]
      self.recurse_index( operations ) # compile expressions
      self.operations = operations[2:]
      
  def __call__( self, elem, points ):
    'evaluate'

    self.compile()
    if isinstance( points, str ):
      points = elem.eval(points)
    values = [ elem, points ]
    try:
      for op, arglist in self.operations:
        args = [ values[arg] if isinstance( arg, StackIndex ) else arg for arg in arglist ]
        values.append( op.eval( *args ) )
    except:
      self.printstack( pointer=(op,arglist), values=values )
      raise
    return values[-1]

  def printstack( self, pointer=None, values=None ):
    'print stack'

    self.compile()
    print 'call stack:'
    for i in range( len(self.operations)+2 ):
      if i < 2:
        objstr = obj2str( values[i] )
      else:
        op, arglist = self.operations[i-2]
        try:
          code = op.eval.func_code
          names = code.co_varnames[ :code.co_argcount ]
          names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(arglist) - len(names) ) )
        except:
          args = [ obj2str(arg) for arg in arglist ]
        else:
          args = [ '%s=%s' % ( name, obj2str(arg) ) for name, arg in zip( names, arglist ) ]
        objstr = '%s( %s )' % ( op.__class__.__name__, ', '.join( args ) )
        if values and len( values ) > i:
          objstr += ' = ' + obj2str( values[i] )
        if pointer and pointer[0] is op and pointer[1] is arglist:
          objstr += ' <-----ERROR'
      print '%2d: %s' % ( i, objstr )

  def __eq__( self, other ):
    'compare'

    return self is other or ( self.__class__ == other.__class__ and self.__args == other.__args )

ELEM   = Evaluable( args=[object()] )
POINTS = Evaluable( args=[object()] )

class Tuple( Evaluable ):
  'combine'

  def __init__( self, items ):
    'constructor'

    self.items = items
    Evaluable.__init__( self, args=items )

  def __iter__( self ):
    'iterate'

    return iter(self.items)

  def __add__( self, other ):
    'add'

    return Tuple( self.items + tuple(other) )

  def __radd__( self, other ):
    'add'

    return Tuple( tuple(other) + self.items )

  @staticmethod
  def eval( *f ):
    'evaluate'

    return f

def Zip( *args ):
  'zip items together'

  args = map( tuple, args )
  n = len( args[0] )
  assert all( len(arg) == n for arg in args[1:] ), 'zipping items of different length'
  return Tuple( [ Tuple(v) for v in zip( *args ) ] )

# ARRAY FUNCTIONS

def merge( func0, *funcs ): # temporary
  'merge disjoint function spaces into one'

  assert func0.__class__ is Function
  shape = func0.shape
  mapping = func0.mapping.copy()
  nelems = len( func0.mapping )
  isfuncsp = shape and isinstance( shape[0], DofAxis )
  if isfuncsp:
    dofmap = shape[0].mapping.copy()
    ndofs = int(shape[0])
    shape = shape[1:]
  for func in funcs:
    assert func.__class__ is Function
    if func.shape and isinstance( func.shape[0], DofAxis ):
      assert isfuncsp and func.shape[1:] == shape
      dofmap.update( (elem,idx+ndofs) for (elem,idx) in func.shape[0].mapping.iteritems() )
      ndofs += int(func.shape[0])
    else:
      assert not isfuncsp and func.shape == shape
    mapping.update( func.mapping )
    nelems += len( func.mapping )
  assert nelems == len( mapping ), 'duplicate elements'
  if isfuncsp:
    shape = ( DofAxis(ndofs,dofmap), ) + shape
  return Function( shape, mapping )

class ArrayFunc( Evaluable ):
  'array function'

  __array_priority__ = 1. # fix numpy's idiotic default behaviour

  def __init__( self, shape, **kwargs ):
    'constructor'

    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    Evaluable.__init__( self, **kwargs )

  def __getitem__( self, item ):
    'get item'
  
    if not isinstance( item, tuple ):
      item = ( item, )

    if all( it == slice(None) for it in item ):
      return self

    if Ellipsis in item:
      idx = item.index( Ellipsis )
      n = len(item) - item.count(_) - 1
      item = item[:idx] + (slice(None),)*(len(self.shape)-n) + item[idx+1:]
      assert Ellipsis not in item

    assert len(item) - item.count(_) == len(self.shape)

    shape = []
    itershape = iter( self.shape )
    for it in item:
      if it == _:
        shape.append( 1 )
        continue
      sh = itershape.next()
      if isinstance( sh, int ) and isinstance( it, int ):
        assert it < sh, 'index out of bounds'
      elif isinstance( it, (list,tuple) ):
        assert all( i < sh for i in it ), 'index out of bounds'
        shape.append( len(it) )
      elif it == slice(None):
        shape.append( sh )
      elif isinstance( sh, int ) and isinstance( it, slice ):
        shape.append( len( numpy.arange(sh)[it] ) )
      else:
        raise Exception, 'invalid slice item: %r' % it
    return GetItem( self, tuple(shape), item )

  def __iter__( self ):
    'split first axis'

    if not self.shape:
      raise TypeError, 'scalar function is not iterable'

    return ( self[i,...] for i in range(self.shape[0]) )

  def find( self, elem, target, start, tol=1e-10, maxiter=999 ):
    'iteratively find x for f(x) = target, starting at x=start'

    points = start
    Jinv = self.localgradient( elem.ndims ).inv(0,1)
    r = target - self( elem, points )
    niter = 0
    while numpy.any( util.contract( r, r, axis=-1 ) > tol ):
      niter += 1
      if niter >= maxiter:
        raise Exception, 'failed to converge in %d iterations' % maxiter
      points = points.offset( util.contract( Jinv( elem, points ), r[:,_,:], axis=-1 ) )
      r = target - self( elem, points )
    return points

  def chain( self, func ):
    'chain function spaces together'
    
    n1 = int(func.shape[0])
    n2 = n1 + int(self.shape[0])
    return OffsetWrapper( self, offset=n1, ndofs=n2 )

  def sum( self, axis=-1 ):
    'sum'

    if self.shape[axis] == 1:
      s = [ slice(None) ] * len(self.shape)
      s[axis] = 0
      return self[ tuple(s) ]

    raise NotImplementedError, 'summation not supported yet!'

  def normalized( self, axis=-1 ):
    'normalize dimension'

    reshape = [ slice(None) ] * len(self.shape)
    reshape[axis] = _
    return self / self.norm2( axis )[ tuple(reshape) ]

  def normal( self, ndims=-1 ):
    'normal'

    assert len(self.shape) == 1
    if ndims <= 0:
      ndims += self.shape[0]

    if self.shape[0] == 2 and ndims == 1:
      grad = self.localgradient( ndims=1 )
      normal = Concatenate([ grad[1,:], -grad[0,:] ])
    elif self.shape[0] == 3 and ndims == 2:
      grad = self.localgradient( ndims=2 )
      normal = Cross( grad[:,0], grad[:,1], axis=0 )
    elif self.shape[0] == 2 and ndims == 0:
      grad = self.localgradient( ndims=1 )
      normal = grad[:,0] * Orientation()
    elif self.shape[0] == 3 and ndims == 1:
      grad = self.localgradient( ndims=1 )
      normal = Cross( grad[:,0], self.normal(), axis=0 )
    elif self.shape[0] == 1 and ndims == 0:
      return StaticArray( 1 ) # TODO fix direction!!!!
    else:
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims )
    return normal.normalized(0)

  def iweights( self, ndims ):
    'integration weights for [ndims] topology'

    J = self.localgradient( ndims )
    cndims, = self.shape
    if cndims == ndims:
      detJ = J.det( 0, 1 )
    elif ndims == 1:
      detJ = J[:,0].norm2( 0 )
    elif cndims == 3 and ndims == 2:
      detJ = Cross( J[:,0], J[:,1], axis=0 ).norm2( 0 )
    elif ndims == 0:
      detJ = 1
    else:
      raise NotImplementedError, 'cannot compute determinant for %dx%d jacobian' % J.shape[:2]
    return detJ * IWeights()

  def indices( self ):
    'get indices for numpy array'

    return Tuple( sh if isinstance(sh,DofAxis) else slice(None) for sh in self.shape )

  def curvature( self, ndims=-1 ):
    'curvature'

    if ndims <= 0:
      ndims += self.shape[0]
    assert ndims == 1 and self.shape == (2,)
    J = self.localgradient( ndims )
    H = J.localgradient( ndims )
    dx, dy = J[:,0]
    ddx, ddy = H[:,0,0]
    return ( dy * ddx - dx * ddy ) / J[:,0].norm2(0)**3

  def __mod__( self, weights ):
    'dot, new notation'

    return self.dot( weights )

  def dot( self, weights ):
    'dot convenience function'

    weights = StaticArray( weights )
    if not weights:
      return ZERO

    #assert int(self.shape[0]) == weights.shape[0]
    #  TODO reinsert range check

    func = self[ (slice(None),) + (_,) * (weights.ndim-1) + (slice(None),) * (self.ndim-1) ]
    shape = self.shape[:1] + weights.shape[1:] + self.shape[1:]
    if self.ndim > 1:
      weights = weights[ (Ellipsis,) + (_,) * (self.ndim-1) ]
    ind = Tuple( (self.shape[0],) + (slice(None),) * (weights.ndim-1) )

    return ( GetItem( weights, shape, ind ) * func ).sum( 0 )

  def inv( self, ax1, ax2 ):
    'inverse'

    return Inverse( self, ax1, ax2 )

  def swapaxes( self, n1, n2 ):
    'swap axes'

    return SwapAxes( self, n1, n2 )

  def grad( self, coords, ndims=0 ):
    'gradient'

    assert len(coords.shape) == 1
    if ndims <= 0:
      ndims += coords.shape[0]
    J = coords.localgradient( ndims )
    if J.shape[0] == J.shape[1]:
      Jinv = J.inv(0,1)
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      print 'WARNING: implementation of stack changed, needs checking'
      Jinv = Stack( [[ J, coords.normal()[:,_] ]] ).inv(0,1)[:-1,:]
    else:
      raise Exception, 'cannot invert jacobian'
    return ( self.localgradient( ndims )[...,_] * Jinv ).sum( -2 )

  def symgrad( self, coords, ndims=0 ):
    'gradient'

    g = self.grad( coords, ndims )
    return .5 * ( g + g.swapaxes(-2,-1) )

  def div( self, coords, ndims=0 ):
    'gradient'

    return self.grad( coords, ndims ).trace( -1, -2 )

  def ngrad( self, coords, ndims=0 ):
    'normal gradient'

    return ( self.grad(coords,ndims) * coords.normal(ndims-1) ).sum()

  def nsymgrad( self, coords, ndims=0 ):
    'normal gradient'

    return ( self.symgrad(coords,ndims) * coords.normal(ndims-1) ).sum()

  def norm2( self, axis=-1 ):
    'norm2'

    return Norm2( self, axis )

  def det( self, ax1, ax2 ):
    'determinant'

    return Determinant( self, ax1, ax2 )

  def __mul__( self, other ):
    'multiply'
  
    if not isinstance( other, Evaluable ):
      other = StaticArray( other )

    if other == 0:
      return ZERO

    if other == 1:
      return self

    return Multiply( self, other )

  def __div__( self, other ):
    'divide'
  
    if isinstance( other, Evaluable ):
      return Divide( self, other )

    return self * numpy.reciprocal( other ) # faster

  def __rdiv__( self, other ):
    'right divide'

    if not isinstance( other, Evaluable ):
      other = StaticArray( other )

    if other == 0:
      return ZERO
    
    return Divide( other, self )

  def __add__( self, other ):
    'add'

    if not isinstance( other, Evaluable ):
      other = StaticArray( other )

    if other == 0:
      return self

    return Add( self, other )

  __rmul__ = __mul__
  __radd__ = __add__

  def __sub__( self, other ):
    'subtract'
  
    if not isinstance( other, Evaluable ):
      other = StaticArray( other )

    if other == 0:
      return self

    return Subtract( self, other )

  def __rsub__( self, other ):
    'right subtract'

    if not isinstance( other, Evaluable ):
      other = StaticArray( other )

    if other == 0:
      return -self

    return Subtract( other, self )

  def __neg__( self ):
    'negate'

    return Negate( self )

  def __pow__( self, n ):
    'power'

    n = StaticArray(n)
    assert n.ndim == 0

    if n == 0:
      return 1

    if n == 1:
      return self

    return Power( self, n )

  @property
  def T( self ):
    'transpose'

    assert len(self.shape) == 2
    return SwapAxes( self, 0, 1 )

  def symmetric( self, n1, n2 ):
    'symmetric'

    return Symmetric( self, n1, n2 )

  def trace( self, n1=-2, n2=-1 ):
    'symmetric'

    return Trace( self, n1, n2 )

class ConstFunc( ArrayFunc ):
  'constant function'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, shape=[1], args=() )

  @staticmethod
  def eval():

    return 1

class OffsetWrapper( ArrayFunc ):
  'chain function spaces'

  def __init__( self, func, offset, ndofs ):
    'constructor'

    self.eval = func.eval
    self.func = func
    self.offset = offset
    ArrayFunc.__init__( self, clone=func, shape=( func.shape[0].transform(ndofs=ndofs,shift=offset), ) + func.shape[1:] )

  def localgradient( self, ndims ):
    'local gradient'

    return OffsetWrapper( self.func.localgradient( ndims ), offset=self.offset, ndofs=int(self.shape[0]) )

  def __str__( self ):
    'string representation'

    return '%s[++%d]' % ( self.func, self.offset )

class IWeights( ArrayFunc ):
  'integration weights'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[POINTS], shape=() )

  @staticmethod
  def eval( points ):
    'evaluate'

    return points.weights

class Orientation( ArrayFunc ):
  'point orientation'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[ELEM], shape=() )

  @staticmethod
  def eval( elem ):
    'evaluate'

    # VERY TEMPORARY
    elem, transform = elem.parent
    return ( transform.offset > .5 ) * 2 - 1

class Const( float ): # for select
  def eval( self, points ):
    return numpy.array(self)[...,_]

def const( *args ):
  'select by topology'

  assert len(args)%2 == 0
  mapping = {}
  for topo, val in zip( args[::2], args[1::2] ):
    mapping.update( dict.fromkeys( topo, Const(val) ) )
  return Function( shape=(), mapping=mapping )

class Function( ArrayFunc ):
  'function'

  def __init__( self, shape, mapping ):
    'constructor'

    self.mapping = mapping
    ArrayFunc.__init__( self, args=(ELEM,POINTS,mapping), shape=shape )

  @staticmethod
  def eval( elem, points, fmap ):
    'evaluate'

    func = fmap.get( elem )
    while func is None:
      elem, transform = elem.parent
      points = transform.eval( points )
      func = fmap.get( elem )
    return func.eval( points )

  def vector( self, ndims ):
    'vectorize'

    return Vectorize( [self]*ndims )

  def localgradient( self, ndims ):
    'local derivative'

    return LocalGradient( self, ndims, level=1 )

  def __str__( self ):
    'string representation'

    return 'F@%x' % id(self.mapping)

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, x, intervals, *funcs ):
    'constructor'

    shapes = [ f.shape for f in funcs if isinstance( f, ArrayFunc ) ]
    shape = shapes.pop()
    assert all( sh == shape for sh in shapes )
    assert len(intervals) == len(funcs)-1
    ArrayFunc.__init__( self, args=(x,intervals)+funcs, shape=x.shape+shape )

  @staticmethod
  def eval( x, intervals, *choices ):
    'evaluate'

    which = 0
    for i in intervals:
      which += ( x > i ).astype( int )
    return numpy.choose( which, choices )

class Choose2D( ArrayFunc ):
  'piecewise function'

  def __init__( self, coords, contour, fin, fout ):
    'constructor'

    shape, (fin,fout) = util.align_arrays( fin, fout )
    ArrayFunc.__init__( self, args=(coords,contour,fin,fout), shape=shape )

  @staticmethod
  def eval( xy, contour, fin, fout ):
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

    ax1, ax2 = normdim( len(func.shape), (ax1,ax2) )
    assert func.shape[ax1] == func.shape[ax2]
    self.func = func
    self.axes = ax1, ax2
    ArrayFunc.__init__( self, args=(func,(ax1-func.ndim,ax2-func.ndim)), shape=func.shape )

  def localgradient( self, ndims ):
    'local gradient'

    ax1, ax2 = self.axes
    G = self.func.localgradient( ndims )
    H = ( self[...,_,_].swapaxes(ax1,-1) * G[...,_].swapaxes(ax2,-1) ).sum()
    I = ( self[...,_,_].swapaxes(ax2,-1) * H[...,_].swapaxes(ax1,-1) ).sum()
    return -I

  eval = staticmethod( util.inv )

  def __str__( self ):
    'string representation'

    return indent( 'Inv:%d,%d' % self.axes, self.func )

class TrivialAxis( object ):
  'trivial axis'

  def __init__( self, end, beg=0 ):
    'new'

    self.beg = beg
    self.end = end

  def transform( self, ndofs, shift ):
    'shift numbering and widen axis'

    return TrivialAxis( ndofs, mapping )

  @staticmethod
  def eval( xi, idxmap ):
    'evaluate'

    index = idxmap.get( xi.elem )
    while index is None:
      xi = xi.next
      index = idxmap.get( xi.elem )
    return index

class DofAxis( Evaluable ):
  'dof axis'

  def __init__( self, end, mapping, beg=0 ):
    'new'

    self.beg = beg
    self.end = end
    self.mapping = mapping
    Evaluable.__init__( self, args=(ELEM,mapping) )

  def transform( self, ndofs, shift ):
    'shift numbering and widen axis'

    mapping = dict( (elem,idx+shift) for elem, idx in self.mapping.iteritems() )
    return DofAxis( ndofs, mapping )

  @staticmethod
  def eval( elem, idxmap ):
    'evaluate'

    index = idxmap.get( elem )
    while index is None:
      elem, transform = elem.parent
      index = idxmap.get( elem )
    return index

  def __int__( self ):
    'integer'

    return self.end

  def __eq__( self, other ):
    'equals'

    if self is other:
      return True

    if not isinstance( other, DofAxis ):
      return False
      
    if set(self.mapping) != set(other.mapping):
      return False

    for elem in self.mapping:
      if list(self.mapping[elem]) != list(other.mapping[elem]):
        return False

    return True

  def __add__( self, other ):
    'add'

    if other == 0:
      return self

    assert isinstance( other, DofAxis )

    assert other.beg == 0
    mapping = self.mapping.copy()
    ndofs = self.end - self.beg
    for elem, idx2 in other.mapping.iteritems():
      idx1 = mapping.get( elem )
      mapping[ elem ] = idx2 + ndofs if idx1 is None \
                   else numpy.hstack([ idx1, idx2 + ndofs ])
    return DofAxis( beg=self.beg, end=self.end+other.end, mapping=mapping )

  def __repr__( self ):
    'string representation'

    return 'DofAxis(%d-%d)' % ( self.beg, self.end )

class Concatenate( ArrayFunc ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    self.axis = axis
    self.funcs = funcs
    shape = ( sum( func.shape[0] for func in funcs ), ) + funcs[0].shape[1:]
    ArrayFunc.__init__( self, args=(axis-funcs[0].ndim,)+tuple(funcs), shape=shape )

  def localgradient( self, ndims ):
    'gradient'

    funcs = [ func.localgradient(ndims) for func in self.funcs ]
    return Concatenate( funcs, axis=self.axis )

  @staticmethod
  def eval( axis, *funcs ):
    'evaluate'

    return numpy.concatenate( funcs, axis=axis )

  def __str__( self ):
    'string representation'

    return indent( 'Concat:%d' % self.axis, *self.funcs )

class Vectorize( ArrayFunc ):
  'vectorize'

  def __init__( self, funcs ):
    'constructor'

    assert all( f.ndim == funcs[0].ndim for f in funcs[1:] )
    shape = (len(funcs),) + funcs[0].shape[1:] # TODO max over all funcs
    self.funcs = funcs
    shape = ( util.sum( func.shape[0] for func in funcs ), len(funcs) ) + funcs[0].shape[1:]
    ArrayFunc.__init__( self, args=(shape[1:],)+tuple(funcs), shape=shape )

  @staticmethod
  def eval( shape, *funcs ):
    'evaluate'

    axis = -len(shape)
    N = sum( func.shape[axis] for func in funcs )
    data = numpy.zeros( funcs[0].shape[:axis] + (N,) + shape )
    count = 0
    for i, func in enumerate( funcs ):
      n = func.shape[axis]
      ind = ( Ellipsis, slice(count,count+n), i ) + (slice(None),) * (len(shape)-1)
      data[ind] = func
      count += n
    assert count == N
    return data

  def localgradient( self, ndims ):
    'gradient'

    return Vectorize([ func.localgradient(ndims) for func in self.funcs ])

  def trace( self, n1, n2 ):
    'trace'

    n1, n2 = normdim( len(self.shape), (n1,n2) )
    assert self.shape[n1] == self.shape[n2]
    if n1 == 1 and n2 == 2:
      trace = Concatenate([ func[:,idim] for idim, func in enumerate( self.funcs ) ])
    else:
      trace = Trace( self, n1, n2 )
    return trace

  def dot( self, weights ):
    'dot'

    if all( func == self.funcs[0] for func in self.funcs[1:] ):
      return self.funcs[0].dot( weights.reshape( len(self.funcs), -1 ).T )

    # TODO broken for merged functions!

    n1 = 0
    funcs = []
    for func in self.funcs:
      n0 = n1
      n1 += int(func.shape[0])
      funcs.append( func.dot( weights[n0:n1,_] ) )
    return Concatenate( funcs )

  def __str__( self ):
    'string representation'

    return indent( 'Vec', *self.funcs )

class Stack( ArrayFunc ):
  'stack functions'

  def __init__( self, funcs, axis=-1 ):
    'constructor'

    shape, funcs = util.align_arrays( *funcs )
    if axis < 0:
      axis += len(shape) + 1
    assert 0 <= axis < len(shape)+1

    self.funcs = funcs
    self.axis = axis
    shape = shape[:axis] + (len(funcs),) + shape[axis:]
    ArrayFunc.__init__( self, args=(axis-len(shape),shape)+tuple(funcs), shape=shape )

  @staticmethod
  def eval( axis, shape, *funcs ):
    'evaluate'

    stacked = numpy.empty( funcs[0].shape[:funcs[0].ndim+1-len(shape)] + shape )
    for array, f in zip( util.ndiag( stacked, [axis] ), funcs ):
      array[:] = f
    return stacked

  def localgradient( self, ndims ):
    'local gradient'

    grads = [ f.localgradient( ndims ) for f in self.funcs ]
    return Stack( grads, self.axis )

  def __str__( self ):
    'string representation'

    return indent( 'Stack:%d' % self.axis, *self.funcs )

#class Stack22( ArrayFunc ):
#  'special 2x2 stack based on unified axes'
#
#  def __init__( self, *funcs ):
#    'constructor'
#
#    for func in funcs:
#      assert len(func.shape) == 2
#
#  @staticmethod
#  def eval( a11, a12, a21, a22 ):
#    'evaluate'
#
#    def unique( n, *args ):
#      N = -1
#      for arg in args:
#        if isinstance( arg, numpy.ndarray ):
#          if N == -1:
#            N = arg.shape[n]
#          else:
#            assert arg.shape[n] == N
#      assert N != -1
#      return N
#
#    nrows = unique(0,a11,a12), unique(0,a21,a22)
#    ncols = unique(1,a11,a21), unique(1,a12,a22)
#    npoints = unique(2,a11,a12,a21,a22) # blatantly assuming 1D points
#
#    data = numpy.empty( [ sum(nrows), sum(ncols), npoints ] )
#    data[:nrows[0],:ncols[0]] = a11
#    data[:nrows[0],ncols[0]:] = a12
#    data[nrows[0]:,:ncols[0]] = a21
#    data[nrows[0]:,ncols[0]:] = a22
#    return data

class UFunc( ArrayFunc ):
  'user function'

  def __init__( self, coords, ufunc, *gradients ):
    'constructor'

    self.coords = coords
    self.gradients = gradients
    ArrayFunc.__init__( self, args=(ufunc,coords), shape=ufunc( numpy.zeros( coords.shape ) ).shape )

  @staticmethod
  def eval( f, x ):
    'evaluate'

    return f( x )

  def localgradient( self, ndims ):
    'local gradient'

    raise NotImplementedError

  def grad( self, coords, topo ):
    'gradient'

    assert coords is self.coords # TODO check tole of topo arg
    return UFunc( self.coords, *self.gradients )

class LocalGradient( ArrayFunc ):
  'local gradient'

  def __init__( self, func, ndims, level ):
    'constructor'

    self.ndims = ndims
    self.func = func
    self.level = level
    ArrayFunc.__init__( self, args=(ELEM,POINTS,func.mapping,ndims,level), shape=func.shape+(ndims,)*level )

  def localgradient( self, ndims ):
    'local gradient'

    assert ndims == self.ndims
    return LocalGradient( self.func, ndims, self.level+1 )

  @staticmethod
  def eval( elem, points, fmap, ndims, level ):
    'evaluate'

    while elem.ndims != ndims:
      elem, transform = elem.parent
      points = transform.eval( points )
    func = fmap.get( elem )
    if func is not None:
      return func.eval( points, grad=level )
    elem, transform = elem.parent
    points = transform.eval( points )
    T = transform.transform
    func = fmap.get( elem )
    while func is None:
      elem, transform = elem.parent
      points = transform.eval( points )
      T = numpy.dot( T, transform.transform )
      func = fmap.get( elem )
    F = func.eval( points, grad=level )
    for axis in range( -level, 0 ): # assumes 1D points!
      F = util.transform( F, T, axis=axis )
    return F

  def __str__( self ):
    'string representation'

    return '%s:g%d/%dd' % ( self.func, self.level, self.ndims )

class Norm2( ArrayFunc ):
  'integration weights'

  def __init__( self, func, axis ):
    'constructor'

    if axis < 0:
      axis += len(func.shape)
    assert 0 <= axis < len(func.shape)

    self.axis = axis
    self.func = func
    shape = list( func.shape )
    shape.pop( axis )
    ArrayFunc.__init__( self, args=(func,axis-func.ndim), shape=shape )

  @staticmethod
  def eval( fval, axis ):
    'evaluate'

    return numpy.sqrt( util.contract( fval, fval, axis ) )

  def __str__( self ):
    'string representation'

    return indent( 'Norm2:%d' % self.axis, self.func )

class Cross( ArrayFunc ):
  'normal'

  def __init__( self, f1, f2, axis ):
    'contructor'

    assert f1.shape == f2.shape
    ArrayFunc.__init__( self, args=(f1,f2,-1,-1,-1,axis-f1.ndim), shape=f1.shape )

  eval = staticmethod( numpy.cross )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, func, ax1, ax2 ):
    'contructor'

    ax1, ax2 = normdim( len(func.shape), (ax1,ax2) )
    shape = list(func.shape)
    shape.pop(ax2)
    shape.pop(ax1)

    self.axes = ax1, ax2
    self.func = func
    ArrayFunc.__init__( self, args=(func,ax1-func.ndim,ax2-func.ndim), shape=shape )

  def localgradient( self, ndims ):
    'local gradient; jacobi formula'

    ax1, ax2 = self.axes
    return self * ( self.func.inv(ax1,ax2).swapaxes(ax1,ax2)[...,_] * self.func.localgradient(ndims) ).sum(ax1,ax2)

  eval = staticmethod( util.det )

  def __str__( self ):
    'string representation'

    return indent( 'Det:%d,%d' % self.axes, self.func )

class GetItem( ArrayFunc ):
  'get item'

  def __init__( self, func, shape, item ):
    'constructor'

    self.func = func
    self.item = item
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+item), shape=shape )

  eval = staticmethod( numpy.ndarray.__getitem__ )

  def localgradient( self, ndims ):
    'local gradient'

    grad = self.func.localgradient( ndims )
    if not grad:
      return ZERO

    index = self.item+(slice(None),)
    return grad[index]

  def __str__( self ):
    'string representation'

    return indent( '[%s]' % ','.join( obj2str(arg) for arg in self.item ), self.func )

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, shape=shape )

  eval = staticmethod( numpy.ndarray.__mul__ )

  def sum( self, ax1=-1, *axes ):
    'sum'

    func1, func2 = self.funcs
    return Dot( func1, func2, (ax1,)+axes )

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1[...,_] * func2.localgradient(ndims) \
         + func2[...,_] * func1.localgradient(ndims)

  def __str__( self ):
    'string representation'

    return indent( 'Mul', *self.funcs )

class Divide( ArrayFunc ):
  'divide'

  def __init__( self, func1, func2 ):
    'constructor'

    if not isinstance( func1, Evaluable ):
      func1 = StaticArray( func1 )
    if not isinstance( func2, Evaluable ):
      func2 = StaticArray( func2 )

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    func1, func2 = self.funcs
    grad1 = func1.localgradient(ndims)
    grad2 = func2.localgradient(ndims) 
    return ( grad1 - func1[...,_] * grad2 / func2[...,_] ) / func2[...,_]

  eval = staticmethod( numpy.ndarray.__div__ )

  def __str__( self ):
    'string representation'

    return indent( 'Div', *self.funcs )

class Negate( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape )

  def localgradient( self, ndims ):
    'local gradient'

    return -self.func.localgradient( ndims )

  eval = staticmethod( numpy.ndarray.__neg__ )

  def __str__( self ):
    'string representation'

    return indent( 'Neg', self.func )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, shape=shape )

  eval = staticmethod( numpy.ndarray.__add__ )

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1.localgradient(ndims) + func2.localgradient(ndims)

  def __str__( self ):
    'string representation'

    return indent( 'Add', *self.funcs )

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, shape=shape )

  def __neg__( self ):
    'negate'

    func1, func2 = self.funcs
    return func2 - func1

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1.localgradient(ndims) - func2.localgradient(ndims)

  eval = staticmethod( numpy.ndarray.__sub__ )

  def __str__( self ):
    'string representation'

    return indent( 'Sub', *self.funcs )

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, axes ):
    'constructor'

    shape, (func1,func2) = util.align_arrays( func1, func2 )
    axes = normdim( len(shape), axes )[::-1]
    shape = list(shape)
    for axis in axes:
      shape.pop( axis )

    self.func1 = func1
    self.func2 = func2
    self.axes = axes
    ArrayFunc.__init__( self, args=(func1,func2,tuple( ax-func1.ndim for ax in axes )), shape=shape )

  eval = staticmethod( util.contract )

  def localgradient( self, ndims ):
    'local gradient'

    return ( self.func1.localgradient(ndims) * self.func2[...,_] ).sum( *self.axes ) \
         + ( self.func1[...,_] * self.func2.localgradient(ndims) ).sum( *self.axes )

  def __str__( self ):
    'string representation'

    return indent( 'Dot:%s' % ','.join(str(a) for a in self.axes), self.func1, self.func2 )

class SwapAxes( ArrayFunc ):
  'swapaxes'

  def __init__( self, func, n1, n2 ):
    'constructor'

    if n1 < 0:
      n1 += len(func.shape)
    if n2 < 0:
      n2 += len(func.shape)
    shape = list( func.shape )
    shape[n1] = func.shape[n2]
    shape[n2] = func.shape[n1]
    self.func = func
    self.axes = n1, n2
    ArrayFunc.__init__( self, args=(func,n1-func.ndim,n2-func.ndim), shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    n1, n2 = self.axes
    return self.func.localgradient( ndims ).swapaxes( n1, n2 )

  eval = staticmethod( numpy.ndarray.swapaxes )

  def __str__( self ):
    'string representation'

    return indent( 'SwapAxes:%d,%d' % self.axes, self.func )

class Trace( ArrayFunc ):
  'trace'

  def __init__( self, func, n1, n2 ):
    'constructor'

    n1, n2 = normdim( len(func.shape), (n1,n2) )
    shape = list( func.shape )
    s1 = shape.pop( n2 )
    s2 = shape.pop( n1 )
    assert s1 == s2
    self.func = func
    self.func = func
    self.axes = n1, n2
    ArrayFunc.__init__( self, args=(func,0,n1-func.ndim,n2-func.ndim), shape=shape )

  eval = staticmethod( numpy.ndarray.trace )

  def localgradient( self, ndims ):
    'local gradient'

    grad = self.func.localgradient( ndims )
    return Trace( grad, *self.axes )

  def __str__( self ):
    'string representation'

    return indent( 'Trace:%d,%d' % self.axes, self.func )

# MATHEMATICAL EXPRESSIONS

class BaseFunc( ArrayFunc ):
  'unary base class'

  def __init__( self, *args ):
    'constructor'

    args = [ arg if isinstance(arg,ArrayFunc) else StaticArray(arg) for arg in args ]
    shape, self.args = util.align_arrays( *args )
    ArrayFunc.__init__( self, args=self.args, shape=shape )

  def __str__( self ):
    'string representation'

    return indent( self.__class__.__name__, *self.args )

class Exp( BaseFunc ):
  'exponent'

  eval = staticmethod( numpy.exp )

  def localgradient( self, ndims ):
    'gradient'

    return self * self.args[0].localgradient(ndims)

class Sin( BaseFunc ):
  'sine'

  eval = staticmethod( numpy.sin )

  def localgradient( self, ndims ):
    'gradient'

    return Cos(self.args[0]) * self.args[0].localgradient(ndims)
    
class Cos( BaseFunc ):
  'cosine'

  eval = staticmethod( numpy.cos )

  def localgradient( self, ndims ):
    'gradient'

    return -Sin(self.args[0]) * self.args[0].localgradient(ndims)

class Log( BaseFunc ):
  'cosine'

  eval = staticmethod( numpy.log )

  def localgradient( self, ndims ):
    'local gradient'

    f, = self.args
    return f.localgradient(ndims) / f

  def __str__( self ):
    'string representation'

    return indent( 'Log', *self.args )

class Arctan2( BaseFunc ):
  'arctan2'

  eval = staticmethod( numpy.arctan2 )

  def localgradient( self, ndims ):
    'local gradient'

    y, x = self.args
    return ( x * y.localgradient(ndims) - y * x.localgradient(ndims) ) / ( x**2 + y**2 )

  def __str__( self ):
    'string representation'

    return indent( 'Atan2', *self.args )

class Power( BaseFunc ):
  'power'

  eval = staticmethod( numpy.ndarray.__pow__ )

  def localgradient( self, ndims ):
    'local gradient'

    func, power = self.args
    return power * ( func**(power-1) )[...,_] * func.localgradient(ndims)

  def __str__( self ):
    'string representation'

    func, power = self.args
    return indent( 'Pow:%.1f' % power, func )

#class Tanh( BaseFunc ):
#  'hyperbolic tangent'
#
#  eval = staticmethod( numpy.tanh )
#
#  def localgradient( self, ndims ):
#    'gradient'
#
#    return (-Tanh( self.args[0] )**2 + 1.) * self.args[0].localgradient(ndims)

class Voigt( ArrayFunc ):
  'turn 3x3 matrix into voight vector'

  def __init__( self, func ):
    'constructor'

    assert func.shape[-2:] == (3,3)
    self.func = func
    ArrayFunc.__init__( self, args=[func], shape=func.shape[:-2]+(6,) )

  @staticmethod
  def eval( data ):
    'evaluation'

    return data.reshape( data.shape[:-2]+(9,) )[...,[0,4,8,5,2,1]]

  def __str__( self ):
    'string representation'

    return indent( 'Voigt', self.func )

def Tanh( x ):

  return 1 - 2 / ( Exp( 2 * x ) + 1 )

# def find( self, x, title=False ):
#   'find physical coordinates in all elements'

#   x = array.asarray( x )
#   assert x.shape[0] == self.topology.ndims
#   ielems = 0
#   coords = []
#   for xi, pi in zip( x, self.gridnodes ):
#     I = numpy.searchsorted( pi, xi )
#     coords.append( ( xi - pi[I-1] ) / ( pi[I] - pi[I-1] ) )
#     ielems = ielems * (len(pi)-1) + (I-1)
#   coords = numpy.array( coords )
#   elems = self.topology.structure.ravel()
#   indices = numpy.arange( len(ielems) )
#   if title:
#     progressbar = util.progressbar( n=indices.size, title=title )
#   while len( ielems ):
#     ielem = ielems[0]
#     select = ( ielems == ielem )
#     elem = elems[ ielem ]
#     xi = elem( coords[:,select] )
#     f = self( xi )
#     f.indices = indices[select]
#     yield f
#     ielems = ielems[~select]
#     coords = coords[:,~select]
#     indices = indices[~select]
#     if title:
#       progressbar.update( progressbar.n-indices.size-1 )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
