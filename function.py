from . import util, element, numpy, _

def obj2str( obj ):
  'convert object to string'

  if isinstance( obj, numpy.ndarray ):
    if obj.size < 4:
      return str(obj.tolist()).replace(' ','')
    return 'array<%s>' % 'x'.join( map( str, obj.shape ) )
  if isinstance( obj, list ):
    return '[#%d]' % len(obj)
  if isinstance( obj, tuple ):
    if len(obj) < 10:
      return '(%s)' % ','.join( obj2str(o) for o in obj )
    return '(#%d)' % len(obj)
  if isinstance( obj, dict ):
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
  if obj is None:
    return '_'
  if obj is POINTS:
    return '<points>'
  if obj is ELEM:
    return '<elem>'
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
    if self.ndim == 0 and isinstance(other,(int,float)):
      return self.view(numpy.ndarray) == other
    if not isinstance( other, numpy.ndarray ) or self.shape != other.shape:
      return False
    return ( self.view(numpy.ndarray) == other.view(numpy.ndarray) ).all()

  def __ne__( self, other ):
    'not equal'

    return not self == other

  def __nonzero__( self ):
    'nonzero'

    return bool( ( self.view( numpy.ndarray ) != 0 ).any() )

  def indices( self ):
    'get indices for numpy array'

    return ()

  def grad( self, coords ):
    'local derivative'

    return ZERO

  def sum( self, ax0=-1, *axes ):
    'sum over multiply axes'

    if self.ndim == 0:
      assert self == 0
      return ZERO

    assert not axes # for now
    return self.view( numpy.ndarray ).sum( ax0 ).view( StaticArray ) # NOTE cast does not work for scalars

  def det( self, *axes ):
    'determinant'

    return util.det( self, *axes )

  def inv( self, *axes ):
    'determinant'

    return util.inv( self, axes )

  def norm2( self, axis=-1 ):
    'norm2'

    return util.norm2( self, axis )

  def ndiag( self, (ax1,ax2), toaxis ):
    'ndiag'

    return util.indiag( self, ax1, ax2, toaxis )

  def localgradient( self, ndims ):
    'local derivative'

    return ZERO

  def concatenate( self, other, axis ):
    'concatenate'

    return numpy.concatenate( [self,other], axis ).view( self.__class__ )

ZERO = StaticArray( 0 )

def normdim( length, n ):
  'sort and make positive'

  return sorted( ni + length if ni < 0 else ni for ni in n )

class Evaluable( object ):
  'evaluable base classs'

  operations = None

  def __init__( self, args, evalf ):
    'constructor'

    self.__args = tuple(args)
    self.__evalf = evalf

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
        idx = -2
      elif arg is POINTS:
        idx = -1
      else:
        data.insert( 0, arg )
        idx = -len(data)-2
      indices[iarg] = idx
    operations.append( (self,indices) )
    return len(operations)-1

  def compile( self ):
    'compile'

    if self.operations is None:
      self.data = []
      self.operations = []
      self.recurse_index( self.data, self.operations ) # compile expressions

  def __call__( self, elem, points ):
    'evaluate'

    self.compile()
    if isinstance( points, str ):
      points = elem.eval(points)

    N = len(self.data) + 2

    values = list( self.data )
    values.append( elem )
    values.append( points )
    for op, indices in self.operations:
      args = [ values[N+i] for i in indices ]
      try:
        retval = op.__evalf( *args )
      except:
        self.printstack( values )
        raise
      values.append( retval )
    return values[-1]

  def graphviz( self, values=None ):
    'create function graph'

    import tempfile, os

    DOT = '/usr/bin/dot'
    if not os.path.isfile( DOT ) or not os.access( DOT, os.X_OK ):
      return False

    self.compile()
    if values is None:
      values = self.data + [ '<elem>', '<points>' ]

    N = len(self.data) + 2
    dofaxes = {} # labels for dofaxes

    fid, dotname = tempfile.mkstemp()
    fileobj = os.fdopen( fid, 'w' )
    print >> fileobj, 'digraph {'

    for i, (op,indices) in enumerate( self.operations ):

      if op.__evalf is numpy.ndarray.__getitem__ and len(indices) == 2 and indices[1] < 0:
        print >> fileobj, ' ', i, '[label="%d. %s",shape="box"];' % ( i, ','.join( obj2str(s) for s in values[N+indices[1]] ) )
        print >> fileobj, ' ', i, '->', indices[0]
        continue

      try:
        code = op.__evalf.func_code
        argnames = code.co_varnames[ :min(code.co_argcount,len(indices)) ]
      except:
        argnames = ()
      argnames += tuple( '%%%d' % n for n in range( len(indices) - len(argnames) ) )

      args = []
      pointers = []
      for argname, idx in zip( argnames, indices ):
        if idx >= 0:
          pointers.append(( argname, idx ))
        else:
          args.append( '%s=%s' % ( argname, obj2str(values[N+idx]) ) )

      label = '%d. %s' % ( i, op.__evalf.__name__.upper() )
      if hasattr( op, 'shape' ):
        label += ' [%s]' % ','.join( dofaxes.setdefault(ax,'$%d'%len(dofaxes)) if isinstance(ax,DofAxis) else str(ax) for ax in op.shape )

      print >> fileobj, ' ', i, '[label="%s"];' % r'\n'.join( [ label ] + args )
      for argname, idx in pointers:
        print >> fileobj, ' ', i, '->', idx, '[label="%s"];' % argname;

    print >> fileobj, '}'
    fileobj.flush()
    svgpath = util.getpath( 'dot{0:03x}.svg' )
    assert os.system( DOT + ' -Tsvg -o%s %s' % ( svgpath, dotname ) ) == 0
    return os.path.basename( svgpath )

  def printstack( self, values=None ):
    'print stack'

    self.compile()
    if values is None:
      values = self.data + ( '<elem>', '<points>' )

    N = len(self.data) + 2

    print 'call stack:'
    for i, (op,indices) in enumerate( self.operations ):
      args = [ '%%%d' % idx if idx >= 0 else obj2str(values[N+idx]) for idx in indices ]
      try:
        code = op.__evalf.func_code
        names = code.co_varnames[ :code.co_argcount ]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], n ) for n in range( len(indices) - len(names) ) )
        args = [ '%s=%s' % item for item in zip( names, args ) ]
      except:
        pass
      objstr = '%s( %s )' % ( op.__evalf.__name__, ', '.join( args ) )
      if N+i < len(values):
        objstr += ' = ' + obj2str( values[N+i] )
      elif N+i == len(values):
        objstr += ' <-----ERROR'
      print '%2d: %s' % ( i, objstr )

  def __eq__( self, other ):
    'compare'

    return self is other or ( isinstance(other,Evaluable) and self.__evalf is other.__evalf and self.__args == other.__args )

  def __str__( self ):
    'string representation'

    key = str(self.__evalf.__name__ )
    lines = []
    indent = '\n' + ' ' + ' ' * len(key)
    for it in reversed( self.__args ):
      lines.append( indent.join( obj2str(it).splitlines() ) )
      indent = '\n' + '|' + ' ' * len(key)
    indent = '\n' + '+' + '-' * (len(key)-1) + ' '
    return key + ' ' + indent.join( reversed( lines ) )

ELEM = object()
POINTS = object()

class Tuple( Evaluable ):
  'combine'

  def __init__( self, items ):
    'constructor'

    self.items = tuple(items)
    Evaluable.__init__( self, args=items, evalf=self.vartuple )

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

def Zip( *args ):
  'zip items together'

  args = map( tuple, args )
  n = len( args[0] )
  assert all( len(arg) == n for arg in args[1:] ), 'zipping items of different length'
  return Tuple( [ Tuple(v) for v in zip( *args ) ] )

#class ArrayFunc2( object ):
#  'array function'
#
#  def __init__( self, .. ):
#    pass
#
#  def __add__( self, other ):
#    'add'
#
#    if not isinstance( other, ArrayFunc2 ):
#      other = StaticArray( other )
#      if not other:
#        return self
#
#    p = _,
#    a = self  [p*(other.ndim-self.ndim)]
#    b = other [p*(self.ndim-other.ndim)]
#
#    assert all( sh1 == sh2 or sh1 == 1 or sh2 == 1 for sh1, sh2 in zip( a.shape, b.shape ) )
#
#    if isinstance( other, ArrayFunc2 ):
#      assert other.indices = self.indices
#    else:
#      assert 
#
#    data = []
#    for da, db in zip( a.data, b.data ):
#      data.append( Add(da,db) )

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

# ARRAY FUNCTIONS

class ArrayFunc( Evaluable ):
  'array function'

  __array_priority__ = 1. # fix numpy's idiotic default behaviour

  def __init__( self, evalf, args, shape ):
    'constructor'

    self.evalf = evalf
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    Evaluable.__init__( self, evalf=evalf, args=args )

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

    try:
      return self._getitem( item )
    except:
      pass

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

  def sum( self, ax0=-1, *axes ):
    'sum'

    sumaxes = []
    s = [ slice(None) ] * self.ndim
    singleton_count = 0
    for ax in normdim( self.ndim, (ax0,)+axes ):
      if self.shape[ax] == 1:
        s[ax] = 0
        singleton_count += 1
      else:
        sumaxes.append( ax - singleton_count )
    func = self[ tuple(s) ]
    if not sumaxes:
      return func

    try:
      return func._sum( sumaxes )
    except:
      pass

    print type(self)
    raise Exception, 'summation not supported yet!'

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
    assert J.shape == (cndims,ndims), 'wrong jacobian shape: got %s, expected %s' % ( J.shape, (cndims, ndims) )
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
      ndim = self.ndim + weights.ndim - 2
      return ZERO.reshape( [1]*ndim )

    # TODO restore:
    #assert int(self.shape[0]) == weights.shape[0]

    func = self[ (slice(None),) + (_,) * (weights.ndim-1) + (slice(None),) * (self.ndim-1) ]
    weights = weights[ (slice(None),) * weights.ndim + (_,) * (self.ndim-1) ]
    assert func.ndim == weights.ndim
    return ( DofIndex( weights, self.shape[0] ) * func ).sum( 0 )

  def inv( self, ax1, ax2 ):
    'inverse'

    return Inverse( self, ax1, ax2 )

  def swapaxes( self, n1, n2 ):
    'swap axes'

    n1, n2 = normdim( self.ndim, (n1,n2) )
    trans = numpy.arange( self.ndim )
    trans[n1] = n2
    trans[n2] = n1
    return Transpose( self, trans )

  def transpose( self, trans ):
    'transpose'

    assert sorted( trans ) == range( self.ndim )
    if list( trans ) == range( self.ndim ):
      return self

    return Transpose( self, trans )

  def grad( self, coords, ndims=0 ):
    'gradient'

    assert len(coords.shape) == 1
    if ndims <= 0:
      ndims += coords.shape[0]
    J = coords.localgradient( ndims )
    if J.shape[0] == J.shape[1]:
      Jinv = J.inv(0,1)
    elif J.shape[0] == J.shape[1] + 1: # gamma gradient
      Jinv = Concatenate( [ J, coords.normal()[:,_] ], axis=1 ).inv(0,1)[:-1,:]
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

    if self.shape == ():
      return self

    return Determinant( self, ax1, ax2 )

  def prod( self, axis ):
    'product'

    if self.shape[axis] == 1:
      s = [slice(None)] * self.ndim
      s[axis] = 0
      return self[tuple(s)]

    try:
      return self._prod( axis )
    except:
      pass

    return Product( self, axis )

  def __mul__( self, other ):
    'multiply'

    if not isinstance( other, ArrayFunc ):
      other = StaticArray( other )

    shape, (func1,func2) = util.align_arrays( self, other )

    if not func2:
      return StaticArray( numpy.zeros(shape) )

    if not func2 - 1 and func1.shape == shape: # TODO replace mul with repeat if possible
      return func1

    try:
      return func1._mul( func2 )
    except:
      pass

    try:
      return func2._mul( func1 )
    except:
      pass

    return Multiply( func1, func2, shape )

  def __rmul__( self, other ):
    'right multiplication'

    return self * other

  def __div__( self, other ):
    'divide'
  
    if isinstance( other, ArrayFunc ):
      return Divide( self, other )

    return self * numpy.reciprocal( other ) # faster

  def __rdiv__( self, other ):
    'right divide'

    if not isinstance( other, ArrayFunc ):
      other = StaticArray( other )

    if not other:
      return ZERO
    
    return Divide( other, self )

  def __add__( self, other ):
    'add'

    if not isinstance( other, ArrayFunc ):
      other = StaticArray( other )

    if not other:
      return self

    shape, (func1,func2) = util.align_arrays( self, other )

    try:
      return func1._add( func2 )
    except:
      pass

    try:
      return func2._add( func1 )
    except:
      pass

    return Add( self, other )

  def __radd__( self, other ):
    'right addition'

    return self + other

  def __sub__( self, other ):
    'subtract'
  
    if not isinstance( other, ArrayFunc ):
      other = StaticArray( other )

    if other == 0:
      return self

    shape, (func1,func2) = util.align_arrays( self, other )

    try:
      return func1._sub( func2 )
    except:
      pass

    return Subtract( self, other )

  def __rsub__( self, other ):
    'right subtract'

    if isinstance( other, ArrayFunc ):
      return other - self

    other = StaticArray( other )
    if not other:
      return -self

    shape, (func1,func2) = util.align_arrays( self, other )

    try:
      return func2._sub( func1 )
    except:
      pass

    return Subtract( func2, func1 )

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

    return Transpose( self, numpy.arange(self.ndim)[::-1] )

  def symmetric( self, n1, n2 ):
    'symmetric'

    return Symmetric( self, n1, n2 )

  def trace( self, n1=-2, n2=-1 ):
    'symmetric'

    return Trace( self, n1, n2 )

  def ndiag( self, axes, toaxis ):
    'ndiag'

    ax1, ax2 = normdim( self.ndim, axes )
    n1 = self.shape[ax1]
    n2 = self.shape[ax2]

    ax = ax1 if n1 == 1 else ax2
    if self.shape[ax] == 1:
      item = [ slice(None) ] * self.ndim
      item[ax] = 0
      return self[ tuple(item) ]

    return NDiag( self, axes, toaxis )

  def concatenate( self, other, axis ):
    'concatenate'

    try:
      return self._concatenate( other, axis )
    except:
      pass

    return Concatenate( [self,other], axis )

class Product( ArrayFunc ):
  'product'

  def __init__( self, func, axis ):
    'constructor'

    axis, = normdim( func.ndim, [axis] )
    shape = list(func.shape)
    shape.pop(axis)
    ArrayFunc.__init__( self, args=[func,axis-func.ndim], evalf=numpy.prod, shape=shape )

class OffsetWrapper( ArrayFunc ):
  'chain function spaces'

  def __init__( self, func, offset, ndofs ):
    'constructor'

    self.__dict__.update( func.__dict__ )
    self.func = func
    self.offset = offset
    self.shape = (func.shape[0].transform(ndofs=ndofs,shift=offset),) + func.shape[1:]

  def localgradient( self, ndims ):
    'local gradient'

    return OffsetWrapper( self.func.localgradient( ndims ), offset=self.offset, ndofs=int(self.shape[0]) )

class IWeights( ArrayFunc ):
  'integration weights'

  def __init__( self ):
    'constructor'

    ArrayFunc.__init__( self, args=[POINTS], evalf=self.iweights, shape=() )

  @staticmethod
  def iweights( points ):
    'evaluate'

    return points.weights

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
    self.__class__.__base__.__init__( self, args=[ELEM,POINTS,mapping], evalf=self.function, shape=shape )

  @staticmethod
  def function( elem, points, fmap ):
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

class Choose( ArrayFunc ):
  'piecewise function'

  def __init__( self, x, intervals, *funcs ):
    'constructor'

    shapes = [ f.shape for f in funcs if isinstance( f, ArrayFunc ) ]
    shape = shapes.pop()
    assert all( sh == shape for sh in shapes )
    assert len(intervals) == len(funcs)-1
    ArrayFunc.__init__( self, args=(x,intervals)+funcs, evalf=self.choose, shape=x.shape+shape )

  @staticmethod
  def choose( x, intervals, *choices ):
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

    ax1, ax2 = normdim( func.ndim, (ax1,ax2) )
    assert func.shape[ax1] == func.shape[ax2]
    self.func = func
    self.axes = ax1, ax2
    ArrayFunc.__init__( self, args=(func,(ax1-func.ndim,ax2-func.ndim)), evalf=util.inv, shape=func.shape )

  def localgradient( self, ndims ):
    'local gradient'

    ax1, ax2 = self.axes
    G = self.func.localgradient( ndims )
    H = ( self[...,_,_].swapaxes(ax1,-1) * G[...,_].swapaxes(ax2,-1) ).sum()
    I = ( self[...,_,_].swapaxes(ax2,-1) * H[...,_].swapaxes(ax1,-1) ).sum()
    return -I

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

  def __init__( self, stop, mapping, start=0 ):
    'new'

    self.start = start
    self.stop = stop
    self.mapping = mapping
    Evaluable.__init__( self, args=(ELEM,mapping), evalf=self.dofaxis )

  def transform( self, ndofs, shift ):
    'shift numbering and widen axis'

    mapping = dict( (elem,idx+shift) for elem, idx in self.mapping.iteritems() )
    return DofAxis( ndofs, mapping )

  @staticmethod
  def dofaxis( elem, idxmap ):
    'evaluate'

    index = idxmap.get( elem )
    while index is None:
      elem, transform = elem.parent
      index = idxmap.get( elem )
    return index

  def __int__( self ):
    'integer'

    return int(self.stop)

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

    assert other.start == 0
    mapping = self.mapping.copy()
    ndofs = self.stop - self.start
    for elem, idx2 in other.mapping.iteritems():
      idx1 = mapping.get( elem )
      mapping[ elem ] = idx2 + ndofs if idx1 is None \
                   else numpy.hstack([ idx1, idx2 + ndofs ])
    return DofAxis( start=self.start, stop=self.stop+other.stop, mapping=mapping )

  def __repr__( self ):
    'string representation'

    return 'DofAxis(%d-%d)' % ( self.start, self.stop )

class Concatenate( ArrayFunc ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    self.funcs = tuple( funcs )
    ndim = funcs[0].ndim
    assert all( func.ndim == ndim for func in funcs[1:] ), 'concatenating functions of unequal dimension'
    shape = numpy.array( funcs[0].shape )
    self.axis, = normdim( ndim, [axis] )
    for func in funcs[1:]:
      assert numpy.all( shape[:self.axis] == func.shape[:self.axis] ) and numpy.all( shape[self.axis+1:] == func.shape[self.axis+1:] ), '%s != %s' % ( shape, func.shape )
      shape[self.axis] += func.shape[self.axis]
    ArrayFunc.__init__( self, args=(self.axis-len(shape),)+self.funcs, evalf=self.concat, shape=shape )

  @staticmethod
  def concat( axis, *arrays ):
    'concatenate, allows for (impl/expl) singleton dims'

    ndim = max( arr.ndim for arr in arrays )
    shape = numpy.zeros( ndim )
    for arr in arrays:
      sh = shape[-arr.ndim:]
      numpy.maximum( sh, arr.shape, out=sh )
    shape[axis] = sum( arr.shape[axis] for arr in arrays )
    concatenated = numpy.empty(shape)
    n0 = 0
    s = [slice(None)]*ndim
    for arr in arrays:
      n1 = n0 + arr.shape[axis]
      s[axis] = slice(n0,n1)
      n0 = n1
      concatenated[tuple(s)] = arr
    assert n1 == shape[axis]
    return concatenated

  def localgradient( self, ndims ):
    'gradient'

    funcs = [ func.localgradient(ndims) for func in self.funcs ]
    return Concatenate( funcs, axis=self.axis )

  def _add( self, other ):
    'addition'

    assert isinstance( other, self.__class__ )
    assert self.axis == other.axis
    fg = zip( self.funcs, other.funcs )
    assert all( f.shape == g.shape for (f,g) in fg )
    return self.__class__( [ f+g for (f,g) in fg ], axis=self.axis )

  def _concatenate( self, other, axis ):
    'concatenate'

    assert self.ndim == other.ndim and axis == self.axis
    return Concatenate( self.funcs + (other,), axis )

  @staticmethod
  def concatenate( axis, *funcs ):
    'evaluate'

    return numpy.concatenate( funcs, axis=axis )

class Vectorize( ArrayFunc ):
  'vectorize'

  def __init__( self, funcs ):
    'constructor'

    assert all( f.ndim == funcs[0].ndim for f in funcs[1:] )
    shape = (len(funcs),) + funcs[0].shape[1:] # TODO max over all funcs
    self.funcs = funcs
    shape = ( util.sum( func.shape[0] for func in funcs ), len(funcs) ) + funcs[0].shape[1:]
    ArrayFunc.__init__( self, args=[shape[1:]]+funcs, evalf=self.vectorize, shape=shape )

  def _sum( self, axes ):
    'sum'

    assert axes == [1]
    return Concatenate( self.funcs, axis=0 )

  def _mul( self, other ):
    'multiply'

    assert other.shape[0] == 1
    funcs = []
    for i, func in enumerate(self.funcs):
      funcs.append( func * other[:,i if other.shape[1] > 1 else 0,...] )
    return Vectorize( funcs )

  @staticmethod
  def vectorize( shape, *funcs ):
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

#   if all( func == self.funcs[0] for func in self.funcs[1:] ):
#     return self.funcs[0].dot( weights.reshape( len(self.funcs), -1 ).T )

    # TODO broken for merged functions!

    n1 = 0
    conc = None
    for func in self.funcs:
      n0 = n1
      n1 += int(func.shape[0])
      f = func.dot( weights[n0:n1,_] )
      conc = f if conc is None else conc.concatenate( f, axis=0 )
    return conc

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
    ArrayFunc.__init__( self, args=(axis-len(shape),shape)+funcs, evalf=self.stack, shape=shape )

  @staticmethod
  def stack( axis, shape, *funcs ):
    'evaluate'

    stacked = numpy.empty( funcs[0].shape[:funcs[0].ndim+1-len(shape)] + shape )
    for array, f in zip( util.ndiag( stacked, [axis] ), funcs ):
      array[:] = f
    return stacked

  def localgradient( self, ndims ):
    'local gradient'

    grads = [ f.localgradient( ndims ) for f in self.funcs ]
    return Stack( grads, self.axis )

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

class UFunc( ArrayFunc ):
  'user function'

  def __init__( self, coords, ufunc, *gradients ):
    'constructor'

    self.coords = coords
    self.gradients = gradients
    ArrayFunc.__init__( self, args=(ufunc,coords), evalf=self.ufunc, shape=ufunc( numpy.zeros( coords.shape ) ).shape )

  @staticmethod
  def ufunc( f, x ):
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
    ArrayFunc.__init__( self, args=(ELEM,POINTS,func.mapping,ndims,level), evalf=self.lgrad, shape=func.shape+(ndims,)*level )

  def localgradient( self, ndims ):
    'local gradient'

    assert ndims == self.ndims
    return LocalGradient( self.func, ndims, self.level+1 )

  @staticmethod
  def lgrad( elem, points, fmap, ndims, level ):
    'evaluate'

    assert elem.ndims <= ndims
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
    for axis in range( -level, 0 ):
      F = util.transform( F, T, axis=axis )
    return F

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
    ArrayFunc.__init__( self, args=(func,axis-func.ndim), evalf=util.norm2, shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    return ( self.func[:,_] * self.func.localgradient( ndims ) ).sum( -2 ) / self

class Cross( ArrayFunc ):
  'normal'

  def __init__( self, f1, f2, axis ):
    'contructor'

    assert f1.shape == f2.shape
    ArrayFunc.__init__( self, args=(f1,f2,-1,-1,-1,axis-f1.ndim), evalf=numpy.cross, shape=f1.shape )

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
    ArrayFunc.__init__( self, args=(func,ax1-func.ndim,ax2-func.ndim), evalf=util.det, shape=shape )

  def localgradient( self, ndims ):
    'local gradient; jacobi formula'

    ax1, ax2 = self.axes
    return self * ( self.func.inv(ax1,ax2).swapaxes(ax1,ax2)[...,_] * self.func.localgradient(ndims) ).sum(ax1,ax2)

class DofIndex( ArrayFunc ):
  '''element-based indexing
  Separate from GetItem because numpy does not allow mixing of newaxis and array indexing'''

  def __init__( self, array, dofaxis ):
    'constructor'

    array = array[:int(dofaxis)] # TODO make strict
    shape = (dofaxis,) + array.shape[1:]
    self.array = array
    self.dofaxis = dofaxis
    ArrayFunc.__init__( self, args=(array,dofaxis), evalf=numpy.ndarray.__getitem__, shape=shape )

  def _add( self, other ):
    'add'

    assert isinstance( other, self.__class__ )
    assert self.dofaxis == other.dofaxis
    return self.__class__( self.array + other.array, self.dofaxis )

  def _sub( self, other ):
    'add'

    assert isinstance( other, self.__class__ )
    assert self.dofaxis == other.dofaxis
    return self.__class__( self.array - other.array, self.dofaxis )

  def _concatenate( self, other, axis ):
    'concatenate'

    assert axis != 0
    assert isinstance( other, self.__class__ )
    assert self.dofaxis == other.dofaxis
    array = self.array.concatenate( other.array, axis )
    return DofIndex( array, self.dofaxis )

  def localgradient( self, ndim ):
    'local gradient'

    return ZERO

class GetItem( ArrayFunc ):
  'get item'

  def __init__( self, func, shape, item ):
    'constructor'

    self.func = func
    self.item = item
    ArrayFunc.__init__( self, args=(func,(Ellipsis,)+item), evalf=numpy.ndarray.__getitem__, shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    grad = self.func.localgradient( ndims )
    if not grad:
      return ZERO

    index = self.item+(slice(None),)
    return grad[index]

  def _getitem( self, item ):
    'get item'

    myitem = iter(self.item)
    myshape = iter(self.shape)
    merged = []
    for it in item:
      if it == _:
        merged.append( _ )
        continue
      myit = myitem.next()
      while isinstance(myit,int):
        merged.append( myit )
        myit = myitem.next()
      mysh = myshape.next()
      if it == slice(None):
        merged.append( myit )
      elif isinstance(it,int):
        assert 0 <= it < mysh
        if myit == _:
          pass
        else:
          assert myit == slice(None) # TODO lift restriction
          merged.append( it )
      else:
        raise NotImplementedError
    merged.extend( myitem )
    return self.func[ tuple(merged) ]

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2, shape ):
    'constructor'

    assert func1.ndim == func2.ndim == len(shape)
    self.funcs = func1, func2
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.multiply, shape=shape )

  def _sum( self, axes ):
    'sum'

    func1, func2 = self.funcs
    dotaxes = []
    shift = 0
    for ax in axes:
      myax = ax - shift
      if func1.shape[myax] == 1 or func2.shape[myax] == 1:
        func1 = func1.sum(ax)
        func2 = func2.sum(ax)
        shift += 1
      else:
        dotaxes.append( myax )
    return Dot( func1, func2, dotaxes ) if dotaxes else func1 * func2

  def _getitem( self, item ):
    'get item'

    func1, func2 = self.funcs
    return func1[item] * func2[item]

  def det( self, ax1, ax2 ):
    'determinant'

    if self.funcs[0].shape == ():
      return self.funcs[1].det( ax1, ax2 ) * self.funcs[0]

    if self.funcs[1].shape == ():
      return self.funcs[0].det( ax1, ax2 ) * self.funcs[1]

    return self.__class__.__base__.det( self, ax1, ax2 )

  def _prod( self, axis ):
    'product'

    func1, func2 = self.funcs
    n = self.shape[axis]
    s = [ slice(None) ] * self.ndim
    s[axis] = 0
    return ( func1[tuple(s)]**n if func1.shape[axis] == 1 else func1.prod(axis) ) \
         * ( func2[tuple(s)]**n if func2.shape[axis] == 1 else func2.prod(axis) )

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1[...,_] * func2.localgradient(ndims) \
         + func2[...,_] * func1.localgradient(ndims)

class Divide( ArrayFunc ):
  'divide'

  def __init__( self, func1, func2 ):
    'constructor'

    if not isinstance( func1, ArrayFunc ):
      func1 = StaticArray( func1 )
    if not isinstance( func2, ArrayFunc ):
      func2 = StaticArray( func2 )

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.divide, shape=shape )

  def _getitem( self, item ):
    'get item'

    func1, func2 = self.funcs
    return func1[item] / func2[item]

# def _mul( self, other ):
#   'multiply'

#   func1, func2 = self.funcs
#   print 'TRANSFORM'
#   print self
#   print '*'
#   print other
#   print 'TO'
#   print ( func1 * other ) / func2
#   return ( func1 * other ) / func2

  def localgradient( self, ndims ):
    'local gradient'

    func1, func2 = self.funcs
    grad1 = func1.localgradient(ndims)
    grad2 = func2.localgradient(ndims) 
    return ( grad1 - func1[...,_] * grad2 / func2[...,_] ) / func2[...,_]

class Negate( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.negative, shape=func.shape )

  def _add( self, other ):
    'addition'

    return other - self.func

  def _sub( self, other ):

    return -( other + self.func )

  def localgradient( self, ndims ):
    'local gradient'

    return -self.func.localgradient( ndims )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.add, shape=shape )

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1.localgradient(ndims) + func2.localgradient(ndims)

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, self.funcs = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=self.funcs, evalf=numpy.subtract, shape=shape )

  def __neg__( self ):
    'negate'

    func1, func2 = self.funcs
    return func2 - func1

  def localgradient( self, ndims ):
    'gradient'

    func1, func2 = self.funcs
    return func1.localgradient(ndims) - func2.localgradient(ndims)

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
    ArrayFunc.__init__( self, args=(func1,func2,tuple( ax-func1.ndim for ax in axes )), evalf=util.contract, shape=shape )
#
#  def _add( self, other ):
#    'addition'
#
#    assert isinstance( other, self.__class__ ) and self.axes == other.axes
#    if self.func1 == other.func1:
#      return Dot( self.func1, self.func2 + other.func2, self.axes )
#    if self.func1 == other.func2:
#      return Dot( self.func1, self.func2 + other.func1, self.axes )
#    if self.func2 == other.func1:
#      return Dot( self.func2, self.func1 + other.func2, self.axes )
#    if self.func2 == other.func2:
#      return Dot( self.func2, self.func1 + other.func1, self.axes )
#    raise Exception
#
#  def _sub( self, other ):
#    'addition'
#
#    assert isinstance( other, self.__class__ ) and self.axes == other.axes
#    if self.func1 == other.func1:
#      return Dot( self.func1, self.func2 - other.func2, self.axes )
#    if self.func1 == other.func2:
#      return Dot( self.func1, self.func2 - other.func1, self.axes )
#    if self.func2 == other.func1:
#      return Dot( self.func2, self.func1 - other.func2, self.axes )
#    if self.func2 == other.func2:
#      return Dot( self.func2, self.func1 - other.func1, self.axes )
#    raise Exception
#
  def _concatenate( self, other, axis ):
    'concatenate'

    assert isinstance( other, self.__class__ )
    for ax in self.axes:
      if ax <= axis:
        axis += 1
    if self.func1 == other.func1 and self.func1.shape[axis] == 1:
      return ( self.func1 * self.func2.concatenate( other.func2, axis ) ).sum( *self.axes )
    if self.func1 == other.func2 and self.func1.shape[axis] == 1:
      return ( self.func1 * self.func2.concatenate( other.func1, axis ) ).sum( *self.axes )
    if self.func2 == other.func1 and self.func2.shape[axis] == 1:
      return ( self.func2 * self.func1.concatenate( other.func2, axis ) ).sum( *self.axes )
    if self.func2 == other.func2 and self.func2.shape[axis] == 1:
      return ( self.func2 * self.func1.concatenate( other.func1, axis ) ).sum( *self.axes )
    raise Exception

  def localgradient( self, ndims ):
    'local gradient'

    return ( self.func1.localgradient(ndims) * self.func2[...,_] ).sum( *self.axes ) \
         + ( self.func1[...,_] * self.func2.localgradient(ndims) ).sum( *self.axes )

class Transpose( ArrayFunc ):
  'swapaxes'

  def __init__( self, func, trans ):
    'constructor'

    assert all( 0 <= t < func.ndim for t in trans )
    self.trans = StaticArray( trans )
    shape = numpy.array( func.shape )[ self.trans ]
    self.func = func
    ArrayFunc.__init__( self, args=(func,self.trans), evalf=self.transpose, shape=shape )

  @staticmethod
  def transpose( array, trans ):
    'transpose'

    n = array.ndim - len(trans)
    T = numpy.arange( array.ndim )
    numpy.add( n, trans, out=T[n:] )
    return numpy.transpose( array, T )

  def _getitem( self, item ):
    'get item'

    newaxes = []
    myitem = []
    n = 0
    for it in item:
      if it == _:
        newaxes.append(n)
      else:
        myitem.append( it )
      if not isinstance( it, int ):
        n += 1

    assert len(myitem) == self.ndim
    origitem = numpy.empty( self.ndim, dtype=object )
    origitem[ self.trans ] = myitem
    trans = numpy.argsort( [ t for it, t in zip(origitem,self.trans) if not isinstance(it,int) ] )

    assert not newaxes # TODO determine optimal place to insert newaxes

    return self.func[ tuple(origitem) ].transpose( trans )

  def localgradient( self, ndims ):
    'local gradient'

    T = numpy.arange( self.ndim+1 )
    T[:-1] = self.transpose
    return self.func.localgradient( ndims ).transpose( T )

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
    ArrayFunc.__init__( self, args=(func,0,n1-func.ndim,n2-func.ndim), evalf=numpy.trace, shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    grad = self.func.localgradient( ndims )
    return Trace( grad, *self.axes )

# MATHEMATICAL EXPRESSIONS

class Exp( ArrayFunc ):
  'exponent'

  def __init__( self, func ):
    'constructor'

    assert isinstance( func, ArrayFunc )
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.exp, shape=func.shape )

  def localgradient( self, ndims ):
    'gradient'

    return self * self.func.localgradient(ndims)

class Sin( ArrayFunc ):
  'sine'

  def __init__( self, func ):
    'constructor'

    assert isinstance( func, ArrayFunc )
    self.arg = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.sin, shape=func.shape )

  def localgradient( self, ndims ):
    'gradient'

    return Cos(self.arg) * self.arg.localgradient(ndims)
    
class Cos( ArrayFunc ):
  'cosine'

  def __init__( self, func ):
    'constructor'

    assert isinstance( func, ArrayFunc )
    self.arg = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.cos, shape=func.shape )

  def localgradient( self, ndims ):
    'gradient'

    return -Sin(self.arg) * self.arg.localgradient(ndims)

class Log( ArrayFunc ):
  'cosine'

  def __init__( self, func ):
    'constructor'

    assert isinstance( func, ArrayFunc )
    self.func = func
    ArrayFunc.__init__( self, args=[func], evalf=numpy.log, shape=func.shape )

  def localgradient( self, ndims ):
    'local gradient'

    return self.func.localgradient(ndims) / self.func

class Arctan2( ArrayFunc ):
  'arctan2'

  def __init__( self, numer, denom ):
    'constructor'

    shape, args = util.align_arrays( numer, denom )
    self.args = numer, denom
    ArrayFunc.__init__( self, args=args, evalf=numpy.arctan2, shape=shape )

  def localgradient( self, ndims ):
    'local gradient'

    y, x = self.args
    return ( x * y.localgradient(ndims) - y * x.localgradient(ndims) ) / ( x**2 + y**2 )

class Power( ArrayFunc ):
  'power'

  def __init__( self, func, power ):
    'constructor'

    assert isinstance( func, ArrayFunc )
    assert isinstance( power, (int,float,numpy.ndarray) )
    self.func = func
    self.power = power
    ArrayFunc.__init__( self, args=[func,power], evalf=numpy.power, shape=func.shape )

  def localgradient( self, ndims ):
    'local gradient'

    return self.power * ( self.func**(self.power-1) )[...,_] * self.func.localgradient(ndims)

class Min( ArrayFunc ):
  'minimum'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, args = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=args, evalf=numpy.minimum, shape=shape )

class Max( ArrayFunc ):
  'maximum'

  def __init__( self, func1, func2 ):
    'constructor'

    shape, args = util.align_arrays( func1, func2 )
    ArrayFunc.__init__( self, args=args, evalf=numpy.maximum, shape=shape )

class NDiag( ArrayFunc ):
  'extract diagonal'

  def __init__( self, func, axes, toaxis ):
    'constructor'

    ax1, ax2 = normdim( func.ndim, axes )
    shape = list( func.shape )
    n = shape.pop(ax2)
    assert n == shape.pop(ax1)
    shape.insert( toaxis, n )
    ArrayFunc.__init__( self, args=[func,ax1-func.ndim,ax2-func.ndim,toaxis-(func.ndim-1)], evalf=util.indiag, shape=shape )

class Diagonalize( ArrayFunc ):
  'diagonal matrix'

  def __init__( self, func, axis, toaxes ):
    'constructor'

    self.axis = axis if axis >= 0 else axis + func.ndim
    assert 0 <= self.axis < func.ndim
    ax1, ax2 = self.toaxes = normdim( func.ndim+1, toaxes )
    assert ax2 > ax1 # strict
    shape = list( func.shape )
    n = shape.pop( self.axis )
    shape = shape[:ax1] + [n] + shape[ax1:ax2-1] + [n] + shape[ax2-1:] # shape[ax1] = shape[ax2] = func.shape[axis]
    self.func = func
    self.__class__.__base__.__init__( self, args=[func,self.axis-func.ndim,ax1-(func.ndim+1),ax2-(func.ndim+1)], evalf=self.diagonalize, shape=shape )

  @staticmethod
  def diagonalize( data, axis, ax1, ax2 ):
    'evaluate'

    shape = list(data.shape)
    n = shape.pop(axis)
    shape.insert( ax2+len(shape)+1, n )
    shape.insert( ax1+len(shape)+1, n )
    assert shape[ax1] == shape[ax2] == n
    diagonalized = numpy.zeros( shape )
    util.indiag( diagonalized, ax1, ax2, axis )[:] = data
    return diagonalized

  def _getitem( self, item ):
    'get item'

    # TODO requires testing!
    assert all( it in [ slice(None), _ ] for it in item )
    func = self.func
    toaxes = list( self.toaxes )
    axis = self.axis
    while _ in item:
      i = item.index(_)
      item = item[:i] + ( slice(None), ) + item[i+1:]
      if i > toaxes[1]:
        i -= 1
      else:
        toaxes[1] += 1
      if i > toaxes[0]:
        i -= 1
      else:
        toaxes[0] += 1
      if i <= axis:
        axis += 1
      s = [ slice(None) ] * (func.ndim+1)
      s[i] = _
      func = func[ tuple(s) ]
    return Diagonalize( func, axis, toaxes )

  def inv( self, *axes ):
    'inverse'

    if normdim( self.ndim, axes ) == self.toaxes:
      return Diagonalize( 1./self.func, self.axis, self.toaxes )

    return self.__class__.__base__.inv( self, *axes )

  def det( self, *axes ):
    'determinant'

    axes = normdim( self.ndim, axes )
    if axes == self.toaxes:
      return self.func.prod( self.axis )

    return self.__class__.__base__.det( self, *axes )

  def _mul( self, other ):
    'multiply'
 
    if other.ndim:
      ax1, ax2 = self.toaxes
      other = other.ndiag( (ax1-self.ndim,ax2-self.ndim), self.axis-(self.ndim-1) )
    return Diagonalize( self.func * other, self.axis, self.toaxes )

  def _sum( self, axes ):
    'sum'

    for ax in axes:
      if ax in self.toaxes:
        break
    else:
      raise Exception

    trans = numpy.arange( self.func.ndim )
    otherax = self.toaxes[0] if ax == self.toaxes[1] else self.toaxes[1]-1
    if otherax < self.axis:
      trans[otherax+1:self.axis+1] = numpy.arange(otherax,self.axis)
    elif otherax > self.axis:
      trans[self.axis:otherax] = numpy.arange(self.axis+1,otherax+1)
    trans[otherax] = self.axis
    func = self.func.transpose( trans )
    if len( axes ) > 1:
      iax = axes.index( ax )
      axes = axes[iax] + axes[iax+1:]
      func = func.sum( *axes )
    return func


#class Tanh( ArrayFunc ):
#  'hyperbolic tangent'
#
#  eval = staticmethod( numpy.tanh )
#
#  def localgradient( self, ndims ):
#    'gradient'
#
#    return (-Tanh( self.args[0] )**2 + 1.) * self.args[0].localgradient(ndims)

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
