from . import util, element, numpy, _

class Zero( int ):
  'zero'

  shape = ()

  def __new__( cls ):
    'constructor'

    return int.__new__( cls, 0 )

  def __getitem__( self, item ):
    'getitem'

    return self

  def sum( self, *axes ):
    'sum'

    return self

  def grad( self, coords, ndims=0 ):
    'gradient'

    return self

  def __sub__( self, other ):
    'subtract'

    return -other

ZERO = Zero()

def unite( *funcs ):
  'create unified consecutive numbering'

  ndofs = sum( f.shape[0].ndofs for f in funcs )
  n = 0
  for f in funcs:
    newf = util.clone( f )
    newf.shape = ( f.shape[0].transform(ndofs=ndofs,shift=n), ) + f.shape[1:]
    yield newf
    n += f.shape[0].ndofs

def align_shapes( *funcs ):
  'align shapes'

  funcs = map( as_evaluable, funcs )
  shape = []
  for f in funcs:
    d = len(shape) - len(f.shape)
    if d < 0:
      shape = list(f.shape[:-d]) + shape
      d = 0
    for i, sh in enumerate( f.shape ):
      if shape[d+i] is nulaxis:
        shape[d+i] = sh
      else:
        assert sh == shape[d+i] or sh is nulaxis
  return shape

def is_zero( obj ):
  'check if equals zero'

  if obj is ZERO:
    return True

  if isinstance( obj, numpy.ndarray ):
    return ( obj == 0 ).all()

  return obj == 0

def is_unit( obj ):
  'check if equals one'

  if isinstance( obj, numpy.ndarray ):
    return obj.ndim == 0 and obj == 1

  return obj == 1

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
  
def as_evaluable( f ):
  'convert to evaluable'

  if isinstance( f, Evaluable ):
    return f

  if isinstance( f, (int,float) ):
    return Scalar( f )

  if isinstance( f, numpy.ndarray ):
    if not f.shape:
      return Scalar( f )
    return StaticArray( f )

  raise Exception, 'not sure how to convert %r to evaluable' % f

class StackIndex( int ):
  'stack index'

  def __str__( self ):
    'string representation'

    return '%%%d' % self

class NulAxis( int ):
  'nul axis'

  __new__ = lambda cls: int.__new__( cls, 1 )

nulaxis = NulAxis()

class Evaluable( object ):
  'evaluable base classs'

  operations = None
  needxi = False

  def recurse_index( self, operations ):
    'compile'

    for i, (op,idx) in enumerate( operations ):
      if op == self:
        return StackIndex( i )

    indices = [ arg.recurse_index( operations ) if isinstance( arg, Evaluable ) else arg for arg in self.args ]
    operations.append(( self, indices ))
    return StackIndex( len(operations)-1 )

  def compile( self ):
    'compile'

    if self.operations is None:
      self.operations = []
      self.recurse_index( self.operations ) # compile expressions
      
  def __call__( self, xi ):
    'evaluate'

    self.compile()
    values = []
    try:
      for op, arglist in self.operations:
        args = [ values[arg] if isinstance( arg, StackIndex ) else arg for arg in arglist ]
        values.append( op.eval( xi, *args ) if op.needxi else op.eval( *args ) )
    except:
      self.printstack( pointer=(op,arglist), values=values )
      raise
    #self.printstack( values=values ) # DEBUG
    #raw_input('press enter to continue')
    return values[-1]

  def printstack( self, pointer=None, values=None ):
    'print stack'

    self.compile()
    print 'call stack:'
    for i, (op,arglist) in enumerate( self.operations ):
      try:
        code = op.eval.func_code
        names = code.co_varnames[ :code.co_argcount ]
        if op.needxi:
          names = names[1:]
        names += tuple( '%s[%d]' % ( code.co_varnames[ code.co_argcount ], i ) for i in range( len(arglist) - len(names) ) )
      except:
        args = [ util.obj2str(arg) for arg in arglist ]
      else:
        args = [ '%s=%s' % ( name, util.obj2str(arg) ) for name, arg in zip( names, arglist ) ]
      shape = ' = ' + util.obj2str( values[i] ) if values and len( values ) > i else ''
      arrow = ' <-----ERROR' if pointer and pointer[0] is op and pointer[1] is arglist else ''
      print '%2d: %s( %s )%s%s' % ( i, op.__class__.__name__, ', '.join( args ), shape, arrow )

  def __eq__( self, other ):
    'compare'

    return self is other or ( self.__class__ == other.__class__ and self.args == other.args )

class Tuple( Evaluable ):
  'combine'

  def __init__( self, args ):
    'constructor'

    self.args = tuple( args )

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

  def __getitem__( self, item ):
    'get item'
  
    if not isinstance( item, tuple ):
      item = ( item, )
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
        shape.append( nulaxis )
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

    return ( self[i] for i in range(self.shape[0]) )

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
      return Scalar( 1 ) # TODO fix direction!!!!
    else:
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % ( self.shape[0], ndims )
    return normal.normalized(0)

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
    'dot'

    if is_zero(weights):
      return ZERO

    return StaticDot( self, weights )

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
  
    if is_zero(other):
      return ZERO

    if is_unit(other):
      return self

    if not isinstance( other, Evaluable ):
      other = numpy.asarray( other )
      other = StaticArray( other ) if other.ndim else Scalar( other )

    return Multiply( self, other )

  def __div__( self, other ):
    'multiply'
  
    assert not is_zero(other)

    if is_unit(other):
      return self

    if isinstance( other, (int,float) ) or isinstance( other, numpy.ndarray ) and other.ndim == 0:
      return self * float(1./other)

    if not isinstance( other, Evaluable ):
      other = numpy.asarray( other )
      if other.ndim:
        other = StaticArray( other )

    return Divide( self, other )

  def __add__( self, other ):
    'add'

    if is_zero(other):
      return self

    if not isinstance( other, Evaluable ):
      other = numpy.asarray( other )
      other = StaticArray( other ) if other.ndim else Scalar( other )

    return Add( self, other )

  __rmul__ = __mul__
  __radd__ = __add__

  def __sub__( self, other ):
    'subtract'
  
    if other == 0:
      return self

    if not isinstance( other, Evaluable ):
      other = numpy.asarray( other )
      other = StaticArray( other ) if other.ndim else Scalar( other )

    return Subtract( self, other )

  def __neg__( self ):
    'negate'

    return Negate( self )

  def __pow__( self, n ):
    'power'

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

class Orientation( ArrayFunc ):
  'point orientation'

  needxi = True
  shape = ()
  args = ()

  def __init__( self ):
    'constructor'

    pass

  @staticmethod
  def eval( xi ):
    'evaluate'

    # VERY TEMPORARY
    elem, transform = xi.elem.parent
    return ( transform.offset > .5 ) * 2 - 1

class StaticDot( ArrayFunc ):
  'dot with static array'

  def __init__( self, func, array ):
    'constructor'

    array = util.UsableArray( array )
    dofaxis = func.shape[0]
    assert isinstance( dofaxis, DofAxis )
    assert int(dofaxis) == array.shape[0]
    shape = array.shape[1:] + func.shape[1:]

    self.func = func
    self.array = array
    self.shape = tuple(shape)
    self.args = func, array, dofaxis

  @staticmethod
  def eval( func, array, I ):
    'evaluate'

    return numpy.tensordot( array[I], func, (0,0) )

  def localgradient( self, ndims ):
    'local gradient'

    return StaticDot( self.func.localgradient(ndims), self.array )

  def __mul__( self, other ):
    'multiply'

    if is_zero(other):
      return ZERO

    if is_unit(other):
      return self

    if isinstance( other, (int,float) ):
      return StaticDot( self.func, self.array * other )

    return ArrayFunc.__mul__( self, other )

  def __add__( self, other ):
    'add'

    if isinstance( other, StaticDot ) and other.func == self.func:
      return StaticDot( self.func, self.array + other.array )

    return ArrayFunc.__add__( self, other )

  def __getitem__( self, item ):
    'get item'

    if isinstance( item, int ):
      item = item,
    for i in range( self.array.ndim-1 ):
      if isinstance( item[i], int ):
        item = list(item)
        index = item.pop( i )
        array = self.array[ (slice(None),)*(i+1) + (index,) ]
        return StaticDot( self.func, array )[ tuple(item) ]

    if all( it == slice(None) for it in item ):
      return self

    return ArrayFunc.__getitem__( self, item )

  def __str__( self ):
    'string representation'

    #return '%s%%[#%s]' % ( self.func, 'x'.join( str(d) for d in self.array.shape ) )
    return indent( 'StaticDot(%s)' % 'x'.join( str(d) for d in self.array.shape ), self.func )

class Const( float ): # for select
  def eval( self, points ):
    return self

def const( *args ):
  'select by topology'

  assert len(args)%2 == 0
  mapping = {}
  for topo, val in zip( args[::2], args[1::2] ):
    mapping.update( dict.fromkeys( topo, Const(val) ) )
  return Function( shape=(), mapping=mapping )

class Function( ArrayFunc ):
  'function'

  needxi = True

  def __init__( self, shape, mapping ):
    'constructor'

    self.shape = shape
    self.mapping = mapping
    self.args = mapping,

  @staticmethod
  def eval( xi, fmap ):
    'evaluate'

    while xi.elem not in fmap:
      xi = xi.next
    return fmap[ xi.elem ].eval( xi.points )

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
    self.shape = x.shape + shape
    self.args = (x,intervals) + funcs

  @staticmethod
  def eval( x, intervals, *choices ):
    'evaluate'

    which = 0
    for i in intervals:
      which += ( x > i ).astype( int )
    return numpy.choose( which, choices )

class PieceWise( ArrayFunc ):
  'differentiate by topology'

  needxi = True

  def __init__( self, *func_and_topo ):
    'constructor'
    
    assert func_and_topo and len(func_and_topo) % 2 == 0
    fmap = {}
    args = ()
    shape = ()
    for topo, func in reversed( zip( func_and_topo[::2], func_and_topo[1::2] ) ):
      if not isinstance( func, ArrayFunc ):
        assert isinstance( func, (numpy.ndarray,int,float,list,tuple) )
        func = StaticArray( func )
      n = len(shape) - len(func.shape)
      if n < 0:
        assert shape == func.shape[-n:]
        shape = func.shape
      else:
        assert func.shape == shape[n:]
      n = len(args)
      args += func.args
      s = slice(n,len(args))
      fmap.update( dict.fromkeys( topo, (func,s) ) )
    self.args = (fmap,)+args
    self.shape = shape

  @staticmethod
  def eval( xi, fmap, *args ):
    'evaluate'

    while xi.elem not in fmap:
      xi = xi.next
    func, s = fmap[ xi.elem ]
    return func.eval( xi, *args[s] ) if func.needxi else func.eval( *args[s] )

class Inverse( ArrayFunc ):
  'inverse'

  def __init__( self, func, ax1, ax2 ):
    'constructor'

    ax1, ax2 = normdim( len(func.shape), (ax1,ax2) )
    assert func.shape[ax1] == func.shape[ax2]
    self.args = func, (ax1,ax2)
    self.shape = func.shape

  def localgradient( self, ndims ):
    'local gradient'

    func, (ax1,ax2) = self.args
    G = func.localgradient( ndims )
    H = ( self[...,_,_].swapaxes(ax1,-1) * G[...,_].swapaxes(ax2,-1) ).sum()
    I = ( self[...,_,_].swapaxes(ax2,-1) * H[...,_].swapaxes(ax1,-1) ).sum()
    return -I

  eval = staticmethod( util.inv )

  def __str__( self ):
    'string representation'

    f, axes = self.args
    return indent( 'Inv:%d,%d' % axes, f )

class DofAxis( ArrayFunc ):
  'dof axis'

  needxi = True

  def __init__( self, ndofs, mapping ):
    'new'

    self.ndofs = ndofs
    self.mapping = mapping
    self.get = mapping.get
    self.args = mapping,
    self.shape = ndofs,

  def transform( self, ndofs, shift ):
    'shift numbering and widen axis'

    mapping = dict( (elem,idx+shift) for elem, idx in self.mapping.iteritems() )
    return DofAxis( ndofs, mapping )

  @staticmethod
  def eval( xi, idxmap ):
    'evaluate'

    index = idxmap.get( xi.elem )
    while index is None:
      xi = xi.next
      index = idxmap.get( xi.elem )
    return index

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

    mapping = self.mapping.copy()
    for elem, idx2 in other.mapping.iteritems():
      idx1 = mapping.get( elem )
      mapping[ elem ] = idx2 + self.ndofs if idx1 is None \
                   else numpy.hstack([ idx1, idx2 + self.ndofs ])
    return DofAxis( self.ndofs + other.ndofs, mapping )

#   other_mapping = other.mapping.copy()
#   try:
#     mapping = dict( ( elem, numpy.hstack([ idx, other_mapping.pop(elem) + self.ndofs ]) )
#                       for elem, idx in self.mapping.iteritems() )
#   except KeyError, e:
#     raise Exception, 'element not in other: %s' % e.args[0]
#   if other_mapping:
#     raise Exception, 'element not in self: %s' % other_mapping.popitem()[0]

#   return DofAxis( self.ndofs + other.ndofs, mapping )

  __radd__ = __add__

  def __int__( self ):
    'int'

    return self.ndofs

  def __repr__( self ):
    'string representation'

    return 'DofAxis(%d)' % self

class Concatenate( ArrayFunc ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    self.args = (axis,) + tuple(funcs)
    self.shape = ( sum( func.shape[0] for func in funcs ), ) + funcs[0].shape[1:]

  def localgradient( self, ndims ):
    'gradient'

    funcs = [ func.localgradient(ndims) for func in self.args[1:] ]
    return Concatenate( funcs, axis=self.args[0] )

  @staticmethod
  def eval( axis, *funcs ):
    'evaluate'

    return numpy.concatenate( funcs, axis=axis )

class Vectorize( ArrayFunc ):
  'vectorize'

  def __init__( self, funcs, shape=False ):
    'constructor'

    self.args = tuple( funcs )
    self.shape = shape or ( sum( func.shape[0] for func in funcs ), len(funcs) ) + funcs[0].shape[1:]

  @staticmethod
  def eval( *funcs ):
    'evaluate'

    N = sum( func.shape[0] for func in funcs )
    shape = ( N, len(funcs) ) + funcs[0].shape[1:]
    data = numpy.zeros( shape )
    count = 0
    for i, func in enumerate( funcs ):
      n = func.shape[0]
      data[count:count+n,i] = func
      count += n
    assert count == N
    return data

  def localgradient( self, ndims ):
    'gradient'

    return Vectorize([ func.localgradient(ndims) for func in self.args ], shape=self.shape+(ndims,))

  def trace( self, n1, n2 ):
    'trace'

    n1, n2 = normdim( len(self.shape), (n1,n2) )
    assert self.shape[n1] == self.shape[n2]
    if n1 == 1 and n2 == 2:
      trace = Concatenate([ func[:,idim] for idim, func in enumerate( self.args ) ])
    else:
      trace = Trace( self, n1, n2 )
    return trace

  def dot( self, weights ):
    'dot'

    if all( func == self.args[0] for func in self.args[1:] ):
      return self.args[0].dot( weights.reshape( len(self.args), -1 ).T )

    # TODO broken for merged functions!

    n1 = 0
    funcs = []
    for func in self.args:
      n0 = n1
      n1 += int(func.shape[0])
      funcs.append( func.dot( weights[n0:n1,_] ) )
    return Concatenate( funcs )

class Stack( ArrayFunc ):
  'stack functions'

  needxi = True

  def __init__( self, funcs, axis=-1 ):
    'constructor'

    funcs = map( as_evaluable, funcs )
    shape = align_shapes( *funcs )
    if axis < 0:
      axis += len(shape) + 1
    assert 0 <= axis < len(shape)+1

    self.funcs = funcs
    self.axis = axis
    self.shape = tuple( shape[:axis] + [len(funcs)] + shape[axis:] )
    self.args = ( axis, self.shape ) + tuple(funcs)

  @staticmethod
  def eval( xi, axis, shape, *funcs ):
    'evaluate'

    n = xi.points.npoints # TODO generalize to shape
    stacked = numpy.empty( shape + (n,) )
    #print stacked.shape, axis
    for array, f in zip( util.ndiag( stacked, [axis] ), funcs ):
      array[:] = f
    return stacked

  def localgradient( self, ndims ):
    'local gradient'

    grads = [ f.localgradient( ndims ) for f in self.funcs ]
    return Stack( grads, self.axis )

  def __str__( self ):
    'string representation'

    #return 'Stack(%s,axis=%d)' % ( ','.join( str(f) for f in self.funcs ), self.axis )

    return indent( 'Stack:%d' % self.axis, *self.funcs )

class Stack22( ArrayFunc ):
  'special 2x2 stack based on unified axes'

  def __init__( self, *funcs ):
    'constructor'

    for func in funcs:
      assert len(func.shape) == 2

  @staticmethod
  def eval( a11, a12, a21, a22 ):
    'evaluate'

    def unique( n, *args ):
      N = -1
      for arg in args:
        if isinstance( arg, numpy.ndarray ):
          if N == -1:
            N = arg.shape[n]
          else:
            assert arg.shape[n] == N
      assert N != -1
      return N

    nrows = unique(0,a11,a12), unique(0,a21,a22)
    ncols = unique(1,a11,a21), unique(1,a12,a22)
    npoints = unique(2,a11,a12,a21,a22) # blatantly assuming 1D points

    data = numpy.empty( [ sum(nrows), sum(ncols), npoints ] )
    data[:nrows[0],:ncols[0]] = a11
    data[:nrows[0],ncols[0]:] = a12
    data[nrows[0]:,:ncols[0]] = a21
    data[nrows[0]:,ncols[0]:] = a22
    return data

class UFunc( ArrayFunc ):
  'user function'

  def __init__( self, coords, ufunc, *gradients ):
    'constructor'

    self.coords = coords
    self.gradients = gradients
    self.shape = ufunc( numpy.zeros( coords.shape ) ).shape
    self.args = ufunc, coords

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

  needxi = True

  def __init__( self, func, ndims, level ):
    'constructor'

    self.ndims = ndims
    self.func = func
    self.level = level
    self.shape = func.shape + (ndims,) * level
    self.args = func.mapping, ndims, level

  def localgradient( self, ndims ):
    'local gradient'

    assert ndims == self.ndims
    return LocalGradient( self.func, ndims, self.level+1 )

  @staticmethod
  def eval( xi, fmap, ndims, level ):
    'evaluate'

    while xi.elem.ndims != ndims:
      xi = xi.next
    T = 1
    while xi.elem not in fmap:
      xi = xi.next
      T = numpy.dot( T, xi.transform )
    F = fmap[ xi.elem ].eval( xi.points, grad=level )
    for axis in range( -1-level, -1 ): # assumes 1D points!
      F = util.transform( F, T, axis=axis )
    return F

  def __str__( self ):
    'string representation'

    return indent( 'Grad:%d;nd%d' % (self.level,self.ndims), self.func )

class Norm2( ArrayFunc ):
  'integration weights'

  def __init__( self, fun, axis ):
    'constructor'

    if axis < 0:
      axis += len(fun.shape)
    assert 0 <= axis < len(fun.shape)

    self.args = fun, axis
    shape = list( fun.shape )
    shape.pop( axis )
    self.shape = tuple(shape)

  @staticmethod
  def eval( fval, axis ):
    'evaluate'

    return numpy.sqrt( util.contract( fval, fval, axis ) )

class Cross( ArrayFunc ):
  'normal'

  def __init__( self, f1, f2, axis ):
    'contructor'

    assert f1.shape == f2.shape
    self.shape = f1.shape
    self.args = f1, f2, -1, -1, -1, axis

  eval = staticmethod( numpy.cross )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, fun, ax1, ax2 ):
    'contructor'

    ax1, ax2 = normdim( len(fun.shape), (ax1,ax2) )
    shape = list(fun.shape)
    shape.pop(ax2)
    shape.pop(ax1)

    self.args = fun, ax1, ax2
    self.shape = tuple(shape)

  def localgradient( self, ndims ):
    'local gradient; jacobi formula'

    fun, ax1, ax2 = self.args
    return self * ( fun.inv(ax1,ax2).swapaxes(ax1,ax2)[...,_] * fun.localgradient(ndims) ).sum(ax1,ax2)

  eval = staticmethod( util.det )

  def __str__( self ):
    'string representation'

    return '%s.det(%d,%d)' % self.args

class GetItem( ArrayFunc ):
  'get item'

  def __init__( self, func, shape, item ):
    'constructor'

    self.shape = shape
    self.args = func, item

  eval = staticmethod( numpy.ndarray.__getitem__ )

  def localgradient( self, ndims ):
    'local gradient'

    func, item = self.args
    grad = func.localgradient( ndims )
    index = item+(slice(None),)
    return grad[index]

  def __str__( self ):
    'string representation'

    #return '%s[%s]' % ( self.args[0], ','.join( util.obj2str(arg) for arg in self.args[1] ) )
    return indent( 'GetItem:%s' % ','.join( util.obj2str(arg) for arg in self.args[1] ), self.args[0] )

class Scalar( float ):
  'scalar'

  shape = ()

  def localgradient( self, ndims ):
    'local gradient'

    return ZERO

  def grad( self, coords, ndims=0 ):
    'gradient'

    # quick insert; is this the best way?
    return ZERO

  def __getitem__( self, item ):
    'get item'

    assert all( it in ( slice(None), _, Ellipsis ) for it in item )
    return self

class StaticArray( ArrayFunc ):
  'static array'

  needxi = True

  def __init__( self, array, shape=None ):
    'constructor'

    array = util.UsableArray( array )
    self.args = array,
    if shape is None:
      shape = array.shape
    else:
      assert len(shape) == array.ndim
      for sh1, sh2 in zip( shape, array.shape ):
        assert int(sh1) == sh2
    self.shape = shape

  def __getitem__( self, item ):
    'get item'

    if not isinstance( item, tuple ):
      item = ( item, )
    if Ellipsis in item:
      idx = item.index( Ellipsis )
      n = len(item) - item.count(_) - 1
      item = item[:idx] + (slice(None),)*(len(self.shape)-n) + item[idx+1:]
      assert Ellipsis not in item

    iter_item = iter( item )
    shape = []
    array = self.args[0]
    for sh in self.shape:
      for it in iter_item:
        if it != _:
          break
        shape.append( nulaxis )
      if sh is nulaxis:
        assert it == slice(None)
        shape.append( sh )
      elif not isinstance( it, int ):
        shape.append( len( numpy.arange(sh)[it] ) )
    for it in iter_item:
      assert it == _
      shape.append( nulaxis )
    return StaticArray( array[item], tuple(shape) )

  def localgradient( self, ndims ):
    'local gradient'

    return ZERO

  @staticmethod
  def eval( xi, array ):

    return util.appendaxes( array, xi.points.coords.shape[1:] )

  def __str__( self ):
    'string representation'

    return 'StaticArray(%s)' % 'x'.join( str(d) for d in self.args[0].shape )

class Multiply( ArrayFunc ):
  'multiply'

  def __init__( self, func1, func2 ):
    'constructor'

    self.args = func1, func2
    self.shape = tuple( align_shapes( func1, func2 ) )

  eval = staticmethod( numpy.ndarray.__mul__ )

  def sum( self, ax1=-1, *axes ):
    'sum'

    func1, func2 = self.args
    return Dot( func1, func2, (ax1,)+axes )

  def localgradient( self, ndims ):
    'gradient'

    return self.args[0][...,_] * self.args[1].localgradient(ndims) \
         + self.args[1][...,_] * self.args[0].localgradient(ndims)

  def __str__( self ):
    'string representation'

    #return '%s * %s' % self.args
    return indent( 'Mul', *self.args )

class Divide( ArrayFunc ):
  'divide'

  def __init__( self, func1, func2 ):
    'constructor'

    if not isinstance( func1, Evaluable ):
      func1 = numpy.asarray( func1 )
      if func1.ndim:
        func1 = StaticArray( func1 )
    if not isinstance( func2, Evaluable ):
      func2 = numpy.asarray( func2 )
      if func2.ndim:
        func2 = StaticArray( func2 )

    shape = align_shapes( func1, func2 )
    self.args = func1, func2
    self.shape = tuple( shape )

  def localgradient( self, ndims ):
    'local gradient'

    func1, func2 = self.args
    grad1 = func1.localgradient(ndims)
    grad2 = func2.localgradient(ndims) 
    return ( grad1 - func1[...,_] * grad2 / func2[...,_] ) / func2[...,_]

  eval = staticmethod( numpy.ndarray.__div__ )

  def __str__( self ):
    'string representation'

    #return '%s / %s' % self.args
    return indent( 'Div', *self.args )

class Negate( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.func = func
    self.shape = func.shape
    self.args = func,

  def localgradient( self, ndims ):
    'local gradient'

    return -self.func.localgradient( ndims )

  eval = staticmethod( numpy.ndarray.__neg__ )

  def __str__( self ):
    'string representation'

    #return '-%s' % self.args[0]
    return indent( 'Neg', *self.args )

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = align_shapes( func1, func2 )
    assert isinstance( func1, (Evaluable,Scalar) )
    assert isinstance( func2, (Evaluable,Scalar) )

    self.args = func1, func2
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.__add__ )

  def localgradient( self, ndims ):
    'gradient'

    return self.args[0].localgradient(ndims) + self.args[1].localgradient(ndims)

  def __str__( self ):
    'string representation'

    #return '(%s + %s)' % self.args
    return indent( 'Add', *self.args )

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    shape = align_shapes( func1, func2 )
    self.args = func1, func2
    self.shape = tuple( shape )

  def __neg__( self ):
    'negate'

    return self.args[1] - self.args[0]

  def localgradient( self, ndims ):
    'gradient'

    return self.args[0].localgradient(ndims) - self.args[1].localgradient(ndims)

  eval = staticmethod( numpy.ndarray.__sub__ )

  def __str__( self ):
    'string representation'

    #return '(%s - %s)' % self.args
    return indent( 'Sub', *self.args )

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, axes ):
    'constructor'

    shape = align_shapes( func1, func2 )
    axes = normdim( len(shape), axes )[::-1]
    for axis in axes:
      shape.pop( axis )

    self.args = func1, func2, tuple(axes)
    self.shape = tuple(shape)

  eval = staticmethod( util.contract )

  def localgradient( self, ndims ):
    'local gradient'

    func1, func2, axes = self.args
    return ( func1.localgradient(ndims) * func2[...,_] ).sum( *axes ) \
         + ( func1[...,_] * func2.localgradient(ndims) ).sum( *axes )

  def __str__( self ):
    'string representation'

    #return '(%s * %s).sum(%s)' % ( self.args[0], self.args[1], ','.join( str(n) for n in self.args[2] ) )
    f1, f2, axes = self.args
    return indent( 'Dot:%s' % ','.join(str(a) for a in axes), f1, f2 )

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
    self.shape = tuple(shape)
    self.args = func, n1, n2

  def localgradient( self, ndims ):
    'local gradient'

    func, n1, n2 = self.args
    return func.localgradient( ndims ).swapaxes( n1, n2 )

  eval = staticmethod( numpy.ndarray.swapaxes )

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
    self.n1 = n1
    self.n2 = n2
    self.args = func, 0, n1, n2
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.trace )

  def localgradient( self, ndims ):
    'local gradient'

    grad = self.func.localgradient( ndims )
    return Trace( grad, self.n1, self.n2 )

  def __str__( self ):
    'string representation'

    return '%s.trace(%d,%d)' % ( self.args[0], self.args[2], self.args[3] )

# MATHEMATICAL EXPRESSIONS

class BaseFunc( ArrayFunc ):
  'unary base class'

  def __init__( self, *funcs ):
    'constructor'

    assert isinstance( funcs[0], ArrayFunc )
    self.shape = funcs[0].shape
    for f in funcs[1:]:
      assert not isinstance( f, ArrayFunc ) or f.shape == self.shape
    self.args = funcs

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
