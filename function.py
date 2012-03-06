from . import util, element, numpy, _

class UsableArray( numpy.ndarray ):
  'array wrapper that can be compared'

  def __new__( self, array ):
    'new'

    return numpy.asarray( array ).view( UsableArray )

  def __eq__( self, other ):
    'compare'

    return isinstance( other, numpy.ndarray ) \
       and other.shape == self.shape \
       and numpy.ndarray.__eq__( self, other ).all()

def normdim( length, *n ):
  'sort and make positive'

  return sorted( ni + length if ni < 0 else ni for ni in n )

class StackIndex( int ):
  'stack index'

  def __str__( self ):
    'string representation'

    return '%%%d' % self

class DofAxis( object ):
  'dof axis'

  def __init__( self, ndofs, mapping ):
    'new'

    self.ndofs = ndofs
    self.mapping = mapping
    self.get = mapping.get

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
    other_mapping = other.mapping.copy()
    try:
      mapping = dict( ( elem, numpy.hstack([ idx, other_mapping.pop(elem) + self.ndofs ]) )
                        for elem, idx in self.mapping.iteritems() )
    except KeyError, e:
      raise Exception, 'element not in other: %s' % e.args[0]
    if other_mapping:
      raise Exception, 'element not in self: %s' % other_mapping.popitem()[0]
    return DofAxis( self.ndofs + other.ndofs, mapping )

  __radd__ = __add__

  def __int__( self ):
    'int'

    return self.ndofs

  def __repr__( self ):
    'string representation'

    return 'DofAxis(%d)' % self

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

    return self.__class__ == other.__class__ and self.args == other.args

class ArrayIndex( Evaluable ):
  'index'

  needxi = True

  def __init__( self, func ):
    'constructor'

    where = numpy.array([ isinstance(sh,DofAxis) for sh in func.shape ])
    count = where.sum()
    reshape = map( tuple, 1 - 2 * numpy.eye( count, dtype=int ) )
    self.args = (reshape,) + func.shape

  @staticmethod
  def eval( xi, reshape, *idxmaps ):
    'evaluate'

    indices = []
    shape = iter(reshape)
    for idxmap in idxmaps:
      if isinstance( idxmap, DofAxis ):
        xi_ = xi
        index = idxmap.get( xi_.elem )
        while index is None:
          xi_ = xi_.next
          index = idxmap.get( xi_.elem )
        indices.append( index.reshape(shape.next()) )
      else:
        indices.append( slice(None) )
    return indices

class StdEval( Evaluable ):
  'stdeval'

  needxi = True

  def __init__( self, func ):
    'constructor'

    self.args = func.mapping,

  @staticmethod
  def eval( xi, mapping ):
    'evaluate'

    while xi.elem not in mapping:
      xi = xi.next
    return mapping[ xi.elem ].eval( xi.points )

  def __str__( self ):
    'string representation'

    return 'Func%x' % id(self)

class Tuple( Evaluable ):
  'combine'

  def __init__( self, args ):
    'constructor'

    self.args = tuple( args )

  @staticmethod
  def eval( *f ):
    'evaluate'

    return f

# ARRAY FUNCTIONS

def merge( funcs ): # temporary
  'merge disjoint function spaces into one'

  shape = {}
  mapping = {}
  ndofs = 0
  nelems = 0
  for func in funcs:
    assert func.__class__ is Function
    shape.update( (elem,idx+ndofs) for (elem,idx) in func.shape[0].mapping.iteritems() )
    mapping.update( func.mapping )
    ndofs += int(func.shape[0])
    nelems += len( func.mapping )
  assert nelems == len( mapping ), 'duplicate elements'
  shape = ( DofAxis( ndofs, shape), ) + func.shape[1:]
  return Function( func.topodims, shape, mapping )

class ArrayFunc( Evaluable ):
  'array function'

  def __getitem__( self, item ):
    'get item'
  
    return GetItem( self, item )

  def __iter__( self ):
    'split first axis'

    return ( self[i] for i in range(self.shape[0]) )

  def swapaxes( self, n1, n2 ):
    'swap axes'

    return SwapAxes( self, n1, n2 )

  def symgrad( self, coords ):
    'gradient'

    g = self.grad( coords )
    return .5 * ( g + g.swapaxes(-2,-1) )

  def div( self, coords ):
    'gradient'

    return self.grad( coords ).trace( -1, -2 )

  def ngrad( self, coords ):
    'normal gradient'

    return ( self.grad(coords) * coords.normal() ).sum()

  def nsymgrad( self, coords ):
    'normal gradient'

    return ( self.symgrad(coords) * coords.normal() ).sum()

  def norm2( self, axis ):
    'norm2'

    return Norm2( self, axis )

  def det( self, ax1, ax2 ):
    'determinant'

    return Determinant( self, ax1, ax2 )

  def __mul__( self, other ):
    'multiply'
  
    if isinstance( other, (int,float) ):
      if other == 0:
        return numpy.zeros( (1,)*len(self.shape), dtype=int )
      if other == 1:
        return self
    return Multiply( self, other )

  def __add__( self, other ):
    'add'
  
    if other == 0:
      return self
    return Add( self, other )

  __rmul__ = __mul__
  __radd__ = __add__

  def __sub__( self, other ):
    'subtract'
  
    if other == 0:
      return self
    return Subtract( self, other )

  def __neg__( self ):
    'negate'

    return Negate( self )

  @property
  def T( self ):
    'transpose'

    assert len(self.shape) == 2
    return SwapAxes( self, 0, 1 )

  def symmetric( self, n1, n2 ):
    'symmetric'

    return Symmetric( self, n1, n2 )

  def trace( self, n1, n2 ):
    'symmetric'

    return Trace( self, n1, n2 )

  def projection( self, fun, topology, coords, **kwargs ):
    'project and return as function'

    weights = self.project( fun, topology, coords, **kwargs )
    return self.dot( weights )

  def project( self, fun, topology, coords=None, ischeme='gauss8', title='projecting', tol=1e-8, exact_boundaries=False, constrain=None, **kwargs ):
    'L2 projection of function onto function space'

    if exact_boundaries:
      assert constrain is None
      constrain = self.project( topology=topology.boundary, coords=coords, ischeme=ischeme, title=None, tol=tol, **kwargs )
    elif constrain is None:
      constrain = util.NanVec( self.shape[0] )
    else:
      assert isinstance( constrain, util.NanVec )
      assert constrain.shape == self.shape[:1]

    if not isinstance( fun, Evaluable ):
      if callable( fun ):
        assert coords
        fun = UFunc( coords, fun )
      else:
        fun = numpy.asarray( fun )

    if len( self.shape ) == 1:
      Afun = self[:,_] * self[_,:]
      bfun = self * fun
    elif len( self.shape ) == 2:
      Afun = ( self[:,_,:] * self[_,:,:] ).sum( 2 )
      bfun = ( self * fun ).sum( 1 )
    else:
      raise Exception
    A, b = topology.integrate( [Afun,bfun], coords=coords, ischeme=ischeme )

    zero = ( numpy.abs( A ) < tol ).all( axis=0 )
    constrain[zero] = 0
    if bfun == 0:
      u = constrain | 0
    else:
      u = util.solve( A, b, constrain )
    u[zero] = numpy.nan
    return u

class IntegrationWeights( ArrayFunc ):
  'integration weights'

  needxi = True

  def __init__( self, coords, ndims ):
    'constructor'

    if coords:
      J = Jacobian( coords )
      cndims, = coords.shape
      if cndims == ndims:
        detJ = J.det( 0, 1 )
      elif ndims == 1:
        detJ = J[:,0].norm2( 0 )
      elif cndims == 3 and ndims == 2:
        detJ = Cross( J[:,0], J[:,1], axis=1 ).norm2( 0 )
      elif ndims == 0:
        detJ = 1.
      else:
        raise NotImplementedError, 'cannot compute determinant for %dx%d jacobian' % J.shape[:2]
    else:
      detJ = 1.

    self.args = detJ,
    self.shape = ()

  @staticmethod
  def eval( xi, detJ ):
    'evaluate'

    return detJ * xi.weights

class Concatenate( ArrayFunc ):
  'concatenate'

  def __init__( self, funcs, axis=0 ):
    'constructor'

    self.args = (axis,) + tuple(funcs)
    self.shape = ( sum( func.shape[0] for func in funcs ), ) + funcs[0].shape[1:]

  def grad( self, coords ):
    'gradient'

    funcs = [ func.grad(coords) for func in self.args[1:] ]
    return Concatenate( funcs, axis=self.args[0] )

  @staticmethod
  def eval( axis, *funcs ):
    'evaluate'

    return numpy.concatenate( funcs, axis=axis )

class Vectorize( ArrayFunc ):
  'vectorize'

  def __init__( self, funcs ):
    'constructor'

    self.args = tuple( funcs )
    self.shape = ( sum( func.shape[0] for func in funcs ), len(funcs) ) + funcs[0].shape[1:]

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

  def grad( self, coords ):
    'gradient'

    return Vectorize([ func.grad( coords ) for func in self.args ])

  def trace( self, n1, n2 ):
    'trace'

    n1, n2 = normdim( len(self.shape), n1, n2 )
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

    n1 = 0
    funcs = []
    for func in self.args:
      n0 = n1
      n1 += int(func.shape[0])
      funcs.append( func.dot( weights[n0:n1,_] ) )
    return Concatenate( funcs )

class Stack( ArrayFunc ):
  'stack'

  def __init__( self, funcs ):
    'constructor'

    funcs = numpy.array( funcs, dtype=object )
    flatfuncs = tuple( funcs.flat )
    shape = []
    indices = []
    partitions = []
    for idim in range( funcs.ndim ):
      n1 = 0
      index = []
      slices = []
      for n in range( funcs.shape[idim] ):
        f = None
        for func in funcs.take( [n], axis=idim ).flat:
          if isinstance( func, ArrayFunc ):
            if not f:
              f = func
            else:
              assert f.shape[idim] == func.shape[idim]
        index.append( flatfuncs.index(f) )
        n0 = n1
        n1 += f.shape[idim]
        slices.append( slice(n0,n1) )
      indices.append( index )
      shape.append( n1 )
      partitions.append( slices )

    self.args = (indices,) + flatfuncs
    self.shape = tuple(shape)
    self.partitions = partitions

  @staticmethod
  def eval( indices, *blocks ):
    'evaluate'

    shape = []
    partitions = []
    for idim, index in enumerate( indices ):
      n1 = 0
      slices = []
      for iblk in index:
        n0 = n1
        n1 += blocks[iblk].shape[idim]
        slices.append( slice(n0,n1) )
      shape.append( n1 )
      partitions.append( slices )

    stacked = numpy.empty( tuple(shape) + blocks[iblk].shape[len(shape):] )
    for I, block in zip( numpy.broadcast( *numpy.ix_( *partitions ) ), blocks ):
      stacked[ I ] = block
    return stacked

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

  def grad( self, coords ):
    'gradient'

    assert coords is self.coords
    return UFunc( self.coords, *self.gradients )

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

class Function( ArrayFunc ):
  'function'

  needxi = True

  def __init__( self, topodims, shape, mapping ):
    'constructor'

    self.topodims = topodims
    self.shape = shape
    self.mapping = mapping
    self.args = mapping,

  @staticmethod
  def eval( xi, fmap ):
    'evaluate'

    while xi.elem not in fmap:
      xi = xi.next
    return fmap[ xi.elem ].eval( xi.points )

  def grad( self, coords ):
    'gradient'

    if self == coords:
      assert len(self.shape) == 1
      return numpy.eye( self.shape[0] )
    return Grad( self, coords )

  def __str__( self ):
    'string representation'

    return 'Function:%x' % id(self.mapping)

  def dot( self, weights ):
    'dot'

    assert weights.shape[0] == int(self.shape[0])
    mapping = {}
    for elem, funceval in self.mapping.iteritems():
      indices = self.shape[0].get( elem )
      mapping[ elem ] = funceval.dot( weights[indices] )
    return Function( topodims=self.topodims, shape=weights.shape[1:]+self.shape[1:], mapping=mapping )

  def normal( self ):
    'normal'

    return Normal( self )

  def vector( self, ndims ):
    'vectorize'

    return Vectorize( [self]*ndims )

class Grad( ArrayFunc ):
  'gradient'

  needxi = True

  def __init__( self, func, coords ):
    'constructor'

    assert len(coords.shape) == 1
    #self.args = StdEval(func), StdEval(coords), Transform(coords,func)
    self.func = func
    self.coords = coords
    self.args = func.mapping, coords.mapping
    self.shape = func.shape + coords.shape

  @staticmethod
  def eval( xi, fmap, cmap ):
    'evaluate'

    while xi.elem not in fmap:
      xi = xi.next
    F = fmap[ xi.elem ].eval( xi.points, grad=1 )
    T = 1
    while xi.elem not in cmap:
      xi = xi.next
      T = numpy.dot( T, xi.transform )
    C = cmap[ xi.elem ].eval( xi.points, grad=1 )
    J = util.transform( C, T, axis=-2 )
    if J.shape[0] == J.shape[1]:
      Jinv = util.inv( J, axes=(0,1) )
    elif J.shape[0] == J.shape[1] + 1:
      J = numpy.concatenate( [ J, c.normal[:,_,:] ], axis=1 )
      Jinv = util.inv( J, axes=(0,1) )[:-1]
    else:
      raise Exception, 'cannot compute inverse of %dx%d jacobian' % ( J.shape[0], J.shape[1] )
    np = Jinv.ndim - 2
    n = F.ndim - np
    index = (slice(None),) * n + (_,Ellipsis)
    return ( F[index] * Jinv ).sum( n-1 )

  def __str__( self ):
    'string representation'

    return '%s.grad(%s)' % ( self.func, self.coords )

class Normal( ArrayFunc ):
  'normal'

  def __init__( self, coords ):
    'constructor'

    self.coords = coords
    self.args = Jacobian( coords ),
    self.shape = coords.shape

  def __str__( self ):
    'string representation'

    return 'Normal(%s)' % self.coords

  @staticmethod
  def eval( J ):
    'evaluate'

    if J.shape[:2] == (2,1):
      normal = numpy.array([ J[1,0], -J[0,0] ])
    elif J.shape[:2] == (3,2):
      normal = numpy.cross( J[:,0], J[:,1], axis=0 )
    elif J.shape[:2] == (3,1):
      normal = numpy.cross( self.next.normal.T, self.J[:,0,:].T ).T
    else:
      raise NotImplementedError, 'cannot compute normal for %dx%d jacobian' % J.shape[:2]
    return normal / util.norm2( normal, axis=0 )

class Norm2( ArrayFunc ):
  'integration weights'

  def __init__( self, fun, axis=0 ):
    'constructor'

    self.args = fun, axis
    shape = list( fun.shape )
    shape.pop( axis )
    self.shape = tuple(shape)

  @staticmethod
  def eval( fval, axis ):
    'evaluate'

    return numpy.sqrt( ( fval**2 ).sum( axis ) )

class Jacobian( ArrayFunc ):
  'jacobian'

  needxi = True

  def __init__( self, coords ):
    'constructor'

    self.args = coords.mapping,
    #self.args = StdEval(coords), Transform(coords)
    assert coords.shape
    self.shape = coords.shape + (coords.topodims,)
    assert len(self.shape) == 2

  @staticmethod
  def eval( xi, cmap ):
    'evaluate'

    T = 1
    while xi.elem not in cmap:
      xi = xi.next
      T = numpy.dot( T, xi.transform )
    C = cmap[ xi.elem ].eval( xi.points, grad=1 )
    return util.transform( C, T, axis=-2 )

  def __str__( self ):
    'string representation'

    return '%s.J' % self.args[0]

class Cross( ArrayFunc ):
  'normal'

  def __init__( self, f1, f2, axis ):
    'contructor'

    assert f1.shape == f2.shape
    self.shape = f1.shape
    self.args = fun1, fun2, -1, -1, -1, axis

  eval = staticmethod( numpy.cross )

class Determinant( ArrayFunc ):
  'normal'

  def __init__( self, fun, ax1, ax2 ):
    'contructor'

    self.args = fun, ax1, ax2

  eval = staticmethod( util.det )

  def __str__( self ):
    'string representation'

    return '%s.det(%d,%d)' % self.args

class GetItem( ArrayFunc ):
  'get item'

  def __init__( self, func, item ):
    'constructor'

    if not isinstance( item, tuple ):
      item = ( item, )
    if Ellipsis in item:
      idx = item.index( Ellipsis )
      n = len(item) - item.count(_) - 1
      item = item[:idx] + (slice(None),)*(len(func.shape)-n) + item[idx+1:]
      assert Ellipsis not in item
    assert len(item) - item.count(_) == len(func.shape)
    shape = []
    itershape = iter( func.shape )
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
        raise Exception, 'invalid slice'
    self.shape = tuple( shape )
    self.args = func, item

  eval = staticmethod( numpy.ndarray.__getitem__ )

  def grad( self, coords ):
    'gradient'

    return self.args[0].grad(coords)[ self.args[1] ]

  def __str__( self ):
    'string representation'

    return '%s[%s]' % ( self.args[0], ','.join( util.obj2str(arg) for arg in self.args[1] ) )

class StaticArray( ArrayFunc ):
  'static array'

  needxi = True

  def __init__( self, array ):
    'constructor'

    array = UsableArray( array )
    self.args = array,
    self.shape = array.shape

  @staticmethod
  def eval( xi, array ):

    return array.reshape( array.shape + (1,)*(xi.points.coords.ndim-1) )

  def __str__( self ):
    'string representation'

    return 'StaticArray(%s)' % self.args[0]

class Multiply( ArrayFunc ):
  'multiply'

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

    self.args = func1, func2
    D = len(func1.shape) - len(func2.shape)
    nul = (nulaxis,)
    shape = []
    for sh1, sh2 in zip( nul*-D + func1.shape, nul*D + func2.shape ):
      if sh1 is nulaxis:
        shape.append( sh2 )
      elif sh2 is nulaxis:
        shape.append( sh1 )
      else:
        assert sh1 == sh2, 'incompatible dimensions: %s and %s' % ( func1.shape, func2.shape )
        shape.append( sh1 )
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.__mul__ )

  def sum( self, ax1=-1, *axes ):
    'sum'

    return Dot( *self.args, axes=(ax1,)+axes )

  def grad( self, coords ):
    'gradient'

    return self.args[0][...,_] * self.args[1].grad( coords ) + self.args[1][...,_] * self.args[0].grad( coords )

  def __str__( self ):
    'string representation'

    return '%s * %s' % self.args

class Negate( ArrayFunc ):
  'negate'

  def __init__( self, func ):
    'constructor'

    self.shape = func.shape
    self.args = func,

  eval = staticmethod( numpy.ndarray.__neg__ )

  def __str__( self ):
    'string representation'

    return '-%s' % self.args[0]

class Add( ArrayFunc ):
  'add'

  def __init__( self, func1, func2 ):
    'constructor'

    if isinstance( func1, (int,float) ):
      func1 = numpy.asarray( func1 )
    func1_shape = func1.shape

    if isinstance( func2, (int,float) ):
      func2 = numpy.asarray( func2 )
    func2_shape = func2.shape

    self.args = func1, func2
    D = len(func1_shape) - len(func2_shape)
    nul = (nulaxis,)
    shape = []
    for sh1, sh2 in zip( nul*-D + func1_shape, nul*D + func2_shape ):
      if sh1 is nulaxis:
        shape.append( sh2 )
      elif sh2 is nulaxis:
        shape.append( sh1 )
      else:
        assert sh1 == sh2, 'incompatible dimensions: %s and %s' % ( func1_shape, func2_shape )
        shape.append( sh1 )
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.__add__ )

  def __str__( self ):
    'string representation'

    return '(%s + %s)' % self.args

class Subtract( ArrayFunc ):
  'subtract'

  def __init__( self, func1, func2 ):
    'constructor'

    if isinstance( func1, (int,float) ):
      func1 = numpy.asarray( func1 )
    func1_shape = func1.shape

    if isinstance( func2, (int,float) ):
      func2 = numpy.asarray( func2 )
    func2_shape = func2.shape

    self.args = func1, func2
    D = len(func1_shape) - len(func2_shape)
    nul = (nulaxis,)
    shape = []
    for sh1, sh2 in zip( nul*-D + func1_shape, nul*D + func2_shape ):
      if sh1 is nulaxis:
        shape.append( sh2 )
      elif sh2 is nulaxis:
        shape.append( sh1 )
      else:
        assert sh1 == sh2, 'incompatible dimensions: %s and %s' % ( func1_shape, func2_shape )
        shape.append( sh1 )
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.__sub__ )

  def __str__( self ):
    'string representation'

    return '(%s - %s)' % self.args

class Dot( ArrayFunc ):
  'dot'

  def __init__( self, func1, func2, axes ):
    'constructor'

    if isinstance( func1, (int,float) ):
      func1 = numpy.asarray( func1 )
    func1_shape = func1.shape

    if isinstance( func2, (int,float) ):
      func2 = numpy.asarray( func2 )
    func2_shape = func2.shape

    D = len(func1_shape) - len(func2_shape)
    nul = (nulaxis,)
    shape = []
    for sh1, sh2 in zip( nul*-D + func1_shape, nul*D + func2_shape ):
      if sh1 is nulaxis:
        shape.append( sh2 )
      elif sh2 is nulaxis:
        shape.append( sh1 )
      else:
        assert sh1 == sh2, 'incompatible dimensions: %s and %s' % ( func1_shape, func2_shape )
        shape.append( sh1 )

    axes = normdim( len(shape), *axes )[::-1]
    self.args = func1, func2, tuple(axes)
    for axis in axes:
      shape.pop( axis )
    self.shape = tuple(shape)

  @staticmethod
  def eval( func1, func2, axes ):
    'evaluate'

    retval = func1 * func2
    for axis in axes:
      retval = retval.sum( axis )
    return retval

  def __str__( self ):
    'string representation'

    return '(%s * %s).sum(%s)' % ( self.args[0], self.args[1], ','.join( str(n) for n in self.args[2] ) )

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

  eval = staticmethod( numpy.ndarray.swapaxes )

class Trace( ArrayFunc ):
  'trace'

  def __init__( self, func, n1, n2 ):
    'constructor'

    n1, n2 = normdim( len(func.shape), n1, n2 )
    shape = list( func.shape )
    s1 = shape.pop( n2 )
    s2 = shape.pop( n1 )
    assert s1 == s2
    self.args = func, 0, n1, n2
    self.shape = tuple( shape )

  eval = staticmethod( numpy.ndarray.trace )

  def __str__( self ):
    'string representation'

    return '%s.trace(%d,%d)' % ( self.args[0], self.args[2], self.args[3] )

# MATHEMATICAL EXPRESSIONS

class UnaryFunc( ArrayFunc ):
  'unary base class'

  def __init__( self, func ):
    'constructor'

    self.args = func,
    self.shape = func.shape

  def __str__( self ):
    'string representation'

    return '%s(%s)' % ( self.__class__.__name__, self.args[0] )

class Exp( UnaryFunc ):
  'exponent'

  eval = staticmethod( numpy.exp )

  def grad( self, coords ):
    'gradient'

    return self * self.args[0].grad( coords )

class Sin( UnaryFunc ):
  'sine'

  eval = staticmethod( numpy.sin )

  def grad( self, coords ):
    'gradient'

    return Cos(self.args[0]) * self.args[0].grad( coords )
    
class Cos( UnaryFunc ):
  'cosine'

  eval = staticmethod( numpy.cos )

  def grad( self, coords ):
    'gradient'

    return -Sin(self.args[0]) * self.args[0].grad( coords )

class Log( UnaryFunc ):
  'cosine'

  eval = staticmethod( numpy.log )

#############################33

def RectilinearFunc( topo, gridnodes ):
  'rectilinear mesh generator'

  assert len( gridnodes ) == topo.ndims
  nodes_structure = numpy.empty( map( len, gridnodes ) + [topo.ndims] )
  for idim, inodes in enumerate( gridnodes ):
    shape = [1,] * topo.ndims
    shape[idim] = -1
    nodes_structure[...,idim] = numpy.asarray( inodes ).reshape( shape )

  return topo.linearfunc().dot( nodes_structure.reshape( -1, topo.ndims ) )

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
