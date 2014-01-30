import numpy, warnings

try:
  from _numeric import _contract, SaneArray
except:
  warnings.warn( '''Failed to load _numeric module.
  Falling back on equivalent python implementation. THIS
  MAY SEVERELY IMPACT PERFORMANCE! Pleace compile the C
  extensions by running 'make' in the nutils directory.''', stacklevel=2 )
  def _contract( A, B, axes ):
    assert A.shape == B.shape and axes > 0
    return ((A*B).reshape(A.shape[:-axes]+(-1,))).sum(-1)
  class SaneArray( numpy.ndarray ):
    __slots__ = ()
    def __new__( cls, arr ):
      return numpy.asarray( arr ).view( cls )
    def __eq__( self, other ):
      return self is other or ( isinstance( other, numpy.ndarray )
        and self.shape == other.shape
        and self.dtype == other.dtype
        and ( self.__array_interface__['data'] == other.__array_interface__['data']
              and self.strides == other.strides
              or numpy.equal( self, other ).all() ) )
    def __hash__( self ):
      assert not self.flage.writeable
      return hash( self.shape + tuple( self.flat[::self.size//4+1] ) ) # incompatible with c hash!

def _wrap( f ):
  return lambda *args, **kwargs: f( *args, **kwargs ).view( SaneArray )

abs           = _wrap( numpy.abs           )
add           = _wrap( numpy.add           )
arange        = _wrap( numpy.arange        )
arccos        = _wrap( numpy.arccos        )
arcsin        = _wrap( numpy.arcsin        )
arctan2       = _wrap( numpy.arctan2       )
arctanh       = _wrap( numpy.arctanh       )
argmax        = _wrap( numpy.argmax        )
argmin        = _wrap( numpy.argmin        )
array         = _wrap( numpy.array         )
asarray       = _wrap( numpy.asarray       )
ceil          = _wrap( numpy.ceil          )
choose        = _wrap( numpy.choose        )
concatenate   = _wrap( numpy.concatenate   )
cos           = _wrap( numpy.cos           )
cosh          = _wrap( numpy.cosh          )
cumsum        = _wrap( numpy.cumsum        )
diag          = _wrap( numpy.diag          )
diag          = _wrap( numpy.diag          )
diagflat      = _wrap( numpy.diagflat      )
diff          = _wrap( numpy.diff          )
divide        = _wrap( numpy.divide        )
empty         = _wrap( numpy.empty         )
empty_like    = _wrap( numpy.empty_like    )
equal         = _wrap( numpy.equal         )
exp           = _wrap( numpy.exp           )
eye           = _wrap( numpy.eye           )
frombuffer    = _wrap( numpy.frombuffer    )
greater       = _wrap( numpy.greater       )
greater_equal = _wrap( numpy.greater_equal )
hstack        = _wrap( numpy.hstack        )
interp        = _wrap( numpy.interp        )
intersect1d   = _wrap( numpy.intersect1d   )
isnan         = _wrap( numpy.isnan         )
less          = _wrap( numpy.less          )
less_equal    = _wrap( numpy.less_equal    )
linspace      = _wrap( numpy.linspace      )
log           = _wrap( numpy.log           )
log10         = _wrap( numpy.log10         )
log2          = _wrap( numpy.log2          )
logical_and   = _wrap( numpy.logical_and   )
logical_not   = _wrap( numpy.logical_not   )
logical_or    = _wrap( numpy.logical_or    )
logical_xor   = _wrap( numpy.logical_xor   )
max           = _wrap( numpy.max           )
maximum       = _wrap( numpy.maximum       )
min           = _wrap( numpy.min           )
minimum       = _wrap( numpy.minimum       )
multiply      = _wrap( numpy.multiply      )
negative      = _wrap( numpy.negative      )
normal        = _wrap( numpy.random.normal )
not_equal     = _wrap( numpy.not_equal     )
ones          = _wrap( numpy.ones          )
ones_like     = _wrap( numpy.ones_like     )
power         = _wrap( numpy.power         )
prod          = _wrap( numpy.prod          )
product       = _wrap( numpy.product       )
reciprocal    = _wrap( numpy.reciprocal    )
repeat        = _wrap( numpy.repeat        )
roll          = _wrap( numpy.roll          )
searchsorted  = _wrap( numpy.searchsorted  )
sign          = _wrap( numpy.sign          )
sign          = _wrap( numpy.sign          )
sin           = _wrap( numpy.sin           )
sinh          = _wrap( numpy.sinh          )
solve         = _wrap( numpy.linalg.solve  )
sqrt          = _wrap( numpy.sqrt          )
subtract      = _wrap( numpy.subtract      )
sum           = _wrap( numpy.sum           )
tan           = _wrap( numpy.tan           )
tanh          = _wrap( numpy.tanh          )
vstack        = _wrap( numpy.vstack        )
zeros         = _wrap( numpy.zeros         )
zeros_like    = _wrap( numpy.zeros_like    )

broadcast = numpy.broadcast
cond = numpy.linalg.cond
float64 = numpy.float64
iinfo = numpy.iinfo
inf = numpy.inf
intc = numpy.intc
isfinite = numpy.isfinite
ix_ = numpy.ix_
nan = numpy.nan
ndarray = numpy.ndarray
ndindex = numpy.ndindex
pi = numpy.pi
random_seed = numpy.random.seed
spacing = numpy.spacing
testing = numpy.testing
unravel_index = numpy.unravel_index

def insane( obj ):
  return obj.view( ndarray ) if isarray( obj ) \
    else list( insane(item) for item in obj ) if isinstance( obj, list ) \
    else tuple( insane(item) for item in obj ) if isinstance( obj, tuple ) \
    else dict( ( insane(key), insane(value) ) for key, value in obj.items() ) if isinstance( obj, dict ) \
    else obj

def grid( shape ):
  return map( asarray, numpy.ogrid[ tuple( slice(n) for n in shape ) ] )

def isarray( A ):
  return isinstance( A, numpy.ndarray )

def isscalar( A ):
  return isinstance( A, (int,float) ) or isarray( A ) and A.ndim == 0

def unique( ar, *args, **kwargs ):
  return numpy.unique( numpy.asarray( ar ), *args, **kwargs ).view( SaneArray )

def eigh( *args, **kwargs ):
  w, v = numpy.linalg.eigh( *args, **kwargs )
  return w.view( SaneArray ), v.view( SaneArray )

def inv( A ):
  'linearized inverse'

  A = asarray( A )
  assert A.shape[-2] == A.shape[-1]
  if A.shape[-1] == 1:
    return reciprocal( A )
  if A.ndim == 2:
    return numpy.linalg.inv( A ).view( SaneArray )
  if A.shape[-1] == 2:
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
    Ainv = empty( A.shape )
    numpy.divide( A[...,1,1],  det, Ainv[...,0,0] )
    numpy.divide( A[...,0,0],  det, Ainv[...,1,1] )
    numpy.divide( A[...,1,0], -det, Ainv[...,1,0] )
    numpy.divide( A[...,0,1], -det, Ainv[...,0,1] )
    return Ainv
  Ainv = empty( A.shape )
  for I in ndindex( A.shape[:-2] ):
    Ainv[I] = numpy.linalg.inv( A[I] )
  return Ainv

def det( A ):
  'determinant'

  A = asarray( A )
  assert A.shape[-2] == A.shape[-1]
  if A.shape[-1] == 1:
    return A[...,0,0]
  if A.ndim == 2:
    return numpy.linalg.det( A ).view( SaneArray )
  if A.shape[-1] == 2:
    return A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
  det = empty( A.shape[:-2] )
  for I in ndindex( A.shape[:-2] ):
    det[I] = numpy.linalg.det( A[I] )
  return det

def getitem( A, axis, indices ):
  indices = (slice(None),) * axis + (indices,) if axis >= 0 \
       else (Ellipsis,indices) + (slice(None),) * (-axis-1)
  return asarray( A )[ indices ]

def as_strided( A, shape, strides ):
  return numpy.lib.stride_tricks.as_strided( A, shape, strides ).view( SaneArray )

def norm2( A, axis=-1 ):
  'L2 norm over specified axis'

  return sqrt( contract( A, A, axis ) )

def findsorted( array, items ):
  indices = searchsorted( array, items )
  assert numpy.less( indices, array.size ).all()
  assert numpy.equal( array[indices], items ).all()
  return indices

def find( arr ):
  nz, = arr.ravel().nonzero()
  return nz.view( SaneArray )

def objmap( func, *arrays ):
  'map numpy arrays'

  arrays = [ asarray( array, dtype=object ) for array in arrays ]
  return numpy.frompyfunc( func, len(arrays), 1 )( *arrays )


#####

def normdim( ndim, n ):
  'check bounds and make positive'

  assert isinstance(ndim,int) and ndim >= 0, 'ndim must be positive integer, got %s' % ndim
  if n < 0:
    n += ndim
  assert 0 <= n < ndim, 'argument out of bounds: %s not in [0,%s)' % (n,ndim)
  return n

def align( arr, trans, ndim ):
  '''create new array of ndim from arr with axes moved accordin
  to trans'''

  # as_strided will check validity of trans
  arr = asarray( arr )
  trans = asarray( trans, dtype=int )
  assert len(trans) == arr.ndim
  strides = numpy.zeros( ndim, dtype=int )
  strides[trans] = arr.strides
  shape = numpy.ones( ndim, dtype=int )
  shape[trans] = arr.shape
  tmp = as_strided( arr, shape, strides )
  return tmp

def expand( arr, *shape ):
  'expand'

  newshape = list( arr.shape )
  for i, sh in enumerate( shape ):
    if sh != None:
      if newshape[i-len(shape)] == 1:
        newshape[i-len(shape)] = sh
      else:
        assert newshape[i-len(shape)] == sh
  expanded = numpy.empty( newshape )
  expanded[:] = arr
  return expanded

def linspace2d( start, stop, steps ):
  'linspace & meshgrid combined'

  start = tuple(start) if isinstance(start,(list,tuple)) else (start,start)
  stop  = tuple(stop ) if isinstance(stop, (list,tuple)) else (stop, stop )
  steps = tuple(steps) if isinstance(steps,(list,tuple)) else (steps,steps)
  assert len(start) == len(stop) == len(steps) == 2
  values = numpy.empty( (2,)+steps )
  values[0] = numpy.linspace( start[0], stop[0], steps[0] )[:,numpy.newaxis]
  values[1] = numpy.linspace( start[1], stop[1], steps[1] )[numpy.newaxis,:]
  return values

def contract( A, B, axis=-1 ):
  'contract'

  A = asarray( A, dtype=float )
  B = asarray( B, dtype=float )

  n = B.ndim - A.ndim
  if n > 0:
    Ashape = list(B.shape[:n]) + list(A.shape)
    Astrides = [0]*n + list(A.strides)
    Bshape = list(B.shape)
    Bstrides = list(B.strides)
  elif n < 0:
    n = -n
    Ashape = list(A.shape)
    Astrides = list(A.strides)
    Bshape = list(A.shape[:n]) + list(B.shape)
    Bstrides = [0]*n + list(B.strides)
  else:
    Ashape = list(A.shape)
    Astrides = list(A.strides)
    Bshape = list(B.shape)
    Bstrides = list(B.strides)

  shape = Ashape
  nd = len(Ashape)
  for i in range( n, nd ):
    if Ashape[i] == 1:
      shape[i] = Bshape[i]
      Astrides[i] = 0
    elif Bshape[i] == 1:
      Bstrides[i] = 0
    else:
      assert Ashape[i] == Bshape[i]

  if isinstance( axis, int ):
    axis = axis,
  axis = sorted( [ ax+nd if ax < 0 else ax for ax in axis ], reverse=True )
  for ax in axis:
    assert 0 <= ax < nd, 'invalid contraction axis'
    shape.append( shape.pop(ax) )
    Astrides.append( Astrides.pop(ax) )
    Bstrides.append( Bstrides.pop(ax) )

  A = as_strided( A, shape, Astrides )
  B = as_strided( B, shape, Bstrides )

  if not A.size:
    return numpy.zeros( A.shape[:-len(axis)] )

  return _contract( A, B, len(axis) )

def contract_fast( A, B, naxes ):
  'contract last n axes'

  A = asarray( A, dtype=float )
  B = asarray( B, dtype=float )

  n = B.ndim - A.ndim
  if n > 0:
    Ashape = list(B.shape[:n]) + list(A.shape)
    Astrides = [0]*n + list(A.strides)
    Bshape = list(B.shape)
    Bstrides = list(B.strides)
  elif n < 0:
    n = -n
    Ashape = list(A.shape)
    Astrides = list(A.strides)
    Bshape = list(A.shape[:n]) + list(B.shape)
    Bstrides = [0]*n + list(B.strides)
  else:
    Ashape = list(A.shape)
    Astrides = list(A.strides)
    Bshape = list(B.shape)
    Bstrides = list(B.strides)

  shape = list(Ashape)
  for i in range( len(Ashape) ):
    if Ashape[i] == 1:
      shape[i] = Bshape[i]
      Astrides[i] = 0
    elif Bshape[i] == 1:
      Bstrides[i] = 0
    else:
      assert Ashape[i] == Bshape[i]

  A = as_strided( A, shape, Astrides )
  B = as_strided( B, shape, Bstrides )

  if not A.size:
    return numpy.zeros( shape[:-naxes] )

  return _contract( A, B, naxes )

def dot( A, B, axis=-1 ):
  '''Transform axis of A by contraction with first axis of B and inserting
     remaining axes. Note: with default axis=-1 this leads to multiplication of
     vectors and matrices following linear algebra conventions.'''

  A = asarray( A, dtype=float )
  B = asarray( B, dtype=float )

  if axis < 0:
    axis += A.ndim
  assert 0 <= axis < A.ndim

  if A.shape[axis] == 1 or B.shape[0] == 1:
    return A.sum(axis)[(slice(None),)*axis+(numpy.newaxis,)*(B.ndim-1)] \
         * B.sum(0)[(Ellipsis,)+(numpy.newaxis,)*(A.ndim-1-axis)]

  assert A.shape[axis] == B.shape[0]

  if B.ndim != 1 or axis != A.ndim-1:
    shape = A.shape[:axis] + B.shape[1:] + A.shape[axis+1:] + A.shape[axis:axis+1]
    Astrides = A.strides[:axis] + (0,) * (B.ndim-1) + A.strides[axis+1:] + A.strides[axis:axis+1]
    A = as_strided( A, shape, Astrides )

  if A.ndim > 1:
    Bstrides = (0,) * axis + B.strides[1:] + (0,) * (A.ndim-B.ndim-axis) + B.strides[:1]
    B = as_strided( B, A.shape, Bstrides )

  if not A.size:
    return zeros( A.shape[:-1] )

  return _contract( A, B, 1 ).view( SaneArray )

def fastrepeat( A, nrepeat, axis=-1 ):
  'repeat axis by 0stride'

  A = asarray( A )
  assert A.shape[axis] == 1
  shape = list( A.shape )
  shape[axis] = nrepeat
  strides = list( A.strides )
  strides[axis] = 0
  return as_strided( A, shape, strides )

def fastmeshgrid( X, Y ):
  'mesh grid based on fastrepeat'

  return fastrepeat(X[numpy.newaxis,:],len(Y),axis=0), fastrepeat(Y[:,numpy.newaxis],len(X),axis=1)

def meshgrid( *args ):
  'multi-dimensional meshgrid generalisation'

  args = map( asarray, args )
  shape = [ len(args) ] + [ arg.size for arg in args if arg.ndim ]
  grid = numpy.empty( shape )
  n = len(shape)-1
  for i, arg in enumerate( args ):
    if arg.ndim:
      n -= 1
      grid[i] = arg[(slice(None),)+(numpy.newaxis,)*n]
    else:
      grid[i] = arg
  assert n == 0
  return grid

def appendaxes( A, shape ):
  'append axes by 0stride'

  shape = (shape,) if isinstance(shape,int) else tuple(shape)
  A = asarray( A )
  return as_strided( A, A.shape + shape, A.strides + (0,)*len(shape) )

def takediag( A ):
  if A.shape[-1] == 1:
    shape = A.shape[:-1]
    strides = A.strides[:-1]
  elif A.shape[-2] == 1:
    shape = A.shape[:-2] + A.shape[-1:]
    strides = A.strides[:-2] + A.strides[-1:]
  else:
    assert A.shape[-1] == A.shape[-2]
    shape = A.shape[:-1]
    strides = A.strides[:-2] + (A.strides[-2]+A.strides[-1],)
  return as_strided( A, shape, strides )

def inverse( A ):
  warnings.warn( 'numeric.inverse is deprecated, use numeric.inv instead' )
  return inv( A )

def determinant( A ):
  warnings.warn( 'numeric.determinant is deprecated, use numeric.det instead' )
  return det( A )

def reshape( A, *shape ):
  'more useful reshape'

  newshape = []
  i = 0
  for s in shape:
    if isinstance( s, (tuple,list) ):
      assert numpy.product( s ) == A.shape[i]
      newshape.extend( s )
      i += 1
    elif s == 1:
      newshape.append( A.shape[i] )
      i += 1
    else:
      assert s > 1
      newshape.append( numpy.product( A.shape[i:i+s] ) )
      i += s
  assert i <= A.ndim
  newshape.extend( A.shape[i:] )
  return A.reshape( newshape )

def mean( A, weights=None, axis=-1 ):
  'generalized mean'

  return A.mean( axis ) if weights is None else dot( A, weights / weights.sum(), axis )

def normalize( A, axis=-1 ):
  'devide by normal'

  s = [ slice(None) ] * A.ndim
  s[axis] = numpy.newaxis
  return A / norm2( A, axis )[ tuple(s) ]

def cross( v1, v2, axis ):
  'cross product'

  if v1.ndim < v2.ndim:
    v1 = v1[ (numpy.newaxis,)*(v2.ndim-v1.ndim) ]
  elif v2.ndim < v1.ndim:
    v2 = v2[ (numpy.newaxis,)*(v1.ndim-v2.ndim) ]
  return numpy.cross( v1, v2, axis=axis )

def stack( arrays, axis=0 ):
  'powerful array stacker with singleton expansion'

  arrays = [ asarray(array,dtype=float) for array in arrays ]
  shape = [1] * max([ array.ndim for array in arrays ])
  axis = normdim( len(shape)+1, axis )
  for array in arrays:
    for i in range(-array.ndim,0):
      if shape[i] == 1:
        shape[i] = array.shape[i]
      else:
        assert array.shape[i] in ( shape[i], 1 )
  stacked = numpy.empty( shape[:axis]+[len(arrays)]+shape[axis:], dtype=float )
  for i, arr in enumerate( arrays ):
    stacked[(slice(None),)*axis+(i,)] = arr
  return stacked

def bringforward( arg, axis ):
  'bring axis forward'

  arg = asarray(arg)
  axis = normdim(arg.ndim,axis)
  if axis == 0:
    return arg
  return arg.transpose( [axis] + range(axis) + range(axis+1,arg.ndim) )

def diagonalize( arg ):
  'append axis, place last axis on diagonal of self and new'

  diagonalized = numpy.zeros( arg.shape + (arg.shape[-1],) )
  takediag( diagonalized )[:] = arg
  return diagonalized

def check_equal_wrapper( f1, f2 ):
  def f12( *args ):
    v1 = f1( *args )
    v2 = f2( *args )
    numpy.testing.assert_array_almost_equal( v1, v2 )
    return v1
  return f12

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
