import numpy, warnings
from _numeric import _contract, NumericArray

_wrap = lambda f: lambda *args, **kwargs: f( *args, **kwargs ).view( NumericArray )
for name in ( 'abs', 'add', 'arange', 'arccos', 'arcsin', 'arctan2', 'arctanh',
  'argmax', 'argmin', 'array', 'asarray', 'ceil', 'choose', 'concatenate',
  'cos', 'cosh', 'cumsum', 'diag', 'diag', 'diagflat', 'diff', 'divide',
  'empty', 'empty_like', 'equal', 'exp', 'eye', 'frombuffer', 'greater',
  'greater_equal', 'hstack', 'interp', 'intersect1d', 'isnan', 'less',
  'less_equal', 'linspace', 'log', 'log10', 'log2', 'logical_and',
  'logical_not', 'logical_or', 'logical_xor', 'max', 'maximum', 'min',
  'minimum', 'multiply', 'negative', 'random.normal', 'not_equal', 'ones',
  'ones_like', 'power', 'prod', 'product', 'reciprocal', 'repeat', 'roll',
  'searchsorted', 'sign', 'sign', 'sin', 'sinh', 'linalg.solve', 'sqrt',
  'subtract', 'sum', 'tan', 'tanh', 'vstack', 'zeros', 'zeros_like' ):
  obj = numpy
  for name in name.split( '.' ):
    obj = getattr( obj, name )
  locals()[ name ] = _wrap( obj )
  
for name in ( 'broadcast', 'linalg.cond', 'float64', 'iinfo', 'inf', 'intc',
  'isfinite', 'ix_', 'nan', 'ndarray', 'ndindex', 'pi', 'random.seed',
  'spacing', 'testing', 'unravel_index', 'newaxis' ):
  obj = numpy
  for name in name.split( '.' ): obj = getattr( obj, name )
  locals()[ name ] = obj

def grid( shape ):
  return map( asarray, numpy.ogrid[ tuple( slice(n) for n in shape ) ] )

def isarray( A ):
  return isinstance( A, numpy.ndarray )

def isscalar( A ):
  return isinstance( A, (int,float) ) or isarray( A ) and A.ndim == 0

def unique( ar, *args, **kwargs ):
  return numpy.unique( numpy.asarray( ar ), *args, **kwargs ).view( NumericArray )

def eigh( *args, **kwargs ):
  w, v = numpy.linalg.eigh( *args, **kwargs )
  return w.view( NumericArray ), v.view( NumericArray )

def inv( A ):
  'linearized inverse'

  A = asarray( A )
  assert A.shape[-2] == A.shape[-1]
  if A.shape[-1] == 1:
    return reciprocal( A )
  if A.ndim == 2:
    return numpy.linalg.inv( A ).view( NumericArray )
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
    return numpy.linalg.det( A ).view( NumericArray )
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
  return numpy.lib.stride_tricks.as_strided( A, shape, strides ).view( NumericArray )

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
  return nz.view( NumericArray )

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
  strides = zeros( ndim, dtype=int )
  strides[trans] = arr.strides
  shape = ones( ndim, dtype=int )
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
  expanded = empty( newshape )
  expanded[:] = arr
  return expanded

def linspace2d( start, stop, steps ):
  'linspace & meshgrid combined'

  start = tuple(start) if isinstance(start,(list,tuple)) else (start,start)
  stop  = tuple(stop ) if isinstance(stop, (list,tuple)) else (stop, stop )
  steps = tuple(steps) if isinstance(steps,(list,tuple)) else (steps,steps)
  assert len(start) == len(stop) == len(steps) == 2
  values = empty( (2,)+steps )
  values[0] = linspace( start[0], stop[0], steps[0] )[:,newaxis]
  values[1] = linspace( start[1], stop[1], steps[1] )[newaxis,:]
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
    return zeros( A.shape[:-len(axis)] )

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
    return zeros( shape[:-naxes] )

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
    return A.sum(axis)[(slice(None),)*axis+(newaxis,)*(B.ndim-1)] \
         * B.sum(0)[(Ellipsis,)+(newaxis,)*(A.ndim-1-axis)]

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

  return _contract( A, B, 1 ).view( NumericArray )

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

  return fastrepeat(X[newaxis,:],len(Y),axis=0), fastrepeat(Y[:,newaxis],len(X),axis=1)

def meshgrid( *args ):
  'multi-dimensional meshgrid generalisation'

  args = map( asarray, args )
  shape = [ len(args) ] + [ arg.size for arg in args if arg.ndim ]
  grid = empty( shape )
  n = len(shape)-1
  for i, arg in enumerate( args ):
    if arg.ndim:
      n -= 1
      grid[i] = arg[(slice(None),)+(newaxis,)*n]
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
      assert product( s ) == A.shape[i]
      newshape.extend( s )
      i += 1
    elif s == 1:
      newshape.append( A.shape[i] )
      i += 1
    else:
      assert s > 1
      newshape.append( product( A.shape[i:i+s] ) )
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
  s[axis] = newaxis
  return A / norm2( A, axis )[ tuple(s) ]

def cross( v1, v2, axis ):
  'cross product'

  if v1.ndim < v2.ndim:
    v1 = v1[ (newaxis,)*(v2.ndim-v1.ndim) ]
  elif v2.ndim < v1.ndim:
    v2 = v2[ (newaxis,)*(v1.ndim-v2.ndim) ]
  return numpy.cross( v1, v2, axis=axis ).view( NumericArray )

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
  stacked = empty( shape[:axis]+[len(arrays)]+shape[axis:], dtype=float )
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

  diagonalized = zeros( arg.shape + (arg.shape[-1],) )
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
