import numpy, warnings

try:
  import _numeric
except:
  warnings.warn( '''Failed to load _numeric module.
  Falling back on equivalent python implementation. THIS
  MAY SEVERELY IMPACT PERFORMANCE! Pleace compile the C
  extensions by running 'make' in the finity directory.''', stacklevel=2 )
  class _numeric:
    @staticmethod
    def contract( A, B, axes ):
      assert A.shape == B.shape and axes > 0
      return ((A*B).reshape(A.shape[:-axes]+(-1,))).sum(-1)

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
  arr = numpy.asarray( arr )
  trans = numpy.asarray( trans, dtype=int )
  assert len(trans) == arr.ndim
  strides = numpy.zeros( ndim, dtype=int )
  strides[trans] = arr.strides
  shape = numpy.ones( ndim, dtype=int )
  shape[trans] = arr.shape
  return numpy.lib.stride_tricks.as_strided( arr, shape, strides )

def get( arr, axis, item ):
  'take single item from array axis'

  arr = numpy.asarray( arr )
  axis = normdim( arr.ndim, axis )
  return arr[ (slice(None),) * axis + (item,) ]

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

  A = numpy.asarray( A, dtype=float )
  B = numpy.asarray( B, dtype=float )

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

  A = numpy.lib.stride_tricks.as_strided( A, shape, Astrides )
  B = numpy.lib.stride_tricks.as_strided( B, shape, Bstrides )

  if not A.size:
    return numpy.zeros( A.shape[:-len(axis)] )

  return _numeric.contract( A, B, len(axis) )

def contract_fast( A, B, naxes ):
  'contract last n axes'

  A = numpy.asarray( A, dtype=float )
  B = numpy.asarray( B, dtype=float )

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

  A = numpy.lib.stride_tricks.as_strided( A, shape, Astrides )
  B = numpy.lib.stride_tricks.as_strided( B, shape, Bstrides )

  if not A.size:
    return numpy.zeros( shape[:-naxes] )

  return _numeric.contract( A, B, naxes )

def dot( A, B, axis=-1 ):
  '''Transform axis of A by contraction with first axis of B and inserting
     remaining axes. Note: with default axis=1 this leads to multiplication of
     vectors and matrices following linear algebra conventions.'''

  A = numpy.asarray( A, dtype=float )
  B = numpy.asarray( B, dtype=float )

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
    A = numpy.lib.stride_tricks.as_strided( A, shape, Astrides )

  if A.ndim > 1:
    Bstrides = (0,) * axis + B.strides[1:] + (0,) * (A.ndim-B.ndim-axis) + B.strides[:1]
    B = numpy.lib.stride_tricks.as_strided( B, A.shape, Bstrides )

  if not A.size:
    return numpy.zeros( A.shape[:-1] )

  return _numeric.contract( A, B, 1 )

def fastrepeat( A, nrepeat, axis=-1 ):
  'repeat axis by 0stride'

  assert A.shape[axis] == 1
  shape = list( A.shape )
  shape[axis] = nrepeat
  strides = list( A.strides )
  strides[axis] = 0
  return numpy.lib.stride_tricks.as_strided( A, shape, strides )

def fastmeshgrid( X, Y ):
  'mesh grid based on fastrepeat'

  return fastrepeat(X[numpy.newaxis,:],len(Y),axis=0), fastrepeat(Y[:,numpy.newaxis],len(X),axis=1)

def appendaxes( A, shape ):
  'append axes by 0stride'

  shape = (shape,) if isinstance(shape,int) else tuple(shape)
  A = numpy.asarray( A )
  return numpy.lib.stride_tricks.as_strided( A, A.shape + shape, A.strides + (0,)*len(shape) )

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
  return numpy.lib.stride_tricks.as_strided( A, shape, strides )

def inverse( A ):
  'linearized inverse'

  assert isinstance( A, numpy.ndarray )
  assert A.shape[-2] == A.shape[-1]
  if A.shape[-1] == 1:
    Ainv = numpy.reciprocal( A )
  elif A.shape[-1] == 2:
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
    Ainv = numpy.empty( A.shape )
    numpy.divide( A[...,1,1],  det, Ainv[...,0,0] )
    numpy.divide( A[...,0,0],  det, Ainv[...,1,1] )
    numpy.divide( A[...,1,0], -det, Ainv[...,1,0] )
    numpy.divide( A[...,0,1], -det, Ainv[...,0,1] )
  else:
    Ainv = numpy.empty( A.shape )
    for I in numpy.lib.index_tricks.ndindex( A.shape[:-2] ):
      Ainv[I] = numpy.linalg.inv( A[I] )
  return Ainv

def determinant( A ):
  'determinant'

  assert isinstance( A, numpy.ndarray )
  assert A.shape[-2] == A.shape[-1]
  if A.shape[-1] == 1:
    det = A[...,0,0]
  elif A.shape[-1] == 2:
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
  else:
    det = numpy.empty( A.shape[:-2] )
    for I in numpy.lib.index_tricks.ndindex( A.shape[:-2] ):
      det[I] = numpy.linalg.det( A[I] )
  return det

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

def norm2( A, axis=-1 ):
  'L2 norm over specified axis'

  return numpy.asarray( numpy.sqrt( contract( A, A, axis ) ) )

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

  arrays = [ numpy.asarray(array,dtype=float) for array in arrays ]
  shape = [1] * max( array.ndim for array in arrays )
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

  arg = numpy.asarray(arg)
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
