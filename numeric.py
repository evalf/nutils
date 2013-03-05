import numpy, _numeric

addsorted = _numeric.addsorted

def align( arr, trans, ndim ):
  '''create new array of ndim from arr with axes moved accordin
  to trans'''

  # as_strided will check validity of trans
  arr = numpy.asarray( arr )
  assert len(trans) == arr.ndim
  strides = numpy.zeros( ndim, dtype=int )
  strides[trans] = arr.strides
  shape = numpy.ones( ndim, dtype=int )
  shape[trans] = arr.shape
  return numpy.lib.stride_tricks.as_strided( arr, shape, strides )

def expand( arr, *shape ):
  'expand'

  shape = list( arr.shape )
  for i, sh in enumerate( shape ):
    if sh != None:
      if shape[i-len(shape)] == 1:
        shape[i-len(shape)] = sh
      else:
        assert shape[i-len(shape)] == sh
  expanded = numpy.empty( shape )
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

def appendaxes( A, shape ):
  'append axes by 0stride'

  shape = (shape,) if isinstance(shape,int) else tuple(shape)
  A = numpy.asarray( A )
  return numpy.lib.stride_tricks.as_strided( A, A.shape + shape, A.strides + (0,)*len(shape) )

def takediag( A, ax1, ax2 ):
  if ax1 < 0:
    ax1 += A.ndim
  if ax2 < 0:
    ax2 += A.ndim
  ax1, ax2 = sorted( [ax1,ax2] )
  assert 0 <= ax1 < ax2 < A.ndim
  shape = list(A.shape)
  n2 = shape.pop(ax2)
  n1 = shape.pop(ax1)
  strides = list(A.strides)
  s2 = strides.pop(ax2)
  s1 = strides.pop(ax1)
  if n1 == 1:
    n = n2
    s = s2
  elif n2 == 1:
    n = n1
    s = s1
  else:
    assert n1 == n2
    n = n1
    s = s1 + s2
  shape.append( n )
  strides.append( s )
  return numpy.lib.stride_tricks.as_strided( A, shape, strides )

def inv( arr, axes ):
  'linearized inverse'

  L = map( numpy.arange, arr.shape )

  ax1, ax2 = sorted( ax + arr.ndim if ax < 0 else ax for ax in axes ) # ax2 > ax1
  L.pop( ax2 )
  L.pop( ax1 )

  indices = list( numpy.ix_( *L ) )
  indices.insert( ax1, slice(None) )
  indices.insert( ax2, slice(None) )

  invarr = numpy.empty_like( arr )
  for index in numpy.broadcast( *indices ):
    invarr[index] = numpy.linalg.inv( arr[index] )

  return invarr

def det( A, ax1, ax2 ):
  'determinant'

  assert isinstance( A, numpy.ndarray )
  ax1, ax2 = sorted( ax + A.ndim if ax < 0 else ax for ax in (ax1,ax2) ) # ax2 > ax1
  assert A.shape[ax1] == A.shape[ax2]
  T = range(A.ndim)
  T.pop(ax2)
  T.pop(ax1)
  T.extend([ax1,ax2])
  A = A.transpose( T )
  if A.shape[-1] == 1:
    det = A[...,0,0]
  elif A.shape[-1] == 2:
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
  else:
    det = numpy.empty( A.shape[:-2] )
    for I in numpy.broadcast( *numpy.ix_( *[ range(n) for n in A.shape[:-2] ] ) ) if A.ndim > 3 else range( A.shape[0] ):
      det[I] = numpy.linalg.det( A[I] )
  return numpy.asarray( det ).view( A.__class__ )

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

  assert v1.ndim == v2.ndim
  return numpy.cross( v1, v2, axis=axis )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
