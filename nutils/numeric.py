# -*- coding: utf8 -*-
#
# Module NUMERIC
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The numeric module provides methods that are lacking from the numpy module.
"""

import numpy, numbers, builtins

_abc = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # indices for einsum

def grid( shape ):
  shape = tuple(shape)
  grid = numpy.empty( (len(shape),)+shape, dtype=int )
  for i, sh in enumerate( shape ):
    grid[i] = numpy.arange(sh)[(slice(None),)+(numpy.newaxis,)*(len(shape)-i-1)]
  return grid

def round( arr ):
  return numpy.round( arr ).astype( int )

def sign( arr ):
  return numpy.sign( arr ).astype( int )

def floor( arr ):
  return numpy.floor( arr ).astype( int )

def ceil( arr ):
  return numpy.ceil( arr ).astype( int )

def overlapping( arr, axis=-1, n=2 ):
  'reinterpret data with overlaps'

  arr = numpy.asarray( arr )
  if axis < 0:
    axis += arr.ndim
  assert 0 <= axis < arr.ndim
  shape = arr.shape[:axis] + (arr.shape[axis]-n+1,n) + arr.shape[axis+1:]
  strides = arr.strides[:axis] + (arr.strides[axis],arr.strides[axis]) + arr.strides[axis+1:]
  overlapping = numpy.lib.stride_tricks.as_strided( arr, shape, strides )
  overlapping.flags.writeable = False
  return overlapping

def normdim( ndim, n ):
  'check bounds and make positive'

  assert isint(ndim) and ndim >= 0, 'ndim must be positive integer, got %s' % ndim
  if n < 0:
    n += ndim
  assert 0 <= n < ndim, 'argument out of bounds: %s not in [0,%s)' % (n,ndim)
  return n

def align( arr, trans, ndim ):
  '''create new array of ndim from arr with axes moved accordin
  to trans'''

  arr = numpy.asarray( arr )
  assert arr.ndim == len(trans)
  if not len(trans):
    return arr[(numpy.newaxis,)*ndim]
  transpose = numpy.empty( ndim, dtype=int )
  trans = numpy.asarray( trans )
  nnew = ndim - len(trans)
  if nnew > 0:
    remaining = numpy.ones( ndim, dtype=bool )
    remaining[trans] = False
    inew, = remaining.nonzero()
    trans = numpy.hstack([ inew, trans ])
    arr = arr[(numpy.newaxis,)*nnew]
  transpose[trans] = numpy.arange(ndim)
  return arr.transpose( transpose )

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

  A = numpy.asarray( A )
  B = numpy.asarray( B )

  maxdim = max( A.ndim, B.ndim )
  m = _abc[maxdim-A.ndim:maxdim]
  n = _abc[maxdim-B.ndim:maxdim]

  axes = sorted( [ normdim(maxdim,axis) ] if isinstance(axis,int) else [ normdim(maxdim,ax) for ax in axis ] )
  o = _abc[:maxdim-len(axes)] if axes == range( maxdim-len(axes), maxdim ) \
    else ''.join( _abc[a+1:b] for a, b in zip( [-1]+axes, axes+[maxdim] ) if a+1 != b )

  return numpy.einsum( '%s,%s->%s' % (m,n,o), A, B )

def contract_fast( A, B, naxes ):
  'contract last n axes'

  assert naxes >= 0
  A = numpy.asarray( A )
  B = numpy.asarray( B )

  maxdim = max( A.ndim, B.ndim )
  m = _abc[maxdim-A.ndim:maxdim]
  n = _abc[maxdim-B.ndim:maxdim]
  o = _abc[:maxdim-naxes]

  return numpy.einsum( '%s,%s->%s' % (m,n,o), A, B )

def dot( A, B, axis=-1 ):
  '''Transform axis of A by contraction with first axis of B and inserting
     remaining axes. Note: with default axis=-1 this leads to multiplication of
     vectors and matrices following linear algebra conventions.'''

  A = numpy.asarray( A )
  B = numpy.asarray( B )

  m = _abc[:A.ndim]
  x = _abc[A.ndim:A.ndim+B.ndim-1]
  n = m[axis] + x
  o = m[:axis] + x
  if axis != -1:
    o += m[axis+1:]

  return numpy.einsum( '%s,%s->%s' % (m,n,o), A, B )

def fastrepeat( A, nrepeat, axis=-1 ):
  'repeat axis by 0stride'

  A = numpy.asarray( A )
  assert A.shape[axis] == 1
  shape = list( A.shape )
  shape[axis] = nrepeat
  strides = list( A.strides )
  strides[axis] = 0
  return numpy.lib.stride_tricks.as_strided( A, shape, strides )

def fastmeshgrid( X, Y ):
  'mesh grid based on fastrepeat'

  return fastrepeat(X[numpy.newaxis,:],len(Y),axis=0), fastrepeat(Y[:,numpy.newaxis],len(X),axis=1)

def meshgrid( *args ):
  'multi-dimensional meshgrid generalisation'

  args = [ numpy.asarray(arg) for arg in args ]
  shape = [ len(args) ] + [ arg.size for arg in args if arg.ndim ]
  dtype = int if all( isintarray(a) for a in args ) else float
  grid = numpy.empty( shape, dtype=dtype )
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
  A = numpy.asarray( A )
  return numpy.lib.stride_tricks.as_strided( A, A.shape + shape, A.strides + (0,)*len(shape) )

def takediag( A ):
  diag = A[...,0] if A.shape[-1] == 1 \
    else A[...,0,:] if A.shape[-2] == 1 \
    else numpy.einsum( '...ii->...i', A )
  diag.flags.writeable = False
  return diag

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
  return A / numpy.linalg.norm( A, axis=axis )[ tuple(s) ]

def cross( v1, v2, axis ):
  'cross product'

  if v1.ndim < v2.ndim:
    v1 = v1[ (numpy.newaxis,)*(v2.ndim-v1.ndim) ]
  elif v2.ndim < v1.ndim:
    v2 = v2[ (numpy.newaxis,)*(v1.ndim-v2.ndim) ]
  return numpy.cross( v1, v2, axis=axis )

def times( A, B ):
  """Times
  Multiply such that shapes are concatenated."""
  A = numpy.asarray( A )
  B = numpy.asarray( B )

  o = _abs[:A.ndim+B.ndim]
  m = o[:A.ndim]
  n = o[A.ndim:]

  return numpy.einsum( '%s,%s->%s' % (m,n,o), A, B )

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

  diagonalized = numpy.zeros(arg.shape + (arg.shape[-1],), arg.dtype)
  diag = numpy.einsum( '...ii->...i', diagonalized )
  assert diag.base is diagonalized
  diag.flags.writeable = True
  diag[:] = arg
  return diagonalized

def eig( A ):
  '''If A has repeated eigenvalues, numpy.linalg.eig sometimes fails to produce
  the complete eigenbasis. This function aims to fix that by identifying the
  problem and completing the basis where necessary.'''

  L, V = numpy.linalg.eig( A )

  # check repeated eigenvalues
  for index in numpy.ndindex( A.shape[:-2] ):
    unique, inverse = numpy.unique( L[index], return_inverse=True )
    if len(unique) < len(inverse): # have repeated eigenvalues
      repeated, = numpy.where( numpy.bincount(inverse) > 1 )
      vectors = V[index].T
      for i in repeated: # indices pointing into unique corresponding to repeated eigenvalues
        where, = numpy.where( inverse == i ) # corresponding eigenvectors
        for j, n in enumerate(where):
          W = vectors[where[:j]]
          vectors[n] -= numpy.dot( numpy.dot( W, vectors[n] ), W ) # gram schmidt orthonormalization
          scale = numpy.linalg.norm(vectors[n])
          if scale < 1e-8: # vectors are near linearly dependent
            u, s, vh = numpy.linalg.svd( A[index] - unique[i] * numpy.eye(len(inverse)) )
            nnz = numpy.argsort( abs(s) )[:len(where)]
            vectors[where] = vh[nnz].conj()
            break
          vectors[n] /= scale

  return L, V

def isbool( a ):
  return isboolarray( a ) and a.ndim == 0 or type(a) == bool

def isboolarray( a ):
  return isinstance( a, numpy.ndarray ) and a.dtype == bool

def isint( a ):
  return isinstance( a, (numbers.Integral,numpy.integer) )

def isnumber( a ):
  return isinstance( a, (numbers.Number,numpy.generic) )

def isintarray( a ):
  return isinstance( a, numpy.ndarray ) and numpy.issubdtype( a.dtype, numpy.integer )

def ortho_complement( A ):
  '''return orthogonal complement to non-square matrix A'''

  m, n = A.shape
  assert n <= m
  if n == 0:
    return numpy.eye( m )
  elif n == m:
    return numpy.empty( (m,0) )
  else:
    u, s, v = numpy.linalg.svd(A)
    return u[:,n:]

asobjvector = lambda v: numpy.array( (None,)+tuple(v), dtype=object )[1:] # 'None' prevents interpretation of objects as axes

def invorder( n ):
  assert n.dtype == int and n.ndim == 1
  ninv = numpy.empty( len(n), dtype=int )
  ninv[n] = numpy.arange( len(n) )
  return ninv

def blockdiag( args ):
  args = [ numpy.asarray(arg) for arg in args ]
  args = [ arg[numpy.newaxis,numpy.newaxis] if arg.ndim == 0 else arg for arg in args ]
  assert all( arg.ndim == 2 for arg in args )
  shapes = numpy.array([ arg.shape for arg in args ])
  blockdiag = numpy.zeros( shapes.sum(0) )
  for arg, (i,j) in zip( args, shapes.cumsum(0) ):
    blockdiag[ i-arg.shape[0]:i, j-arg.shape[1]:j ] = arg
  return blockdiag

def nanjoin( args, axis=0 ):
  args = [ numpy.asarray(arg) for arg in args ]
  assert args
  assert axis >= 0
  shape = list( args[0].shape )
  shape[axis] = sum( arg.shape[axis] for arg in args ) + len(args) - 1
  concat = numpy.empty( shape, dtype=float )
  concat[:] = numpy.nan
  i = 0
  for arg in args:
    j = i + arg.shape[axis]
    concat[(slice(None),)*axis+(slice(i,j),)] = arg
    i = j + 1
  return concat

def broadcasted( f ):
  def wrapped( *args, **kwargs ):
    bcast = broadcast( *args )
    return asobjvector( f(*_args,**kwargs) for _args in bcast ).reshape( bcast.shape )
  return wrapped

def ix( args ):
  'version of :func:`numpy.ix_` that allows for scalars'
  args = tuple( numpy.asarray(arg) for arg in args )
  assert all( 0 <= arg.ndim <= 1 for arg in args )
  idims = numpy.cumsum( [0] + [ arg.ndim for arg in args ] )
  ndims = idims[-1]
  return [ arg.reshape((1,)*idim+(arg.size,)+(1,)*(ndims-idim-1)) for idim, arg in zip( idims, args ) ]

def kronecker( arr, axis, length, pos ):
  axis = normdim( arr.ndim+1, axis )
  kron = numpy.zeros( arr.shape[:axis]+(length,)+arr.shape[axis:], arr.dtype )
  kron[ (slice(None),)*axis + (pos,) ] = arr
  return kron

class Broadcast1D( object ):
  def __init__( self, arg ):
    self.arg = numpy.asarray( arg )
    self.shape = self.arg.shape
    self.size = self.arg.size
  def __iter__( self ):
    return ( (item,) for item in self.arg.flat )

broadcast = lambda *args: numpy.broadcast( *args ) if len(args) > 1 else Broadcast1D( args[0] )

def searchsorted( items, item ):
  '''Find indices where elements should be inserted to maintain order.

  Find the index into a sorted array `items` such that, if `item` were inserted
  before the index, the order of `items` would be preserved.'''

  n = 1
  while (n<<1) <= len(items):
    n <<= 1
  i = 0
  while n:
    j = i|n
    if j <= len(items) and item > items[j-1]:
      i = j
    n >>= 1
  return i

# EXACT OPERATIONS ON FLOATS

def solve_exact( A, *B ):
  A = numpy.asarray( A )
  assert A.ndim == 2
  B = [ numpy.asarray(b) for b in B ]
  assert all( b.shape[0] == A.shape[0] and b.ndim in (1,2) for b in B )
  n = A.shape[1]
  S = [ slice(i,i+b.shape[1]) if b.ndim == 2 else i for b, i in zip( B, numpy.cumsum([0]+[ b[0].size for b in B[:-1] ]) ) ]
  Ab = numpy.concatenate( [ A ] + [ b.reshape(len(b),-1) for b in B ], axis=1 )
  for icol in range(n):
    if not Ab[icol,icol]:
      Ab[icol:] = Ab[icol+numpy.argsort([ abs(v) if v else numpy.inf for v in Ab[icol:,icol] ])]
    Ab[:icol] = Ab[:icol] * Ab[icol,icol] - Ab[:icol,icol,numpy.newaxis] * Ab[icol,:]
    Ab[icol+1:] = Ab[icol+1:] * Ab[icol,icol] - Ab[icol+1:,icol,numpy.newaxis] * Ab[icol,:]
  if Ab[n:].any():
    raise numpy.linalg.LinAlgError( 'linear system has no solution' )
  try:
    Y = div_exact( Ab[:n,n:], numpy.diag( Ab[:n,:n] )[:,numpy.newaxis] )
  except:
    raise numpy.linalg.LinAlgError( 'linear system has no base2 solution' )
  X = [ Y[:,s] for s in S ]
  assert all( numpy.all( dot(A,x) == b ) for (x,b) in zip(X,B) )
  if len(B) == 1:
    X, = X
  return X

def adj_exact( A ):
  '''adj(A) = inv(A) * det(A)'''
  A = numpy.asarray(A)
  assert A.ndim == 2 and A.shape[0] == A.shape[1]
  if len(A) == 1:
    adj = numpy.ones( (1,1) )
  elif len(A) == 2:
    ((a,b),(c,d)) = A
    adj = numpy.array(((d,-b),(-c,a)))
  elif len(A) == 3:
    ((a,b,c),(d,e,f),(g,h,i)) = A
    adj = numpy.array(((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)))
  else:
    raise NotImplementedError( 'shape={}'.format(A.shape) )
  return adj

def det_exact( A ):
  # for some reason, numpy.linalg.det suffers from rounding errors
  A = numpy.asarray( A )
  assert A.ndim == 2 and A.shape[0] == A.shape[1]
  if len(A) == 1:
    det = A[0,0]
  elif len(A) == 2:
    ((a,b),(c,d)) = A
    det = a*d - b*c
  elif len(A) == 3:
    ((a,b,c),(d,e,f),(g,h,i)) = A
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
  else:
    raise NotImplementedError( 'shape=' + str(A.shape) )
  return det

def div_exact( A, B ):
  Am, Ae = fextract( A )
  Bm, Be = fextract( B )
  assert Bm.all(), 'division by zero'
  Cm, rem = divmod( Am, Bm )
  assert not rem.any(), 'indivisible arguments'
  Ce = Ae - Be
  return fconstruct( Cm, Ce )

def inv_exact( A ):
  A = numpy.asarray( A )
  return div_exact( adj_exact(A), det_exact(A) )

def ext( A ):
  """Exterior
  For array of shape (n,n-1) return n-vector ex such that ex.array = 0 and
  det(arr;ex) = ex.ex"""
  A = numpy.asarray(A)
  assert A.ndim == 2 and A.shape[0] == A.shape[1]+1
  if len(A) == 1:
    ext = numpy.ones( 1 )
  elif len(A) == 2:
    ((a,),(b,)) = A
    ext = numpy.array((b,-a))
  elif len(A) == 3:
    ((a,b),(c,d),(e,f)) = A
    ext = numpy.array((c*f-e*d,e*b-a*f,a*d-c*b))
  else:
    raise NotImplementedError( 'shape=%s' % (A.shape,) )
  return ext

def fextract( A, single=False ):
  A = numpy.asarray( A, dtype=numpy.float64 )
  bits = A.view( numpy.int64 ).ravel()
  nz = ( bits & 0x7fffffffffffffff ).astype(bool)
  if not nz.any():
    return ( numpy.zeros( A.shape, dtype=int ), 0 ) if single else numpy.zeros( (2,)+A.shape, dtype=int )
  bits = bits[nz]
  sign = numpy.sign( bits )
  exponent = ( (bits>>52) & 0x7ff ) - 1075
  mantissa = 0x10000000000000 | ( bits & 0xfffffffffffff )
  # from here on A.flat[nz] == sign * mantissa * 2**exponent
  for shift in 32, 16, 8, 4, 2, 1:
    I = mantissa & ((1<<shift)-1) == 0
    if I.any():
      mantissa[I] >>= shift
      exponent[I] += shift
  if not single:
    retval = numpy.zeros( (2,)+A.shape, dtype=int )
    retval.reshape(2,-1)[:,nz] = sign * mantissa, exponent
    return retval
  minexp = numpy.min( exponent )
  shift = exponent - minexp
  assert not numpy.any( mantissa >> (63-shift) )
  fullmantissa = numpy.zeros( A.shape, dtype=int )
  fullmantissa.flat[nz] = sign * (mantissa << shift)
  return fullmantissa, minexp

def fconstruct( m, e ):
  return numpy.asarray( m ) * numpy.power( 2., e )

def fstr( A ):
  if A.ndim:
    return '[{}]'.format( ','.join( fstr(a) for a in A ) )
  mantissa, exp = fextract( A )
  return str( mantissa << exp ) if exp >= 0 else '{}/{}'.format( mantissa, 1<<(-exp) )

def fhex( A ):
  if A.ndim:
    return '[{}]'.format( ','.join( fhex(a) for a in A ) )
  mantissa, exp = fextract( A )
  div, mod = divmod( exp, 4 )
  h = '{:+x}'.format( mantissa << mod )[1:]
  return ( '-' if mantissa < 0 else '' ) + '0x' + ( h.ljust( len(h)+div, '0' ) if div >= 0 else ( h[:div] or '0' ) + '.' + h[div:].rjust( -div, '0' ) )

def power( a, b ):
  a = numpy.asarray( a )
  b = numpy.asarray( b )
  if a.dtype == int and b.dtype == int:
    b = b.astype( float )
  return numpy.power( a, b )

def serialized(array, nsig, ndec):
  if array.ndim > 0:
    return '[{}]'.format(','.join(serialized(a, nsig, ndec) for a in array))
  if not numpy.isfinite(array): # nan, inf
    return str(array)
  a = builtins.round(float(array) * 10**ndec)
  if a == 0:
    return '0'
  while abs(a) >= 10**nsig:
    a //= 10
    ndec -= 1
  return '{}e{}'.format(a, -ndec)

def encode64(array, nsig, ndec):
  import zlib, binascii
  assert isinstance(array, numpy.ndarray) and array.dtype == float
  binary = zlib.compress('{},{},{}'.format(nsig, ndec, serialized(array, nsig, ndec)).encode(), 9)
  data = binascii.b2a_base64(binary).decode().rstrip()
  assert_allclose64(array, data)
  return data

def decode64(data):
  import zlib, binascii
  serialized = zlib.decompress(binascii.a2b_base64(data))
  nsig, ndec, array = eval(serialized, numpy.__dict__)
  return nsig, ndec, numpy.array(array, dtype=float)

def assert_allclose64(actual, data=None):
  try:
    nsig, ndec, desired = decode64(data)
  except Exception as e:
    status = str(e)
    nsig = 4
    ndec = 15
  else:
    try:
      numpy.testing.assert_allclose(actual, desired, atol=1.5*10**-ndec, rtol=10**(1-nsig))
    except Exception as e:
      status = str(e)
    else:
      return
  status += '\n\nIf this is expected, use the following base64 string to test up to nsig={}, ndec={}:'.format(nsig, ndec)
  data = encode64(actual, nsig=nsig, ndec=ndec)
  while data:
    status += '\n{!r}'.format(data[:80])
    data = data[80:]
  raise Exception(status)

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
