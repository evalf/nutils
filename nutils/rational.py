# -*- coding: utf8 -*-
#
# Module RATIONAL
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The rational module.
"""

from __future__ import print_function, division
import numpy


## OPERATIONS

def _logical( op, A, B ):
  A = asarray( A )
  B = asarray( B )
  if not isrational( A ) or not isrational( B ):
    return op( A.astype(float), B.astype(float) )
  common = gcd( A.denom, B.denom )
  return op( A.numer * (B.denom//common), B.numer * (A.denom//common) )

greater       = lambda A, B: _logical( numpy.greater,       A, B )
greater_equal = lambda A, B: _logical( numpy.greater_equal, A, B )
less          = lambda A, B: _logical( numpy.less,          A, B )
less_equal    = lambda A, B: _logical( numpy.less_equal,    A, B )
equal         = lambda A, B: _logical( numpy.equal,         A, B )
not_equal     = lambda A, B: _logical( numpy.not_equal,     A, B )

def _addsub( op, A, B ):
  A = asarray( A )
  B = asarray( B )
  if not isrational( A ) or not isrational( B ):
    return op( A.astype(float), B.astype(float) )
  common = gcd( A.denom, B.denom )
  return Rational( op( A.numer * (B.denom//common), B.numer * (A.denom//common) ), A.denom * (B.denom//common) )

add      = lambda A, B: _addsub( numpy.add,      A, B )
subtract = lambda A, B: _addsub( numpy.subtract, A, B )

def multiply( A, B ):
  A = asarray( A )
  B = asarray( B )
  if not isrational( A ) or not isrational( B ):
    return A.astype(float) * B.astype(float)
  return Rational( A.numer * B.numer, A.denom * B.denom )

def divide( A, B ):
  A = asarray( A )
  B = asarray( B )
  if not isrational( A ) or not isrational( B ):
    return A.astype(float) / B.astype(float)
  assert B.ndim == 0, 'only scalar division supported for now'
  sign = 1 if B.numer > 0 else -1
  return Rational( sign * A.numer * B.denom, sign * A.denom * B.numer )

def power( A, n ):
  assert isint( n ) and numpy.ndim(n) == 0
  A = asarray( A )
  if not isrational( A ):
    return numpy.power( A, n )
  return Rational( A.numer**n, A.denom**n ) if n > 1 \
    else A if n == 1 \
    else ones( A.shape ) if n == 0 \
    else 1 / power( A, -n )

def _unary( op, A ):
  A = asarray( A )
  if not isrational( A ):
    return op( A )
  return Rational( op(A.numer), A.denom, isfactored=True )

negative = lambda A: _unary( numpy.negative, A )
transpose = lambda A, axes=None: _unary( lambda A: numpy.transpose(A,axes), A )
absolute = lambda A: _unary( numpy.absolute, A )

## RATIONAL CLASS

class Rational( object ):

  __array_priority__ = 1

  def __init__( self, numer, denom=1, isfactored=False ):
    assert isint(denom) and numpy.ndim(denom) == 0 and denom > 0
    self.denom = numpy.int64( denom )
    self.numer = numpy.array( numer, dtype=numpy.int64 )
    if self.denom != 1 and not isfactored:
      common = gcd( self.denom, *self.numer.flat )
      if common != 1:
        self.numer //= common
        self.denom //= common
    self.numer.flags.writeable = False

  def __iter__( self ):
    for array in self.numer:
      yield Rational( array, self.denom )

  def __nonzero__( self ):
    return bool(self.numer)

  __bool__ = __nonzero__ # python3

  def __getitem__( self, item ):
    return Rational( self.numer[item], self.denom )

  def __int__( self ):
    assert self.ndim == 0 and self.denom == 1
    return int(self.numer)

  def __float__( self ):
    assert self.ndim == 0
    return float(self.numer) / self.denom

  def astype( self, tp ):
    if tp == int:
      assert self.denom == 1
      return self.numer
    assert tp == float
    return self.numer / float(self.denom)

  @property
  def size( self ):
    return self.numer.size

  @property
  def ndim( self ):
    return self.numer.ndim

  @property
  def shape( self ):
    return self.numer.shape

  def __len__( self ):
    return len(self.numer)

  T = property( transpose )
  __neg__ = negative
  __gt__ = greater
  __ge__ = greater_equal
  __lt__ = less
  __le__ = less_equal
  __eq__ = equal
  __ne__ = not_equal
  __add__ = add
  __radd__ = add
  __sub__ = subtract
  __rsub__ = lambda self, other: subtract( other, self )
  __mul__ = multiply
  __rmul__ = multiply
  __div__ = divide
  __truediv__ = divide
  __rdiv__ = lambda self, other: divide( other, self )
  __rtruediv__ = __rdiv__
  __pow__ = power
  __abs__ = absolute

  def __str__( self ):
    return '%s/%s' % ( str(self.numer.tolist()).replace(' ',''), self.denom )

  def __hash__( self ):
    raise TypeError( "unhashable type: 'Rational'" )
    # Actually being immutable there is no reason why Rational should be
    # unhashable, except if we implement __hash__ we must also implement __eq__
    # such that it returns True on equality. Sounds fair enough, except numpy
    # decided differently. We choose to cripple our object for consistency.


## UTILITY FUNCTIONS

def gcd( *numbers ):
  uniqdesc = numpy.unique( numpy.abs(numbers) )[::-1].tolist() # unique descending
  if uniqdesc[-1] == 0:
    uniqdesc.pop() # ignore zero
  gcd = uniqdesc.pop()
  while uniqdesc and gcd > 1:
    n = uniqdesc.pop()
    while n: # Euclid's algorithm
      gcd, n = n, gcd % n
  return gcd

def det( array ):
  array = asarray( array )
  if not isrational( array ):
    return numpy.linalg.det( array )
  assert array.ndim == 2 and array.shape[0] == array.shape[1]
  zeros = array.numer == 0
  if zeros.any():
    nzcols = zeros.sum( axis=0 )
    nzrows = zeros.sum( axis=1 )
    if max(nzcols) > max(nzrows):
      j = numpy.argmax( nzcols )
      IJ = [ (i,j) for i in (~zeros[:,j]).nonzero()[0] ]
    else:
      i = numpy.argmax( nzrows )
      IJ = [ (i,j) for j in (~zeros[i,:]).nonzero()[0] ]
    n = numpy.arange( len(array) )
    # laplace's formula
    return sum( det(array[numpy.ix_(n!=i,n!=j)]) * ( array[i,j] if (i+j)%2==0 else -array[i,j] )
      for i, j in IJ ) if IJ else zero
  if len(array) == 1:
    retval = array[0,0]
  elif len(array) == 2:
    ((a,b),(c,d)) = array.numer
    retval = Rational( a*d - b*c, array.denom**2 )
  elif len(array) == 3:
    ((a,b,c),(d,e,f),(g,h,i)) = array.numer
    retval = Rational( a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h, array.denom**3 )
  else:
    raise NotImplementedError( 'shape=' + str(array.shape) )
  return retval

def invdet( array ):
  '''invdet(array) = inv(array) * det(array)'''
  array = asarray(array)
  if not isrational( array ):
    return numpy.linalg.inv(array) * numpy.linalg.det(array)
  if array.shape == (1,1):
    invdet = ones( (1,1) )
  elif array.shape == (2,2):
    ((a,b),(c,d)) = array.numer
    invdet = Rational( ((d,-b),(-c,a)), array.denom, isfactored=True )
  elif array.shape == (3,3):
    ((a,b,c),(d,e,f),(g,h,i)) = array.numer
    invdet = Rational( ((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)), array.denom**2 )
  else:
    raise NotImplementedError( 'shape={}'.format(array.shape) )
  return invdet
  
def inv( array ):
  array = asarray(array)
  if not isrational( array ):
    return numpy.linalg.inv( array )
  return invdet( array ) / det( array )

def ext( array ):
  """Exterior
  For array of shape (n,n-1) return n-vector ex such that ex.array = 0 and
  det(arr;ex) = ex.ex"""
  array = asarray(array)
  assert isrational(array)
  assert array.ndim == 2 and array.shape[0] == array.shape[1]+1
  zeros = ( array.numer == 0 ).all( axis=1 )
  if len(array) == 1:
    ext = ones( 1 )
  elif len(array) == 2:
    ((a,),(b,)) = array.numer
    ext = Rational( (b,-a), array.denom, isfactored=True )
  elif any(zeros):
    alpha = det(array[~zeros])
    (i,), = zeros.nonzero()
    v = zeros * ( alpha.numer if i%2==0 else -alpha.numer )
    ext = Rational( v, alpha.denom, isfactored=True )
  elif len(array) == 3:
    ((a,b),(c,d),(e,f)) = array.numer
    ext = Rational( (c*f-e*d,e*b-a*f,a*d-c*b), array.denom**2 )
  else:
    raise NotImplementedError( 'shape=%s' % (array.shape,) )
  # VERIFY
  Av = concatenate( [ext[:,numpy.newaxis],array], axis=1 )
  assert equal( dot( ext, array ), 0 ).all()
  assert equal( det(Av), dot(ext,ext) ).all()
  return ext

isint = lambda a: numpy.issubdtype( a.dtype if isinstance(a,numpy.ndarray) else type(a), numpy.integer )
isrational = lambda arr: isinstance( arr, Rational )

def asarray( arr ):
  if not isinstance( arr, Rational ):
    if not isinstance( arr, numpy.ndarray ):
      arr = numpy.asarray( arr )
      if not arr.size:
        arr = numpy.zeros( arr.shape, dtype=int )
    if numpy.issubdtype( arr.dtype, numpy.integer ):
      arr = Rational( arr )
  return arr

def frac( a, b ):
  return asarray(a) / asarray(b)

def dot( A, B ):
  A = asarray( A )
  B = asarray( B )
  if not isrational( A ) or not isrational( B ):
    return numpy.dot( A.astype(float), B.astype(float) )
  return Rational( numpy.dot( A.numer, B.numer ), A.denom * B.denom )

def eye( ndims ):
  return Rational( numpy.eye(ndims,dtype=int) )

def zeros( shape ):
  return Rational( numpy.zeros(shape,dtype=int) )

def ones( shape ):
  return Rational( numpy.ones(shape,dtype=int) )

def concatenate( args, axis=0 ):
  args = [ asarray(arg) for arg in args ]
  if not all( isrational(arg) for arg in args ):
    return numpy.concatenate( [ arg.astype(float) for arg in args ], axis=axis )
  arg1, arg2 = args
  return Rational( numpy.concatenate([ arg1.numer * arg2.denom, arg2.numer * arg1.denom ], axis=axis ), arg1.denom * arg2.denom )

def blockdiag( args ):
  args = [ asarray(arg) for arg in args ]
  assert all( isrational(arg) for arg in args )
  arg1, arg2 = args
  assert arg1.ndim == arg2.ndim == 2
  blockdiag = numpy.zeros( (arg1.shape[0]+arg2.shape[0],arg1.shape[1]+arg2.shape[1]), dtype=int )
  blockdiag[:arg1.shape[0],:arg1.shape[1]] = arg1.numer * arg2.denom
  blockdiag[arg1.shape[0]:,arg1.shape[1]:] = arg2.numer * arg1.denom
  return Rational( blockdiag, arg1.denom * arg2.denom )

def round( array, denom=1 ):
  array = asarray( array )
  if isrational( array ):
    return array
  numer = array * denom
  return Rational( ( numer - numpy.less(numer,0) + .5 ).astype( int ), denom )

def solve( A, *B ):
  A = asarray( A )
  assert A.ndim == 2
  B = [ asarray(b) for b in B ]
  assert all( b.shape[0] == A.shape[0] and b.ndim in (1,2) for b in B )
  S = [ slice(i,i+b.shape[1]) if b.ndim == 2 else i for b, i in zip( B, numpy.cumsum([0]+[ b[0].size for b in B[:-1] ]) ) ]
  if not isrational( A ) or not all( isrational( b ) for b in B ):
    A = A.astype(float)
    B = numpy.concatenate( [ b.astype(float).reshape(len(b),-1) for b in B ], axis=1 )
    Y = numpy.linalg.solve( A, B )
    X = [ Y[:,s] for s in S ]
  else:
    Ab = numpy.concatenate( [ A.numer ] + [ b.numer.reshape(len(b),-1) for b in B ], axis=1 )
    n = A.shape[1]
    for icol in range(n):
      if not Ab[icol,icol]:
        Ab[icol:] = Ab[icol+numpy.argsort([ abs(v) if v else numpy.inf for v in Ab[icol:,icol] ])]
      Ab[:icol] = Ab[:icol] * Ab[icol,icol] - Ab[:icol,icol,numpy.newaxis] * Ab[icol,:]
      Ab[icol+1:] = Ab[icol+1:] * Ab[icol,icol] - Ab[icol+1:,icol,numpy.newaxis] * Ab[icol,:]
    if Ab[n:].any():
      raise numpy.linalg.LinAlgError( 'linear system has no solution' )
    w = numpy.diag( Ab[:n,:n] )
    denom = gcd(*w)
    numer = Ab[:n,n:] * ( denom // w[:,numpy.newaxis] )
    X = [ Rational( numer[:,s] * A.denom, denom * b.denom ) for (s,b) in zip(S,B) ]
    assert not any( ( dot( A, x ) - b ).numer.any() for (x,b) in zip(X,B) )
  if len(B) == 1:
    X, = X
  return X


zero = Rational( 0 )
unit = Rational( 1 )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
