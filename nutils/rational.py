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

from __future__ import division
from . import cache, log
import numpy, collections


## PRIMES

masks = [ 0, 0 ] # primes per number encoded as bitmask

class Primes( object ):

  def __init__( self ):
    self.__primes = [] # all prime numbers found thus far
    self.__masks = collections.deque([ 0 ]) # primes per number in progress

  def __iter__( self ):
    for prime in self.__primes:
      yield prime
    while True:
      mask = self.__masks.popleft()
      if mask:
        masks.append( mask )
        iprime = 0
        while mask:
          if mask & 1:
            self.__mark( iprime )
          mask >>= 1
          iprime += 1
      else: # found prime
        iprime = len(self.__primes)
        prime = len(masks)
        self.__primes.append( prime )
        self.__mark( iprime )
        masks.append( 1 << iprime )
        yield prime

  def __mark( self, iprime ):
    i = self.__primes[iprime] - 1
    if i < len( self.__masks ):
      self.__masks[ i ] |= 1 << iprime
    else:
      self.__masks.extend( [0] * ( i - len(self.__masks) ) + [ 1 << iprime ] )

primes = Primes()


## SCALAR

class Scalar( object ):

  __array_priority__ = 1

  def __init__( self, init ):
    while len(init) and init[-1] == 0:
      init = init[:-1]
    if isinstance( init, numpy.ndarray ):
      assert init.dtype in (int,long)
    else:
      assert all( isinstance( n, (int,long) ) for n in init )
    self.__n = tuple( init )

  def __hash__( self ):
    return hash( self.__n )

  def __eq__( self, other ):
    return self is other or isscalar(other) and self.__n == other.__n

  def __op( self, other, op ):
    assert self.__n or other.__n, 'trivial cases should be handled by caller'
    return Scalar( numpy.hstack([
      op( self.__n, other.__n[:len(self.__n)] ),
      op( 0, other.__n[len(self.__n):] ) ]) if len(self.__n) < len(other.__n)
                else numpy.hstack([
      op( self.__n[:len(other.__n)], other.__n ),
      op( self.__n[len(other.__n):], 0 ) ]) if len(self.__n) > len(other.__n)
                else
      op( self.__n, other.__n ) )

  def __mul__( self, other ):
    if not isexact( other ):
      return float(self) * other
    other = asrational( other )
    if isarray( other ):
      return other * self
    assert isscalar( other )
    return other if not self.__n \
      else self if not other.__n \
      else self.__op( other, numpy.add )

  def __neg__( self ):
    raise TypeError, 'rational cannot be negated'

  def __add__( self, other ):
    if not isexact( other ):
      return float(self) + other
    other = asrational( other )
    if isarray( other ):
      return other + self
    raise NotImplementedError

  def __sub__( self, other ):
    if not isexact( other ):
      return float(self) - other
    other = asrational( other )
    if isarray( other ):
      return other - self
    raise NotImplementedError

  def __div__( self, other ):
    if not isexact( other ):
      return float(self) / other
    other = asrational( other )
    assert isscalar( other )
    return self if not other.__n \
      else Scalar( numpy.negative(other.__n) ) if not self.__n \
      else self.__op( other, numpy.subtract )

  def __rdiv__( self, other ):
    return Scalar( numpy.negative(self.__n) ) * other

  __rmul__ = __mul__
  __radd__ = __add__
  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def __pow__( self, n ):
    assert isinstance( n, (int,long) )
    if n == 0 or not self.__n:
      return unit
    return Scalar( numpy.multiply( self.__n, n ) )

  def __float__( self ):
    numer, denom = self.frac
    return numer / denom
  
  def __int__( self ):
    numer, denom = self.frac
    assert denom == 1
    return numer
  
  def __str__( self ):
    return '%d/%d' % self.frac

  def __repr__( self ):
    return 'Scalar(%s)' % str(self)

  def factor( self ):
    return zip( primes, self.__n )

  @cache.property
  def frac( self ):
    numer = denom = 1
    for p, n in zip( primes, self.__n ):
      if n > 0:
        numer *= p**+n
      elif n < 0:
        denom *= p**-n
    return numer, denom

  @property
  def numer( self ):
    return Scalar( self.__n and numpy.maximum( 0, self.__n ) )
  
  @property
  def denom( self ):
    return Scalar( self.__n and numpy.minimum( 0, self.__n ) )

  def gcd( self, other ):
    """a.gcd(b)

    Return largest rational c such that a/c and b/c are integer."""

    other = asscalar( other )
    if self == other:
      return self
    d = len(other.__n) - len(self.__n)
    return Scalar( numpy.minimum( self.__n + (0,)*d, other.__n + (0,)*-d ) )


unit = Scalar(())
half = Scalar((-1,))


## ARRAY

class Array( object ):

  __array_priority__ = 1

  def __init__( self, array, factor=unit, isfactored=False ):
    assert isscalar( factor )
    array = numpy.asarray( array )
    assert array.dtype in (int,long)
    self.__array, self.__factor = ( array, factor ) if isfactored else gcd( array, factor )
    self.__nonzero = array.any()
    assert self.__nonzero or self.__factor == unit

  def __iter__( self ):
    for array in self.__array:
      yield Array( array, self.__factor )

  def __getitem__( self, item ):
    return Array( self.__array[item], self.__factor )

  @property
  def ndim( self ):
    return self.__array.ndim

  @property
  def shape( self ):
    return self.__array.shape

  @property
  def T( self ):
    return Array( self.__array.T, self.__factor, isfactored=True )

  def __len__( self ):
    return len(self.__array)

  @property
  def __cmpdata( self ):
    return self.__array.shape, tuple(self.__array.flat), self.__factor

  def __hash__( self ):
    return hash( self.__cmpdata )

  def __eq__( self, other ):
    return self is other or isarray(other) and self.__cmpdata == other.__cmpdata

  def __neg__( self ):
    return Array( -self.__array, self.__factor, isfactored=True )

  def __add__( self, other ):
    if not isexact( other ):
      return self.__array * float(self.__factor) + other
    other = asrational( other )
    assert isarray( other )
    common = self.__factor.gcd( other.__factor )
    return Array( self.__array * int(self.__factor/common)
              + other.__array * int(other.__factor/common), common )

  def __sub__( self, other ):
    if not isexact( other ):
      return self.__array * float(self.__factor) - other
    other = asrational( other )
    assert isarray( other )
    common = self.__factor.gcd( other.__factor )
    return Array( self.__array * int(self.__factor/common)
              - other.__array * int(other.__factor/common), common )

  def __mul__( self, other ):
    if not isexact( other ):
      return self.__array * ( other * float(self.__factor) )
    other = asrational( other )
    if isscalar( other ):
      if not self.__nonzero:
        return self # 0 * a = 0
      return Array( self.__array, self.__factor*other, isfactored=True )
    raise NotImplementedError

  def __div__( self, other ):
    if not isexact( other ):
      return self.__array * ( float(self.__factor) / other )
    other = asrational( other )
    if isscalar( other ):
      return Array( self.__array, self.__factor/other, isfactored=True ) if self.__nonzero else self
    raise NotImplementedError

  def __rdiv__( self, other ):
    if not isexact( other ):
      return ( other / float(self.__factor) ) / self.__array
    raise NotImplementedError

  __rmul__ = __mul__
  __radd__ = __add__
  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def decompose( self ):
    return self.__array.copy(), self.__factor

  @cache.property
  def frac( self ):
    numbers, indices = numpy.unique( abs(self.__array.ravel()), return_inverse=True )
    frac = numpy.array(
      [ (0,1) ] + [ ( factor(n) * self.__factor ).frac for n in numbers[1:] ] if numbers[0] == 0
             else [ ( factor(n) * self.__factor ).frac for n in numbers ]
      ).T[:,indices].reshape( (2,)+self.shape )
    frac[0] *= numpy.sign( self.__array )
    return frac

  def __str__( self ):
    numer, denom = self.__factor.frac
    s = _a2s( self.__array * numer )
    return s if denom == 1 else '%s/%s' % ( s, denom )

_a2s = lambda array: '[%s]' % ','.join( _a2s(a) for a in array ) if isinstance(array,numpy.ndarray) else str(array)


## UTILITY FUNCTIONS

def det( array ):
  '''returns determinant as Scalar (must be >0)'''
  assert isarray(array)
  if array.shape == (1,1):
    ((a,),), scale = array.decompose()
    assert a == 1
    det = scale
  elif array.shape == (2,2):
    ((a,b),(c,d)), scale = array.decompose()
    det = factor( a*d - b*c ) * scale**2
  elif array.shape == (3,3):
    ((a,b,c),(d,e,f),(g,h,i)), scale = array.decompose()
    det = factor( a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h ) * scale**3
  else:
    raise NotImplementedError, 'shape=' + str(array.shape)
  return det

def invdet( array ):
  '''invdet(array) = inv(array) * det(array)'''
  assert isarray(array)
  if array.shape == (1,1):
    inv = ones( (1,1) )
  elif array.shape == (2,2):
    ((a,b),(c,d)), scale = array.decompose()
    inv = Array( ((d,-b),(-c,a)), scale, isfactored=True )
  elif array.shape == (3,3):
    ((a,b,c),(d,e,f),(g,h,i)), scale = array.decompose()
    inv = Array( ((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)), scale**2 )
  else:
    raise NotImplementedError, 'shape=' + tuple(array.shape)
  return inv
  
def inv( array ):
  return invdet( array ) / det( array )

def ext( array ):
  """Exterior
  For array of shape (n,n-1) return n-vector ex such that ex.array = 0 and
  det(arr;ex) = ex.ex"""
  assert isarray(array)
  if array.shape == (1,0):
    ext = Array( (1,), unit, isfactored=True )
  elif array.shape == (2,1):
    ((a,),(b,)), scale = array.decompose()
    ext = Array( (-b,a), unit/scale, isfactored=True )
  elif array.shape == (3,2):
    ((a,b),(c,d),(e,f)), scale = array.decompose()
    ext = Array( (c*f-e*d,e*b-a*f,a*d-c*b), unit/scale )
  else:
    raise NotImplementedError, 'shape=%s' % (array.shape,)
  # VERIFY
  A = asfloat( array )
  v = asfloat( ext )
  Av = numpy.concatenate( [A,v[:,numpy.newaxis]], axis=1 )
  numpy.testing.assert_almost_equal( numpy.dot( v, A ), 0 )
  numpy.testing.assert_almost_equal( numpy.linalg.det(Av), numpy.dot(v,v) )
  return ext

def factor( n ):
  assert isinstance( n, (int,long) ) and n > 0
  factors = []
  if n >= len(masks):
    __log__ = log.iter( 'extending primes', primes )
    for prime in __log__:
      assert n >= prime
      count = 0
      while n % prime == 0:
        n //= prime
        count += 1
      factors.append( count )
      if n < len(masks):
        break
  factor = Scalar( factors )
  while n > 1:
    mask = masks[n]
    nums = []
    while mask:
      nums.append( int(mask&1) )
      mask >>= 1
    r = Scalar( nums )
    factor *= r
    n //= int(r)
  assert n == 1
  return factor

def isscalar( num ):
  return isinstance( num, Scalar )

def asscalar( num ):
  if isinstance( num, (int,long) ):
    num = factor( num )
  assert isscalar( num )
  return num

def isarray( arr ):
  return isinstance( arr, Array )

def asarray( arr ):
  if isarray( arr ):
    return arr
  arr = numpy.asarray( arr )
  assert arr.dtype in (int,long)
  return Array( arr, unit )

def isrational(obj):
  return isscalar(obj) or isarray(obj)

def isexact( obj ):
  return isinstance(obj,numpy.ndarray) and obj.dtype in (int,long) or isinstance(obj,(int,long)) or isrational(obj)

def asrational( obj ):
  return obj if isrational( obj ) \
    else asscalar( obj ) if isinstance( obj, (int,long) ) \
    else asarray( obj )

def frac( numer, denom ):
  return asrational( numer ) / asscalar( denom )

def gcd( numbers, scalar=unit ):
  numbers = numpy.asarray( numbers )
  assert numbers.dtype in (int,long)
  if not numbers.size:
    return numbers, unit
  unique = numpy.unique( numpy.abs(numbers.flat) )
  if unique[0] == 0:
    unique = unique[1:]
  if not unique.size:
    return numbers, unit
  common = factor( unique[0] )
  unique = unique[1:]
  if unique.size:
    factors = []
    for prime, count in common.factor():
      n = 0
      while count > n and not numpy.any( unique % prime ):
        unique //= prime
        n += 1
      factors.append( n )
    common = Scalar( factors )
  return numbers // int(common), common * scalar

def asfloat( obj ):
  if isinstance(obj,(int,long)) or isscalar(obj):
    return float(obj)
  if isarray( obj ):
    array, factor = obj.decompose()
    return array * float(factor)
  return numpy.asarray( obj, dtype=float )

def asint( obj ):
  if isinstance(obj,numpy.ndarray) and obj.dtype in (int,long) or isinstance(obj,(int,long)):
    return obj
  if isarray( obj ):
    ints, scale = obj.decompose()
    return ints * int(scale)
  if isscalar( obj ):
    return int(obj)
  raise Exception, 'cannot convert to int: %r' % obj

def dot( A, B ):
  if not isexact( A ) or not isexact( B ):
    return numpy.dot( asfloat(A), asfloat(B) )
  A, a = asarray( A ).decompose()
  B, b = asarray( B ).decompose()
  return Array( numpy.dot( A, B ), a * b )

def eye( ndims ):
  return Array( numpy.eye(ndims,dtype=int), unit, isfactored=True )

def zeros( shape ):
  return Array( numpy.zeros(shape,dtype=int), unit, isfactored=True )

def ones( shape ):
  return Array( numpy.ones(shape,dtype=int), unit, isfactored=True )

def common_factor( arr1, arr2 ):
  if not isexact(arr1) or not isexact(arr2):
    return asfloat(arr1), asfloat(arr2), None
  int1, factor1 = asarray(arr1).decompose()
  int2, factor2 = asarray(arr2).decompose()
  common = factor1.gcd( factor2 )
  return int1 * int(factor1/common), int2 * int(factor2/common), common


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
