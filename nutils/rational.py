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

class Rational( object ):

  __array_priority__ = 1

  def __init__( self, init=() ):
    while len(init) and init[-1] == 0:
      init = init[:-1]
    if isinstance( init, numpy.ndarray ):
      assert init.dtype == int
    else:
      assert all( isinstance(n,int) for n in init )
    self.__n = tuple( init )

  def __hash__( self ):
    return hash( self.__n )

  def __eq__( self, other ):
    return self is other or isrational(other) and self.__n == other.__n

  def __op( self, other, op ):
    assert self.__n or other.__n, 'trivial cases should be handled by caller'
    return Rational( numpy.hstack([
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
    if isarray( other ):
      return other * self
    assert isrational( other )
    return other if not self.__n \
      else self if not other.__n \
      else self.__op( other, numpy.add )

  def __neg__( self ):
    raise TypeError, 'rational cannot be negated'

  def __add__( self, other ):
    if not isexact( other ):
      return float(self) + other
    if isarray( other ):
      return other + self
    raise NotImplementedError

  def __sub__( self, other ):
    if not isexact( other ):
      return float(self) - other
    if isarray( other ):
      return other - self
    raise NotImplementedError

  def __div__( self, other ):
    if not isexact( other ):
      return float(self) / other
    assert isrational( other )
    return self if not other.__n \
      else Rational( numpy.negative(other.__n) ) if not self.__n \
      else self.__op( other, numpy.subtract )

  def __rdiv__( self, other ):
    return Rational( numpy.negative(self.__n) ) * other

  __rmul__ = __mul__
  __radd__ = __add__
  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def __pow__( self, n ):
    assert isinstance( n, int )
    if n == 0 or not self.__n:
      return Rational()
    return Rational( numpy.multiply( self.__n, n ) )

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
    return 'Rational(%s)' % str(self)

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
    return Rational( self.__n and numpy.maximum( 0, self.__n ) )
  
  @property
  def denom( self ):
    return Rational( self.__n and numpy.minimum( 0, self.__n ) )

unit = Rational()
half = Rational((-1,))

class Array( object ):

  __array_priority__ = 1

  def __init__( self, array, factor=unit, isfactored=False ):
    assert isrational( factor )
    array = numpy.asarray( array )
    assert array.dtype == int
    if isfactored:
      self.__array = array
      self.__factor = factor
    else:
      numbers, common = gcd( array.ravel() )
      self.__array = numbers.reshape( array.shape )
      self.__factor = factor * common

  @property
  def ndim( self ):
    return self.__array.ndim

  @property
  def shape( self ):
    return self.__array.shape

  @property
  def T( self ):
    return Array( self.__array.T, self.__factor, True )

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
    return Array( -self.__array, self.__factor, True )

  def __add__( self, other ):
    if not isexact( other ):
      return self.__array * float(self.__factor) + other
    raise NotImplementedError

  def __sub__( self, other ):
    if not isexact( other ):
      return self.__array * float(self.__factor) - other
    raise NotImplementedError

  def __mul__( self, other ):
    if not isexact( other ):
      return self.__array * ( other * float(self.__factor) )
    if isrational( other ):
      return Array( self.__array, self.__factor*other, True )
    raise NotImplementedError

  def __div__( self, other ):
    if not isexact( other ):
      return self.__array * ( float(self.__factor) / other )
    if isrational( other ):
      return Array( self.__array, self.__factor/other, True )
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
    return self.__array, self.__factor

  def __str__( self ):
    return str(self.__factor) + str(self.__array.tolist()).replace( ' ', '' )

def det( array ):
  assert isarray(array)
  if array.shape == (2,2):
    ((a,b),(c,d)), scale = array.decompose()
    det = a*d - b*c
  elif array.shape == (3,3):
    ((a,b,c),(d,e,f),(g,h,i)), scale = array.decompose()
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
  else:
    raise NotImplementedError, 'shape=' + str(array.shape)
  return factor(det) * scale**len(array)

def inv( array ):
  assert isarray(array)
  if array.shape == (2,2):
    ((a,b),(c,d)), scale = array.decompose()
    inv = Array( ((d,-b),(-c,a)), scale / det(array), True )
  elif array.shape == (3,3):
    raise NotImplementedError
    ((a,b,c),(d,e,f),(g,h,i)), scale = array.decompose()
    inv = Array( ((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)), scale / det(array), False )
  else:
    raise NotImplementedError, 'shape=' + tuple(array.shape)
  return inv

def factor( n ):
  assert isinstance( n, int ) and n > 0
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
  factor = Rational( factors )
  while n > 1:
    mask = masks[n]
    nums = []
    while mask:
      nums.append( int(mask&1) )
      mask >>= 1
    r = Rational( nums )
    factor *= r
    n //= int(r)
  return factor

def isrational( num ):
  return isinstance( num, Rational )

def asrational( num ):
  if isinstance( num, int ):
    num = factor( num )
  assert isrational( num )
  return num

def isarray( arr ):
  return isinstance( arr, Array )

def asarray( arr ):
  if isarray( arr ):
    return arr
  arr = numpy.asarray( arr )
  assert arr.dtype == int
  return Array( arr, unit, False )

def isexact( obj ):
  return isrational(obj) or isarray(obj)

def asexact( obj ):
  return obj if isexact( obj ) \
    else asrational( obj ) if isinstance( obj, int ) \
    else asarray( obj )

def frac( numer, denom ):
  return asexact( numer ) / asrational( denom )

def gcd( numbers ):
  numbers = numpy.asarray( numbers )
  assert numbers.dtype == int
  if not numbers.size:
    return numbers, Rational()
  unique = numpy.unique( numpy.abs(numbers.flat) )
  if unique[0] == 0:
    unique = unique[1:]
  if not unique.size:
    return numbers, Rational()
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
    common = Rational( factors )
  return numbers // int(common), common

def asfloat( obj ):
  if isinstance(obj,int) or isrational(obj):
    return float(obj)
  if isarray( obj ):
    array, factor = obj.decompose()
    return array * float(factor)
  if isinstance( obj, numpy.ndarray ) and obj.dtype == float:
    return obj # don't touch, might be of derived type
  return numpy.asarray( obj, dtype=float )

def dot( A, B ):
  if not isexact( A ) or not isexact( B ):
    Aflt = asfloat(A)
    Bflt = asfloat(B)
    return numpy.dot( Aflt, Bflt ).view( Aflt.__class__ ) # .view necesssary for 1D Aflt (numpy exception)
  A, a = asarray( A ).decompose()
  B, b = asarray( B ).decompose()
  return Array( numpy.dot( A, B ), a * b, False )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
