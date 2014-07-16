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

class Rational( tuple ):

  def __new__( cls, init=() ):
    while len(init) and init[-1] == 0:
      init = init[:-1]
    assert all( isinstance(n,int) for n in init )
    return tuple.__new__( cls, init )

  def __mul__( self, other ):
    if not isrational( other ):
      return float(self) * other
    if not self:
      return other
    if not other:
      return self
    return Rational( tuple( numpy.add( self,  other[:len(self)] ) ) + other[len(self):] if len(self) < len(other)
                else tuple( numpy.add( self[:len(other)], other ) ) + self[len(other):] if len(self) > len(other)
                else numpy.add( self, other ) )

  def __div__( self, other ):
    if not isrational( other ):
      return float(self) / other
    if not self:
      return Rational( numpy.negative(other) )
    if not other:
      return self
    return Rational( tuple( numpy.subtract( self,  other[:len(self)] ) ) + tuple( numpy.negative(other[len(self):]) ) if len(self) < len(other)
                else tuple( numpy.subtract( self[:len(other)], other ) ) + self[len(other):] if len(self) > len(other)
                else numpy.subtract( self, other ) )

  def __rdiv__( self, other ):
    return Rational( numpy.negative(self) ) * other

  __rmul__ = __mul__
  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def __pow__( self, n ):
    assert isinstance( n, int )
    if n == 0:
      return Rational()
    return Rational( numpy.multiply( self, n ) )

  def __float__( self ):
    numer, denom = self.frac
    return numer / denom
  
  def __int__( self ):
    numer, denom = self.frac
    assert denom == 1
    return numer

  @cache.property
  def frac( self ):
    numer = denom = 1
    for p, n in zip( primes, self ):
      if n > 0:
        numer *= p**+n
      elif n < 0:
        denom *= p**-n
    return numer, denom

  @property
  def numer( self ):
    return Rational( numpy.maximum( 0, self ) )
  
  @property
  def denom( self ):
    return Rational( numpy.minimum( 0, self ) )
  
  def __str__( self ):
    return '%d/%d' % self.frac

  def __repr__( self ):
    return 'Rational(%s)' % str(self)

unit = Rational()
half = Rational((-1,))

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

def rational( numer, denom=unit ):
  return asrational( numer ) / asrational( denom )

def gcd( numbers ):
  numbers = numpy.array( numbers )
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
    for prime, count in zip( primes, common ):
      n = 0
      while count > n and not numpy.any( unique % prime ):
        unique //= prime
        n += 1
      factors.append( n )
    common = Rational( factors )
  return numbers // int(common), common

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
