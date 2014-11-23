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

class Rational( object ):

  __array_priority__ = 1

  def __init__( self, numer, denom=1, isfactored=False ):
    assert isint(denom) and denom > 0
    if not isinstance( numer, numpy.ndarray ):
      numer = numpy.array( numer )
      if not numer.size:
        numer = numpy.astype( int )
      numer.flags.writeable = False
    assert isint(numer)
    if not numer.size:
      denom = 1
    elif denom != 1 and not isfactored:
      absnumers = numpy.unique( abs(numer) )[::-1].tolist() # unique descending
      if not absnumers[-1]:
        absnumers.pop() # ignore zero
      common = denom
      while absnumers and common > 1:
        n = absnumers.pop()
        while n: # GCD: Euclid's algorithm
          common, n = n, common % n
      if common != 1:
        numer = numer // common
        if numer.flags.writeable:
          numer.flags.writeable = False
        denom //= common
    if numer.flags.writeable:
      numer = numer.copy()
      numer.flags.writeable = False
    self.numer = numer
    self.denom = denom

  def __iter__( self ):
    for array in self.numer:
      yield Rational( array, self.denom )

  def __abs__( self ):
    return Rational( abs(self.numer), self.denom, isfactored=True )

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

  @property
  def T( self ):
    return Rational( self.numer.T, self.denom, isfactored=True )

  def __len__( self ):
    return len(self.numer)

  @property
  def __cmpdata( self ):
    return self.numer.shape, tuple(self.numer.flat), self.denom

  def __hash__( self ):
    return hash( self.__cmpdata )

  def __eq__( self, other ):
    return self is other or isrational(other) and self.__cmpdata == other.__cmpdata

  def __neg__( self ):
    return Rational( -self.numer, self.denom, isfactored=True )

  def __add__( self, other ):
    if other is 0:
      return self
    other = asarray( other )
    if not isrational( other ):
      return self.numer / float(self.denom) + other
    return Rational( self.numer * other.denom + other.numer * self.denom, self.denom * other.denom )

  def __sub__( self, other ):
    if other is 0:
      return self
    other = asarray( other )
    if not isrational( other ):
      return self.numer / float(self.denom) - other
    return Rational( self.numer * other.denom - other.numer * self.denom, self.denom * other.denom )

  def __rsub__( self, other ):
    if other is 0:
      return -self
    other = asarray( other )
    if not isrational( other ):
      return other - self.numer / float(self.denom)
    return Rational( other.numer * self.denom - self.numer * other.denom, self.denom * other.denom )

  def __mul__( self, other ):
    if other is 1:
      return self
    other = asarray( other )
    if not isrational( other ):
      return self.numer * ( other / float(self.denom) )
    return Rational( self.numer * other.numer, self.denom * other.denom )

  def __div__( self, other ):
    if other is 1:
      return self
    other = asarray( other )
    if not isrational( other ):
      return self.numer / ( other * float(self.denom) )
    assert other.size == 1, 'only scalar division supported for now'
    numer, = other.numer.flat
    denom = other.denom
    assert numer != 0
    if numer < 0:
      numer = -numer
      denum = -denom
    return Rational( self.numer * denom, self.denom * numer )

  def __rdiv__( self, other ):
    other = asarray( other )
    if not isrational( other ):
      return ( other * float(self.denom) ) / self.numer
    return other / self

  __rmul__ = __mul__
  __radd__ = __add__
  __truediv__ = __div__
  __rtruediv__ = __rdiv__

  def __pow__( self, n ):
    assert isint( n )
    return Rational( self.numer**n, self.denom**n ) if n > 1 \
      else self if n == 1 \
      else ones( self.shape ) if n == 0 \
      else 1 / (self**-n)

  def __str__( self ):
    return '%s/%s' % ( str(self.numer.tolist()).replace(' ',''), self.denom )



## UTILITY FUNCTIONS

isint = lambda a: numpy.issubdtype( a.dtype if isinstance(a,numpy.ndarray) else type(a), numpy.integer )

unit = Rational( 1 )

def det( array ):
  array = asrational( array )
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
  array = asrational(array)
  if array.shape == (1,1):
    invdet = ones( (1,1) )
  elif array.shape == (2,2):
    ((a,b),(c,d)) = array.numer
    invdet = Rational( ((d,-b),(-c,a)), array.denom, isfactored=True )
  elif array.shape == (3,3):
    ((a,b,c),(d,e,f),(g,h,i)) = array.numer
    invdet = Rational( ((e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)), array.denom**2 )
  else:
    raise NotImplementedError( 'shape=' + tuple(array.shape) )
  return invdet
  
def inv( array ):
  return invdet( array ) / det( array )

def ext( array ):
  """Exterior
  For array of shape (n,n-1) return n-vector ex such that ex.array = 0 and
  det(arr;ex) = ex.ex"""
  array = asrational(array)
  assert array.ndim == 2 and array.shape[0] == array.shape[1]+1
  zeros = ( array.numer == 0 ).all( axis=1 )
  if len(array) == 1:
    ext = ones( 1 )
  elif len(array) == 2:
    ((a,),(b,)) = array.numer * array.denom
    ext = Rational( (b,-a) )
  elif any(zeros):
    alpha = det(array[~zeros])
    (i,), = zeros.nonzero()
    v = zeros * ( alpha.numer if i%2==0 else -alpha.numer )
    ext = Rational( v, alpha.denom, True )
  elif len(array) == 3:
    ((a,b),(c,d),(e,f)) = array.numer * array.denom
    ext = Rational( (c*f-e*d,e*b-a*f,a*d-c*b) )
  else:
    raise NotImplementedError( 'shape=%s' % (array.shape,) )
  # VERIFY
  Av = concatenate( [ext[:,numpy.newaxis],array], axis=1 )
  assert equal( dot( ext, array ), 0 ).all()
  assert equal( det(Av), dot(ext,ext) ).all()
  return ext

def isrational( arr ):
  return isinstance( arr, Rational )

def asrational( arr ):
  return arr if isrational( arr ) else Rational( arr )

def frac( a, b ):
  return asrational(a) / asrational(b)

def asarray( arr ):
  if isrational( arr ):
    return arr
  arr = numpy.asarray( arr )
  if isint(arr):
    return Rational( arr )
  return arr

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

def equal( A, B ):
  A = asrational( A )
  B = asrational( B )
  return numpy.equal( A.numer * B.denom, B.numer * A.denom )

def concatenate( args, axis=0 ):
  arg1, arg2 = map( asrational, args )
  return Rational( numpy.concatenate([ arg1.numer * arg2.denom, arg2.numer * arg1.denom ], axis=axis ), arg1.denom * arg2.denom )

def blockdiag( args ):
  arg1, arg2 = args
  arg1 = asrational( arg1 )
  arg2 = asrational( arg2 )
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

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
