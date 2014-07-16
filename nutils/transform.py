# -*- coding: utf8 -*-
#
# Module ELEMENT
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The transform module.
"""

from __future__ import division
from . import cache, rational
import numpy


class Transform( cache.Immutable ):

  def __init__( self, todims, fromdims ):
    self.todims = todims
    self.fromdims = fromdims

  def __rshift__( self, other ):
    assert isinstance( other, Transform )
    assert self.todims == other.fromdims
    return Compound( self, other )

  def __str__( self ):
    return '?'

  def __repr__( self ):
    return '%s(%s)' % ( self.__class__.__name__, self )

class Identity( Transform ):

  def __init__( self, ndims ):
    Transform.__init__( self, ndims, ndims )

  @property
  def det( self ):
    return rational.unit

  @property
  def inv( self ):
    return self

  @property
  def matrix( self ):
    return numpy.eye( self.fromdims )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points

  def __str__( self ):
    return '='

class Compound( Transform ):

  def __init__( self, trans1, trans2 ):
    assert isinstance( trans1, Transform )
    assert isinstance( trans2, Transform )
    assert trans1.todims == trans2.fromdims
    Transform.__init__( self, trans2.todims, trans1.fromdims )
    self.trans1 = trans1
    self.trans2 = trans2

  @cache.property
  def det( self ):
    return self.trans1.det * self.trans2.det

  @cache.property
  def matrix( self ):
    return numpy.dot( self.trans2.matrix, self.trans1.matrix )

  @cache.property
  def inv( self ):
    return self.trans2.inv >> self.trans1.inv

  def apply( self, points ):
    return self.trans2.apply( self.trans1.apply( points ) )

  def __str__( self ):
    return '%s << %s' % ( self.trans1, self.trans2 )

class Shift( Transform ):

  def __init__( self, shift, factor ):
    Transform.__init__( self, len(shift), len(shift) )
    self.shift = numpy.array( shift )
    assert self.shift.dtype == int
    self.factor = factor

  @cache.property
  def inv( self ):
    # y = x + n/d b <=> x = y + n/d (-b)
    return shift( -self.shift, self.factor )

  @property
  def det( self ):
    return rational.unit

  @property
  def matrix( self ):
    return numpy.eye( self.todims )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points + self.shift * float(self.factor)

  def __str__( self ):
    return '+%s:%s' % ( self.factor, ','.join( str(i) for i in self.shift ) )

class Scale( Transform ):

  def __init__( self, ndims, factor ):
    Transform.__init__( self, ndims, ndims )
    self.factor = factor
    
  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points * float(self.factor)

  @property
  def matrix( self ):
    return numpy.eye(self.todims) * float(self.factor)

  @property
  def det( self ):
    return self.factor**self.fromdims

  @property
  def inv( self ):
    return scale( self.fromdims, rational.unit/self.factor )

  def __str__( self ):
    return '*%s' % str(self.factor)

class Linear( Transform ):

  def __init__( self, shape, numbers, factor ):
    Transform.__init__( self, *shape )
    self.linear = numpy.array( numbers ).reshape( shape )
    self.factor = factor

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return numpy.dot( points, self.linear.T ) * float(self.factor)

  @property
  def matrix( self ):
    return self.linear * float(self.factor)

  @cache.property
  def det( self ):
    assert self.fromdims == self.todims
    if self.fromdims == 2:
      (a,b),(c,d) = self.linear
      intdet = a*d - b*c
    elif self.fromdims == 3:
      (a,b,c),(d,e,f),(g,h,i) = self.linear
      intdet = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
    else:
      raise NotImplementedError, 'ndims=%d' % self.ndims
    det = rational.factor(intdet) * self.factor**self.fromdims
    #assert abs( numpy.linalg.det(self.matrix) - float(det) ) < 1e-15
    return det
  
  @cache.property
  def inv( self ):
    assert self.fromdims == self.todims
    if self.todims == 2:
      (a,b),(c,d) = self.linear
      invlinear = (d,-b),(-c,a)
    elif self.todims == 3:
      raise NotImplementedError
      (a,b,c),(d,e,f),(g,h,i) = self.linear
      invlinear = (e*i-f*h,c*h-b*i,b*f-c*e),(f*g-d*i,a*i-c*g,c*d-a*f),(d*h-e*g,b*g-a*h,a*e-b*d)
    else:
      raise NotImplementedError, 'ndims=%d' % self.ndims
    inv = linear( invlinear, self.factor/self.det )
    #assert numpy.linalg.norm( numpy.dot( self.matrix, inv.matrix ) - numpy.eye(self.todims) ) < 1e-15
    return inv

  def __str__( self ):
    return '*%s:%s' % ( self.factor, ';'.join( '%s' % ','.join( str(i) for i in row ) for row in self.linear ) )


## UTILITY FUNCTIONS


def identity( ndims ):
  return Identity( ndims )

def scale( ndims, factor ):
  if factor == rational.unit:
    return identity( ndims )
  return Scale( ndims, rational.asrational(factor) )

def half( ndims ):
  return Scale( ndims, rational.half )

def shift( shift, factor=rational.unit ):
  if not any( shift ):
    return identity( len(shift) )
  shift, common = rational.gcd( shift )
  common *= rational.asrational( factor )
  return Shift( tuple(shift), common )

def linear( matrix, factor=rational.unit ):
  matrix = numpy.asarray(matrix)
  numbers, common = rational.gcd( matrix.ravel() )
  assert any( numbers )
  common *= rational.asrational( factor )
  return Linear( matrix.shape, tuple(numbers), common )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
