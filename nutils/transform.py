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
    return rational.dot( self.trans2.matrix, self.trans1.matrix )

  @cache.property
  def inv( self ):
    return self.trans2.inv >> self.trans1.inv

  def apply( self, points ):
    return self.trans2.apply( self.trans1.apply( points ) )

  def __str__( self ):
    return '%s >> %s' % ( self.trans1, self.trans2 )

class Shift( Transform ):

  def __init__( self, shift ):
    assert rational.isarray( shift ) and shift.ndim == 1
    Transform.__init__( self, len(shift), len(shift) )
    self.shift = shift

  @cache.property
  def inv( self ):
    return shift( -self.shift )

  @property
  def det( self ):
    return rational.unit

  @property
  def matrix( self ):
    return numpy.eye( self.todims )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points + self.shift

  def __str__( self ):
    return '+%s' % self.shift

class Scale( Transform ):

  def __init__( self, ndims, factor ):
    Transform.__init__( self, ndims, ndims )
    self.factor = factor
    
  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points * self.factor

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

  def __init__( self, linear ):
    assert rational.isarray( linear ) and linear.ndim == 2
    Transform.__init__( self, *linear.shape )
    self.linear = linear

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return rational.dot( points, self.linear.T )

  @property
  def matrix( self ):
    return rational.asfloat( self.linear )

  @cache.property
  def det( self ):
    return rational.det( self.linear )
  
  @cache.property
  def inv( self ):
    return linear( rational.inv( self.linear ) )

  def __str__( self ):
    return '*%s' % self.linear


## UTILITY FUNCTIONS


def identity( ndims ):
  return Identity( ndims )

def scale( ndims, factor ):
  if factor == rational.unit:
    return identity( ndims )
  return Scale( ndims, rational.asrational(factor) )

def half( ndims ):
  return Scale( ndims, rational.half )

def shift( shift, numer=rational.unit ):
  return Shift( rational.frac(shift,numer) )

def linear( matrix, numer=rational.unit ):
  return Linear( rational.frac(matrix,numer) )


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
