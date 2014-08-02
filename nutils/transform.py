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

  def __init__( self, todims, fromdims, sign ):
    self.todims = todims
    self.fromdims = fromdims
    assert abs(sign) == 1
    self.sign = sign

  @property
  def flipped( self ):
    return self

  def __iter__( self ):
    yield identity(self.fromdims), self

  def __getitem__( self, i ):
    assert i >= 0
    for item in self:
      if not i:
        return item
      i -= 1

  def rstrip( self, trans ):
    assert self == trans, 'cannot find sub-transformation'
    return identity( self.fromdims )

  @property
  def det( self ):
    raise NotImplementedError

  @property
  def inv( self ):
    raise NotImplementedError

  def __rshift__( self, other ):
    assert isinstance( other, Transform )
    assert self.todims == other.fromdims
    return Compound( self, other )

  def __str__( self ):
    return '#%x' % id(self)

  def __repr__( self ):
    return '%s(%s)' % ( self.__class__.__name__, self )

class Identity( Transform ):

  def __init__( self, ndims ):
    Transform.__init__( self, ndims, ndims, 1 )

  @property
  def det( self ):
    return rational.unit

  @property
  def inv( self ):
    return self

  @property
  def matrix( self ):
    return rational.eye( self.fromdims )

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
    Transform.__init__( self, trans2.todims, trans1.fromdims, trans1.sign * trans2.sign )
    self.trans1 = trans1
    self.trans2 = trans2

  @property
  def flipped( self ):
    return Compound( self.trans1.flipped, self.trans2.flipped )

  def __iter__( self ):
    yield identity(self.fromdims), self
    for t1, t2 in self.trans2:
      yield self.trans1 >> t1, t2

  def rstrip( self, trans ):
    return identity( self.fromdims ) if self == trans \
      else self.trans1 >> self.trans2.rstrip( trans )

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
    Transform.__init__( self, len(shift), len(shift), 1 )
    self.shift = shift

  @cache.property
  def inv( self ):
    return shift( -self.shift )

  @property
  def det( self ):
    return rational.unit

  @property
  def matrix( self ):
    return rational.eye( self.todims )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points + self.shift

  def __str__( self ):
    return '+%s' % self.shift

class Scale( Transform ):

  def __init__( self, ndims, factor ):
    Transform.__init__( self, ndims, ndims, 1 )
    self.factor = factor
    
  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points * self.factor

  @property
  def matrix( self ):
    return rational.eye(self.todims) * self.factor

  @property
  def det( self ):
    return self.factor**self.fromdims

  @property
  def inv( self ):
    return scale( self.fromdims, rational.unit/self.factor )

  def __str__( self ):
    return '*%s' % str(self.factor)

class Updim( Transform ):

  def __init__( self, matrix, sign ):
    assert rational.isarray( matrix ) and matrix.ndim == 2
    Transform.__init__( self, matrix.shape[0], matrix.shape[1], sign )
    self.matrix = matrix

  @property
  def flipped( self ):
    return Updim( self.matrix, -self.sign )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return rational.dot( points, self.matrix.T )

  def __str__( self ):
    return '*%s' % self.matrix

class Linear( Updim ):

  def __init__( self, matrix ):
    assert matrix.shape[0] == matrix.shape[1]
    Updim.__init__( self, matrix, 1 )

  @cache.property
  def det( self ):
    return rational.det( self.matrix )
  
  @cache.property
  def inv( self ):
    return linear( rational.inv( self.matrix ) )

class Tensor( Transform ):

  def __init__( self, trans1, trans2 ):
    assert isinstance( trans1, Transform )
    assert isinstance( trans2, Transform )
    Transform.__init__( self, trans1.todims+trans2.todims, trans1.fromdims+trans2.fromdims, trans1.sign * trans2.sign )
    self.trans1 = trans1
    self.trans2 = trans2

  @property
  def flipped( self ):
    return Tensor( self.trans1.flipped, self.trans2.flipped )

  def apply( self, points ):
    points1 = self.trans1.apply( points[...,:self.trans1.fromdims] )
    points2 = self.trans2.apply( points[...,self.trans1.fromdims:] )
    points1, points2, factor = rational.common_factor( points1, points2 )
    points = numpy.concatenate( [ points1, points2 ], axis=-1 )
    return points if factor is None else rational.Array( points, factor, True )

  @cache.property
  def matrix( self ):
    matrix1, matrix2, factor = rational.common_factor( self.trans1.matrix, self.trans2.matrix )
    matrix = numpy.zeros( [self.todims,self.fromdims], dtype=int )
    matrix[:self.trans1.todims,:self.trans1.fromdims] = matrix1
    matrix[self.trans1.todims:,self.trans1.fromdims:] = matrix2
    return rational.Array( matrix, factor, True )

  @property
  def det( self ):
    return self.trans1.det * self.trans2.det

  @property
  def inv( self ):
    return Tensor( self.trans1.inv, self.trans2.inv )

  def __str__( self ):
    return '[%s; %s]' % ( self.trans1, self.trans2 )

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

def updim( matrix, sign, numer=rational.unit ):
  return Updim( rational.frac(matrix,numer), sign )

def tensor( trans1, trans2 ):
  return Tensor( trans1, trans2 )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
