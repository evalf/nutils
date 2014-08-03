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


class Transform( object ):

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


## IMMUTABLE TRANSFORMS

class ImmutableTransform( Transform ):

  __metaclass__ = cache.Meta

class Identity( ImmutableTransform ):

  def __init__( self, ndims ):
    ImmutableTransform.__init__( self, ndims, ndims, 1 )

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

class Compound( ImmutableTransform ):

  def __init__( self, trans1, trans2 ):
    assert isinstance( trans1, Transform )
    assert isinstance( trans2, Transform )
    assert trans1.todims == trans2.fromdims
    ImmutableTransform.__init__( self, trans2.todims, trans1.fromdims, trans1.sign * trans2.sign )
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

class Shift( ImmutableTransform ):

  def __init__( self, shift ):
    assert rational.isarray( shift ) and shift.ndim == 1
    ImmutableTransform.__init__( self, len(shift), len(shift), 1 )
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

class Scale( ImmutableTransform ):

  def __init__( self, ndims, factor ):
    ImmutableTransform.__init__( self, ndims, ndims, 1 )
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

class Updim( ImmutableTransform ):

  def __init__( self, matrix, sign ):
    assert rational.isarray( matrix ) and matrix.ndim == 2
    ImmutableTransform.__init__( self, matrix.shape[0], matrix.shape[1], sign )
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

class Tensor( ImmutableTransform ):

  def __init__( self, trans1, trans2 ):
    assert isinstance( trans1, Transform )
    assert isinstance( trans2, Transform )
    ImmutableTransform.__init__( self, trans1.todims+trans2.todims, trans1.fromdims+trans2.fromdims, trans1.sign * trans2.sign )
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


## VERTEX TRANSFORMS

class VertexTransform( Transform ):

  def __init__( self, fromdims ):
    Transform.__init__( self, None, fromdims, 1 )

class MapTrans( VertexTransform ):

  def __init__( self, coords, vertices ):
    self.coords = coords
    self.vertices = rational.asarray(vertices)
    nverts, ndims = vertices.shape
    VertexTransform.__init__( self, ndims )

  def apply( self, coords ):
    assert coord.ndim == 1
    mycoords, coords, common = rational.common_factor( self.coords, coord )
    return [ self.vertices[mycoords.tolist().index(coord.tolist())] for coord in coords ]

class RootTrans( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.I, = numpy.where( shape )
    self.w = rational.asarray( numpy.take( shape, self.I ) )
    self.fmt = name+'{}'

  def apply( self, coords ):
    assert coords.ndim == 2
    if self.I.size:
      ci, wi, factor = rational.common_factor( coords, self.w )
      ci[:,self.I] = ci[:,self.I] % wi
      coords = rational.Array( ci, factor, True )
    return map( self.fmt.format, coords )

  def __str__( self ):
    return repr( self.fmt.format('*') )

class RootTransEdges( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.shape = shape
    assert isinstance( name, numpy.ndarray )
    assert name.shape == (3,)*len(shape)
    self.name = name.copy()

  def apply( self, coords ):
    assert coords.ndim == 2
    labels = []
    for coord in coords.T.frac.T:
      right = (coord[:,1]==1) & (coord[:,0]==self.shape)
      left = coord[:,0]==0
      where = (1+right)-left
      s = self.name[tuple(where)] + '[%s]' % ','.join( str(n) if d == 1 else '%d/%d' % (n,d) for n, d in coord[where==1] )
      labels.append( s )
    return labels

  def __str__( self ):
    return repr( ','.join(self.name.flat)+'*' )


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
