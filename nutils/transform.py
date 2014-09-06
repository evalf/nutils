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


class TransformChain( tuple ):

  __slots__ = ()

  @property
  def todims( self ):
    return self[0].todims

  @property
  def fromdims( self ):
    return self[-1].fromdims

  @property
  def isflipped( self ):
    return any( trans.isflipped for trans in self )

  def lookup( self, transforms ):
    # to be replaced by bisection soon
    if self in transforms:
      return self, identity(self.fromdims)
    headtrans = self[:-1]
    tail = self[-1:]
    while headtrans:
      head = TransformChain(headtrans)
      if head in transforms:
        return head, TransformChain(tail)
      tail = headtrans[-1:] + tail
      headtrans = headtrans[:-1]
    return None

  def slice( self, fromdims=None, todims=None ):
    if fromdims is None:
      fromdims = self.fromdims
    if todims is None:
      todims = self.todims
    if fromdims == self.fromdims and todims == self.todims:
      return self
    transforms = self
    while transforms and transforms[-1].fromdims != fromdims:
      transforms = transforms[:-1]
    while transforms and transforms[0].todims != todims:
      transforms = transforms[1:]
    if transforms:
      return TransformChain( transforms )
    assert fromdims == todims, 'invalid slice (%d,%d) of %s' % ( todims, fromdims, self )
    return identity(fromdims)

  def __lshift__( self, other ):
    # self << other
    assert isinstance( other, TransformChain )
    assert self.fromdims == other.todims
    return TransformChain( self + other )

  def __rshift__( self, other ):
    # self >> other
    assert isinstance( other, TransformChain )
    assert self.todims == other.fromdims
    return TransformChain( other + self )

  @property
  def flipped( self ):
    return TransformChain( tuple( trans.flipped for trans in self ) )

  @property
  def det( self ):
    det = self[0].det
    for trans in self[1:]:
      det *= trans.det
    return det

  @property
  def offset( self ):
    offset = self[-1].offset
    for trans in self[-2::-1]:
      offset = trans.apply( offset ) + trans.offset
    return offset

  @property
  def matrix( self ):
    matrix = self[0].matrix
    for trans in self[1:]:
      matrix = rational.dot( matrix, trans.matrix )
    return matrix

  @property
  def invmatrix( self ):
    invmatrix = self[-1].invmatrix
    for trans in self[:-1]:
      invmatrix = rational.dot( invmatrix, trans.invmatrix )
    return invmatrix

  def apply( self, points ):
    for trans in reversed(self):
      points = trans.apply( points )
    return points

  @property
  def parent( self ):
    assert len( self ) >= 2
    return TransformChain( self[:-1] ), TransformChain( self[-1:] )

  def __str__( self ):
    return ' << '.join( str(trans) for trans in self )


class TransformItem( object ):

  def __init__( self, todims, fromdims, isflipped ):
    self.todims = todims
    self.fromdims = fromdims
    if todims == fromdims:
      assert not isflipped
    self.isflipped = isflipped

  @property
  def det( self ):
    raise NotImplementedError

  def __str__( self ):
    return '<TransformItem>' # should be redefined

  def __repr__( self ):
    return '%s(%s)' % ( self.__class__.__name__, self )


## IMMUTABLE TRANSFORMS

class ImmutableTransform( TransformItem ):

  __metaclass__ = cache.Meta

class Identity( ImmutableTransform ):

  def __init__( self, ndims ):
    ImmutableTransform.__init__( self, ndims, ndims, False )

  @property
  def flipped( self ):
    return self

  @property
  def det( self ):
    return rational.unit

  @property
  def matrix( self ):
    return rational.eye( self.todims )

  @property
  def invmatrix( self ):
    return rational.eye( self.todims )

  @property
  def offset( self ):
    return rational.zeros( self.todims )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return points

  def __str__( self ):
    return '='

class Affine( ImmutableTransform ):

  def __init__( self, matrix, offset, isflipped ):
    self.matrix = matrix
    self.offset = offset
    assert matrix.ndim == 2 and offset.shape == matrix.shape[:1]
    ImmutableTransform.__init__( self, matrix.shape[0], matrix.shape[1], isflipped )

  @property
  def flipped( self ):
    return self if self.fromdims == self.todims \
      else Affine( self.matrix, self.offset, not self.isflipped )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return rational.dot( points, self.matrix.T ) + self.offset

  @cache.property
  def det( self ):
    return rational.det( self.matrix )

  @cache.property
  def invmatrix( self ):
    return rational.invdet( self.matrix ) / self.det

  def __str__( self ):
    return '%s x + %s' % ( self.matrix, self.offset )


## VERTEX TRANSFORMS

class VertexTransform( TransformItem ):

  def __init__( self, fromdims ):
    TransformItem.__init__( self, None, fromdims, False )

class MapTrans( VertexTransform ):

  def __init__( self, coords, vertices ):
    self.coords = rational.asarray(coords)
    nverts, ndims = coords.shape
    assert len(vertices) == nverts
    self.vertices = tuple(vertices)
    VertexTransform.__init__( self, ndims )

  def apply( self, coords ):
    assert coords.ndim == 2
    self_coords, coords, common = rational.common_factor( self.coords, coords )
    indices = map( self_coords.tolist().index, coords.tolist() )
    return [ self.vertices[n] for n in indices ]

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

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
  return TransformChain(( Identity( ndims ), ))

def affine( matrix, offset=None, numer=rational.unit, isflipped=False ):
  r_matrix = rational.frac( matrix, numer )
  r_offset = rational.frac( offset, numer ) if offset is not None \
        else rational.zeros( r_matrix.shape[0] )
  return TransformChain(( Affine( r_matrix, r_offset, isflipped ), ))

def roottrans( name, shape ):
  return TransformChain(( RootTrans( name, shape ), ))

def roottransedges( name, shape ):
  return TransformChain(( RootTransEdges( name, shape ), ))

def maptrans( coords, vertices ):
  return TransformChain(( MapTrans( coords, vertices ), ))

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
