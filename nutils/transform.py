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

from __future__ import print_function, division
from . import cache, rational, numeric
import numpy

_noarg = object()


class TransformChain( tuple ):

  __slots__ = ()

  def __getitem__( self, item ):
    trans = tuple.__getitem__( self, item )
    return TransformChain( trans ) if isinstance( trans, tuple ) else trans

  def __getslice__( self, i, j ):
    return TransformChain( tuple.__getslice__( self, i, j ) )

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
    headtrans = self
    while headtrans:
      if headtrans in transforms:
        return headtrans
      headtrans = headtrans[:-1]
    return None

  def split( self, ndims=_noarg ):
    if ndims is _noarg:
      ndims = self.fromdims
    if self.todims == ndims:
      return TransformChain(), self
    for i, trans in enumerate(self):
      if trans.fromdims == ndims:
        return self[:i+1], self[i+1:]
    raise Exception( 'dimension not found in chain: %s' % ndims )

  def __lshift__( self, other ):
    # self << other
    assert isinstance( other, TransformChain )
    if not self:
      return other
    if not other:
      return self
    assert self.fromdims == other.todims
    return TransformChain( self + other )

  def __rshift__( self, other ):
    # self >> other
    assert isinstance( other, TransformChain )
    return other << self

  @property
  def flipped( self ):
    return TransformChain( trans.flipped for trans in self )

  @property
  def det( self ):
    det = 1
    for trans in self:
      det *= trans.det
    return det

  @property
  def offset( self ):
    offset = self[-1].offset
    for trans in self[-2::-1]:
      offset = trans.apply( offset )
    return offset

  @property
  def linear( self ):
    linear = rational.unit
    for trans in self:
      linear = rational.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
          else linear * trans.linear
    return linear

  @property
  def invlinear( self ):
    invlinear = rational.unit
    for trans in self:
      invlinear = rational.dot( trans.invlinear, invlinear ) if invlinear.ndim and trans.linear.ndim \
             else trans.invlinear * invlinear
    return invlinear

  def apply( self, points ):
    for trans in reversed(self):
      points = trans.apply( points )
    return points

  def __str__( self ):
    return ' << '.join( str(trans) for trans in self ) if self else '='

  def __repr__( self ):
    return 'TransformChain( %s )' % (self,)

  @property
  def flat( self ):
    return self if len(self) == 1 \
      else affine( self.linear, self.offset, isflipped=self.isflipped )


## TRANSFORM ITEMS

class TransformItem( cache.Immutable ):

  def __init__( self, todims, fromdims, isflipped ):
    self.todims = todims
    self.fromdims = fromdims
    self.isflipped = isflipped

  __lt__ = lambda self, other: id(self) <  id(other)
  __gt__ = lambda self, other: id(self) >  id(other)
  __le__ = lambda self, other: id(self) <= id(other)
  __ge__ = lambda self, other: id(self) >= id(other)

  def __repr__( self ):
    return '%s( %s )' % ( self.__class__.__name__, self )

class Shift( TransformItem ):

  def __init__( self, offset ):
    self.linear = self.invlinear = self.det = rational.unit
    self.offset = offset
    assert offset.ndim == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0], False )

  @property
  def flipped( self ):
    return self

  def apply( self, points ):
    return points + self.offset

  def __str__( self ):
    return 'x + %s' % self.offset

class Scale( TransformItem ):

  def __init__( self, linear, offset ):
    assert linear.ndim == 0 and offset.ndim == 1
    self.linear = linear
    self.offset = offset
    TransformItem.__init__( self, offset.shape[0], offset.shape[0], False )

  @property
  def flipped( self ):
    return self

  def apply( self, points ):
    return self.linear * points + self.offset

  @property
  def det( self ):
    return self.linear**self.todims

  @property
  def invlinear( self ):
    return 1 / self.linear

  def __str__( self ):
    return '%s x + %s' % ( self.linear, self.offset )

class Matrix( TransformItem ):

  def __init__( self, linear, offset, isflipped ):
    self.linear = linear
    self.offset = offset
    assert linear.ndim == 2 and offset.shape == linear.shape[:1]
    TransformItem.__init__( self, linear.shape[0], linear.shape[1], isflipped )

  @property
  def flipped( self ):
    return self if self.fromdims == self.todims \
      else Matrix( self.linear, self.offset, not self.isflipped )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return rational.dot( points, self.linear.T ) + self.offset

  @cache.property
  def det( self ):
    return rational.det( self.linear )

  @cache.property
  def invlinear( self ):
    return rational.invdet( self.linear ) / self.det

  def __str__( self ):
    return '%s x + %s' % ( self.linear, self.offset )

class VertexTransform( TransformItem ):

  def __init__( self, fromdims ):
    TransformItem.__init__( self, None, fromdims, False )

class MapTrans( VertexTransform ):

  def __init__( self, coords, vertices ):
    self.coords = numpy.asarray(coords)
    assert numeric.isintarray( self.coords )
    nverts, ndims = coords.shape
    assert len(vertices) == nverts
    self.vertices = tuple(vertices)
    VertexTransform.__init__( self, ndims )

  def apply( self, coords ):
    assert coords.ndim == 2
    coords = rational.asarray( coords )
    assert coords.denom == 1 # for now
    indices = map( self.coords.tolist().index, coords.numer.tolist() )
    return [ self.vertices[n] for n in indices ]

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

class RootTrans( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.I, = numpy.where( shape )
    self.w = numpy.take( shape, self.I )
    self.fmt = name+'{}'

  def apply( self, coords ):
    assert coords.ndim == 2
    coords = rational.asarray( coords )
    if self.I.size:
      ci = coords.numer.copy()
      wi = self.w * coords.denom
      ci[:,self.I] = ci[:,self.I] % wi
      coords = rational.Rational( ci, coords.denom )
    return [ self.fmt.format(c) for c in coords ]

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


## CONSTRUCTORS

def affine( linear=None, offset=None, numer=1, isflipped=False ):
  r_offset = rational.asarray( offset ) / numer if offset is not None else rational.zeros( len(linear) )
  r_linear = rational.asarray( linear ) / numer if linear is not None else rational.unit
  return TransformChain((
         Matrix( r_linear, r_offset, isflipped ) if r_linear.ndim
    else Scale( r_linear, r_offset ) if r_linear != rational.unit
    else Shift( r_offset ), ))

def roottrans( name, shape ):
  return TransformChain(( RootTrans( name, shape ), ))

def roottransedges( name, shape ):
  return TransformChain(( RootTransEdges( name, shape ), ))

def maptrans( coords, vertices ):
  return TransformChain(( MapTrans( coords, vertices ), ))

def equivalent( trans1, trans2 ):
  trans1 = TransformChain( trans1 )
  trans2 = TransformChain( trans2 )
  return trans1.linear == trans2.linear and trans1.offset == trans2.offset


## UTILITY FUNCTIONS

identity = TransformChain()

def canonical( transchain ):
  # keep at highest ndims possible
  chain = []
  trans1 = transchain[ 0 ]
  for trans2 in transchain[ 1: ]:
    if isinstance( trans2, Scale ) and trans1.todims == trans1.fromdims + 1:
      newscale = Scale( trans2.linear, trans1.apply( trans2.offset ) - trans2.linear * trans1.offset )
      assert equivalent( (trans1,trans2), (newscale,trans1) )
      chain.append( newscale )
    else:
      chain.append( trans1 )
      trans1 = trans2
  chain.append( trans1 )
  return TransformChain( chain )

def solve( transA, transB ): # A << X = B
  A = transA.linear
  B = transB.linear
  assert A.ndim == B.ndim == 2
  a = transA.offset
  b = transB.offset
  AAinv = rational.inv( rational.dot( A.T, A ) )
  X = rational.dot( AAinv, rational.dot( A.T, B ) ) # A X = B
  x = rational.dot( AAinv, rational.dot( A.T, b-a ) ) # A x + a = b
  transX = affine( X, x )
  transAX = transA << transX
  if numpy.any( transAX.linear != transB.linear ) or numpy.any( transAX.offset != transB.offset ):
    return None
  return transX

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
