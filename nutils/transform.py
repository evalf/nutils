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

from . import cache, numeric, core
import numpy


class TransformChain( tuple ):

  __slots__ = ()

  def slicefrom( self, i ):
    return TransformChain( self[i:] )

  def sliceto( self, j ):
    return TransformChain( self[:j] )

  def fromtrans( self, trans ):
    # assuming self and trans both canonical
    mytrans = self.promote( trans.fromdims )
    assert mytrans[:len(trans)] == trans
    return mytrans[len(trans):]

  @property
  def todims( self ):
    return self[0].todims

  @property
  def fromdims( self ):
    return self[-1].fromdims

  @property
  def isflipped( self ):
    return sum( trans.isflipped for trans in self ) % 2 == 1

  @property
  def orientation( self ):
    return -1 if self.isflipped else +1

  def lookup( self, transforms ):
    # to be replaced by bisection soon
    headtrans = self
    while headtrans:
      if headtrans in transforms:
        return headtrans
      headtrans = headtrans.sliceto(-1)
    return None

  def rsplit( self, ndims ):
    if self.fromdims == ndims:
      return self, CanonicalTransformChain()
    for i in range( len(self)-1, -1, -1 ):
      if self[i].todims == ndims:
        return self.sliceto(i), self.slicefrom(i)
    raise Exception( 'failed to split transformation' )

  def split( self, ndims, after=True ):
    # split before/after the first occurrence of .fromdims==ndims. For
    # after=True (default) the base part represents the coordinate system for
    # integration/gradients at the level specified.
    i = core.index( trans.fromdims == ndims for trans in self ) + after
    return self.sliceto(i), self.slicefrom(i)

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
    assert self.todims == self.fromdims+1
    return TransformChain( trans.flipped if trans.todims == trans.fromdims+1 else trans for trans in self )

  @property
  def det( self ):
    det = 1
    for trans in self:
      det *= trans.det
    return det

  @property
  def ext( self ):
    ext = numeric.ext( self.linear )
    return ext if not self.isflipped else -ext

  @property
  def offset( self ):
    offset = self[-1].offset
    for trans in self[-2::-1]:
      offset = trans.apply( offset )
    return offset

  @property
  def linear( self ):
    linear = numpy.array( 1. )
    for trans in self:
      linear = numpy.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
          else linear * trans.linear
    return linear

  @property
  def invlinear( self ):
    invlinear = numpy.array( 1. )
    for trans in self:
      invlinear = numpy.dot( trans.invlinear, invlinear ) if invlinear.ndim and trans.linear.ndim \
             else trans.invlinear * invlinear
    return invlinear

  def apply( self, points ):
    for trans in reversed(self):
      points = trans.apply( points )
    return points

  def solve( self, points ):
    return numeric.solve_exact( self.linear, (points - self.offset).T ).T

  def __str__( self ):
    return ' << '.join( str(trans) for trans in self ) if self else '='

  def __repr__( self ):
    return '{}( {} )'.format( self.__class__.__name__, self )

  @property
  def flat( self ):
    return self if len(self) == 1 \
      else affine( self.linear, self.offset, isflipped=self.isflipped )

  @property
  def canonical( self ):
    # Keep at lowest ndims possible. The reason is that in this form we can do
    # lookups of embedded elements.
    items = list( self )
    for i in range(len(items)-1)[::-1]:
      trans1, trans2 = items[i:i+2]
      if mayswap( trans1, trans2 ):
        trans12 = TransformChain(( trans1, trans2 )).flat
        try:
          newlinear, newoffset = numeric.solve_exact( trans2.linear, trans12.linear, trans12.offset - trans2.offset )
        except numpy.linalg.LinAlgError:
          pass
        else:
          trans21 = TransformChain( (trans2,) + affine( newlinear, newoffset ) )
          assert trans21.flat == trans12
          items[i:i+2] = trans21
    return CanonicalTransformChain( items )

  def promote( self, ndims ):
    raise Exception( 'promotion only possible from canonical form' )

class CanonicalTransformChain( TransformChain ):

  def slicefrom( self, i ):
    return CanonicalTransformChain( TransformChain.slicefrom( self, i ) )

  def sliceto( self, j ):
    return CanonicalTransformChain( TransformChain.sliceto( self, j ) )

  def __lshift__( self, other ):
    # self << other
    joint = TransformChain.__lshift__( self, other )
    if self and other and isinstance( other, CanonicalTransformChain ) and not mayswap( self[-1], other[0] ):
      joint = CanonicalTransformChain( joint )
    return joint

  @property
  def flipped( self ):
    return CanonicalTransformChain( TransformChain.flipped.fget( self ) )

  @property
  def canonical( self ):
    return self

  def promote( self, ndims ):
    if ndims == self.fromdims:
      return self
    index = core.index( trans.fromdims == self.fromdims for trans in self )
    uptrans = self[index]
    if index == len(self)-1 or not mayswap( self[index+1], uptrans ):
      A = self.sliceto(index)
      B = self.slicefrom(index)
    else:
      body = list( self[:index] )
      for i in range( index+1, len(self) ):
        scale = self[i]
        if not mayswap( scale, uptrans ):
          break
        newscale = Scale( scale.linear, uptrans.apply(scale.offset) - scale.linear * uptrans.offset )
        body.append( newscale )
      else:
        i = len(self)+1
      assert equivalent( body[index:]+[uptrans], self[index:i] )
      A = CanonicalTransformChain( body )
      B = CanonicalTransformChain( (uptrans,) + self[i:] )
    return A.promote(ndims) << B

mayswap = lambda trans1, trans2: isinstance( trans1, Scale ) and trans1.linear == .5 and trans2.todims == trans2.fromdims + 1 and trans2.fromdims > 0


## TRANSFORM ITEMS

class TransformItem( cache.Immutable ):

  def __init__( self, todims, fromdims ):
    self.todims = todims
    self.fromdims = fromdims

  __lt__ = lambda self, other: id(self) <  id(other)
  __gt__ = lambda self, other: id(self) >  id(other)
  __le__ = lambda self, other: id(self) <= id(other)
  __ge__ = lambda self, other: id(self) >= id(other)

  def __repr__( self ):
    return '{}( {} )'.format( self.__class__.__name__, self )

class Shift( TransformItem ):

  def __init__( self, offset ):
    self.linear = self.invlinear = self.det = numpy.array(1.)
    self.offset = offset
    self.isflipped = False
    assert offset.ndim == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0] )

  def apply( self, points ):
    return points + self.offset

  def __str__( self ):
    return '{}+x'.format( numeric.fstr(self.offset) )

class Scale( TransformItem ):

  def __init__( self, linear, offset ):
    assert linear.ndim == 0 and offset.ndim == 1
    self.linear = linear
    self.offset = offset
    self.isflipped = linear < 0 and len(offset)%2 == 1
    TransformItem.__init__( self, offset.shape[0], offset.shape[0] )

  def apply( self, points ):
    return self.linear * points + self.offset

  @property
  def det( self ):
    return self.linear**self.todims

  @property
  def invlinear( self ):
    return 1 / self.linear

  def __str__( self ):
    return '{}+{}*x'.format( numeric.fstr(self.offset), numeric.fstr(self.linear) )

class Matrix( TransformItem ):

  def __init__( self, linear, offset ):
    self.linear = linear
    self.offset = offset
    assert linear.ndim == 2 and offset.shape == linear.shape[:1]
    TransformItem.__init__( self, linear.shape[0], linear.shape[1] )

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return numpy.dot( points, self.linear.T ) + self.offset

  def __str__( self ):
    return '{}{}{}'.format( '~' if self.isflipped else '', numeric.fstr(self.offset), ''.join( '+{}*x{}'.format( numeric.fstr(v), i ) for i, v in enumerate(self.linear.T) ) )

class Square( Matrix ):

  def __init__( self, linear, offset ):
    Matrix.__init__( self, linear, offset )
    assert self.fromdims == self.todims

  @property
  def isflipped( self ):
    return self.det < 0

  @cache.property
  def det( self ):
    return numeric.det_exact( self.linear )

  @cache.property
  def invlinear( self ):
    return numpy.linalg.inv( self.linear )

class Updim( Matrix ):

  def __init__( self, linear, offset, isflipped ):
    assert isflipped in (True,False)
    self.isflipped = isflipped
    Matrix.__init__( self, linear, offset )
    assert self.todims > self.fromdims

  @property
  def flipped( self ):
    return Updim( self.linear, self.offset, not self.isflipped )

class VertexTransform( TransformItem ):

  def __init__( self, fromdims ):
    TransformItem.__init__( self, None, fromdims )
    self.isflipped = False

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
    coords = numpy.asarray( coords )
    indices = map( self.coords.tolist().index, coords.tolist() )
    return tuple( self.vertices[n] for n in indices )

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

class RootTrans( VertexTransform ):

  def __init__( self, name, shape ):
    VertexTransform.__init__( self, len(shape) )
    self.I, = numpy.where( shape )
    self.w = numpy.take( shape, self.I )
    self.name = name

  def apply( self, coords ):
    coords = numpy.asarray(coords)
    assert coords.ndim == 2
    if self.I.size:
      coords = coords.copy()
      coords[:,self.I] %= self.w
    return tuple( self.name + str(c) for c in coords.tolist() )

  def __str__( self ):
    return repr( self.name + '[*]' )

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

def affine( linear, offset, denom=1, isflipped=None ):
  r_offset = numpy.asarray( offset, dtype=float ) / denom
  r_linear = numpy.asarray( linear, dtype=float ) / denom
  n, = r_offset.shape
  if r_linear.ndim == 2:
    assert r_linear.shape[0] == n
    if r_linear.shape[1] != n:
      trans = Updim( r_linear, r_offset, isflipped )
    elif n == 0:
      trans = Shift( r_offset )
    elif n == 1 or r_linear[0,-1] == 0 and numpy.all( r_linear == r_linear[0,0] * numpy.eye(n) ):
      trans = Scale( r_linear[0,0], r_offset ) if r_linear[0,0] != 1 else Shift( r_offset )
    else:
      trans = Square( r_linear, r_offset )
  else:
    assert r_linear.ndim == 0
    trans = Scale( r_linear, r_offset ) if r_linear != 1 else Shift( r_offset )
  if isflipped is not None:
    assert trans.isflipped == isflipped
  return CanonicalTransformChain( [trans] )

def simplex( coords, isflipped=None ):
  coords = numpy.asarray(coords)
  offset = coords[0]
  return affine( (coords[1:]-offset).T, offset, isflipped=isflipped )

def roottrans( name, shape ):
  return CanonicalTransformChain(( RootTrans( name, shape ), ))

def roottransedges( name, shape ):
  return CanonicalTransformChain(( RootTransEdges( name, shape ), ))

def maptrans( coords, vertices ):
  return CanonicalTransformChain(( MapTrans( coords, vertices ), ))

def equivalent( trans1, trans2, flipped=False ):
  trans1 = TransformChain( trans1 )
  trans2 = TransformChain( trans2 )
  if trans1 == trans2:
    return not flipped
  while trans1 and trans2 and trans1[0] == trans2[0]:
    trans1 = trans1.slicefrom(1)
    trans2 = trans2.slicefrom(1)
  return numpy.all( trans1.linear == trans2.linear ) and numpy.all( trans1.offset == trans2.offset ) and trans1.isflipped^trans2.isflipped == flipped


## INSTANCES

identity = CanonicalTransformChain()

def solve( T1, T2 ): # T1 << x == T2
  assert isinstance( T1, TransformChain )
  assert isinstance( T2, TransformChain )
  while T1 and T2 and T1[0] == T2[0]:
    T1 = T1.slicefrom(1)
    T2 = T2.slicefrom(1)
  if not T1:
    return T2
  # A1 * ( Ax * xi + bx ) + b1 == A2 * xi + b2 => A1 * Ax = A2, A1 * bx + b1 = b2
  Ax, bx = numeric.solve_exact( T1.linear, T2.linear, T2.offset - T1.offset )
  return affine( Ax, bx )


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
