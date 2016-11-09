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

from . import cache, numeric, core, _
import numpy


class TransformChain( tuple ):

  __slots__ = ()

  def startswith( self, other ):
    return self[:len(other)] == other

  @property
  def trimmed( self ):
    for i in range( len(self)-1, -1, -1 ):
      if self[i].todims != self.fromdims:
        return self.slicefrom( i+1 )
    return self

  def slicefrom( self, i ):
    return self.__class__( self[i:] )

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
    return self.__class__( trans.flipped if trans.todims == trans.fromdims+1 else trans for trans in self )

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
    return linear( self )

  def apply( self, points ):
    return apply( self, points )

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

  def promote_trim( self, ndims ):
    raise Exception( 'promotion only possible from canonical form' )

  def lookup_split( self, transforms ):
    # to be replaced by bisection soon
    head = self
    while head:
      if head in transforms:
        return head, self.slicefrom(len(head))
      head = head[:-1]
    return CanonicalTransformChain(), self

  def promote_and_lookup( self, transforms ):
    if isinstance( transforms, (list,tuple) ):
      transforms = dict( zip( transforms, range(len(transforms)) ) )
    assert isinstance( transforms, dict )
    for trans in transforms:
      ndims = trans.fromdims
      break
    head, tail = self.canonical.promote_split( ndims )
    while head and head[-1].fromdims == ndims:
      try:
        item = transforms[head]
      except KeyError:
        pass
      else:
        return item, TransformChain(tail)
      tail = head[-1:] + tail
      head = head[:-1]
    raise KeyError( self )

  def contained_in( self, transforms ):
    for trans in transforms:
      ndims = trans.fromdims
      break
    head, tail = self.canonical.promote_split( ndims )
    while head and head[-1].fromdims == ndims:
      if head in transforms:
        return True
      head = head[:-1]
    return False

class CanonicalTransformChain( TransformChain ):

  __slots__ = ()

  def __lshift__( self, other ):
    # self << other
    joint = TransformChain.__lshift__( self, other )
    if self and other and isinstance( other, CanonicalTransformChain ) and not mayswap( self[-1], other[0] ):
      joint = CanonicalTransformChain( joint )
    return joint

  @property
  def canonical( self ):
    return self

  def promote_helper( self ):
    index = core.index( trans.fromdims == self.fromdims for trans in self )
    head = self[:index]
    uptrans = self[index]
    if index == len(self)-1 or not mayswap( self[index+1], uptrans ):
      tail = self[index:]
    else:
      for i in range( index+1, len(self) ):
        scale = self[i]
        if not mayswap( scale, uptrans ):
          break
        head += Scale( scale.linear, uptrans.apply(scale.offset) - scale.linear * uptrans.offset ),
      else:
        i = len(self)+1
      assert equivalent( head[index:]+(uptrans,), self[index:i] )
      tail = (uptrans,) + self[i:]
    return CanonicalTransformChain(head), CanonicalTransformChain(tail)

  def promote_split( self, ndims ):
    head = self
    tail = ()
    while head.fromdims < ndims:
      head, tmp = head.promote_helper()
      tail = tmp + tail
    return head, CanonicalTransformChain(tail)

  def promote_trim( self, ndims ):
    head, tail = self.promote_split( ndims )
    return head

  def promote( self, ndims ):
    head, tail = self.promote_split( ndims )
    return head << tail

  def lookup( self, transforms ):
    # to be replaced by bisection soon
    head, tail = self.lookup_split( transforms )
    if head:
      return CanonicalTransformChain( head )

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
    self.linear = self.det = numpy.array(1.)
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

def linear( chain ):
  linear = numpy.array( 1. )
  for trans in chain:
    linear = numpy.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
        else linear * trans.linear
  return linear

def fulllinear( chain ):
  linear = numpy.eye( chain[-1].fromdims )
  for trans in reversed(chain):
    if trans.linear.ndim == 0:
      linear = trans.linear * linear
    else:
      linear = numpy.dot( trans.linear, linear )
      if linear.shape[0] != linear.shape[1]:
        n = numeric.ext( linear )
        if trans.isflipped:
          n = -n
        linear = numpy.concatenate( [ linear, n[:,_] ], axis=1 )
  return linear

def apply( chain, points ):
  for trans in reversed(chain):
    points = trans.apply( points )
  return points


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
