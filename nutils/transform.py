# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
The transform module.
"""

from . import cache, numeric, core, _
import numpy, collections, itertools, functools, operator


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
    return isflipped( self )

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
    return offset( self )

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
  def n_ascending(self):
    # number of ascending transform items counting from root (0). this is a
    # temporary hack required to deal with Bifurcate/Slice; as soon as we have
    # proper tensorial topologies we can switch back to strictly ascending
    # transformation chains.
    for n, trans in enumerate(self):
      if trans.todims is not None and trans.todims < trans.fromdims:
        return n
    return len(self)

  @property
  def canonical( self ):
    # keep at lowest ndims possible; this is the required form for bisection
    n = self.n_ascending
    if n < 2:
      return CanonicalTransformChain(self)
    items = list(self)
    i = 0
    while items[i].fromdims > items[n-1].fromdims:
      swapped = items[i+1].swapdown(items[i])
      if swapped:
        items[i:i+2] = swapped
        i -= i > 0
      else:
        i += 1
    return CanonicalTransformChain(items)

  def promote(self, ndims):
    # split self into chain1 and chain2 such that self == chain1 << chain2 and
    # chain1.fromdims == chain2.todims == ndims, where chain1 is canonical and
    # chain2 climbs to ndims as fast as possible.
    n = self.n_ascending
    assert ndims >= self[n-1].fromdims
    items = list(self)
    i = n
    while items[i-1].fromdims < ndims:
      swapped = items[i-2].swapup(items[i-1])
      if swapped:
        items[i-2:i] = swapped
        i += i < n
      else:
        i -= 1
    assert items[i-1].fromdims == ndims
    return TransformChain(items[:i]).canonical, TransformChain(items[i:])

  def lookup( self, transforms ):
    if not transforms:
      return
    for trans in transforms:
      ndims = trans.fromdims
      break
    head, tail = self.canonical.promote( ndims )
    while head:
      if head in transforms:
        return CanonicalTransformChain(head), TransformChain(tail)
      tail = head[-1:] + tail
      head = head[:-1]

  def lookup_item( self, transforms ):
    head_tail = self.lookup( transforms )
    if not head_tail:
      raise KeyError( self )
    head, tail = head_tail
    item = transforms[head] if isinstance(transforms, collections.Mapping) \
      else transforms.index( head )
    return item, tail

class CanonicalTransformChain( TransformChain ):

  __slots__ = ()

  @property
  def canonical( self ):
    return self


## TRANSFORM ITEMS

class TransformItem( cache.Immutable ):

  def __init__(self, todims, fromdims:int):
    self.todims = todims
    self.fromdims = fromdims

  def __repr__( self ):
    return '{}( {} )'.format( self.__class__.__name__, self )

  def swapup(self, other):
    return None

  def swapdown(self, other):
    return None

class Shift( TransformItem ):

  def __init__(self, offset:numeric.const):
    self.linear = self.det = numpy.array(1.)
    self.offset = offset
    self.isflipped = False
    assert offset.ndim == 1
    super().__init__(offset.shape[0], offset.shape[0])

  def apply( self, points ):
    return numeric.const(points + self.offset, copy=False)

  def __str__( self ):
    return '{}+x'.format( numeric.fstr(self.offset) )

class Scale( TransformItem ):

  def __init__(self, scale:float, offset:numeric.const):
    assert offset.ndim == 1
    assert scale != 1
    self.scale = scale
    self.linear = numpy.array(scale)
    self.offset = offset
    self.isflipped = scale < 0 and len(offset)%2 == 1
    self._transform_matrix = {}
    super().__init__(offset.shape[0], offset.shape[0])

  def apply( self, points ):
    return numeric.const(self.scale * points + self.offset, copy=False)

  @property
  def det( self ):
    return self.linear**self.todims

  def __str__( self ):
    return '{}+{}*x'.format( numeric.fstr(self.offset), numeric.fstr(self.linear) )

  def transform_poly(self, coeffs):
    n, *p = coeffs.shape
    ndim = coeffs.ndim-1
    p, = set(p)
    p -= 1
    try:
      M = self._transform_matrix[p,ndim]
    except KeyError:
      M = numpy.zeros((p+1,)*(2*ndim), dtype=float)
      for i in itertools.product(*[range(p+1)]*ndim):
        if sum(i) <= p:
          for j in itertools.product(*(range(k+1) for k in i)):
            M[j+i] = functools.reduce(operator.mul, (numeric.binom(i[k], j[k])*self.offset[k]**(i[k]-j[k]) for k in range(ndim)), self.scale**sum(j))
      M = self._transform_matrix[p,ndim] = M.reshape([(p+1)**ndim]*2)
    return numpy.einsum('jk,ik', self._transform_matrix[p,ndim], coeffs.reshape(n,-1)).reshape(coeffs.shape)

class Matrix( TransformItem ):

  def __init__(self, linear:numeric.const, offset:numeric.const):
    self.linear = linear
    self.offset = offset
    assert linear.ndim == 2 and offset.shape == linear.shape[:1]
    super().__init__(linear.shape[0], linear.shape[1])

  def apply( self, points ):
    assert points.shape[-1] == self.fromdims
    return numeric.const(numpy.dot( points, self.linear.T ) + self.offset, copy=False)

  def __str__( self ):
    return '{}{}{}'.format( '~' if self.isflipped else '', numeric.fstr(self.offset), ''.join( '+{}*x{}'.format( numeric.fstr(v), i ) for i, v in enumerate(self.linear.T) ) )

class Square( Matrix ):

  def __init__(self, linear:numeric.const, offset:numeric.const):
    assert linear.shape[0] == linear.shape[1]
    super().__init__(linear, offset)

  @property
  def isflipped( self ):
    return self.det < 0

  @cache.property
  def det( self ):
    return numeric.det_exact( self.linear )

class Updim( Matrix ):

  def __init__(self, linear:numeric.const, offset:numeric.const, isflipped:bool):
    assert linear.shape[0] > linear.shape[1]
    self.isflipped = isflipped
    super().__init__(linear, offset)

  @cache.property
  def ext( self ):
    ext = numeric.ext( self.linear )
    return -ext if self.isflipped else ext

  @property
  def flipped( self ):
    return Updim( self.linear, self.offset, not self.isflipped )

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if self.todims == self.fromdims + 1 and isinstance(other, Scale) and other.scale == .5:
      return Scale(other.linear, self.apply(other.offset) - other.linear * self.offset), self

  def swapdown(self, other):
    # prioritize decending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, Scale) and other.scale == .5 and self.todims == self.fromdims + 1 and self.fromdims > 0:
      trans12 = TransformChain((other, self)).flat
      try:
        newlinear, newoffset = numeric.solve_exact(self.linear, trans12.linear, trans12.offset - self.offset)
      except numpy.linalg.LinAlgError:
        pass
      else:
        trans21 = TransformChain((self,) + affine(newlinear, newoffset))
        assert trans21.flat == trans12
        return trans21

class Bifurcate( TransformItem ):
  'bifurcate'

  def __init__(self, trans1, trans2):
    assert trans1.fromdims == trans2.fromdims
    self.trans1 = trans1
    self.trans2 = trans2
    super().__init__(todims=trans1.todims if trans1.todims == trans2.todims else None, fromdims=trans1.fromdims)

  def apply( self, points ):
    return (self.trans1.apply(points), self.trans2.apply(points))

class Slice( Matrix ):
  'slice'

  def __init__(self, i1:int, i2:int, fromdims:int):
    todims = i2-i1
    assert 0 <= todims <= fromdims
    self.s = slice(i1,i2)
    self.isflipped = False
    super().__init__(numpy.eye(fromdims)[self.s], numpy.zeros(todims))

  def apply( self, points ):
    return numeric.const(points[:,self.s])

class VertexTransform( TransformItem ):

  def __init__(self, fromdims:int):
    self.isflipped = False
    super().__init__(None, fromdims)

class MapTrans( VertexTransform ):

  def __init__(self, linear:numeric.const, offset:numeric.const, vertices:numeric.const):
    assert len(linear) == len(offset) == len(vertices)
    self.vertices, self.linear, self.offset = map( numpy.array, zip( *sorted( zip( vertices, linear, offset ) ) ) ) # sort vertices
    super().__init__(self.linear.shape[1])

  def apply( self, points ):
    barycentric = numpy.dot( points, self.linear.T ) + self.offset
    return tuple( tuple( (v,float(c)) for v, c in zip( self.vertices, coord ) if c ) for coord in barycentric )

  def __str__( self ):
    return ','.join( str(v) for v in self.vertices )

class RootTrans( VertexTransform ):

  def __init__(self, name, shape:tuple):
    self.I, = numpy.where( shape )
    self.w = numpy.take( shape, self.I )
    self.name = name
    super().__init__(len(shape))

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

  def __init__(self, name, shape:tuple):
    self.shape = shape
    assert numeric.isarray(name)
    assert name.shape == (3,)*len(shape)
    self.name = name.copy()
    super().__init__(len(shape))

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
    elif n == 1 or r_linear[0,-1] == 0 and numpy.equal(r_linear, r_linear[0,0] * numpy.eye(n)).all():
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

def maptrans( linear, offset, vertices ):
  return CanonicalTransformChain(( MapTrans( linear, offset, vertices ), ))

def equivalent( trans1, trans2 ):
  trans1 = TransformChain( trans1 )
  trans2 = TransformChain( trans2 )
  if trans1 == trans2:
    return True
  while trans1 and trans2 and trans1[0] == trans2[0]:
    trans1 = trans1[1:]
    trans2 = trans2[1:]
  return numpy.equal(fulllinear(trans1), fulllinear(trans2)).all() and numpy.equal(offset(trans1), offset(trans2)).all()


## INSTANCES

identity = CanonicalTransformChain()

def solve( T1, T2 ): # T1 << x == T2
  assert isinstance( T1, TransformChain )
  assert isinstance( T2, TransformChain )
  while T1 and T2 and T1[0] == T2[0]:
    T1 = T1[1:]
    T2 = T2[1:]
  if not T1:
    return TransformChain(T2)
  # A1 * ( Ax * xi + bx ) + b1 == A2 * xi + b2 => A1 * Ax = A2, A1 * bx + b1 = b2
  Ax, bx = numeric.solve_exact( linear(T1), linear(T2), offset(T2) - offset(T1) )
  return affine( Ax, bx )

def tensor( trans1, trans2 ):
  if not trans1 and not trans2:
    return identity
  return affine( trans1.linear if trans1.linear.ndim == 0 and trans2.linear.ndim == 0 and trans1.linear == trans2.linear
            else numeric.blockdiag([ trans1.linear, trans2.linear ]), numpy.concatenate([ trans1.offset, trans2.offset ]) )

def isflipped( chain ):
  return sum( trans.isflipped for trans in chain ) % 2 == 1

def linear( chain ):
  linear = numpy.array( 1. )
  for trans in chain:
    linear = numpy.dot( linear, trans.linear ) if linear.ndim and trans.linear.ndim \
        else linear * trans.linear
  return linear

def fulllinear( chain ):
  scale = 1
  linear = numpy.eye( chain[-1].fromdims )
  for trans in reversed(chain):
    if trans.linear.ndim == 0:
      scale *= trans.linear
    else:
      linear = numpy.dot( trans.linear, linear )
      if trans.todims > trans.fromdims:
        linear = numpy.concatenate( [ linear, trans.ext[:,_] ], axis=1 )
  return linear * scale

def linearfrom( chain, ndims ):
  if chain and ndims < chain[-1].fromdims:
    for i in reversed(range(len(chain))):
      if chain[i].todims == ndims:
        chain = chain[:i]
        break
    else:
      raise Exception( 'failed to find {}D coordinate system'.format(ndims) )
  if not chain:
    return numpy.eye( ndims )
  linear = fulllinear( chain )
  n, m = linear.shape
  if m >= ndims:
    return linear[:,:ndims]
  return numpy.concatenate( [ linear, numpy.zeros((n,ndims-m)) ], axis=1 )

def apply( chain, points ):
  for trans in reversed(chain):
    points = trans.apply( points )
  return points

def offset( chain ):
  offset = chain[-1].offset
  for trans in chain[-2::-1]:
    offset = trans.apply( offset )
  return offset

def slicetrans( i1, i2, n ):
  return CanonicalTransformChain( [ Slice(i1,i2,n) ] )

def stack( trans1, trans2 ):
  fromdims = trans1.fromdims + trans2.fromdims
  return bifurcate( trans1.canonical << slicetrans(0,trans1.fromdims,fromdims), trans2.canonical << slicetrans(trans1.fromdims,fromdims,fromdims) )

def bifurcate( trans1, trans2 ):
  return CanonicalTransformChain([ Bifurcate( trans1, trans2 ) ])

def invapply( trans, points ):
  A = linear(trans)
  b = points - offset(trans)
  return b / A if isinstance(A,float) else numpy.linalg.solve( A, b )

def transform_poly(trans, coeffs):
  for item in trans:
    coeffs = item.transform_poly(coeffs)
  return coeffs

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
