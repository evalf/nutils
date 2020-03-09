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

from . import cache, numeric, util, types, _
import numpy, collections, itertools, functools, operator


## TRANSFORM CHAIN OPERATIONS

def apply(chain, points):
  for trans in reversed(chain):
    points = trans.apply(points)
  return points

def n_ascending(chain):
  # number of ascending transform items counting from root (0). this is a
  # temporary hack required to deal with Bifurcate/Slice; as soon as we have
  # proper tensorial topologies we can switch back to strictly ascending
  # transformation chains.
  for n, trans in enumerate(chain):
    if trans.todims is not None and trans.todims < trans.fromdims:
      return n
  return len(chain)

def canonical(chain):
  # keep at lowest ndims possible; this is the required form for bisection
  n = n_ascending(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = 0
  while items[i].fromdims > items[n-1].fromdims:
    swapped = items[i+1].swapdown(items[i])
    if swapped:
      items[i:i+2] = swapped
      i -= i > 0
    else:
      i += 1
  return tuple(items)

def uppermost(chain):
  # bring to highest ndims possible
  n = n_ascending(chain)
  if n < 2:
    return tuple(chain)
  items = list(chain)
  i = n
  while items[i-1].todims < items[0].todims:
    swapped = items[i-2].swapup(items[i-1])
    if swapped:
      items[i-2:i] = swapped
      i += i < n
    else:
      i -= 1
  return tuple(items)

def promote(chain, ndims):
  # swap transformations such that ndims is reached as soon as possible, and
  # then maintained as long as possible (i.e. proceeds as canonical).
  for i, item in enumerate(chain): # NOTE possible efficiency gain using bisection
    if item.fromdims == ndims:
      return canonical(chain[:i+1]) + uppermost(chain[i+1:])
  return chain # NOTE at this point promotion essentially failed, maybe it's better to raise an exception

def linearfrom(chain, fromdims):
  todims = chain[0].todims if chain else fromdims
  while chain and fromdims < chain[-1].fromdims:
    chain = chain[:-1]
  if not chain:
    assert todims == fromdims
    return numpy.eye(fromdims)
  linear = numpy.eye(chain[-1].fromdims)
  for transitem in reversed(uppermost(chain)):
    linear = numpy.dot(transitem.linear, linear)
    if transitem.todims == transitem.fromdims + 1:
      linear = numpy.concatenate([linear, transitem.ext[:,_]], axis=1)
  assert linear.shape[0] == todims
  return linear[:,:fromdims] if linear.shape[1] >= fromdims \
    else numpy.concatenate([linear, numpy.zeros((todims, fromdims-linear.shape[1]))], axis=1)

## TRANSFORM ITEMS

class TransformItem(types.Singleton):
  '''Affine transformation.

  Base class for transformations of the type :math:`x ↦ A x + b`.

  Args
  ----
  todims : :class:`int`
      Dimension of the affine transformation domain.
  fromdims : :class:`int`
      Dimension of the affine transformation range.
  '''

  __slots__ = 'todims', 'fromdims'

  @types.apply_annotations
  def __init__(self, todims, fromdims:int):
    super().__init__()
    self.todims = todims
    self.fromdims = fromdims

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self)

  def swapup(self, other):
    return None

  def swapdown(self, other):
    return None

stricttransformitem = types.strict[TransformItem]
stricttransform = types.tuple[stricttransformitem]

class Bifurcate(TransformItem):

  __slots__ = 'trans1', 'trans2'

  @types.apply_annotations
  def __init__(self, trans1:canonical, trans2:canonical):
    fromdims = trans1[-1].fromdims + trans2[-1].fromdims
    self.trans1 = trans1 + (Slice(0, trans1[-1].fromdims, fromdims),)
    self.trans2 = trans2 + (Slice(trans1[-1].fromdims, fromdims, fromdims),)
    super().__init__(todims=trans1[0].todims if trans1[0].todims == trans2[0].todims else None, fromdims=fromdims)

  def __str__(self):
    return '{}<>{}'.format(self.trans1, self.trans2)

  def apply(self, points):
    return apply(self.trans1, points), apply(self.trans2, points)

class Matrix(TransformItem):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×m` matrix, :math:`n≥m`

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'linear', 'offset'

  @types.apply_annotations
  def __init__(self, linear:types.frozenarray, offset:types.frozenarray):
    assert linear.ndim == 2 and linear.dtype == float
    assert offset.ndim == 1 and offset.dtype == float
    assert len(offset) == len(linear)
    self.linear = linear
    self.offset = offset
    super().__init__(linear.shape[0], linear.shape[1])

  def apply(self, points):
    assert points.shape[-1] == self.fromdims
    return types.frozenarray(numpy.dot(points, self.linear.T) + self.offset, copy=False)

  def __mul__(self, other):
    assert isinstance(other, Matrix) and self.fromdims == other.todims
    linear = numpy.dot(self.linear, other.linear)
    offset = self.apply(other.offset)
    return Square(linear, offset) if self.todims == other.fromdims \
      else Updim(linear, offset, self.isflipped^other.isflipped) if self.todims == other.fromdims+1 \
      else Matrix(linear, offset)

  def __str__(self):
    if not hasattr(self, 'offset') or not hasattr(self, 'linear'):
      return '<uninitialized>'
    return util.obj2str(self.offset) + ''.join('+{}*x{}'.format(util.obj2str(v), i) for i, v in enumerate(self.linear.T))

class Square(Matrix):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` square

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = '_transform_matrix',
  __cache__ ='det',

  @types.apply_annotations
  def __init__(self, linear:types.frozenarray, offset:types.frozenarray):
    assert linear.shape[0] == linear.shape[1]
    self._transform_matrix = {}
    super().__init__(linear, offset)

  def invapply(self, points):
    return types.frozenarray(numpy.linalg.solve(self.linear, points - self.offset), copy=False)

  @property
  def det(self):
    return numpy.linalg.det(self.linear)

  @property
  def isflipped(self):
    return self.fromdims > 0 and self.det < 0

  def transform_poly(self, coeffs):
    assert coeffs.ndim == self.fromdims + 1
    degree = coeffs.shape[1] - 1
    assert all(n == degree+1 for n in coeffs.shape[2:])
    try:
      M = self._transform_matrix[degree]
    except KeyError:
      eye = numpy.eye(self.fromdims, dtype=int)
      # construct polynomials for affine transforms of individual dimensions
      polys = numpy.zeros((self.fromdims,)+(2,)*self.fromdims)
      polys[(slice(None),)+(0,)*self.fromdims] = self.offset
      for idim, e in enumerate(eye):
        polys[(slice(None),)+tuple(e)] = self.linear[:,idim]
      # reduces polynomials to smallest nonzero power
      polys = [poly[tuple(slice(None if p else 1) for p in poly[tuple(eye)])] for poly in polys]
      # construct transform poly by transforming all monomials separately and summing
      M = numpy.zeros((degree+1,)*(2*self.fromdims), dtype=float)
      for powers in numpy.ndindex(*[degree+1]*self.fromdims):
        if sum(powers) <= degree:
          M_power = functools.reduce(numeric.poly_mul, [numeric.poly_pow(poly, power) for poly, power in zip(polys, powers)])
          M[tuple(slice(n) for n in M_power.shape)+powers] += M_power
      self._transform_matrix[degree] = M
    return numpy.einsum('jk,ik', M.reshape([(degree+1)**self.fromdims]*2), coeffs.reshape(coeffs.shape[0],-1)).reshape(coeffs.shape)

class Shift(Square):
  '''Shift transformation :math:`x ↦ x + b`

  Parameters
  ----------
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = ()

  det = 1.

  @types.apply_annotations
  def __init__(self, offset:types.frozenarray):
    assert offset.ndim == 1 and offset.dtype == float
    super().__init__(numpy.eye(len(offset)), offset)

  def apply(self, points):
    return types.frozenarray(points + self.offset, copy=False)

  def invapply(self, points):
    return types.frozenarray(points - self.offset, copy=False)

  def __str__(self):
    return '{}+x'.format(util.obj2str(self.offset))

class Identity(Shift):
  '''Identity transformation :math:`x ↦ x`

  Parameters
  ----------
  ndims : :class:`int`
      Dimension of :math:`x`.
  '''

  __slots__ = ()

  def __init__(self, ndims):
    super().__init__(numpy.zeros(ndims))

  def apply(self, points):
    return points

  def invapply(self, points):
    return points

  def __str__(self):
    return 'x'

class Scale(Square):
  '''Affine transformation :math:`x ↦ a x + b`, with :math:`a` a scalar

  Parameters
  ----------
  scale : :class:`float`
      The scalar :math:`a`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'scale',

  @types.apply_annotations
  def __init__(self, scale:float, offset:types.frozenarray):
    assert offset.ndim == 1 and offset.dtype == float
    self.scale = scale
    super().__init__(numpy.eye(len(offset)) * scale, offset)

  def apply(self, points):
    return types.frozenarray(self.scale * points + self.offset, copy=False)

  def invapply(self, points):
    return types.frozenarray((points - self.offset) / self.scale, copy=False)

  @property
  def det(self):
    return self.scale**self.todims

  def __str__(self):
    return '{}+{}*x'.format(util.obj2str(self.offset), self.scale)

  def __mul__(self, other):
    assert isinstance(other, Matrix) and self.fromdims == other.todims
    if isinstance(other, Scale):
      return Scale(self.scale * other.scale, self.apply(other.offset))
    return super().__mul__(other)

class Updim(Matrix):
  '''Affine transformation :math:`x ↦ A x + b`, with :math:`A` an :math:`n×(n-1)` matrix

  Parameters
  ----------
  linear : :class:`numpy.ndarray`
      The transformation matrix :math:`A`.
  offset : :class:`numpy.ndarray`
      The offset :math:`b`.
  '''

  __slots__ = 'isflipped',
  __cache__ = 'ext',

  @types.apply_annotations
  def __init__(self, linear:types.frozenarray, offset:types.frozenarray, isflipped:bool):
    assert linear.shape[0] == linear.shape[1] + 1
    self.isflipped = isflipped
    super().__init__(linear, offset)

  @property
  def ext(self):
    ext = numeric.ext(self.linear)
    return types.frozenarray(-ext if self.isflipped else ext, copy=False)

  @property
  def flipped(self):
    return Updim(self.linear, self.offset, not self.isflipped)

  def swapdown(self, other):
    if isinstance(other, TensorChild):
      return ScaledUpdim(other, self), Identity(self.fromdims)

class SimplexEdge(Updim):

  __slots__ = 'iedge', 'inverted'

  swap = (
    ((1,0), (2,0), (3,0), (7,1)),
    ((0,1), (2,1), (3,1), (6,1)),
    ((0,2), (1,2), (3,2), (5,1)),
    ((0,3), (1,3), (2,3), (4,3)),
  )

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, iedge:types.strictint, inverted:bool=False):
    assert ndims >= iedge >= 0
    self.iedge = iedge
    self.inverted = inverted
    vertices = numpy.concatenate([numpy.zeros(ndims)[_,:], numpy.eye(ndims)], axis=0)
    coords = vertices[list(range(iedge))+list(range(iedge+1,ndims+1))]
    super().__init__((coords[1:]-coords[0]).T, coords[0], inverted^(iedge%2))

  @property
  def flipped(self):
    return SimplexEdge(self.todims, self.iedge, not self.inverted)

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, SimplexChild):
      ichild, iedge = self.swap[self.iedge][other.ichild]
      return SimplexChild(self.todims, ichild), SimplexEdge(self.todims, iedge, self.inverted)

  def swapdown(self, other):
    # prioritize decending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, SimplexChild):
      key = other.ichild, self.iedge
      for iedge, children in enumerate(self.swap[:self.todims+1]):
        try:
          ichild = children[:2**self.fromdims].index(key)
        except ValueError:
          pass
        else:
          return SimplexEdge(self.todims, iedge, self.inverted), SimplexChild(self.fromdims, ichild)

class SimplexChild(Square):

  __slots__ = 'ichild',

  def __init__(self, ndims, ichild):
    self.ichild = ichild
    if ichild <= ndims:
      linear = numpy.eye(ndims) * .5
      offset = linear[ichild-1] if ichild else numpy.zeros(ndims)
    elif ndims == 2 and ichild == 3:
      linear = (-.5,0), (.5,.5)
      offset = .5, 0
    elif ndims == 3 and ichild == 4:
      linear = (-.5,0,-.5), (.5,.5,0), (0,0,.5)
      offset = .5, 0, 0
    elif ndims == 3 and ichild == 5:
      linear = (0,-.5,0), (.5,0,0), (0,.5,.5)
      offset = .5, 0, 0
    elif ndims == 3 and ichild == 6:
      linear = (.5,0,0), (0,-.5,0), (0,.5,.5)
      offset = 0, .5, 0
    elif ndims == 3 and ichild == 7:
      linear = (-.5,0,-.5), (-.5,-.5,0), (.5,.5,.5)
      offset = .5, .5, 0
    else:
      raise NotImplementedError('SimplexChild(ndims={}, ichild={})'.format(ndims, ichild))
    super().__init__(linear, offset)

class Slice(Matrix):

  __slots__ = 's',

  @types.apply_annotations
  def __init__(self, i1:int, i2:int, fromdims:int):
    todims = i2-i1
    assert 0 <= todims <= fromdims
    self.s = slice(i1,i2)
    super().__init__(numpy.eye(fromdims)[self.s], numpy.zeros(todims))

  def apply(self, points):
    return types.frozenarray(points[:,self.s])

class ScaledUpdim(Updim):

  __slots__ = 'trans1', 'trans2'

  def __init__(self, trans1, trans2):
    assert trans1.todims == trans1.fromdims == trans2.todims == trans2.fromdims + 1
    self.trans1 = trans1
    self.trans2 = trans2
    super().__init__(numpy.dot(trans1.linear, trans2.linear), trans1.apply(trans2.offset), trans1.isflipped^trans2.isflipped)

  def swapup(self, other):
    if type(other) is Identity:
      return self.trans1, self.trans2

  @property
  def flipped(self):
    return ScaledUpdim(self.trans1, self.trans2.flipped)

class TensorEdge1(Updim):

  __slots__ = 'trans',

  def __init__(self, trans1, ndims2):
    self.trans = trans1
    super().__init__(linear=numeric.blockdiag([trans1.linear, numpy.eye(ndims2)]), offset=numpy.concatenate([trans1.offset, numpy.zeros(ndims2)]), isflipped=trans1.isflipped)

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, TensorChild) and self.trans.fromdims == other.trans1.todims:
      swapped = self.trans.swapup(other.trans1)
      trans2 = other.trans2
    elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdims == other.todims and not self.trans.fromdims:
      swapped = self.trans.swapup(SimplexChild(0, 0))
      trans2 = other
    else:
      swapped = None
    if swapped:
      child, edge = swapped
      return TensorChild(child, trans2), TensorEdge1(edge, trans2.fromdims)

  def swapdown(self, other):
    # prioritize ascending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, TensorChild) and other.trans1.fromdims == self.trans.todims:
      swapped = self.trans.swapdown(other.trans1)
      if swapped:
        edge, child = swapped
        return TensorEdge1(edge, other.trans2.todims), TensorChild(child, other.trans2) if child.fromdims else other.trans2
      return ScaledUpdim(other, self), Identity(self.fromdims)

  @property
  def flipped(self):
    return TensorEdge1(self.trans.flipped, self.fromdims-self.trans.fromdims)

class TensorEdge2(Updim):

  __slots__ = 'trans'

  def __init__(self, ndims1, trans2):
    self.trans = trans2
    super().__init__(linear=numeric.blockdiag([numpy.eye(ndims1), trans2.linear]), offset=numpy.concatenate([numpy.zeros(ndims1), trans2.offset]), isflipped=trans2.isflipped^(ndims1%2))

  def swapup(self, other):
    # prioritize ascending transformations, i.e. change updim << scale to scale << updim
    if isinstance(other, TensorChild) and self.trans.fromdims == other.trans2.todims:
      swapped = self.trans.swapup(other.trans2)
      trans1 = other.trans1
    elif isinstance(other, (TensorChild, SimplexChild)) and other.fromdims == other.todims and not self.trans.fromdims:
      swapped = self.trans.swapup(SimplexChild(0, 0))
      trans1 = other
    else:
      swapped = None
    if swapped:
      child, edge = swapped
      return TensorChild(trans1, child), TensorEdge2(trans1.fromdims, edge)

  def swapdown(self, other):
    # prioritize ascending transformations, i.e. change scale << updim to updim << scale
    if isinstance(other, TensorChild) and other.trans2.fromdims == self.trans.todims:
      swapped = self.trans.swapdown(other.trans2)
      if swapped:
        edge, child = swapped
        return TensorEdge2(other.trans1.todims, edge), TensorChild(other.trans1, child) if child.fromdims else other.trans1
      return ScaledUpdim(other, self), Identity(self.fromdims)

  @property
  def flipped(self):
    return TensorEdge2(self.fromdims-self.trans.fromdims, self.trans.flipped)

class TensorChild(Square):

  __slots__ = 'trans1', 'trans2'
  __cache__ = 'det',

  def __init__(self, trans1, trans2):
    assert trans1.fromdims and trans2.fromdims
    self.trans1 = trans1
    self.trans2 = trans2
    linear = numeric.blockdiag([trans1.linear, trans2.linear])
    offset = numpy.concatenate([trans1.offset, trans2.offset])
    super().__init__(linear, offset)

  @property
  def det(self):
    return self.trans1.det * self.trans2.det

class Identifier(Identity):
  '''Generic identifier

  This transformation serves as an element-specific or topology-specific token
  to form the basis of transformation lookups. Otherwise, the transform behaves
  like an identity.
  '''

  __slots__ = 'token'

  @types.apply_annotations
  def __init__(self, ndims:int, token):
    self.token = token
    super().__init__(ndims)

  def __str__(self):
    return ':'.join(map(str, self._args))

# vim:sw=2:sts=2:et
