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
The element module defines reference elements such as the
:class:`LineReference` and :class:`TriangleReference`, but also more exotic
objects like the :class:`MosaicReference`. A set of (interconnected) elements
together form a :class:`nutils.topology.Topology`. Elements have edges and
children (for refinement), which are in turn elements and map onto self by an
affine transformation. They also have a well defined reference coordinate
system, and provide pointsets for purposes of integration and sampling.
"""

from . import util, numpy, numeric, cache, transform, warnings, types, points, _
import re, math, itertools, operator, functools


## REFERENCE ELEMENTS

class Reference(types.Singleton):
  'reference element'

  __slots__ = 'ndims',
  __cache__ = 'connectivity', 'edgechildren', 'ribbons', 'volume', 'centroid', '_linear_bernstein', 'getpoints'

  @types.apply_annotations
  def __init__(self, ndims:int):
    super().__init__()
    self.ndims = ndims

  @property
  def nverts(self):
    return len(self.vertices)

  __and__ = lambda self, other: self if self == other else NotImplemented
  __or__ = lambda self, other: self if self == other else NotImplemented
  __rand__ = lambda self, other: self.__and__(other)
  __ror__ = lambda self, other: self.__or__(other)
  __sub__ = __rsub__ = lambda self, other: self.empty if self == other else NotImplemented
  __bool__ = __nonzero__ = lambda self: bool(self.volume)

  @property
  def empty(self):
    return EmptyLike(self)

  def __mul__(self, other):
    assert isinstance(other, Reference)
    return self if not other.ndims else other if not self.ndims else TensorReference(self, other)

  def __pow__(self, n):
    assert numeric.isint(n) and n >= 0
    return getsimplex(0) if n == 0 \
      else self if n == 1 \
      else self * self**(n-1)

  @property
  def nedges(self):
    return len(self.edge_transforms)

  @property
  def nchildren(self):
    return len(self.child_transforms)

  @property
  def edges(self):
    return list(zip(self.edge_transforms, self.edge_refs))

  @property
  def children(self):
    return list(zip(self.child_transforms, self.child_refs))

  @property
  def connectivity(self):
    # Nested tuple with connectivity information about edges of children:
    # connectivity[ichild][iedge] = ioppchild (interface) or -1 (boundary).
    connectivity = [-numpy.ones(child.nedges, dtype=int) for child in self.child_refs]
    vmap = {}
    for ichild, (ctrans, cref) in enumerate(self.children):
      for iedge, etrans in enumerate(cref.edge_transforms):
        v = ctrans * etrans
        try:
          jchild, jedge = vmap.pop(v.flipped)
        except KeyError:
          vmap[v] = ichild, iedge
        else:
          connectivity[jchild][jedge] = ichild
          connectivity[ichild][iedge] = jchild
    for etrans, eref in self.edges:
      for ctrans in eref.child_transforms:
        vmap.pop(etrans * ctrans, None)
    assert not any(self.child_refs[ichild].edge_refs[iedge] for ichild, iedge in vmap.values()), 'not all boundary elements recovered'
    return tuple(types.frozenarray(c, copy=False) for c in connectivity)

  @property
  def edgechildren(self):
    edgechildren = []
    for iedge, (etrans, eref) in enumerate(self.edges):
      children = []
      for ichild, ctrans in enumerate(eref.child_transforms):
        ctrans_, etrans_ = etrans.swapup(ctrans)
        ichild_ = self.child_transforms.index(ctrans_)
        iedge_ = self.child_refs[ichild].edge_transforms.index(etrans_)
        children.append((ichild_, iedge_))
      edgechildren.append(types.frozenarray(children))
    return tuple(edgechildren)

  @property
  def ribbons(self):
    # tuples of (iedge1,jedge1), (iedge2,jedge2) pairs
    assert self.ndims >= 2
    transforms = {}
    ribbons = []
    for iedge1, (etrans1,edge1) in enumerate(self.edges):
      if edge1:
        for iedge2, (etrans2,edge2) in enumerate(edge1.edges):
          if edge2:
            key = tuple(sorted(tuple(p) for p in (etrans1 * etrans2).apply(edge2.vertices)))
            try:
              jedge1, jedge2 = transforms.pop(key)
            except KeyError:
              transforms[key] = iedge1, iedge2
            else:
              assert self.edge_refs[jedge1].edge_refs[jedge2] == edge2
              ribbons.append(((iedge1,iedge2), (jedge1,jedge2)))
    assert not transforms
    return tuple(ribbons)

  def getischeme(self, ischeme):
    ischeme, degree = parse_legacy_ischeme(ischeme)
    points = self.getpoints(ischeme, degree)
    return points.coords, getattr(points, 'weights', None)

  def getpoints(self, ischeme, degree):
    raise Exception('unsupported ischeme for {}: {!r}'.format(self.__class__.__name__, ischeme))

  def with_children(self, child_refs):
    child_refs = tuple(child_refs)
    if not any(child_refs):
      return self.empty
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference(self, child_refs)

  @property
  def volume(self):
    return self.getpoints('gauss', 1).weights.sum()

  @property
  def centroid(self):
    gauss = self.getpoints('gauss', 1)
    return gauss.coords.T.dot(gauss.weights) / gauss.weights.sum()

  def trim(self, levels, maxrefine, ndivisions):
    'trim element along levelset'

    assert len(levels) == self.nvertices_by_level(maxrefine)
    return self if not self or numpy.greater_equal(levels, 0).all() \
      else self.empty if numpy.less_equal(levels, 0).all() \
      else self.with_children(cref.trim(clevels, maxrefine-1, ndivisions)
            for cref, clevels in zip(self.child_refs, self.child_divide(levels,maxrefine))) if maxrefine > 0 \
      else self.slice(lambda vertices: numeric.dot(numeric.poly_eval(self._linear_bernstein[_], vertices), levels), ndivisions)

  @property
  def _linear_bernstein(self):
    return self.get_poly_coeffs('bernstein', degree=1)

  def slice(self, levelfunc, ndivisions):
    # slice along levelset by recursing over dimensions

    levels = levelfunc(self.vertices)

    assert numeric.isint(ndivisions)
    assert len(levels) == self.nverts
    if numpy.greater_equal(levels, 0).all():
      return self
    if numpy.less_equal(levels, 0).all():
      return self.empty

    nbins = 2**ndivisions

    if self.ndims == 1:

      l0, l1 = levels
      xi = numpy.round(l0/(l0-l1) * nbins)
      if xi in (0,nbins):
        return self.empty if xi == 0 and l1 < 0 or xi == nbins and l0 < 0 else self
      v0, v1 = self.vertices
      midpoint = v0 + (xi/nbins) * (v1-v0)
      refs = [edgeref if levelfunc(edgetrans.apply(numpy.zeros((1,0)))) > 0 else edgeref.empty for edgetrans, edgeref in self.edges]

    else:

      refs = [edgeref.slice(lambda vertices: levelfunc(edgetrans.apply(vertices)), ndivisions) for edgetrans, edgeref in self.edges]
      if sum(ref != baseref for ref, baseref in zip(refs, self.edge_refs)) <= 1:
        return self
      if sum(bool(ref) for ref in refs) <= 1:
        return self.empty

      clevel = levelfunc(self.centroid[_])[0]

      select   = clevel*levels<=0 if clevel!=0 else levels!=0
      levels   = levels[select]
      vertices = self.vertices[select]

      xi = numpy.round(levels/(levels-clevel) * nbins)
      midpoint = numpy.mean(vertices + (self.centroid-vertices)*(xi/nbins)[:,_], axis=0)

    if tuple(refs) == tuple(self.edge_refs):
      return self
    if not any(refs):
      return self.empty

    mosaic = MosaicReference(self, refs, midpoint)
    return self.empty if mosaic.volume == 0 else mosaic if mosaic.volume < self.volume else self

  def cone(self, trans, tip):
    return Cone(self, trans, tip)

  def check_edges(self, tol=1e-15, print=print):
    volume = 0
    zero = 0
    for trans, edge in self.edges:
      if edge:
        gauss = edge.getpoints('gauss', 1)
        w_normal = gauss.weights[:,_] * trans.ext
        zero += w_normal.sum(0)
        volume += numeric.contract(trans.apply(gauss.coords), w_normal, axis=0)
    if numpy.greater(abs(zero), tol).any():
      print('divergence check failed: {} != 0'.format(zero))
    if numpy.greater(abs(volume - self.volume), tol).any():
      print('divergence check failed: {} != {}'.format(volume, self.volume))

  def vertex_cover(self, ctransforms, maxrefine):
    if maxrefine < 0:
      raise Exception('maxrefine is too low')
    npoints = self.nvertices_by_level(maxrefine)
    allindices = numpy.arange(npoints)
    if len(ctransforms) == 1:
      ctrans, = ctransforms
      assert not ctrans
      return ((), self.getpoints('vertex', maxrefine).coords, allindices),
    if maxrefine == 0:
      raise Exception('maxrefine is too low')
    cbins = [set() for ichild in range(self.nchildren)]
    for ctrans in ctransforms:
      ichild = self.child_transforms.index(ctrans[0])
      cbins[ichild].add(ctrans[1:])
    if not all(cbins):
      raise Exception('transformations to not form an element cover')
    fcache = cache.WrapperCache()
    return tuple(((ctrans,) + trans, points, cindices[indices])
      for ctrans, cref, cbin, cindices in zip(self.child_transforms, self.child_refs, cbins, self.child_divide(allindices,maxrefine))
        for trans, points, indices in fcache[cref.vertex_cover](frozenset(cbin), maxrefine-1))

  def __str__(self):
    return self.__class__.__name__

  __repr__ = __str__

  def get_ndofs(self, degree):
    raise NotImplementedError

  def get_poly_coeffs(self, basis, **kwargs):
    raise NotImplementedError

  def get_edge_dofs(self, degree, iedge):
    raise NotImplementedError

strictreference = types.strict[Reference]

class EmptyLike(Reference):
  'inverse reference element'

  __slots__ = 'baseref',

  volume = 0

  @property
  def empty(self):
    return self

  @types.apply_annotations
  def __init__(self, baseref:strictreference):
    self.baseref = baseref
    super().__init__(baseref.ndims)

  @property
  def vertices(self):
    return self.baseref.vertices

  @property
  def edge_transforms(self):
    return self.baseref.edge_transforms

  @property
  def edge_refs(self):
    return tuple(eref.empty for eref in self.baseref.edge_refs)

  @property
  def child_transforms(self):
    return self.baseref.child_transforms

  @property
  def child_refs(self):
    return tuple(cref.empty for cref in self.baseref.child_refs)

  __and__ = __sub__ = lambda self, other: self if other.ndims == self.ndims else NotImplemented
  __or__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented
  __rsub__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented

  def trim(self, levels, maxrefine, ndivisions):
    return self

  def inside(self, point, eps=0):
    return False

class RevolutionReference(Reference):
  'modify gauss integration to always return a single point'

  __slots__ = ()
  __cache__ = 'getpoints',

  def __init__(self):
    super().__init__(ndims=1)

  @property
  def vertices(self):
    return types.frozenarray([[0.]])

  @property
  def edge_transforms(self): # only used in check_edges
    return transform.Updim(numpy.zeros((1,0)), [-numpy.pi], isflipped=True), transform.Updim(numpy.zeros((1,0)), [+numpy.pi], isflipped=False)

  @property
  def edge_refs(self): # idem edge_transforms
    return PointReference(), PointReference()

  @property
  def child_transforms(self):
    return transform.Identity(1),

  @property
  def child_refs(self):
    return self,

  @property
  def simplices(self):
    return (transform.Identity(self.ndims), self),

  def getpoints(self, ischeme, degree):
    return points.CoordsWeightsPoints([[0.]], [2 * numpy.pi])

  def inside(self, point, eps=0):
    return True

  def nvertices_by_level(self, n):
    return 1

  def child_divide(self, vals, n):
    return vals,

  def get_poly_coeffs(self, basis, **kwargs):
    return numpy.ones((1,1)) # single, constant basis function

class SimplexReference(Reference):
  'simplex reference'

  __slots__ = ()
  __cache__ = 'edge_refs', 'edge_transforms', 'ribbons', '_get_poly_coeffs_bernstein', '_get_poly_coeffs_lagrange', '_integer_barycentric_coordinates'

  @property
  def vertices(self):
    return types.frozenarray(numpy.concatenate([numpy.zeros(self.ndims)[_,:], numpy.eye(self.ndims)], axis=0), copy=False)

  @property
  def edge_refs(self):
    assert self.ndims > 0
    return (getsimplex(self.ndims-1),) * (self.ndims+1)

  @property
  def edge_transforms(self):
    assert self.ndims > 0
    return tuple(transform.SimplexEdge(self.ndims, i) for i in range(self.ndims+1))

  @property
  def child_refs(self):
    return tuple([self] * (2**self.ndims))

  @property
  def child_transforms(self):
    return tuple(transform.SimplexChild(self.ndims, ichild) for ichild in range(2**self.ndims))

  @property
  def ribbons(self):
    return tuple(((iedge1,iedge2),(iedge2+1,iedge1)) for iedge1 in range(self.ndims+1) for iedge2 in range(iedge1,self.ndims))

  def getpoints(self, ischeme, degree):
    if ischeme == 'gauss':
      return points.SimplexGaussPoints(self.ndims, degree if numeric.isint(degree) else sum(degree))
    if ischeme == 'vtk':
      return points.SimplexBezierPoints(self.ndims, 2)
    if ischeme == 'vertex':
      return points.SimplexBezierPoints(self.ndims, 2**(degree or 0) + 1)
    if ischeme == 'bezier':
      return points.SimplexBezierPoints(self.ndims, degree)
    return super().getpoints(ischeme, degree)

  @property
  def simplices(self):
    return (transform.Identity(self.ndims), self),

  def get_ndofs(self, degree):
    prod = lambda start, stop: functools.reduce(operator.mul, range(start, stop), 1)
    return prod(degree+1, degree+1+self.ndims) // prod(1, self.ndims+1)

  def get_poly_coeffs(self, basis, **kwargs):
    f = getattr(self, '_get_poly_coeffs_{}'.format(basis), None)
    if f:
      return f(**kwargs)
    else:
      raise ValueError('basis {!r} undefined on {}'.format(basis, type(self).__qualname__))

  def _integer_barycentric_coordinates(self, degree):
    return tuple(
      (degree-sum(i),*i[::-1])
      for i in itertools.product(*[range(degree+1)]*self.ndims)
      if sum(i) <= degree)

  def _get_poly_coeffs_bernstein(self, degree):
    ndofs = self.get_ndofs(degree)
    if self.ndims == 0:
      return types.frozenarray(numpy.ones((ndofs,), dtype=int), copy=False)
    coeffs = numpy.zeros((ndofs,)+(degree+1,)*self.ndims, dtype=int)
    for i, p in enumerate(self._integer_barycentric_coordinates(degree)):
      p = p[1:]
      for q in itertools.product(*[range(degree+1)]*self.ndims):
        if sum(p+q) <= degree:
          coeffs[(i,)+tuple(map(operator.add, p, q))] = (-1)**sum(q)*math.factorial(degree)//(math.factorial(degree-sum(p+q))*util.product(map(math.factorial, p+q)))
    assert i == ndofs - 1
    return types.frozenarray(coeffs, copy=False)

  def _get_poly_coeffs_lagrange(self, degree):
    if self.ndims == 0:
      coeffs = numpy.ones((1,))
    elif degree == 0:
      coeffs = numpy.ones((1,*[1]*self.ndims))
    else:
      P = numpy.array(tuple(self._integer_barycentric_coordinates(degree)), dtype=int)[:,1:]
      coeffs_ = numpy.linalg.inv(((P[:,_,:]/degree)**P[_,:,:]).prod(-1))
      coeffs = numpy.zeros((len(P),*[degree+1]*self.ndims), dtype=float)
      for i, p in enumerate(P):
        coeffs[(slice(None),*p)] = coeffs_[i]
    return types.frozenarray(coeffs, copy=False)

  def get_edge_dofs(self, degree, iedge):
    return types.frozenarray(tuple(i for i, j in enumerate(self._integer_barycentric_coordinates(degree)) if j[iedge] == 0), dtype=int)

  def inside(self, point, eps=0):
    return numpy.greater_equal(point, -eps).all(axis=0) and numpy.less_equal(numpy.sum(point, axis=0), 1+eps)

class PointReference(SimplexReference):
  '0D simplex'

  __slots__ = ()
  __cache__ = 'getpoints',

  def __init__(self):
    super().__init__(ndims=0)

  def getpoints(self, ischeme, degree):
    return points.CoordsWeightsPoints(numpy.empty([1,0]), [1.])

  def nvertices_by_level(self, n):
    return 1

  def child_divide(self, vals, n):
    return vals,

class LineReference(SimplexReference):
  '1D simplex'

  __slots__ = '_bernsteincache',

  def __init__(self):
    self._bernsteincache = [] # TEMPORARY
    super().__init__(ndims=1)

  def getpoints(self, ischeme, degree):
    if ischeme == 'uniform':
      return points.CoordsUniformPoints(numpy.arange(.5, degree)[:,_] / degree, 1)
    return super().getpoints(ischeme, degree)

  def nvertices_by_level(self, n):
    return 2**n + 1

  def child_divide(self, vals, n):
    assert n > 0
    assert len(vals) == self.nvertices_by_level(n)
    m = (len(vals)+1) // 2
    return vals[:m], vals[m-1:]

class TriangleReference(SimplexReference):
  '2D simplex'

  __slots__ = ()

  def __init__(self):
    super().__init__(ndims=2)

  def getpoints(self, ischeme, degree):
    if ischeme == 'uniform':
      p = numpy.arange(1./3, degree) / degree
      C = numpy.empty([2, degree, degree])
      C[0] = p[:,_]
      C[1] = p[_,:]
      coords = C.reshape(2, -1)
      flip = numpy.greater(coords.sum(0), 1)
      coords[:,flip] = 1 - coords[::-1,flip]
      return points.CoordsUniformPoints(coords.T, .5)
    return super().getpoints(ischeme, degree)

  def nvertices_by_level(self, n):
    m = 2**n + 1
    return ((m+1)*m) // 2

  def child_divide(self, vals, n):
    assert len(vals) == self.nvertices_by_level(n)
    np = 1 + 2**n # points along parent edge
    mp = 1 + 2**(n-1) # points along child edge
    cvals = []
    for i in range(mp):
      j = numpy.arange(mp-i)
      cvals.append([vals[b+a*np-(a*(a-1))//2] for a, b in [(i,j),(i,mp-1+j),(mp-1+i,j),(i+j,mp-1-j)]])
    return numpy.concatenate(cvals, axis=1)

class TetrahedronReference(SimplexReference):
  '3D simplex'

  # TETRAHEDRON:
  # c\d
  # a-b
  #
  # EDGES:
  # d\  d\  d\  c\
  # b-c a-c a-b a-b

  # SUBDIVIDED TETRAHEDRON:
  # f\  i\j
  # d-e\g-h
  # a-b-c
  #
  # SUBDIVIDED EDGES:
  # j\    j\    j\    f\
  # h-i\  g-i\  g-h\  d-e\
  # c-e-f a-d-f a-b-c a-b-c
  #
  # CHILDREN:
  # d\g e\h f\i i\j e\g g\h g\i h\i
  # a-b b-c d-e g-h b-d b-e d-e e-g

  __slots__ = ()

  _children_vertices = [0,1,3,6], [1,2,4,7], [3,4,5,8], [6,7,8,9], [1,3,4,6], [1,4,6,7], [3,4,6,8], [4,6,7,8]

  def __init__(self):
    super().__init__(ndims=3)

  def getindices_vertex(self, n):
    m = 2**n+1
    indis = numpy.arange(m)
    return numpy.array([[i,j,k] for k in indis for j in indis[:m-k] for i in indis[:m-j-k]])

  def nvertices_by_level(self, n):
    m = 2**n+1
    return ((m+2)*(m+1)*m)//6

  def child_divide(self, vals, n):
    assert len(vals) == self.nvertices_by_level(n)

    child_indices =  self.getindices_vertex(1)

    offset = numpy.array([1,0,0,0])
    linear = numpy.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])

    m = 2**n+1
    cvals = []
    for child_ref, child_vertices in zip(self.child_refs,self._children_vertices):
      V = child_indices[child_vertices]

      child_offset = (2**(n-1))*V.T.dot(offset)
      child_linear = V.T.dot(linear)

      original    = child_ref.getindices_vertex(n-1)
      transformed = original.dot(child_linear.T) + child_offset

      i, j, k = transformed.T
      cvals.append(vals[((k-1)*k*(2*k-1)//6 - (1+2*m)*(k-1)*k//2 + m*(m+1)*k)//2 + ((2*(m-k)+1)*j-j**2)//2 + i])

    return numpy.array(cvals)

class TensorReference(Reference):
  'tensor reference'

  __slots__ = 'ref1', 'ref2'
  __cache__ = 'vertices', 'edge_transforms', 'ribbons', 'child_transforms', 'getpoints', 'get_poly_coeffs'

  def __init__(self, ref1, ref2):
    assert not isinstance(ref1, TensorReference)
    self.ref1 = ref1
    self.ref2 = ref2
    super().__init__(ref1.ndims + ref2.ndims)

  def __mul__(self, other):
    assert isinstance(other, Reference)
    return TensorReference(self.ref1, self.ref2 * other)

  @property
  def vertices(self):
    vertices = numpy.empty((self.ref1.nverts, self.ref2.nverts, self.ndims), dtype=float)
    vertices[:,:,:self.ref1.ndims] = self.ref1.vertices[:,_]
    vertices[:,:,self.ref1.ndims:] = self.ref2.vertices[_,:]
    return types.frozenarray(vertices.reshape(self.ref1.nverts*self.ref2.nverts, self.ndims), copy=False)

  @property
  def centroid(self):
    return numpy.concatenate([self.ref1.centroid, self.ref2.centroid])

  def nvertices_by_level(self, n):
    return self.ref1.nvertices_by_level(n) * self.ref2.nvertices_by_level(n)

  def child_divide(self, vals, n):
    np1 = self.ref1.nvertices_by_level(n)
    np2 = self.ref2.nvertices_by_level(n)
    return [v2.swapaxes(0,1).reshape((-1,)+vals.shape[1:])
      for v1 in self.ref1.child_divide(vals.reshape((np1,np2)+vals.shape[1:]), n)
        for v2 in self.ref2.child_divide(v1.swapaxes(0,1), n)]

  def __str__(self):
    return '{}*{}'.format(self.ref1, self.ref2)

  def getpoints(self, ischeme, degree):
    if self.ref1.ndims == 0:
      return self.ref2.getpoints(ischeme, degree)
    if self.ref2.ndims == 0:
      return self.ref1.getpoints(ischeme, degree)
    if ischeme != 'vtk':
      ischeme1, ischeme2 = ischeme.split('*', 1) if '*' in ischeme else (ischeme, ischeme)
      degree1 = degree if not isinstance(degree, tuple) else degree[0]
      degree2 = degree if not isinstance(degree, tuple) else degree[1] if len(degree) == 2 else degree[1:]
      return points.TensorPoints(self.ref1.getpoints(ischeme1, degree1), self.ref2.getpoints(ischeme2, degree2))
    if self.ref1.ndims == self.ref2.ndims == 1:
      coords = numpy.empty([2, 2, 2])
      coords[...,:1] = self.ref1.vertices[:,_]
      coords[0,:,1:] = self.ref2.vertices
      coords[1,:,1:] = self.ref2.vertices[::-1]
    elif self.ref1.ndims <= 1 and self.ref2.ndims >= 1:
      coords = numpy.empty([self.ref1.nverts, self.ref2.nverts, self.ndims])
      coords[...,:self.ref1.ndims] = self.ref1.vertices[:,_]
      coords[...,self.ref1.ndims:] = self.ref2.vertices[_,:]
    elif self.ref1.ndims >= 1 and self.ref2.ndims <= 1:
      coords = numpy.empty([self.ref2.nverts, self.ref1.nverts, self.ndims])
      coords[...,:self.ref1.ndims] = self.ref1.vertices[_,:]
      coords[...,self.ref1.ndims:] = self.ref2.vertices[:,_]
    else:
      raise NotImplementedError
    return points.CoordsPoints(coords.reshape(self.nverts, self.ndims))

  @property
  def edge_transforms(self):
    edge_transforms = []
    if self.ref1.ndims:
      edge_transforms.extend(transform.TensorEdge1(trans1, self.ref2.ndims) for trans1 in self.ref1.edge_transforms)
    if self.ref2.ndims:
      edge_transforms.extend(transform.TensorEdge2(self.ref1.ndims, trans2) for trans2 in self.ref2.edge_transforms)
    return tuple(edge_transforms)

  @property
  def edge_refs(self):
    edge_refs = []
    if self.ref1.ndims:
      edge_refs.extend(edge1 * self.ref2 for edge1 in self.ref1.edge_refs)
    if self.ref2.ndims:
      edge_refs.extend(self.ref1 * edge2 for edge2 in self.ref2.edge_refs)
    return tuple(edge_refs)

  @property
  def ribbons(self):
    if self.ref1.ndims == 0:
      return self.ref2.ribbons
    if self.ref2.ndims == 0:
      return self.ref1.ribbons
    ribbons = []
    for iedge1 in range(self.ref1.nedges):
      #iedge = self.ref1.edge_refs[iedge] * self.ref2
      for iedge2 in range(self.ref2.nedges):
        #jedge = self.ref1 * self.ref2.edge_refs[jedge]
        jedge1 = self.ref1.nedges + iedge2
        jedge2 = iedge1
        if self.ref1.ndims > 1:
          iedge2 += self.ref1.edge_refs[iedge1].nedges
        ribbons.append(((iedge1,iedge2),(jedge1,jedge2)))
    if self.ref1.ndims >= 2:
      ribbons.extend(self.ref1.ribbons)
    if self.ref2.ndims >= 2:
      ribbons.extend(((iedge1+self.ref1.nedges,iedge2+self.ref1.nedges),
                       (jedge1+self.ref1.nedges,jedge2+self.ref1.nedges)) for (iedge1,iedge2), (jedge1,jedge2) in self.ref2.ribbons)
    return tuple(ribbons)

  @property
  def child_transforms(self):
    return tuple(transform.TensorChild(trans1, trans2) for trans1 in self.ref1.child_transforms for trans2 in self.ref2.child_transforms)

  @property
  def child_refs(self):
    return tuple(child1 * child2 for child1 in self.ref1.child_refs for child2 in self.ref2.child_refs)

  def inside(self, point, eps=0):
    return self.ref1.inside(point[:self.ref1.ndims],eps) and self.ref2.inside(point[self.ref1.ndims:],eps)

  @property
  def simplices(self):
    return tuple((transform.TensorChild(trans1, trans2), TensorReference(simplex1, simplex2)) for trans1, simplex1 in self.ref1.simplices for trans2, simplex2 in self.ref2.simplices)

  def get_ndofs(self, degree):
    return self.ref1.get_ndofs(degree)*self.ref2.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return numeric.poly_outer_product(self.ref1.get_poly_coeffs(basis, **kwargs), self.ref2.get_poly_coeffs(basis, **kwargs))

  def get_edge_dofs(self, degree, iedge):
    if not numeric.isint(iedge) or iedge < 0 or iedge >= self.nedges:
      raise IndexError('edge index out of range')
    nd2 = self.ref2.get_ndofs(degree)
    if iedge < self.ref1.nedges:
      dofs1 = self.ref1.get_edge_dofs(degree, iedge)
      dofs2 = range(self.ref2.get_ndofs(degree))
    else:
      dofs1 = range(self.ref1.get_ndofs(degree))
      dofs2 = self.ref2.get_edge_dofs(degree, iedge-self.ref1.nedges)
    return types.frozenarray(tuple(d1*nd2+d2 for d1, d2 in itertools.product(dofs1, dofs2)), dtype=int)

  @property
  def _flat_refs(self):
    for ref in self.ref1, self.ref2:
      if isinstance(ref, TensorReference):
        yield from ref._flat_refs
      else:
        yield ref

class Cone(Reference):
  'cone'

  __slots__ = 'edgeref', 'etrans', 'tip', 'extnorm', 'height'
  __cache__ = 'vertices', 'edge_transforms', 'edge_refs', 'volume'

  @types.apply_annotations
  def __init__(self, edgeref, etrans, tip:types.frozenarray):
    assert etrans.fromdims == edgeref.ndims
    assert etrans.todims == len(tip)
    super().__init__(len(tip))
    self.edgeref = edgeref
    self.etrans = etrans
    self.tip = tip
    ext = etrans.ext
    self.extnorm = numpy.linalg.norm(ext)
    self.height = numpy.dot(etrans.offset - tip, ext) / self.extnorm
    assert self.height >= 0, 'tip is positioned at the negative side of edge'

  @property
  def vertices(self):
    return types.frozenarray(numpy.vstack([[self.tip], self.etrans.apply(self.edgeref.vertices)]), copy=False)

  @property
  def edge_transforms(self):
    edge_transforms = [self.etrans]
    if self.edgeref.ndims > 0:
      for trans, edge in self.edgeref.edges:
        if edge:
          b = self.etrans.apply(trans.offset)
          A = numpy.hstack([numpy.dot(self.etrans.linear, trans.linear), (self.tip-b)[:,_]])
          newtrans = transform.Updim(A, b, isflipped=self.etrans.isflipped^trans.isflipped^(self.ndims%2==1)) # isflipped logic tested up to 3D
          edge_transforms.append(newtrans)
    else:
      edge_transforms.append(transform.Updim(numpy.zeros((1,0)), self.tip, isflipped=not self.etrans.isflipped))
    return edge_transforms

  @property
  def edge_refs(self):
    edge_refs = [self.edgeref]
    if self.edgeref.ndims > 0:
      extrudetrans = transform.Updim(numpy.eye(self.ndims-1)[:,:-1], numpy.zeros(self.ndims-1), isflipped=self.ndims%2==0)
      tip = numpy.array([0]*(self.ndims-2)+[1], dtype=float)
      edge_refs.extend(edge.cone(extrudetrans, tip) for edge in self.edgeref.edge_refs if edge)
    else:
      edge_refs.append(getsimplex(0))
    return edge_refs

  def getpoints(self, ischeme, degree):
    if ischeme == 'gauss':
      if self.nverts == self.ndims+1: # use optimal gauss schemes for simplex-like cones
        trans = transform.Square((self.etrans.apply(self.edgeref.vertices) - self.tip).T, self.tip)
        return points.TransformPoints(getsimplex(self.ndims).getpoints(ischeme, degree), trans)
      epoints = self.edgeref.getpoints('gauss', degree)
      tx, tw = points.gauss((degree + self.ndims - 1)//2)
      wx = tx**(self.ndims-1) * tw * self.extnorm * self.height
      return points.CoordsWeightsPoints((tx[:,_,_] * (self.etrans.apply(epoints.coords)-self.tip)[_,:,:] + self.tip).reshape(-1, self.ndims), (epoints.weights[_,:] * wx[:,_]).ravel())
    if ischeme == 'uniform':
      coords = numpy.concatenate([(self.etrans.apply(self.edgeref.getpoints('uniform', i+1).coords) - self.tip) * ((i+.5)/degree) + self.tip for i in range(degree)])
      return points.CoordsUniformPoints(coords, self.volume)
    if ischeme == 'vtk' and self.nverts == 5 and self.ndims==3: # pyramid
      return points.CoordsPoints(self.vertices[[1,2,4,3,0]])
    return points.ConePoints(self.edgeref.getpoints(ischeme, degree), self.etrans, self.tip)

  @property
  def volume(self):
    return self.edgeref.volume * self.extnorm * self.height / self.ndims

  @property
  def simplices(self):
    if self.nverts == self.ndims+1 or self.edgeref.ndims == 2 and self.edgeref.nverts == 4: # simplices and square-based pyramids are ok
      return [(transform.Identity(self.ndims), self)]
    return tuple((transform.Identity(self.ndims), ref.cone(self.etrans*trans,self.tip)) for trans, ref in self.edgeref.simplices)

  def inside(self, point, eps=0):
    # point = etrans.apply(epoint) * xi + tip * (1-xi) => etrans.apply(epoint) = tip + (point-tip) / xi
    xi = numpy.dot(self.etrans.ext, point-self.tip) / numpy.dot(self.etrans.ext, self.etrans.offset-self.tip)
    return -eps <= xi <= 1+eps and self.edgeref.inside(numpy.linalg.solve(
      numpy.dot(self.etrans.linear.T, self.etrans.linear),
      numpy.dot(self.etrans.linear.T, self.tip + (point-self.tip)/xi - self.etrans.offset)), eps=eps)

class OwnChildReference(Reference):
  'forward self as child'

  __slots__ = 'baseref', 'child_refs', 'child_transforms'

  def __init__(self, baseref):
    self.baseref = baseref
    self.child_refs = baseref,
    self.child_transforms = transform.Identity(baseref.ndims),
    super().__init__(baseref.ndims)

  @property
  def vertices(self):
    return self.baseref.vertices

  @property
  def edge_transforms(self):
    return self.baseref.edge_transforms

  @property
  def edge_refs(self):
    return [OwnChildReference(edge) for edge in self.baseref.edge_refs]

  def getpoints(self, ischeme, degree):
    return self.baseref.getpoints(ischeme, degree)

  @property
  def simplices(self):
    return self.baseref.simplices

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)

class WithChildrenReference(Reference):
  'base reference with explicit children'

  __slots__ = 'baseref', 'child_transforms', 'child_refs'
  __cache__ = '__extra_edges', 'edge_transforms', 'edge_refs', 'connectivity'

  @types.apply_annotations
  def __init__(self, baseref, child_refs:tuple):
    assert len(child_refs) == baseref.nchildren and any(child_refs) and child_refs != baseref.child_refs
    assert all(isinstance(child_ref,Reference) for child_ref in child_refs)
    assert all(child_ref.ndims == baseref.ndims for child_ref in child_refs)
    self.baseref = baseref
    self.child_transforms = baseref.child_transforms
    self.child_refs = child_refs
    super().__init__(baseref.ndims)

  def check_edges(self, tol=1e-15, print=print):
    super().check_edges(tol=tol, print=print)
    for cref in self.child_refs:
      cref.check_edges(tol=tol, print=print)

  @property
  def vertices(self):
    return self.baseref.vertices

  def nvertices_by_level(self, n):
    return self.baseref.nvertices_by_level(n)

  def child_divide(self, vals, n):
    return self.baseref.child_divide(vals, n)

  __sub__ = lambda self, other: self.empty if other in (self,self.baseref) else self.baseref.with_children(self_child-other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref in (self,self.baseref) else NotImplemented
  __rsub__ = lambda self, other: self.baseref.with_children(other_child - self_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if other == self.baseref else NotImplemented
  __and__ = lambda self, other: self if other == self.baseref else other if isinstance(other,WithChildrenReference) and self == other.baseref else self.baseref.with_children(self_child & other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented
  __or__ = lambda self, other: other if other == self.baseref else self.baseref.with_children(self_child | other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented

  @property
  def __extra_edges(self):
    extra_edges = [(ichild, iedge, cref.edge_refs[iedge])
      for ichild, cref in enumerate(self.child_refs) if cref
        for iedge in range(self.baseref.child_refs[ichild].nedges, cref.nedges)]
    for ichild, edges in enumerate(self.baseref.connectivity):
      cref = self.child_refs[ichild]
      if not cref:
        continue # new child is empty
      for iedge, jchild in enumerate(edges):
        if jchild == -1:
          continue # iedge already is an external boundary
        coppref = self.child_refs[jchild]
        if coppref == self.baseref.child_refs[jchild]:
          continue # opposite is complete, so iedge cannot form a new external boundary
        eref = cref.edge_refs[iedge]
        if coppref: # opposite new child is not empty
          eref -= coppref.edge_refs[self.baseref.connectivity[jchild].index(ichild)]
        if eref:
          extra_edges.append((ichild, iedge, eref))
    return extra_edges

  def subvertex(self, ichild, i):
    assert 0<=ichild<self.nchildren
    npoints = 0
    for childindex, child in enumerate(self.child_refs):
      if child:
        points = child.getpoints('vertex', i-1).coords
        if childindex == ichild:
          rng = numpy.arange(npoints, npoints+len(points))
        npoints += len(points)
      elif ichild==childindex:
        rng = numpy.array([],dtype=int)
    return npoints, rng

  def getpoints(self, ischeme, degree):
    if ischeme == 'vertex':
      return self.baseref.getpoints(ischeme, degree)
    if ischeme == 'bezier':
      childpoints = [points.TransformPoints(ref.getpoints('bezier', degree//2+1), trans) for trans, ref in self.children if ref]
      return points.ConcatPoints(childpoints, points.find_duplicates(childpoints))
    return points.ConcatPoints(points.TransformPoints(ref.getpoints(ischeme, degree), trans) for trans, ref in self.children if ref)

  @property
  def simplices(self):
    return [(trans2*trans1, simplex) for trans2, child in self.children for trans1, simplex in (child.simplices if child else [])]

  @property
  def edge_transforms(self):
    return tuple(self.baseref.edge_transforms) \
         + tuple(transform.ScaledUpdim(self.child_transforms[ichild], self.child_refs[ichild].edge_transforms[iedge]) for ichild, iedge, ref in self.__extra_edges)

  @property
  def edge_refs(self):
    refs = []
    for etrans, eref in self.baseref.edges:
      children = []
      if eref:
        for ctrans, cref in eref.children:
          ctrans_, etrans_ = etrans.swapup(ctrans)
          ichild = self.baseref.child_transforms.index(ctrans_)
          cref = self.child_refs[ichild]
          children.append(cref.edge_refs[cref.edge_transforms.index(etrans_)])
      refs.append(eref.with_children(children))
    for ichild, iedge, ref in self.__extra_edges:
      refs.append(OwnChildReference(ref))
    return tuple(refs)

  @property
  def connectivity(self):
    return tuple(types.frozenarray(edges.tolist() + [-1] * (self.child_refs[ichild].nedges - len(edges))) for ichild, edges in enumerate(self.baseref.connectivity))

  def inside(self, point, eps=0):
    return any(cref.inside(ctrans.invapply(point), eps=eps) for ctrans, cref in self.children)

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)

class MosaicReference(Reference):
  'triangulation'

  __slots__ = 'baseref', '_edge_refs', '_midpoint', 'edge_refs', 'edge_transforms'
  __cache__ = 'vertices', 'subrefs'

  @types.apply_annotations
  def __init__(self, baseref, edge_refs:tuple, midpoint:types.frozenarray):
    assert len(edge_refs) == baseref.nedges
    assert edge_refs != tuple(baseref.edge_refs)

    self.baseref = baseref
    self._edge_refs = edge_refs
    self._midpoint = midpoint
    self.edge_refs = list(edge_refs)
    self.edge_transforms = list(baseref.edge_transforms)

    if baseref.ndims == 1:

      assert any(edge_refs) and not all(edge_refs), 'invalid 1D mosaic: exactly one edge should be non-empty'
      iedge, = [i for i, edge in enumerate(edge_refs) if edge]
      self.edge_refs.append(getsimplex(0))
      self.edge_transforms.append(transform.Updim(linear=numpy.zeros((1,0)), offset=midpoint, isflipped=not baseref.edge_transforms[iedge].isflipped))

    else:

      newedges = [(etrans1, etrans2, edge) for (etrans1,orig), new in zip(baseref.edges, edge_refs) for etrans2, edge in new.edges[orig.nedges:]]
      for (iedge1,iedge2), (jedge1,jedge2) in baseref.ribbons:
        Ei = edge_refs[iedge1]
        ei = Ei.edge_refs[iedge2]
        Ej = edge_refs[jedge1]
        ej = Ej.edge_refs[jedge2]
        ejsubi = ej - ei
        if ejsubi:
          newedges.append((self.edge_transforms[jedge1], Ej.edge_transforms[jedge2], ejsubi))
        eisubj = ei - ej
        if eisubj:
          newedges.append((self.edge_transforms[iedge1], Ei.edge_transforms[iedge2], eisubj))

      extrudetrans = transform.Updim(numpy.eye(baseref.ndims-1)[:,:-1], numpy.zeros(baseref.ndims-1), isflipped=baseref.ndims%2==0)
      tip = numpy.array([0]*(baseref.ndims-2)+[1], dtype=float)
      for etrans, trans, edge in newedges:
        b = etrans.apply(trans.offset)
        A = numpy.hstack([numpy.dot(etrans.linear, trans.linear), (midpoint-b)[:,_]])
        newtrans = transform.Updim(A, b, isflipped=etrans.isflipped^trans.isflipped^(baseref.ndims%2==1)) # isflipped logic tested up to 3D
        self.edge_transforms.append(newtrans)
        self.edge_refs.append(edge.cone(extrudetrans, tip))

    super().__init__(baseref.ndims)

  @property
  def vertices(self):
    vertices = []
    for etrans, eref in self.edges:
      if eref:
        for vertex in etrans.apply(eref.vertices):
          if vertex not in vertices:
            vertices.append(vertex)
    return types.frozenarray(vertices)

  def __and__(self, other):
    if other in (self,self.baseref):
      return self
    if isinstance(other, MosaicReference) and other.baseref == self:
      return other
    if isinstance(other, MosaicReference) and self.baseref == other.baseref and numpy.equal(other._midpoint, self._midpoint).all():
      isect_edge_refs = [selfedge & otheredge for selfedge, otheredge in zip(self._edge_refs, other._edge_refs)]
      if not any(isect_edge_refs):
        return self.empty
      return MosaicReference(self.baseref, isect_edge_refs, self._midpoint)
    return NotImplemented

  def __or__(self, other):
    if other in (self,self.baseref):
      return other
    if isinstance(other, MosaicReference) and self.baseref == other.baseref and numpy.equal(other._midpoint, self._midpoint).all():
      union_edge_refs = [selfedge | otheredge for selfedge, otheredge in zip(self._edge_refs, other._edge_refs)]
      if tuple(union_edge_refs) == tuple(self.baseref.edge_refs):
        return self.baseref
      return MosaicReference(self.baseref, union_edge_refs, self._midpoint)
    return NotImplemented

  def __sub__(self, other):
    if other in (self,self.baseref):
      return self.empty
    if isinstance(other, MosaicReference) and other.baseref == self:
      inv_edge_refs = [baseedge - edge for baseedge, edge in zip(self.edge_refs, other._edge_refs)]
      return MosaicReference(self, inv_edge_refs, other._midpoint)
    return NotImplemented

  def __rsub__(self, other):
    if other == self.baseref:
      inv_edge_refs = [baseedge - edge for baseedge, edge in zip(other.edge_refs, self._edge_refs)]
      return MosaicReference(other, inv_edge_refs, self._midpoint)
    return NotImplemented

  def nvertices_by_level(self, n):
    return self.baseref.nvertices_by_level(n)

  @property
  def subrefs(self):
    return [ref.cone(trans,self._midpoint) for trans, ref in zip(self.baseref.edge_transforms, self._edge_refs) if ref]

  @property
  def simplices(self):
    return [simplex for subvol in self.subrefs for simplex in subvol.simplices]

  def getpoints(self, ischeme, degree):
    if ischeme == 'vertex':
      return self.baseref.getpoints(ischeme, degree)
    subpoints = [subvol.getpoints(ischeme, degree) for subvol in self.subrefs]
    dups = points.find_duplicates(subpoints) if ischeme == 'bezier' else ()
    return points.ConcatPoints(subpoints, dups)

  def inside(self, point, eps=0):
    return any(subref.inside(point, eps=eps) for subref in self.subrefs)

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)


## UTILITY FUNCTIONS

def parse_legacy_ischeme(ischeme):
  matches = list(map(re.compile('^([a-zA-Z]+)(.*)$').match, ischeme.split('*')))
  assert all(matches), 'cannot parse integration scheme {!r}'.format(ischeme)
  ischeme = '*'.join(match.group(1) for match in matches)
  degree = eval(','.join(match.group(2) or 'None' for match in matches))
  return ischeme, degree

def getsimplex(ndims):
  Simplex_by_dim = PointReference, LineReference, TriangleReference, TetrahedronReference
  return Simplex_by_dim[ndims]()

def index_or_append(items, item):
  try:
    index = items.index(item)
  except ValueError:
    index = len(items)
    items.append(item)
  return index

def arglexsort(triangulation):
  return numpy.argsort(numeric.asobjvector(tuple(tri) for tri in triangulation))


# vim:sw=2:sts=2:et
