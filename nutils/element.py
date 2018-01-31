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
The element module defines reference elements such as the :class:`QuadElement`
and :class:`TriangularElement`, but also more exotic objects like the
:class:`TrimmedElement`. A set of (interconnected) elements together form a
:mod:`nutils.topology`. Elements have edges and children (for refinement), which
are in turn elements and map onto self by an affine transformation. They also
have a well defined reference coordinate system, and provide pointsets for
purposes of integration and sampling.
"""

from . import log, util, numpy, config, numeric, function, cache, transform, warnings, _
import re, math, itertools, operator, functools


## ELEMENT

class Element:
  'element class'

  __slots__ = 'reference', 'transform', 'opposite'

  def __init__(self, reference, trans, opptrans=None, oriented=False):
    assert isinstance(reference, Reference)
    trans = transform.canonical(trans)
    assert trans[-1].fromdims == reference.ndims and trans[0].todims == None
    if opptrans is not None:
      opptrans = transform.canonical(opptrans)
      assert opptrans[-1].fromdims == reference.ndims and opptrans[0].todims == None
      if not oriented:
        vtx1 = transform.apply(trans, reference.vertices)
        if vtx1 != transform.apply(opptrans, reference.vertices):
          for ptrans in reference.permutation_transforms:
            if vtx1 == transform.apply(opptrans + (ptrans,), reference.vertices):
              opptrans += ptrans,
              break
          else:
            raise Exception('Did not find a conforming permutation for the opposing transformation')
    self.reference = reference
    self.transform = trans
    self.opposite = opptrans or trans

  def withopposite(self, opp, oriented=False):
    if isinstance(opp, tuple):
      return Element(self.reference, self.transform, opp, oriented)
    assert isinstance(opp, Element) and opp.reference == self.reference
    return Element(self.reference, self.transform, opp.transform, oriented or opp.opposite==self.transform)

  def __mul__(self, other):
    self_is_iface = self.opposite != self.transform
    other_is_iface = other.opposite != other.transform
    trans = transform.Bifurcate(self.transform, other.transform),
    if self_is_iface != other_is_iface:
      opptrans = transform.Bifurcate(self.opposite, other.opposite),
    else:
      opptrans = None
    return Element(self.reference * other.reference, trans, opptrans, oriented=True)

  def __getnewargs__(self):
    return self.reference, self.transform, self.opposite, True

  def __hash__(self):
    return hash((self.reference, self.transform, self.opposite))

  def __eq__(self, other):
    return self is other or isinstance(other,Element) \
      and self.reference == other.reference \
      and self.transform == other.transform \
      and self.opposite == other.opposite

  @property
  def vertices(self):
    return transform.apply(self.transform, self.reference.vertices)

  @property
  def ndims(self):
    return self.reference.ndims

  @property
  def nverts(self):
    return self.reference.nverts

  @property
  def nedges(self):
    return self.reference.nedges

  @property
  def edges(self):
    return [self.edge(i) for i in range(self.nedges)]

  def edge(self, iedge):
    trans, edge = self.reference.edges[iedge]
    return Element(edge, self.transform + (trans,), self.opposite and self.opposite + (trans,), oriented=True) if edge else None

  @property
  def children(self):
    return [Element(child, self.transform + (trans,), self.opposite and self.opposite + (trans,), oriented=True)
      for trans, child in self.reference.children if child]

  @property
  def flipped(self):
    assert self.opposite, 'element does not define an opposite'
    return Element(self.reference, self.opposite, self.transform, oriented=True)

  @property
  def simplices(self):
    return [Element(reference, self.transform + (trans,), self.opposite and self.opposite + (trans,), oriented=True)
      for trans, reference in self.reference.simplices]

  def __str__(self):
    return 'Element({})'.format(self.vertices)


## REFERENCE ELEMENTS

class Reference(cache.Immutable):
  'reference element'

  def __init__(self, ndims:int):
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
    return EmptyReference(self.ndims)

  def __mul__(self, other):
    assert isinstance(other, Reference)
    return TensorReference(self, other)

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

  @cache.property
  def connectivity(self):
    # Nested tuple with connectivity information about edges of children:
    # connectivity[ichild][iedge] = ioppchild (interface) or -1 (boundary).
    childmap = [[-1] * child.nedges for child in self.child_refs]
    vmap = {}
    for ichild, (ctrans, cref) in enumerate(self.children):
      for iedge, etrans in enumerate(cref.edge_transforms):
        v = ctrans * etrans
        try:
          jchild, jedge = vmap.pop(v.flipped)
        except KeyError:
          vmap[v] = ichild, iedge
        else:
          childmap[jchild][jedge] = ichild
          childmap[ichild][iedge] = jchild
    for etrans, eref in self.edges:
      for ctrans in eref.child_transforms:
        vmap.pop(etrans * ctrans, None)
    assert not any(self.child_refs[ichild].edge_refs[iedge] for ichild, iedge in vmap.values()), 'not all boundary elements recovered'
    return tuple(map(tuple, childmap))

  @cache.property
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

  permutation_transforms = ()

  def getischeme(self, ischeme):
    match = re.match('([a-zA-Z]+)(.*)', ischeme)
    assert match, 'cannot parse integration scheme {!r}'.format(ischeme)
    ptype, args = match.groups()
    get = getattr(self, 'getischeme_'+ptype)
    ipoints, iweights = get(eval(args)) if args else get()
    return numeric.const(ipoints, copy=False), numeric.const(iweights, copy=False) if iweights is not None else None

  @classmethod
  def register(cls, ptype, func):
    setattr(cls, 'getischeme_{}'.format(ptype), func)

  def with_children(self, child_refs):
    child_refs = tuple(child_refs)
    if not any(child_refs):
      return self.empty
    if child_refs == self.child_refs:
      return self
    return WithChildrenReference(self, child_refs)

  @cache.property
  def volume(self):
    ipoints, iweights = self.getischeme('gauss{}'.format(1))
    return iweights.sum()

  @cache.property
  def centroid(self):
    ipoints, iweights = self.getischeme('gauss{}'.format(1))
    return ipoints.T.dot(iweights)/iweights.sum()

  def trim(self, levels, maxrefine, ndivisions):
    'trim element along levelset'

    assert len(levels) == self.nvertices_by_level(maxrefine)
    return self if not self or numpy.greater_equal(levels, 0).all() \
      else self.empty if numpy.less_equal(levels, 0).all() \
      else self.with_children(cref.trim(clevels, maxrefine-1, ndivisions)
            for cref, clevels in zip(self.child_refs, self.child_divide(levels,maxrefine))) if maxrefine > 0 \
      else self.slice(lambda vertices: numeric.dot(numeric.poly_eval(self._linear_bernstein[_], vertices), levels), ndivisions)

  @cache.property
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
        xe, we = edge.getischeme('gauss1')
        w_normal = we[:,_] * trans.ext
        zero += w_normal.sum(0)
        volume += numeric.contract(trans.apply(xe), w_normal, axis=0)
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
      assert not ctransforms[0]
      return ((), self.getischeme('vertex{}'.format(maxrefine))[0], allindices),
    if maxrefine == 0:
      raise Exception('maxrefine is too low')
    cbins = [[] for ichild in range(self.nchildren)]
    for ctrans in ctransforms:
      ichild = self.child_transforms.index(ctrans[0])
      cbins[ichild].append(ctrans[1:])
    if not all(cbins):
      raise Exception('transformations to not form an element cover')
    fcache = cache.WrapperCache()
    return tuple(((ctrans,) + trans, points, cindices[indices])
      for ctrans, cref, cbin, cindices in zip(self.child_transforms, self.child_refs, cbins, self.child_divide(allindices,maxrefine))
        for trans, points, indices in fcache[cref.vertex_cover](tuple(sorted(cbin)), maxrefine-1))

  def __str__(self):
    return self.__class__.__name__

  __repr__ = __str__

  def get_ndofs(self, degree):
    raise NotImplementedError

  def get_poly_coeffs(self, basis, **kwargs):
    raise NotImplementedError

  def get_edge_dofs(self, degree, iedge):
    raise NotImplementedError

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    raise NotImplementedError

class EmptyReference(Reference):
  'inverse reference element'

  volume = 0

  edge_transforms = ()
  edge_refs = ()
  child_transforms = ()
  child_refs = ()

  @property
  def vertices(self):
    return numeric.const(numpy.zeros((0, self.ndims)), copy=False)

  __and__ = __sub__ = lambda self, other: self if other.ndims == self.ndims else NotImplemented
  __or__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented
  __rsub__ = lambda self, other: other if other.ndims == self.ndims else NotImplemented

  def trim(self, levels, maxrefine, ndivisions):
    return self

  def inside(self, point, eps=0):
    return False

class RevolutionReference(Reference):
  'modify gauss integration to always return a single point'

  def __init__(self):
    super().__init__(ndims=1)

  def vertices(self):
    return numeric.const([[0.]]) # NOTE unclear if this is the desired outcome

  @property
  def edge_transforms(self): # only used in check_edges
    return transform.Updim(numpy.zeros((1,0)), [-numpy.pi], isflipped=True), transform.Updim(numpy.zeros((1,0)), [+numpy.pi], isflipped=False)

  @property
  def edge_refs(self): # idem edge_transforms
    return PointReference(), PointReference()

  @property
  def simplices(self):
    return (transform.Identity(self.ndims), self),

  def getischeme(self, ischeme):
    return numeric.const([[0.]]), numeric.const([2 * numpy.pi])

  def inside(self, point, eps=0):
    return True

class SimplexReference(Reference):
  'simplex reference'

  @property
  def vertices(self):
    return numeric.const(numpy.concatenate([numpy.zeros(self.ndims)[_,:], numpy.eye(self.ndims)], axis=0), copy=False)

  @cache.property
  def edge_refs(self):
    assert self.ndims > 0
    return (getsimplex(self.ndims-1),) * (self.ndims+1)

  @cache.property
  def edge_transforms(self):
    assert self.ndims > 0
    return tuple(transform.SimplexEdge(self.ndims, i) for i in range(self.ndims+1))

  @property
  def child_refs(self):
    return tuple([self] * (2**self.ndims))

  @property
  def child_transforms(self):
    return tuple(transform.SimplexChild(self.ndims, ichild) for ichild in range(2**self.ndims))

  @cache.property
  def permutation_transforms(self):
    transforms = []
    for verts in itertools.permutations(tuple(v for v in self.vertices)):
      offset = verts[0]
      linear = verts[1:]-verts[0]
      transforms.append(transform.Square(linear.T, offset))
    return tuple(transforms)

  @cache.property
  def ribbons(self):
    return tuple(((iedge1,iedge2),(iedge2+1,iedge1)) for iedge1 in range(self.ndims+1) for iedge2 in range(iedge1,self.ndims))

  def getischeme_vtk(self):
    return self.vertices, None

  def getischeme_vertex(self, n=0):
    if n == 0:
      return self.vertices, None
    return self.getischeme_bezier(2**n+1)

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
    return (
      (degree-sum(i),*i[::-1])
      for i in itertools.product(*[range(degree+1)]*self.ndims)
      if sum(i) <= degree)

  def _get_poly_coeffs_bernstein(self, degree):
    ndofs = self.get_ndofs(degree)
    if self.ndims == 0:
      return numpy.ones((ndofs,), dtype=int)
    coeffs = numpy.zeros((ndofs,)+(degree+1,)*self.ndims, dtype=int)
    for i, p in enumerate(self._integer_barycentric_coordinates(degree)):
      p = p[1:]
      for q in itertools.product(*[range(degree+1)]*self.ndims):
        if sum(p+q) <= degree:
          coeffs[(i,)+tuple(map(operator.add, p, q))] = (-1)**sum(q)*math.factorial(degree)//(math.factorial(degree-sum(p+q))*util.product(map(math.factorial, p+q)))
    assert i == ndofs - 1
    return numeric.const(coeffs, copy=False)

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
    return numeric.const(coeffs, copy=False)

  def get_edge_dofs(self, degree, iedge):
    return numeric.const(tuple(i for i, j in enumerate(self._integer_barycentric_coordinates(degree)) if j[iedge] == 0), dtype=int)

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    vertex_transpose_map = tuple(vertex_transpose_map)
    if len(vertex_transpose_map) != self.nverts or set(vertex_transpose_map) != set(range(self.nverts)):
      raise ValueError('invalid vertex indices: {!r}'.format(vertex_transpose_map))
    return numeric.const(tuple(i for i, j in sorted(enumerate(self._integer_barycentric_coordinates(degree)), key=lambda ij: tuple(map(ij[1].__getitem__, vertex_transpose_map[::-1])))), dtype=int)

class PointReference(SimplexReference):
  '0D simplex'

  def __init__(self):
    super().__init__(ndims=0)

  def getischeme(self, ischeme):
    return numeric.const(numpy.empty([1,0])), numeric.const([1.])

  def inside(self, point, eps=0):
    return True

  def nvertices_by_level(self, n):
    return 1

  def child_divide(self, vals, n):
    return vals,

class LineReference(SimplexReference):
  '1D simplex'

  def __init__(self):
    self._bernsteincache = [] # TEMPORARY
    super().__init__(ndims=1)

  def getischeme_gauss(self, degree):
    assert isinstance(degree, int) and degree >= 0
    x, w = gauss(degree)
    return x[:,_], w

  def getischeme_uniform(self, n):
    return (numpy.arange(.5,n) / n)[:,_], numeric.const.full([n], 1/n)

  def getischeme_bezier(self, np):
    return numpy.linspace(0, 1, np)[:,_], None

  def nvertices_by_level(self, n):
    return 2**n + 1

  def child_divide(self, vals, n):
    assert n > 0
    assert len(vals) == self.nvertices_by_level(n)
    m = (len(vals)+1) // 2
    return vals[:m], vals[m-1:]

  def inside(self, point, eps=0):
    x, = point
    return -eps <= x <= 1+eps

class TriangleReference(SimplexReference):
  '2D simplex'

  def __init__(self):
    super().__init__(ndims=2)

  def getischeme_gauss(self, degree):
    '''get integration scheme
    http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''
    if isinstance(degree, tuple):
      assert len(degree) == self.ndims
      degree = sum(degree)
    assert isinstance(degree, int) and degree >= 0

    I = [0,0],
    J = [1,1],[0,1],[1,0]
    K = [1,2],[2,0],[0,1],[2,1],[1,0],[0,2]

    icw = [
      (I, [1/3], 1)
    ] if degree <= 1 else [
      (J, [2/3,1/6], 1/3)
    ] if degree == 2 else [
      (I, [1/3], -9/16),
      (J, [3/5,1/5], 25/48),
    ] if degree == 3 else [
      (J, [0.816847572980458,0.091576213509771], 0.109951743655322),
      (J, [0.108103018168070,0.445948490915965], 0.223381589678011),
    ] if degree == 4 else [
      (I, [1/3], 0.225),
      (J, [0.797426985353088,0.101286507323456], 0.125939180544827),
      (J, [0.059715871789770,0.470142064105115], 0.132394152788506),
    ] if degree == 5 else [
      (J, [0.873821971016996,0.063089014491502], 0.050844906370207),
      (J, [0.501426509658180,0.249286745170910], 0.116786275726379),
      (K, [0.636502499121399,0.310352451033785,0.053145049844816], 0.082851075618374),
    ] if degree == 6 else [
      (I, [1/3.], -0.149570044467671),
      (J, [0.479308067841924,0.260345966079038], 0.175615257433204),
      (J, [0.869739794195568,0.065130102902216], 0.053347235608839),
      (K, [0.638444188569809,0.312865496004875,0.048690315425316], 0.077113760890257),
    ]

    if degree > 7:
      warnings.warn('inexact integration for polynomial of degree {}'.format(degree))

    return numpy.concatenate([numpy.take(c,i) for i, c, w in icw], axis=0), \
           numpy.concatenate([[w*.5] * len(i) for i, c, w in icw])

  def getischeme_uniform(self, n):
    points = numpy.arange(1./3, n) / n
    nn = n**2
    C = numpy.empty([2,n,n])
    C[0] = points[:,_]
    C[1] = points[_,:]
    coords = C.reshape(2, nn)
    flip = coords.sum(0) > 1
    coords[:,flip] = 1 - coords[::-1,flip]
    weights = numeric.const.full([nn], .5/nn)
    return coords.T, weights

  def getischeme_bezier(self, np):
    points = numpy.linspace(0, 1, np)
    return numpy.array([[x,y] for i, y in enumerate(points) for x in points[:np-i]]), None

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

  def inside(self, point, eps=0):
    x, y = point
    return x >= -eps and y >= -eps and 1-x-y >= -eps

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

  _children_vertices = [0,1,3,6], [1,2,4,7], [3,4,5,8], [6,7,8,9], [1,3,4,6], [1,4,6,7], [3,4,6,8], [4,6,7,8]

  def __init__(self):
    super().__init__(ndims=3)

  def getindices_vertex(self, n):
    m = 2**n+1
    indis = numpy.arange(m)
    return numpy.array([[i,j,k] for k in indis for j in indis[:m-k] for i in indis[:m-j-k]])

  def getischeme_vertex(self, n):
    return self.getindices_vertex(n)/(2**n), None

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

  def getischeme_gauss(self, degree):
    '''get integration scheme
    http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''
    if isinstance(degree, tuple):
      assert len(degree) == 3
      degree = sum(degree)
    assert isinstance(degree, int) and degree >= 0

    I = [0,0,0],
    J = [1,1,1],[0,1,1],[1,1,0],[1,0,1]
    K = [0,1,1],[1,0,1],[1,1,0],[1,0,0],[0,1,0],[0,0,1]
    L = [0,1,1],[1,0,1],[1,1,0],[2,1,1],[1,2,1],[1,1,2],[1,0,2],[0,2,1],[2,1,0],[1,2,0],[0,1,2],[2,0,1]

    icw = [
      (I, [1/4], 1),
    ] if degree == 1 else [
      (J, [0.5854101966249685,0.1381966011250105], 1/4),
    ] if degree == 2 else [
      (I, [.25], -.8),
      (J, [.5,1/6], .45),
    ] if degree == 3 else [
      (I, [.25], -.2368/3),
      (J, [0.7857142857142857,0.0714285714285714], .1372/3),
      (K, [0.1005964238332008,0.3994035761667992], .448/3),
    ] if degree == 4 else [
      (I, [.25], 0.1817020685825351),
      (J, [0,1/3.], 0.0361607142857143),
      (J, [8/11.,1/11.], 0.0698714945161738),
      (K, [0.4334498464263357,0.0665501535736643], 0.0656948493683187),
    ] if degree == 5 else [
      (J, [0.3561913862225449,0.2146028712591517], 0.0399227502581679),
      (J, [0.8779781243961660,0.0406739585346113], 0.0100772110553207),
      (J, [0.0329863295731731,0.3223378901422757], 0.0553571815436544),
      (L, [0.2696723314583159,0.0636610018750175,0.6030056647916491], 0.0482142857142857),
    ] if degree == 6 else [
      (I, [.25], 0.1095853407966528),
      (J, [0.7653604230090441,0.0782131923303186],  0.0635996491464850),
      (J, [0.6344703500082868,0.1218432166639044], -0.3751064406859797),
      (J, [0.0023825066607383,0.3325391644464206],  0.0293485515784412),
      (K, [0,.5], 0.0058201058201058),
      (L, [.2,.1,.6], 0.1653439153439105)
    ] if degree == 7 else [
      (I, [.25], -0.2359620398477557),
      (J, [0.6175871903000830,0.1274709365666390], 0.0244878963560562),
      (J, [0.9037635088221031,0.0320788303926323], 0.0039485206398261),
      (K, [0.4502229043567190,0.0497770956432810], 0.0263055529507371),
      (K, [0.3162695526014501,0.1837304473985499], 0.0829803830550589),
      (L, [0.0229177878448171,0.2319010893971509,0.5132800333608811], 0.0254426245481023),
      (L, [0.7303134278075384,0.0379700484718286,0.1937464752488044], 0.0134324384376852),
    ]

    if degree > 8:
      warnings.warn('inexact integration for polynomial of degree {}'.format(degree))

    return numpy.concatenate([numpy.take(c,i) for i, c, w in icw], axis=0), \
           numpy.concatenate([[w/6] * len(i) for i, c, w in icw])

class TensorReference(Reference):
  'tensor reference'

  _re_ischeme = re.compile('([a-zA-Z]+)(.*)')

  def __init__(self, ref1, ref2):
    assert not isinstance(ref1, TensorReference)
    self.ref1 = ref1
    self.ref2 = ref2
    super().__init__(ref1.ndims + ref2.ndims)

  def __mul__(self, other):
    assert isinstance(other, Reference)
    return TensorReference(self.ref1, self.ref2 * other)

  @cache.property
  def vertices(self):
    vertices = numpy.empty((self.ref1.nverts, self.ref2.nverts, self.ndims), dtype=float)
    vertices[:,:,:self.ref1.ndims] = self.ref1.vertices[:,_]
    vertices[:,:,self.ref1.ndims:] = self.ref2.vertices[_,:]
    return numeric.const(vertices.reshape(self.ref1.nverts*self.ref2.nverts, self.ndims), copy=False)

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

  def getischeme_vtk(self):
    if self.ref1.ndims == self.ref2.ndims == 1:
      points = numpy.empty([2, 2, 2])
      points[...,:1] = self.ref1.vertices[:,_]
      points[0,:,1:] = self.ref2.vertices
      points[1,:,1:] = self.ref2.vertices[::-1]
    elif self.ref1.ndims <= 1 and self.ref2.ndims >= 1:
      points = numpy.empty([self.ref1.nverts, self.ref2.nverts, self.ndims])
      points[...,:self.ref1.ndims] = self.ref1.vertices[:,_]
      points[...,self.ref1.ndims:] = self.ref2.vertices[_,:]
    elif self.ref1.ndims >= 1 and self.ref2.ndims <= 1:
      points = numpy.empty([self.ref2.nverts, self.ref1.nverts, self.ndims])
      points[...,:self.ref1.ndims] = self.ref1.vertices[_,:]
      points[...,self.ref1.ndims:] = self.ref2.vertices[:,_]
    else:
      raise NotImplementedError
    return points.reshape(self.nverts, self.ndims), None

  def getischeme(self, ischeme):
    if '*' in ischeme:
      ischeme1, ischeme2 = ischeme.split('*', 1)
    else:
      match = self._re_ischeme.match(ischeme)
      assert match, 'cannot parse integration scheme {!r}'.format(ischeme)
      ptype, args = match.groups()
      get = getattr(self, 'getischeme_'+ptype, None)
      if get:
        return get(eval(args)) if args else get()
      if args and ',' in args:
        args = eval(args)
        assert len(args) == self.ndims
        ischeme1 = ptype+','.join(str(n) for n in args[:self.ref1.ndims])
        ischeme2 = ptype+','.join(str(n) for n in args[self.ref1.ndims:])
      else:
        ischeme1 = ischeme2 = ischeme
    ipoints1, iweights1 = self.ref1.getischeme(ischeme1)
    ipoints2, iweights2 = self.ref2.getischeme(ischeme2)
    ipoints = numpy.empty((len(ipoints1), len(ipoints2), self.ndims))
    ipoints[:,:,0:self.ref1.ndims] = ipoints1[:,_,:self.ref1.ndims]
    ipoints[:,:,self.ref1.ndims:self.ndims] = ipoints2[_,:,:self.ref2.ndims]
    iweights = numeric.const((iweights1[:,_] * iweights2[_,:]).ravel(), copy=False) if iweights1 is not None and iweights2 is not None else None
    return numeric.const(ipoints.reshape(len(ipoints1) * len(ipoints2), self.ndims), copy=False), iweights

  @cache.property
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

  @cache.property
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

  @cache.property
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
    return numeric.const(tuple(d1*nd2+d2 for d1, d2 in itertools.product(dofs1, dofs2)), dtype=int)

  @property
  def _flat_refs(self):
    for ref in self.ref1, self.ref2:
      if isinstance(ref, TensorReference):
        yield from ref._flat_refs
      else:
        yield ref

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    vertex_transpose_map = tuple(vertex_transpose_map)
    if len(vertex_transpose_map) != self.nverts:
      raise ValueError('invalid vertex indices: {!r}'.format(vertex_transpose_map))
    refs = tuple(ref for ref in self._flat_refs if ref.nverts > 1)

    # Let `ref_verts[i]` be a permutation of `range(refs[i].nverts)`.  The
    # `vertex_transpose_map` should be the tensor product of the
    # `ref_verts[i]*vertex_strides[i]` for all `i`, permuted by `perm` and
    # flattened.  The `ref_strides` recovers the original structure from the
    # permuted and flattened `vertex_transpose_map`.  We reverse engineer the
    # per ref vertices, `ref_verts`, and permutation of the references, `perm`,
    # and apply the same permutation and flattening to the tensor product of
    # the dofs.

    stride = 1
    vertex_strides = []
    for ref in refs[::-1]:
      vertex_strides.insert(0, stride)
      stride *= ref.nverts

    verts = numpy.array(0, dtype=int)
    dofs = numpy.array(0, dtype=int)
    ref_strides = []
    for ref, stride in zip(refs, vertex_strides):
      ref_idx = [vertex_transpose_map.index(i*stride) for i in range(ref.nverts)]
      ref_verts = numpy.argsort(ref_idx)
      verts = verts[...,None]*len(ref_verts)+ref_verts
      ref_dofs = ref.get_dof_transpose_map(degree, ref_verts)
      dofs = dofs[...,None]*len(ref_dofs)+ref_dofs
      ref_strides.append(ref_idx[ref_verts[1]]-ref_idx[ref_verts[0]])
    perm = numpy.argsort(ref_strides)[::-1]
    # Verify that `vertex_transpose_map` is in fact a tensor product of the
    # `ref_verts`.
    if not numpy.all(numpy.equal(numpy.transpose(verts, perm).ravel(), vertex_transpose_map)):
      raise ValueError('invalid transformation: {!r}'.format(vertex_transpose_map))
    return numeric.const(numpy.transpose(dofs, perm).ravel())

class Cone(Reference):
  'cone'

  def __init__(self, edgeref, etrans, tip:numeric.const):
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

  @cache.property
  def vertices(self):
    return numeric.const(numpy.vstack([[self.tip], self.etrans.apply(self.edgeref.vertices)]), copy=False)

  @cache.property
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

  @cache.property
  def edge_refs(self):
    edge_refs = [self.edgeref]
    if self.edgeref.ndims > 0:
      extrudetrans = transform.Updim(numpy.eye(self.ndims-1)[:,:-1], numpy.zeros(self.ndims-1), isflipped=self.ndims%2==0)
      tip = numpy.array([0]*(self.ndims-2)+[1], dtype=float)
      edge_refs.extend(edge.cone(extrudetrans, tip) for edge in self.edgeref.edge_refs if edge)
    else:
      edge_refs.append(getsimplex(0))
    return edge_refs

  def getischeme_gauss(self, degree):
    if self.nverts == self.ndims+1: # simplex
      spoints, sweights = getsimplex(self.ndims).getischeme_gauss(degree)
      offset = self.vertices[0,:]
      linear = self.vertices[1:,:] - offset
      points = numpy.dot(spoints, linear) + offset
      weights = sweights * abs(numpy.linalg.det(linear))
    else:
      epoints, eweights = self.edgeref.getischeme('gauss{}'.format(degree))
      tpoints, tweights = getsimplex(1).getischeme_gauss(degree + self.ndims - 1)
      tx, = tpoints.T
      points = (tx[:,_,_] * (self.etrans.apply(epoints)-self.tip)[_,:,:] + self.tip).reshape(-1, self.ndims)
      wx = tx**(self.ndims-1) * tweights * self.extnorm * self.height
      weights = (eweights[_,:] * wx[:,_]).ravel()
    return points, weights

  def getischeme_bezier(self, degree):
    assert self.nverts == self.ndims+1
    spoints, none = getsimplex(self.ndims).getischeme_bezier(degree)
    offset = self.vertices[0,:]
    linear = self.vertices[1:,:] - offset
    return numpy.dot(spoints, linear) + offset, None

  def getischeme_vtk(self):
    if self.nverts == 4 and self.ndims==3: # tetrahedron
      I = slice(None)
    elif self.nverts == 5 and self.ndims==3: # pyramid
      I = numpy.array([1,2,4,3,0])
    elif self.nverts == 3 and self.ndims==2: # triangle
      I = slice(None)
    else:
      raise Exception('invalid number of points: {}'.format(self.nverts))
    return self.vertices[I], None

  @property
  def simplices(self):
    if self.nverts == self.ndims+1 or self.edgeref.ndims == 2 and self.edgeref.nverts == 4: # simplices and square-based pyramids are ok
      return [(transform.Identity(self.ndims), self)]
    return tuple((transform.Identity(self.ndims), ref.cone(self.etrans*trans,self.tip)) for trans, ref in self.edgeref.simplices)

  def inside(self, point, eps=0):
    # point = etrans.apply(epoint) * xi + tip * (1-xi) => etrans.apply(epoint) = tip + (point-tip) / xi
    xi = numpy.dot(self.etrans.ext, point-self.tip) / numpy.dot(self.etrans.ext, self.etrans.offset-self.tip)
    return 0 < xi <= 1+eps and self.edgeref.inside(numpy.linalg.solve(
      numpy.dot(self.etrans.linear.T, self.etrans.linear),
      numpy.dot(self.etrans.linear.T, self.tip + (point-self.tip)/xi - self.etrans.offset)), eps=eps)

class NeighborhoodTensorReference(TensorReference):
  'product reference element'

  def __init__(self, ref1, ref2, neighborhood, transf):
    '''Neighborhood of elem1 and elem2 and transformations to get mutual
    overlap in right location. Returns 3-element tuple:
    * neighborhood, as given by Element.neighbor(),
    * transf1, required rotation of elem1 map: {0:0, 1:pi/2, 2:pi, 3:3*pi/2},
    * transf2, required rotation of elem2 map (is indep of transf1 in Topology.'''

    super().__init__(ref1, ref2)
    self.neighborhood = neighborhood
    self.transf = transf

  def singular_ischeme_quad(self, points):
    transfpoints = numpy.empty(points.shape)
    def transform(points, transf):
      x, y = points[:,0], points[:,1]
      tx = x if transf in (0,1,6,7) else 1-x
      ty = y if transf in (0,3,4,7) else 1-y
      return function.stack((ty, tx) if transf%2 else (tx, ty), axis=1)
    transfpoints[:,:2] = transform(points[:,:2], self.transf[0])
    transfpoints[:,2:] = transform(points[:,2:], self.transf[1])
    return transfpoints

  def get_tri_bem_ischeme(self, ischeme):
    'Some cached quantities for the singularity quadrature scheme.'
    points, weights = (getsimplex(1)**4).getischeme(ischeme)
    eta1, eta2, eta3, xi = points.T
    if self.neighborhood == 0:
      temp = xi*eta1*eta2*eta3
      pts0 = xi*eta1*(1 - eta2)
      pts1 = xi - pts0
      pts2 = xi - temp
      pts3 = xi*(1 - eta1)
      pts4 = pts0 + temp
      pts5 = xi*(1 - eta1*eta2)
      pts6 = xi*eta1 - temp
      points = numpy.array(
        [[1-xi,   1-pts2, 1-xi,   1-pts5, 1-pts2, 1-xi  ],
         [pts1, pts3, pts4, pts0, pts6, pts0],
         [1-pts2, 1-xi,   1-pts5, 1-xi,   1-xi,   1-pts2],
         [pts3, pts1, pts0, pts4, pts0, pts6]]).reshape(4, -1).T
      points = points * [-1,1,-1,1] + [1,0,1,0] # flipping in x -GJ
      weights = numpy.concatenate(6*[xi**3*eta1**2*eta2*weights])
    elif self.neighborhood == 1:
      A = xi*eta1
      B = A*eta2
      C = A*eta3
      D = B*eta3
      E = xi - B
      F = A - B
      G = xi - D
      H = B - D
      I = A - D
      points = numpy.array(
        [[1-xi, 1-xi, 1-E,  1-G,  1-G ],
         [C,  G,  F,  H,  I ],
         [1-E,  1-G,  1-xi, 1-xi, 1-xi],
         [F,  H,  D,  A,  B ]]).reshape(4, -1).T
      temp = xi*A
      weights = numpy.concatenate([A*temp*weights] + 4*[B*temp*weights])
    elif self.neighborhood == 2:
      A = xi*eta2
      B = A*eta3
      C = xi*eta1
      points = numpy.array(
        [[1-xi, 1-A ],
         [C,  B ],
         [1-A,  1-xi],
         [B,  C ]]).reshape(4, -1).T
      weights = numpy.concatenate(2*[xi**2*A*weights])
    else:
      assert self.neighborhood == -1, 'invalid neighborhood {!r}'.format(self.neighborhood)
      points = numpy.array([eta1*eta2, 1-eta2, eta3*xi, 1-xi]).T
      weights = eta2*xi*weights
    return points, weights

  def get_quad_bem_ischeme(self, ischeme):
    'Some cached quantities for the singularity quadrature scheme.'
    quad = getsimplex(1)**4
    points, weights = quad.getischeme(ischeme)
    eta1, eta2, eta3, xi = points.T
    if self.neighborhood == 0:
      xe = xi*eta1
      A = (1 - xi)*eta3
      B = (1 - xe)*eta2
      C = xi + A
      D = xe + B
      points = numpy.array(
        [[A, B, A, D, B, C, C, D],
         [B, A, D, A, C, B, D, C],
         [C, D, C, B, D, A, A, B],
         [D, C, B, C, A, D, B, A]]).reshape(4, -1).T
      weights = numpy.concatenate(8*[xi*(1-xi)*(1-xe)*weights])
    elif self.neighborhood == 1:
      ox = 1 - xi
      A = xi*eta1
      B = xi*eta2
      C = ox*eta3
      D = C + xi
      E = 1 - A
      F = E*eta3
      G = A + F
      points = numpy.array(
        [[D,  C,  G,  G,  F,  F ],
         [B,  B,  B,  xi, B,  xi],
         [C,  D,  F,  F,  G,  G ],
         [A,  A,  xi, B,  xi, B ]]).reshape(4, -1).T
      weights = numpy.concatenate(2*[xi**2*ox*weights] + 4*[xi**2*E*weights])
    elif self.neighborhood == 2:
      A = xi*eta1
      B = xi*eta2
      C = xi*eta3
      points = numpy.array(
        [[xi, A,  A,  A ],
         [A,  xi, B,  B ],
         [B,  B,  xi, C ],
         [C,  C,  C,  xi]]).reshape(4, -1).T
      weights = numpy.concatenate(4*[xi**3*weights])
    else:
      assert self.neighborhood == -1, 'invalid neighborhood {!r}'.format(self.neighborhood)
    return points, weights

  def getischeme_singular(self, n):
    'get integration scheme'

    gauss = 'gauss{}'.format(n*2-2)
    assert self.ref1 == self.ref2 == getsimplex(1)**2
    points, weights = self.get_quad_bem_ischeme(gauss)
    return self.singular_ischeme_quad(points), weights

class OwnChildReference(Reference):
  'forward self as child'

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

  def getischeme(self, ischeme):
    return self.baseref.getischeme(ischeme)

  @property
  def simplices(self):
    return self.baseref.simplices

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    return self.baseref.get_dof_transpose_map(degree, vertex_transpose_map)

class WithChildrenReference(Reference):
  'base reference with explicit children'

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

  @property
  def permutation_transforms(self):
    return self.baseref.permutation_transforms

  def nvertices_by_level(self, n):
    return self.baseref.nvertices_by_level(n)

  def child_divide(self, vals, n):
    return self.baseref.child_divide(vals, n)

  __sub__ = lambda self, other: self.empty if other in (self,self.baseref) else self.baseref.with_children(self_child-other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref in (self,self.baseref) else NotImplemented
  __rsub__ = lambda self, other: self.baseref.with_children(other_child - self_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if other == self.baseref else NotImplemented
  __and__ = lambda self, other: self if other == self.baseref else other if isinstance(other,WithChildrenReference) and self == other.baseref else self.baseref.with_children(self_child & other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented
  __or__ = lambda self, other: other if other == self.baseref else self.baseref.with_children(self_child | other_child for self_child, other_child in zip(self.child_refs, other.child_refs)) if isinstance(other, WithChildrenReference) and other.baseref == self.baseref else NotImplemented

  @cache.property
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
        points, weights = child.getischeme('vertex{}'.format(i-1))
        assert weights is None
        if childindex == ichild:
          rng = numpy.arange(npoints, npoints+len(points))
        npoints += len(points)
      elif ichild==childindex:
        rng = numpy.array([],dtype=int)
    return npoints, rng

  def getischeme(self, ischeme):
    'get integration scheme'

    if ischeme.startswith('vertex'):
      return self.baseref.getischeme(ischeme)

    allcoords = []
    allweights = []
    for trans, ref in self.children:
      if ref:
        points, weights = ref.getischeme(ischeme)
        allcoords.append(trans.apply(points))
        if weights is not None:
          allweights.append(weights * abs(float(trans.det)))

    coords = numpy.concatenate(allcoords, axis=0)
    weights = numpy.concatenate(allweights, axis=0) \
      if len(allweights) == len(allcoords) else None

    return coords, weights

  @property
  def simplices(self):
    return [(trans2*trans1, simplex) for trans2, child in self.children for trans1, simplex in (child.simplices if child else [])]

  @cache.property
  def edge_transforms(self):
    return tuple(self.baseref.edge_transforms) \
         + tuple(transform.ScaledUpdim(self.child_transforms[ichild], self.child_refs[ichild].edge_transforms[iedge]) for ichild, iedge, ref in self.__extra_edges)

  @cache.property
  def edge_refs(self):
    refs = []
    for etrans, eref in self.baseref.edges:
      children = []
      if eref:
        for ctrans, cref in eref.children:
          ctrans_, etrans_ = etrans.swapup(ctrans)
          ichild = self.baseref.child_transforms.index(ctrans_)
          cref = self.child_refs[ichild]
          children.append(cref.edge_refs[cref.edge_transforms.index(etrans_)] if cref else EmptyReference(self.ndims-1))
      refs.append(eref.with_children(children))
    for ichild, iedge, ref in self.__extra_edges:
      refs.append(OwnChildReference(ref))
    return tuple(refs)

  @cache.property
  def connectivity(self):
    # same as base implementation but cheaper
    return tuple(tuple(edges[iedge] if iedge < len(edges) and edges[iedge] != -1 and self.child_refs[edges[iedge]] else -1 for iedge in range(self.child_refs[ichild].nedges))
      for ichild, edges in enumerate(self.baseref.connectivity))

  def inside(self, point, eps=0):
    return any(cref.inside(ctrans.invapply(point), eps=eps) for ctrans, cref in self.children)

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    return self.baseref.get_dof_transpose_map(degree, vertex_transpose_map)

class MosaicReference(Reference):
  'triangulation'

  def __init__(self, baseref, edge_refs:tuple, midpoint:numeric.const):
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
        ei = Ei.edge_refs[iedge2] if Ei else EmptyReference(Ei.ndims-1)
        Ej = edge_refs[jedge1]
        ej = Ej.edge_refs[jedge2] if Ej else EmptyReference(Ej.ndims-1)
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

  @cache.property
  def vertices(self):
    vertices = []
    for etrans, eref in self.edges:
      indices = []
      for vertex in etrans.apply(eref.vertices).tolist():
        try:
          index = vertices.index(vertex)
        except ValueError:
          index = len(vertices)
          vertices.append(vertex)
        indices.append(index)
    return numeric.const(vertices)

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

  @cache.property
  def subrefs(self):
    return [ref.cone(trans,self._midpoint) for trans, ref in zip(self.baseref.edge_transforms, self._edge_refs) if ref]

  @property
  def simplices(self):
    return [simplex for subvol in self.subrefs for simplex in subvol.simplices]

  def getischeme(self, ischeme):
    'get integration scheme'

    if ischeme.startswith('vertex'):
      return self.baseref.getischeme(ischeme)

    allpoints, allweights = zip(*[subvol.getischeme(ischeme) for subvol in self.subrefs])
    points = numpy.concatenate(allpoints, axis=0)
    weights = None if any(w is None for w in allweights) else numpy.concatenate(allweights, axis=0)
    return points, weights

  def inside(self, point, eps=0):
    return any(subref.inside(point, eps=eps) for subref in self.subrefs)

  def get_ndofs(self, degree):
    return self.baseref.get_ndofs(degree)

  def get_poly_coeffs(self, basis, **kwargs):
    return self.baseref.get_poly_coeffs(basis, **kwargs)

  def get_edge_dofs(self, degree, iedge):
    return self.baseref.get_edge_dofs(degree, iedge)

  def get_dof_transpose_map(self, degree, vertex_transpose_map):
    return self.baseref.get_dof_transpose_map(degree, vertex_transpose_map)


# UTILITY FUNCTIONS

_gauss = []
def gauss(degree):
  n = degree // 2
  while len(_gauss) <= n:
    _gauss.append(None)
  gaussn = _gauss[n]
  if gaussn is None:
    k = numpy.arange(n) + 1
    d = k / numpy.sqrt(4*k**2-1)
    x, w = numpy.linalg.eigh(numpy.diagflat(d,-1)) # eigh operates (by default) on lower triangle
    _gauss[n] = gaussn = numeric.const((x+1) * .5, copy=False), numeric.const(w[0]**2, copy=False)
  return gaussn

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


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
