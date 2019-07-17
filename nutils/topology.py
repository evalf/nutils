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
The topology module defines the topology objects, notably the
:class:`StructuredTopology` and :class:`UnstructuredTopology`. Maintaining
strict separation of topological and geometrical information, the topology
represents a set of elements and their interconnectivity, boundaries,
refinements, subtopologies etc, but not their positioning in physical space. The
dimension of the topology represents the dimension of its elements, not that of
the the space they are embedded in.

The primary role of topologies is to form a domain for :mod:`nutils.function`
objects, like the geometry function and function bases for analysis, as well as
provide tools for their construction. It also offers methods for integration and
sampling, thus providing a high level interface to operations otherwise written
out in element loops. For lower level operations topologies can be used as
:mod:`nutils.element` iterators.
"""

from . import element, elementseq, function, util, parallel, config, numeric, cache, transform, transformseq, warnings, matrix, types, sample, points, log, _
import numpy, functools, collections.abc, itertools, functools, operator, numbers, pathlib, abc

_identity = lambda x: x

class Topology(types.Singleton):
  'topology base class'

  __slots__ = 'references', 'transforms', 'opposites', 'ndims'
  __cache__ = 'border_transforms', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, references:elementseq.strictreferences, transforms:transformseq.stricttransforms, opposites:transformseq.stricttransforms):
    assert references.ndims == opposites.fromdims == transforms.fromdims
    assert len(references) == len(transforms) == len(opposites)
    self.references = references
    self.transforms = transforms
    self.opposites = opposites
    self.ndims = transforms.fromdims
    super().__init__()

  def __str__(self):
    'string representation'

    return '{}(#{})'.format(self.__class__.__name__, len(self))

  def __len__(self):
    return len(self.references)

  def getitem(self, item):
    return EmptyTopology(self.ndims)

  def __getitem__(self, item):
    if numeric.isintarray(item):
      item = types.frozenarray(item)
      return UnstructuredTopology(self.references[item], self.transforms[item], self.opposites[item])
    if not isinstance(item, tuple):
      item = item,
    if all(it in (...,slice(None)) for it in item):
      return self
    topo = self.getitem(item) if len(item) != 1 or not isinstance(item[0],str) \
       else functools.reduce(operator.or_, map(self.getitem, item[0].split(',')), EmptyTopology(self.ndims))
    if not topo:
      raise KeyError(item)
    return topo

  def __invert__(self):
    return OppositeTopology(self)

  def __or__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other if not self \
      else self if not other \
      else NotImplemented if isinstance(other, UnionTopology) \
      else UnionTopology((self,other))

  __ror__ = lambda self, other: self.__or__(other)

  def __and__(self, other):
    keep_self = numpy.array(list(map(other.transforms.contains_with_tail, self.transforms)), dtype=bool)
    if keep_self.all():
      return self
    keep_other = numpy.array(list(map(self.transforms.contains_with_tail, other.transforms)), dtype=bool)
    if keep_other.all():
      return other
    ind_self = types.frozenarray(keep_self.nonzero()[0], copy=False)
    ind_other = types.frozenarray([i for i, trans in enumerate(other.transforms) if keep_other[i] and not self.transforms.contains(trans)], dtype=int)
    # The last condition is to avoid duplicate elements. Note that we could
    # have reused the result of an earlier lookup to avoid a new (using index
    # instead of contains) but we choose to trade some speed for simplicity.
    references = elementseq.chain([self.references[ind_self], other.references[ind_other]], self.ndims)
    transforms = transformseq.chain([self.transforms[ind_self], other.transforms[ind_other]], self.ndims)
    opposites = transformseq.chain([self.opposites[ind_self], other.opposites[ind_other]], self.ndims)
    return UnstructuredTopology(references, transforms, opposites, ndims=self.ndims)

  __rand__ = lambda self, other: self.__and__(other)

  def __add__(self, other):
    return self | other

  def __sub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other.__rsub__(self)

  def __rsub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other - other.subset(self, newboundary=getattr(self,'boundary',None))

  def __mul__(self, other):
    return ProductTopology(self, other)

  @property
  def border_transforms(self):
    indices = set()
    for btrans in self.boundary.transforms:
      try:
        ielem, tail = self.transforms.index_with_tail(btrans)
      except ValueError:
        pass
      else:
        indices.add(ielem)
    return self.transforms[numpy.array(sorted(indices), dtype=int)]

  @property
  def refine_iter(self):
    topo = self
    while True:
      yield topo
      topo = topo.refined

  def basis(self, name, *args, **kwargs):
    '''
    Create a basis.
    '''
    if self.ndims == 0:
      return function.PlainBasis([[1]], [[0]], 1, self.transforms)
    split = name.split('-', 1)
    if len(split) == 2 and split[0] in ('h', 'th'):
      name = split[1] # default to non-hierarchical bases
      if split[0] == 'th':
        kwargs.pop('truncation_tolerance', None)
    f = getattr(self, 'basis_' + name)
    return f(*args, **kwargs)

  def sample(self, ischeme, degree):
    'Create sample.'

    points = [ischeme(reference, degree) for reference in self.references] if callable(ischeme) \
        else self.references.getpoints(ischeme, degree)
    offset = numpy.cumsum([0] + [p.npoints for p in points])
    transforms = self.transforms,
    if len(self.transforms) == 0 or self.opposites != self.transforms:
      transforms += self.opposites,
    return sample.Sample(transforms, points, map(numpy.arange, offset[:-1], offset[1:]))

  @util.single_or_multiple
  def integrate_elementwise(self, funcs, *, asfunction=False, **kwargs):
    'element-wise integration'

    ielem = function.TransformsIndexWithTail(self.transforms, function.TRANS).index
    with matrix.backend('numpy'):
      retvals = self.integrate([function.Inflate(function.asarray(func)[_], dofmap=ielem[_], length=len(self), axis=0) for func in funcs], **kwargs)
    retvals = [retval.export('dense') if len(retval.shape) == 2 else retval for retval in retvals]
    return [function.elemwise(self.transforms, retval) for retval in retvals] if asfunction \
      else retvals

  @util.single_or_multiple
  def elem_mean(self, funcs, geometry=None, ischeme='gauss', degree=None, **kwargs):
    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    funcs = (1,)+funcs
    if geometry is not None:
      funcs = [func * function.J(geometry, self.ndims) for func in funcs]
    area, *integrals = self.integrate_elementwise(funcs, ischeme=ischeme, degree=degree, **kwargs)
    return [integral / area[(slice(None),)+(_,)*(integral.ndim-1)] for integral in integrals]

  @util.single_or_multiple
  def integrate(self, funcs, ischeme='gauss', degree=None, edit=None, *, arguments=None, title='integrate'):
    'integrate functions'

    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    if edit is not None:
      funcs = [edit(func) for func in funcs]
    return self.sample(ischeme, degree).integrate(funcs, **arguments or {})

  def integral(self, func, ischeme='gauss', degree=None, edit=None):
    'integral'

    ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
    if edit is not None:
      funcs = edit(func)
    return self.sample(ischeme, degree).integral(func)

  def projection(self, fun, onto, geometry, **kwargs):
    'project and return as function'

    weights = self.project(fun, onto, geometry, **kwargs)
    return onto.dot(weights)

  @log.withcontext
  def project(self, fun, onto, geometry, ischeme='gauss', degree=None, droptol=1e-12, exact_boundaries=False, constrain=None, verify=None, ptype='lsqr', edit=None, *, arguments=None, **solverargs):
    'L2 projection of function onto function space'

    log.debug('projection type:', ptype)

    if degree is not None:
      ischeme += str(degree)
    if constrain is None:
      constrain = util.NanVec(onto.shape[0])
    else:
      constrain = constrain.copy()
    if exact_boundaries:
      constrain |= self.boundary.project(fun, onto, geometry, constrain=constrain, ischeme=ischeme, droptol=droptol, ptype=ptype, edit=edit, arguments=arguments)
    assert isinstance(constrain, util.NanVec)
    assert constrain.shape == onto.shape[:1]

    avg_error = None # setting this depends on projection type

    if ptype == 'lsqr':
      assert ischeme is not None, 'please specify an integration scheme for lsqr-projection'
      fun2 = function.asarray(fun)**2
      if len(onto.shape) == 1:
        Afun = function.outer(onto)
        bfun = onto * fun
      elif len(onto.shape) == 2:
        Afun = function.outer(onto).sum(2)
        bfun = function.sum(onto * fun, -1)
        if fun2.ndim:
          fun2 = fun2.sum(-1)
      else:
        raise Exception
      assert fun2.ndim == 0
      J = function.J(geometry, self.ndims)
      A, b, f2, area = self.integrate([Afun*J,bfun*J,fun2*J,J], ischeme=ischeme, edit=edit, arguments=arguments)
      N = A.rowsupp(droptol)
      if numpy.equal(b, 0).all():
        constrain[~constrain.where&N] = 0
        avg_error = 0.
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve(b, constrain=solvecons, **solverargs)
        constrain[N] = u[N]
        err2 = f2 - numpy.dot(2 * b - A @ u, u) # can be negative ~zero due to rounding errors
        avg_error = numpy.sqrt(err2) / area if err2 > 0 else 0

    elif ptype == 'convolute':
      assert ischeme is not None, 'please specify an integration scheme for convolute-projection'
      if len(onto.shape) == 1:
        ufun = onto * fun
        afun = onto
      elif len(onto.shape) == 2:
        ufun = function.sum(onto * fun, axis=-1)
        afun = function.norm2(onto)
      else:
        raise Exception
      J = function.J(geometry, self.ndims)
      u, scale = self.integrate([ufun*J, afun*J], ischeme=ischeme, edit=edit, arguments=arguments)
      N = ~constrain.where & (scale > droptol)
      constrain[N] = u[N] / scale[N]

    elif ptype == 'nodal':

      ## data = function.Tuple([fun, onto])
      ## F = W = 0
      ## for elem in self:
      ##   f, w = data(elem, 'bezier2')
      ##   W += w.sum(axis=-1).sum(axis=0)
      ##   F += numeric.contract(f[:,_,:], w, axis=[0,2])
      ## I = (W!=0)

      F = numpy.zeros(onto.shape[0])
      W = numpy.zeros(onto.shape[0])
      I = numpy.zeros(onto.shape[0], dtype=bool)
      fun = function.asarray(fun).prepare_eval()
      data = function.Tuple(function.Tuple([fun, onto_f.simplified, function.Tuple(onto_ind)]) for onto_ind, onto_f in function.blocks(onto.prepare_eval()))
      for ref, trans, opp in zip(self.references, self.transforms, self.opposites):
        ipoints, iweights = ref.getischeme('bezier2')
        for fun_, onto_f_, onto_ind_ in data.eval(_transforms=(trans, opp), _points=ipoints, **arguments or {}):
          onto_f_ = onto_f_.swapaxes(0,1) # -> dof axis, point axis, ...
          indfun_ = fun_[(slice(None),)+numpy.ix_(*onto_ind_[1:])]
          assert onto_f_.shape[0] == len(onto_ind_[0])
          assert onto_f_.shape[1:] == indfun_.shape
          W[onto_ind_[0]] += onto_f_.reshape(onto_f_.shape[0],-1).sum(1)
          F[onto_ind_[0]] += (onto_f_ * indfun_).reshape(onto_f_.shape[0],-1).sum(1)
          I[onto_ind_[0]] = True

      I[constrain.where] = False
      constrain[I] = F[I] / W[I]

    else:
      raise Exception('invalid projection {!r}'.format(ptype))

    numcons = constrain.where.sum()
    info = 'constrained {}/{} dofs'.format(numcons, constrain.size)
    if avg_error is not None:
      info += ', error {:.2e}/area'.format(avg_error)
    log.info(info)
    if verify is not None:
      assert numcons == verify, 'number of constraints does not meet expectation: {} != {}'.format(numcons, verify)

    return constrain

  def refined_by(self, refine):
    'create refined space by refining dofs in existing one'

    return HierarchicalTopology(self, [numpy.arange(len(self))]).refined_by(refine)

  @property
  def refined(self):
    return RefinedTopology(self)

  def refine(self, n):
    'refine entire topology n times'

    if numpy.iterable(n):
      assert len(n) == self.ndims
      assert all(ni == n[0] for ni in n)
      n = n[0]
    return self if n <= 0 else self.refined.refine(n-1)

  @log.withcontext
  def trim(self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None, *, arguments=None):
    'trim element along levelset'

    if arguments is None:
      arguments = {}

    levelset = levelset.prepare_eval().simplified
    refs = []
    if leveltopo is None:
      with log.context('progress'):
        for ielem, (ref, trans, opp) in enumerate(zip(self.references, self.transforms, self.opposites)):
          log.recontext('progress {:.0f}%'.format(100*ielem/len(self)))
          levels = levelset.eval(_transforms=(trans, opp), _points=ref.getpoints('vertex', maxrefine).coords, **arguments)
          refs.append(ref.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
    else:
      log.info('collecting leveltopo elements')
      bins = [set() for ielem in range(len(self))]
      for trans in leveltopo.transforms:
        ielem, tail = self.transforms.index_with_tail(trans)
        bins[ielem].add(tail)
      fcache = cache.WrapperCache()
      with log.context('trimming'):
        for ielem, (ref, trans, ctransforms) in enumerate(zip(self.references, self.transforms, bins)):
          log.recontext('trimming {:.0f}%'.format(100*ielem/len(self)))
          levels = numpy.empty(ref.nvertices_by_level(maxrefine))
          cover = list(fcache[ref.vertex_cover](frozenset(ctransforms), maxrefine))
          # confirm cover and greedily optimize order
          mask = numpy.ones(len(levels), dtype=bool)
          while mask.any():
            imax = numpy.argmax([mask[indices].sum() for tail, points, indices in cover])
            tail, points, indices = cover.pop(imax)
            levels[indices] = levelset.eval(_transforms=(trans + tail,), _points=points, **arguments)
            mask[indices] = False
          refs.append(ref.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
      log.debug('cache', fcache.stats)
    return SubsetTopology(self, refs, newboundary=name)

  def subset(self, topo, newboundary=None, strict=False):
    'intersection'
    refs = [ref.empty for ref in self.references]
    for ref, trans in zip(topo.references, topo.transforms):
      try:
        ielem = self.transforms.index(trans)
      except ValueError:
        assert not strict, 'elements do not form a strict subset'
      else:
        subref = self.references[ielem] & ref
        if strict:
          assert subref == ref, 'elements do not form a strict subset'
        refs[ielem] = subref
    if not any(refs):
      return EmptyTopology(self.ndims)
    return SubsetTopology(self, refs, newboundary)

  def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
    return WithGroupsTopology(self, vgroups, bgroups, igroups, pgroups) if vgroups or bgroups or igroups or pgroups else self

  withsubdomain  = lambda self, **kwargs: self.withgroups(vgroups=kwargs)
  withboundary   = lambda self, **kwargs: self.withgroups(bgroups=kwargs)
  withinterfaces = lambda self, **kwargs: self.withgroups(igroups=kwargs)
  withpoints     = lambda self, **kwargs: self.withgroups(pgroups=kwargs)

  @log.withcontext
  @util.single_or_multiple
  def elem_project(self, funcs, degree, ischeme=None, check_exact=False, *, arguments=None):

    if arguments is None:
      arguments = {}

    if ischeme is None:
      ischeme = 'gauss{}'.format(degree*2)

    blocks = function.Tuple([function.Tuple([function.Tuple((function.Tuple(ind), f.simplified))
      for ind, f in function.blocks(func.prepare_eval())])
        for func in funcs])

    bases = {}
    extractions = [[] for ifunc in range(len(funcs))]

    for ielem, (ref, trans, opp) in enumerate(zip(self.references, self.transforms, self.opposites)):
      with log.context('elem', ielem, '({:.0f}%)'.format(100*ielem/len(self))):

        try:
          points, projector, basis = bases[ref]
        except KeyError:
          points, weights = ref.getischeme(ischeme)
          coeffs = ref.get_poly_coeffs('bernstein', degree=degree)
          basis = numeric.poly_eval(coeffs[_], points)
          npoints, nfuncs = basis.shape
          A = numeric.dot(weights, basis[:,:,_] * basis[:,_,:])
          projector = numpy.linalg.solve(A, basis.T * weights)
          bases[ref] = points, projector, basis

        for ifunc, ind_val in enumerate(blocks.eval(_transforms=(trans, opp), _points=points, **arguments)):

          if len(ind_val) == 1:
            (allind, sumval), = ind_val
          else:
            allind, where = zip(*[numpy.unique([i for ind, val in ind_val for i in ind[iax]], return_inverse=True) for iax in range(funcs[ifunc].ndim)])
            sumval = numpy.zeros([len(n) for n in (points,) + allind])
            for ind, val in ind_val:
              I, where = zip(*[(w[:len(n)], w[len(n):]) for w, n in zip(where, ind)])
              numpy.add.at(sumval, numpy.ix_(range(len(points)), *I), val)
            assert not any(where)

          ex = numeric.dot(projector, sumval)
          if check_exact:
            numpy.testing.assert_almost_equal(sumval, numeric.dot(basis, ex), decimal=15)

          extractions[ifunc].append((allind, ex))

    return extractions

  @log.withcontext
  def volume(self, geometry, ischeme='gauss', degree=1, *, arguments=None):
    return self.integrate(function.J(geometry, self.ndims), ischeme=ischeme, degree=degree, arguments=arguments)

  @log.withcontext
  def check_boundary(self, geometry, elemwise=False, ischeme='gauss', degree=1, tol=1e-15, print=print, *, arguments=None):
    if elemwise:
      for ref in self.references:
        ref.check_edges(tol=tol, print=print)
    volume = self.volume(geometry, ischeme=ischeme, degree=degree, arguments=arguments)
    J = function.J(geometry, self.ndims-1)
    zeros, volumes = self.boundary.integrate([geometry.normal()*J, geometry*geometry.normal()*J], ischeme=ischeme, degree=degree, arguments=arguments)
    if numpy.greater(abs(zeros), tol).any():
      print('divergence check failed: {} != 0'.format(zeros))
    if numpy.greater(abs(volumes - volume), tol).any():
      print('divergence check failed: {} != {}'.format(volumes, volume))

  def indicator(self, subtopo):
    if isinstance(subtopo, str):
      subtopo = self[subtopo]
    values = numpy.zeros([len(self)], dtype=int)
    values[numpy.fromiter(map(self.transforms.index, subtopo.transforms), dtype=int)] = 1
    return function.Get(values, axis=0, item=function.TransformsIndexWithTail(self.transforms, function.TRANS).index)

  def select(self, indicator, ischeme='bezier2', **kwargs):
    sample = self.sample(*element.parse_legacy_ischeme(ischeme))
    isactive = numpy.greater(sample.eval(indicator, **kwargs), 0)
    selected = types.frozenarray(tuple(i for i, index in enumerate(sample.index) if isactive[index].any()), dtype=int)
    return self[selected]

  @log.withcontext
  def locate(self, geom, coords, *, ischeme='vertex', scale=1, tol=1e-12, eps=0, maxiter=100, arguments=None):
    '''Create a sample based on physical coordinates.

    In a finite element application, functions are commonly evaluated in points
    that are defined on the topology. The reverse, finding a point on the
    topology based on a function value, is often a nonlinear process and as
    such involves Newton iterations. The ``locate`` function facilitates this
    search process and produces a :class:`nutils.sample.Sample` instance that
    can be used for the subsequent evaluation of any function in the given
    physical points.

    Example:

    >>> from . import mesh
    >>> domain, geom = mesh.unitsquare(nelems=3, etype='mixed')
    >>> sample = domain.locate(geom, [[.9, .4]])
    >>> sample.eval(geom).tolist()
    [[0.9, 0.4]]

    Locate has a long list of arguments that can be used to steer the nonlinear
    search process, but the default values should be fine for reasonably
    standard situations.

    Args
    ----
    geom : 1-dimensional :class:`nutils.function.Array`
        Geometry function of length ``ndims``.
    coords : 2-dimensional :class:`float` array
        Array of coordinates with ``ndims`` columns.
    ischeme : :class:`str` (default: "vertex")
        Sample points used to determine bounding boxes.
    scale : :class:`float` (default: 1)
        Bounding box amplification factor, useful when element shapes are
        distorted. Setting this to >1 can increase computational effort but is
        otherwise harmless.
    tol : :class:`float` (default: 1e-12)
        Newton tolerance.
    eps : :class:`float` (default: 0)
        Epsilon radius around element within which a point is considered to be
        inside.
    maxiter : :class:`int` (default: 100)
        Maximum allowed number of Newton iterations.
    arguments : :class:`dict` (default: None)
        Arguments for function evaluation.

    Returns
    -------
    located : :class:`nutils.sample.Sample`
    '''

    coords = numpy.asarray(coords, dtype=float)
    if geom.ndim == 0:
      geom = geom[_]
      coords = coords[...,_]
    if not geom.shape == coords.shape[1:] == (self.ndims,):
      raise Exception('invalid geometry or point shape for {}D topology'.format(self.ndims))
    bboxsample = self.sample(*element.parse_legacy_ischeme(ischeme))
    vertices = map(bboxsample.eval(geom, **arguments or {}).__getitem__, bboxsample.index)
    bboxes = numpy.array([numpy.mean(v,axis=0) * (1-scale) + numpy.array([numpy.min(v,axis=0), numpy.max(v,axis=0)]) * scale
      for v in vertices]) # nelems x {min,max} x ndims
    vref = element.getsimplex(0)
    ielems = parallel.shempty(len(coords), dtype=int)
    xis = parallel.shempty((len(coords),len(geom)), dtype=float)
    ipoints = parallel.range(len(coords))
    with parallel.fork(min(config.nprocs, len(coords))), log.context('progress'):
      for ipoint in ipoints:
        log.recontext('progress {:.0f}%'.format(100*ipoint/len(coords)))
        coord = coords[ipoint]
        ielemcandidates, = numpy.logical_and(numpy.greater_equal(coord, bboxes[:,0,:]), numpy.less_equal(coord, bboxes[:,1,:])).all(axis=-1).nonzero()
        for ielem in sorted(ielemcandidates, key=lambda i: numpy.linalg.norm(bboxes[i].mean(0)-coord)):
          converged = False
          ref = self.references[ielem]
          p = ref.getpoints('gauss', 1)
          xi = p.coords
          w = p.weights
          xi = (numpy.dot(w,xi) / w.sum())[_] if len(xi) > 1 else xi.copy()
          J = function.localgradient(geom, self.ndims)
          geom_J = function.Tuple((geom, J)).prepare_eval().simplified
          for iiter in range(maxiter):
            coord_xi, J_xi = geom_J.eval(_transforms=(self.transforms[ielem], self.opposites[ielem]), _points=xi, **arguments or {})
            err = numpy.linalg.norm(coord - coord_xi)
            if err < tol:
              converged = True
              break
            if iiter and err > prev_err:
              break
            prev_err = err
            xi += numpy.linalg.solve(J_xi, coord - coord_xi)
          if converged and ref.inside(xi[0], eps=eps):
            ielems[ipoint] = ielem
            xis[ipoint], = xi
            break
        else:
          raise LocateError('failed to locate point: {}'.format(coord))
    return self._sample(ielems, xis)

  def _sample(self, ielems, coords):
    uielems = numpy.unique(ielems)
    points_ = []
    index = []
    for ielem in uielems:
      w, = numpy.equal(ielems, ielem).nonzero()
      points_.append(points.CoordsPoints(coords[w]))
      index.append(w)
    transforms = self.transforms[uielems],
    if len(self.transforms) == 0 or self.opposites != self.transforms:
      transforms += self.opposites[uielems],
    return sample.Sample(transforms, points_, index)

  def revolved(self, geom):
    assert geom.ndim == 1
    revdomain = self * RevolutionTopology()
    angle = function.RevolutionAngle()
    geom, angle = function.bifurcate(geom, angle)
    revgeom = function.concatenate([geom[0] * function.trignormal(angle), geom[1:]])
    simplify = _identity
    return revdomain, revgeom, simplify

  def extruded(self, geom, nelems, periodic=False, bnames=('front','back')):
    assert geom.ndim == 1
    root = transform.Identifier(self.ndims+1, 'extrude')
    extopo = self * StructuredLine(root, i=0, j=nelems, periodic=periodic, bnames=bnames)
    exgeom = function.concatenate(function.bifurcate(geom, function.rootcoords(1)))
    return extopo, exgeom

  @property
  @log.withcontext
  def boundary(self):
    '''
    :class:`Topology`:
      The boundary of this topology.
    '''

    references = []
    transforms = []
    for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
      for (edgetrans, edgeref), ioppelem in zip(elemref.edges, ioppelems):
        if edgeref:
          if ioppelem == -1:
            references.append(edgeref)
            transforms.append(elemtrans+(edgetrans,))
          else:
            ioppedge = self.connectivity[ioppelem].index(ielem)
            ref = edgeref - self.references[ioppelem].edge_refs[ioppedge]
            if ref:
              references.append(ref)
              transforms.append(elemtrans+(edgetrans,))
    return UnstructuredTopology(references, transforms, transforms, ndims=self.ndims-1)

  @property
  @log.withcontext
  def interfaces(self):
    references = []
    transforms = []
    opposites = []
    for ielem, (ioppelems, elemref, elemtrans) in enumerate(zip(self.connectivity, self.references, self.transforms)):
      for (edgetrans, edgeref), ioppelem in zip(elemref.edges, ioppelems):
        if edgeref and -1 < ioppelem < ielem:
          ioppedge = self.connectivity[ioppelem].index(ielem)
          oppedgetrans, oppedgeref = self.references[ioppelem].edges[ioppedge]
          ref = oppedgeref and edgeref & oppedgeref
          if ref:
            references.append(ref)
            transforms.append(elemtrans+(edgetrans,))
            opposites.append(self.transforms[ioppelem]+(oppedgetrans,))
    return UnstructuredTopology(references, transforms, opposites, ndims=self.ndims-1)

  def basis_spline(self, degree):
    assert degree == 1
    return self.basis('std', degree)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    assert numeric.isint(degree) and degree >= 0
    if self.references.isuniform:
      coeffs = [self.references[0].get_poly_coeffs('bernstein', degree=degree)]*len(self.references)
    else:
      coeffs = [ref.get_poly_coeffs('bernstein', degree=degree) for ref in self.references]
    return function.DiscontBasis(coeffs, self.transforms)

  def _basis_c0_structured(self, name, degree):
    'C^0-continuous shape functions with lagrange stucture'

    assert numeric.isint(degree) and degree >= 0

    if degree == 0:
      raise ValueError('Cannot build a C^0-continuous basis of degree 0.  Use basis \'discont\' instead.')

    coeffs = [ref.get_poly_coeffs(name, degree=degree) for ref in self.references]
    offsets = numpy.cumsum([0] + [len(c) for c in coeffs])
    dofmap = numpy.repeat(-1, offsets[-1])
    for ielem, ioppelems in enumerate(self.connectivity):
      for iedge, jelem in enumerate(ioppelems): # loop over element neighbors and merge dofs
        if jelem < ielem:
          continue # either there is no neighbor along iedge or situation will be inspected from the other side
        jedge = self.connectivity[jelem].index(ielem)
        idofs = offsets[ielem] + self.references[ielem].get_edge_dofs(degree, iedge)
        jdofs = offsets[jelem] + self.references[jelem].get_edge_dofs(degree, jedge)
        for idof, jdof in zip(idofs, jdofs):
          while dofmap[idof] != -1:
            idof = dofmap[idof]
          while dofmap[jdof] != -1:
            jdof = dofmap[jdof]
          if idof != jdof:
            dofmap[max(idof, jdof)] = min(idof, jdof) # create left-looking pointer
    # assign dof numbers left-to-right
    ndofs = 0
    for i, n in enumerate(dofmap):
      if n == -1:
        dofmap[i] = ndofs
        ndofs += 1
      else:
        dofmap[i] = dofmap[n]

    elem_slices = map(slice, offsets[:-1], offsets[1:])
    dofs = tuple(types.frozenarray(dofmap[s]) for s in elem_slices)
    return function.PlainBasis(coeffs, dofs, ndofs, self.transforms)

  def basis_lagrange(self, degree):
    'lagrange shape functions'
    return self._basis_c0_structured('lagrange', degree)

  def basis_bernstein(self, degree):
    'bernstein shape functions'
    return self._basis_c0_structured('bernstein', degree)

  basis_std = basis_bernstein

stricttopology = types.strict[Topology]

class LocateError(Exception):
  pass

class WithGroupsTopology(Topology):
  'item topology'

  __slots__ = 'basetopo', 'vgroups', 'bgroups', 'igroups', 'pgroups'
  __cache__ = 'refined',

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, vgroups:types.frozendict={}, bgroups:types.frozendict={}, igroups:types.frozendict={}, pgroups:types.frozendict={}):
    assert vgroups or bgroups or igroups or pgroups
    self.basetopo = basetopo
    self.vgroups = vgroups
    self.bgroups = bgroups
    self.igroups = igroups
    self.pgroups = pgroups
    super().__init__(basetopo.references, basetopo.transforms, basetopo.opposites)
    assert all(topo is Ellipsis or isinstance(topo, str) or isinstance(topo, Topology) and topo.ndims == basetopo.ndims and set(self.basetopo.transforms).issuperset(topo.transforms) for topo in self.vgroups.values())

  def __len__(self):
    return len(self.basetopo)

  def getitem(self, item):
    if isinstance(item, str) and item in self.vgroups:
      itemtopo = self.vgroups[item]
      return itemtopo if isinstance(itemtopo, Topology) else self.basetopo[itemtopo]
    return self.basetopo.getitem(item)

  @property
  def border_transforms(self):
    return self.basetopo.border_transforms

  @property
  def connectivity(self):
    return self.basetopo.connectivity

  @property
  def boundary(self):
    return self.basetopo.boundary.withgroups(self.bgroups)

  @property
  def interfaces(self):
    baseitopo = self.basetopo.interfaces
    igroups = self.igroups.copy()
    for name, topo in self.igroups.items():
      if isinstance(topo, Topology):
        # last minute orientation fix
        s = []
        for transs in zip(topo.transforms, topo.opposites):
          for trans in transs:
            try:
              s.append(baseitopo.transforms.index(trans))
              break
            except ValueError:
              continue
          else:
            raise ValueError('group is not a subset of topology')
        s = types.frozenarray(tuple(sorted(s)), dtype=int)
        igroups[name] = UnstructuredTopology(baseitopo.references[s], baseitopo.transforms[s], baseitopo.opposites[s])
    return baseitopo.withgroups(igroups)

  @property
  def points(self):
    ptopos = []
    pnames = []
    topo = self
    while isinstance(topo, WithGroupsTopology):
      for pname, ptopo in topo.pgroups.items():
        if pname not in pnames:
          pnames.append(pname)
          ptopos.append(ptopo)
      topo = topo.basetopo
    return UnionTopology(ptopos, pnames)

  def basis(self, name, *args, **kwargs):
    return self.basetopo.basis(name, *args, **kwargs)

  @property
  def refined(self):
    groups = [{name: topo.refined if isinstance(topo,Topology) else topo for name, topo in groups.items()} for groups in (self.vgroups,self.bgroups,self.igroups,self.pgroups)]
    return self.basetopo.refined.withgroups(*groups)

class OppositeTopology(Topology):
  'opposite topology'

  __slots__ = 'basetopo',

  def __init__(self, basetopo):
    self.basetopo = basetopo
    super().__init__(basetopo.references, basetopo.opposites, basetopo.transforms)

  def getitem(self, item):
    return ~(self.basetopo.getitem(item))

  def __len__(self):
    return len(self.basetopo)

  def __invert__(self):
    return self.basetopo

class EmptyTopology(Topology):
  'empty topology'

  __slots__ = ()

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__(elementseq.PlainReferences((), ndims), transformseq.PlainTransforms((), ndims), transformseq.PlainTransforms((), ndims))

  def __or__(self, other):
    assert self.ndims == other.ndims
    return other

  def __rsub__(self, other):
    return other

class Point(Topology):
  'point'

  __slots__ = ()

  @types.aspreprocessor
  @types.apply_annotations
  def _preprocess_init(self, trans:transform.stricttransform, opposite:transform.stricttransform=None):
    return (self, trans, trans if opposite is None else opposite), {}

  @_preprocess_init
  def __init__(self, trans, opposite):
    assert trans[-1].fromdims == 0
    references = elementseq.asreferences([element.getsimplex(0)], 0)
    transforms = transformseq.PlainTransforms((trans,), 0)
    opposites = transforms if opposite is None else transformseq.PlainTransforms((opposite,), 0)
    super().__init__(references, transforms, opposites)

class StructuredLine(Topology):
  'structured topology'

  __slots__ = 'root', 'i', 'j', 'periodic', 'bnames'
  __cache__ = 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, i:types.strictint, j:types.strictint, periodic:bool=False, bnames:types.tuple[types.strictstr]=None):
    'constructor'

    assert j > i
    assert not bnames or len(bnames) == 2
    self.root = root
    self.i = i
    self.j = j
    self.periodic = periodic
    self.bnames = bnames or ()

    references = elementseq.asreferences([element.getsimplex(1)], 1)*(self.j-self.i)
    transforms = transformseq.PlainTransforms(tuple((self.root, transform.Shift([float(offset)])) for offset in range(self.i, self.j)), 1)
    super().__init__(references, transforms, transforms)

  @property
  def boundary(self):
    if self.periodic:
      return EmptyTopology(ndims=0)
    outsideleft = self.root, transform.Shift([float(self.i-1)])
    outsideright = self.root, transform.Shift([float(self.j)])
    right, left = element.LineReference().edge_transforms
    bnd = Point(self.transforms[0] + (left,), outsideleft + (right,)), Point(self.transforms[-1] + (right,), outsideright + (left,))
    return DisjointUnionTopology(bnd, self.bnames)

  @property
  def interfaces(self):
    right, left = element.LineReference().edge_transforms
    return DisjointUnionTopology(tuple(Point(b+(left,), a+(right,)) for a, b in util.pairwise(self.transforms, periodic=self.periodic)))

  @classmethod
  def _bernstein_poly(cls, degree):
    'bernstein polynomial coefficients'


  @classmethod
  def _spline_coeffs(cls, p, n):
    'spline polynomial coefficients'

    assert p >= 0, 'invalid polynomial degree {}'.format(p)
    if p == 0:
      assert n == -1
      return numpy.array([[[1.]]])

    assert 1 <= n < 2*p
    extractions = numpy.empty((n, p+1, p+1))
    extractions[0] = numpy.eye(p+1)
    for i in range(1, n):
      extractions[i] = numpy.eye(p+1)
      for j in range(2, p+1):
        for k in reversed(range(j, p+1)):
          alpha = 1. / min(2+k-j, n-i+1)
          extractions[i-1,:,k] = alpha * extractions[i-1,:,k] + (1-alpha) * extractions[i-1,:,k-1]
        extractions[i,-j-1:-1,-j-1] = extractions[i-1,-j:,-1]

    # magic bernstein triangle
    poly = numpy.zeros([p+1,p+1], dtype=int)
    for k in range(p//2+1):
      poly[k,k] = root = (-1)**p if k == 0 else (poly[k-1,k] * (k*2-1-p)) / k
      for i in range(k+1,p+1-k):
        poly[i,k] = poly[k,i] = root = (root * (k+i-p-1)) / i
    poly = poly[::-1].astype(float)

    return types.frozenarray(numeric.contract(extractions[:,_,:,:], poly[_,:,_,:], axis=-1).transpose(0,2,1), copy=False)

  def basis_spline(self, degree, periodic=None, removedofs=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numpy.iterable(degree):
      degree, = degree

    if numpy.iterable(removedofs):
      removedofs, = removedofs

    strides = 1, 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      assert ndofs >= 2 * degree
      dofs[-degree:] = dofs[:degree]
      ndofs -= degree
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = types.frozenarray(dofs, copy=False)

    p = degree
    n = 2*p-1
    nelems = len(self)
    if periodic:
      if nelems == 1: # periodicity on one element can only mean a constant
        coeffs = list(self._spline_coeffs(0, n))
        dofs = types.frozenarray([[0]], copy=False)
      else:
        coeffs = list(self._spline_coeffs(p, n)[p-1:p]) * nelems
    else:
      coeffs = list(self._spline_coeffs(p, min(nelems,n)))
      if len(coeffs) < nelems:
        coeffs = coeffs[:p-1] + coeffs[p-1:p] * (nelems-2*(p-1)) + coeffs[p:]
    coeffs = types.frozenarray(coeffs, copy=False)

    func = function.PlainBasis(coeffs, dofs, ndofs, self.transforms)
    if removedofs:
      func = func[numpy.setdiff1d(numpy.arange(len(func)), removedofs)]
    return func

  def basis_std(self, degree, periodic=None, removedofs=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    strides = max(1, degree), 1
    shape = len(self), degree+1
    ndofs = sum(s*(n-1) for s, n in zip(strides, shape))+1
    dofs = numpy.arange(ndofs)
    if periodic and degree > 0:
      dofs[-1] = dofs[0]
      ndofs -= 1
    dofs = numpy.lib.stride_tricks.as_strided(dofs, shape=shape, strides=tuple(s*dofs.strides[0] for s in strides))
    dofs = types.frozenarray(dofs, copy=False)

    coeffs = [element.LineReference().get_poly_coeffs('bernstein', degree=degree)]*len(self)
    func = function.PlainBasis(coeffs, dofs, ndofs, self.transforms)
    if removedofs:
      func = func[numpy.setdiff1d(numpy.arange(len(func)), removedofs)]
    return func

  def __str__(self):
    'string representation'

    return '{}({}:{})'.format(self.__class__.__name__, self.i, self.j)

class StructuredTopology(Topology):
  'structured topology'

  __slots__ = 'root', 'axes', 'nrefine', 'shape', '_bnames'
  __cache__ = 'connectivity', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, axes:types.tuple[types.strict[transformseq.Axis]], nrefine:types.strictint=0, bnames:types.tuple[types.tuple[types.strictstr]]=(('left', 'right'), ('bottom', 'top'), ('front', 'back'))):
    'constructor'

    assert all(len(bname) == 2 for bname in bnames)

    self.root = root
    self.axes = axes
    self.nrefine = nrefine
    self.shape = tuple(axis.j - axis.i for axis in self.axes if axis.isdim)
    self._bnames = bnames

    references = elementseq.asreferences([util.product(element.getsimplex(1 if axis.isdim else 0) for axis in self.axes)], len(self.shape))*len(self)
    transforms = transformseq.StructuredTransforms(self.root, self.axes, self.nrefine)
    nbounds = len(self.axes) - len(self.shape)
    if nbounds == 0:
      opposites = transforms
    else:
      axes = [transformseq.BndAxis(axis.i, axis.j, axis.ibound, not axis.side) if not axis.isdim and axis.ibound==nbounds-1 else axis for axis in self.axes]
      opposites = transformseq.StructuredTransforms(self.root, axes, self.nrefine)

    super().__init__(references, transforms, opposites)

  def __repr__(self):
    return '{}<{}>'.format(type(self).__qualname__, 'x'.join(str(axis.j-axis.i)+('p' if axis.isperiodic else '') for axis in self.axes if isinstance(axis, transformseq.DimAxis)))

  def __len__(self):
    return numpy.prod(self.shape, dtype=int)

  def getitem(self, item):
    if not isinstance(item, tuple):
      return EmptyTopology(self.ndims)
    assert all(isinstance(it,slice) for it in item) and len(item) <= self.ndims
    if all(it == slice(None) for it in item): # shortcut
      return self
    axes = []
    idim = 0
    for axis in self.axes:
      if axis.isdim and idim < len(item):
        s = item[idim]
        if s != slice(None):
          start, stop, stride = s.indices(axis.j - axis.i)
          assert stride == 1
          assert stop > start
          axis = transformseq.DimAxis(axis.i+start, axis.i+stop, isperiodic=False)
        idim += 1
      axes.append(axis)
    return StructuredTopology(self.root, axes, self.nrefine, bnames=self._bnames)

  @property
  def periodic(self):
    dimaxes = (axis for axis in self.axes if axis.isdim)
    return tuple(idim for idim, axis in enumerate(dimaxes) if axis.isdim and axis.isperiodic)

  @property
  def connectivity(self):
    connectivity = numpy.empty(self.shape+(self.ndims,2), dtype=int)
    connectivity[...] = -1
    ielems = numpy.arange(len(self)).reshape(self.shape)
    for idim in range(self.ndims):
      s = (slice(None),)*idim
      s1 = s + (slice(1,None),)
      s2 = s + (slice(0,-1),)
      connectivity[s2+(...,idim,0)] = ielems[s1]
      connectivity[s1+(...,idim,1)] = ielems[s2]
      if idim in self.periodic:
        connectivity[s+(-1,...,idim,0)] = ielems[s+(0,)]
        connectivity[s+(0,...,idim,1)] = ielems[s+(-1,)]
    return types.frozenarray(connectivity.reshape(len(self), self.ndims*2), copy=False)

  @property
  def boundary(self):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    btopos = [StructuredTopology(root=self.root, axes=self.axes[:idim] + (transformseq.BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:], nrefine=self.nrefine, bnames=self._bnames)
      for idim, axis in enumerate(self.axes) if axis.isdim and not axis.isperiodic
        for side, n in enumerate((axis.i,axis.j))]
    if not btopos:
      return EmptyTopology(self.ndims-1)
    bnames = [bname for bnames, axis in zip(self._bnames, self.axes) if axis.isdim and not axis.isperiodic for bname in bnames]
    return DisjointUnionTopology(btopos, bnames)

  @property
  def interfaces(self):
    'interfaces'

    assert self.ndims > 0, 'zero-D topology has no interfaces'
    itopos = []
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      intaxis = lambda side: (transformseq.PIntAxis if idim in self.periodic else transformseq.IntAxis)(axis.i, axis.j, nbounds, side)
      axes = (*self.axes[:idim], intaxis(True), *self.axes[idim+1:])
      oppaxes = (*self.axes[:idim], intaxis(False), *self.axes[idim+1:])
      itransforms = transformseq.StructuredTransforms(self.root, axes, self.nrefine)
      iopposites = transformseq.StructuredTransforms(self.root, oppaxes, self.nrefine)
      ireferences = elementseq.asreferences([util.product(element.getsimplex(1 if a.isdim else 0) for a in axes)], self.ndims-1)*len(itransforms)
      itopos.append(UnstructuredTopology(ireferences, itransforms, iopposites))
    assert len(itopos) == self.ndims
    return DisjointUnionTopology(itopos, names=['dir{}'.format(idim) for idim in range(self.ndims)])

  def _basis_spline(self, degree, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
    'spline with structure information'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = [degree]*self.ndims

    assert len(degree) == self.ndims

    if knotvalues is None or isinstance(knotvalues[0], (int,float)):
      knotvalues = [knotvalues] * self.ndims
    else:
      assert len(knotvalues) == self.ndims

    if knotmultiplicities is None or isinstance(knotmultiplicities[0], int):
      knotmultiplicities = [knotmultiplicities] * self.ndims
    else:
      assert len(knotmultiplicities) == self.ndims

    if not numpy.iterable(continuity):
      continuity = [continuity] * self.ndims
    else:
      assert len(continuity) == self.ndims

    vertex_structure = numpy.array(0)
    stdelems = []
    dofshape = []
    slices = []
    cache = {}
    for idim in range(self.ndims):
      p = degree[idim]
      n = self.shape[idim]
      isperiodic = idim in periodic

      c = continuity[idim]
      if c < 0:
        c += p
      assert -1 <= c < p

      k = knotvalues[idim]
      if k is None: #Defaults to uniform spacing
        k = numpy.arange(n+1)
      else:
        k = numpy.array(k)
        while len(k) < n+1:
          k_ = numpy.empty(len(k)*2-1)
          k_[::2] = k
          k_[1::2] = (k[:-1] + k[1:]) / 2
          k = k_
        assert len(k) == n+1, 'knot values do not match the topology size'

      m = knotmultiplicities[idim]
      if m is None: #Defaults to open spline without internal repetitions
        m = numpy.repeat(p-c, n+1)
        if not isperiodic:
          m[0] = m[-1] = p+1
      else:
        m = numpy.array(m)
        assert min(m) >0 and max(m) <= p+1, 'incorrect multiplicity encountered'
        while len(m) < n+1:
          m_ = numpy.empty(len(m)*2-1, dtype=int)
          m_[::2] = m
          m_[1::2] = p-c
          m = m_
        assert len(m) == n+1, 'knot multiplicity do not match the topology size'

      if not isperiodic:
        nd = sum(m)-p-1
        npre  = p+1-m[0]  #Number of knots to be appended to front
        npost = p+1-m[-1] #Number of knots to be appended to rear
        m[0] = m[-1] = p+1
      else:
        assert m[0]==m[-1], 'Periodic spline multiplicity expected'
        assert m[0]<p+1, 'Endpoint multiplicity for periodic spline should be p or smaller'

        nd = sum(m[:-1])
        npre = npost = 0
        k = numpy.concatenate([k[-p-1:-1]+k[0]-k[-1], k, k[1:1+p]-k[0]+k[-1]])
        m = numpy.concatenate([m[-p-1:-1], m, m[1:1+p]])

      km = numpy.array([ki for ki,mi in zip(k,m) for cnt in range(mi)],dtype=float)
      assert len(km)==sum(m)
      assert nd>0, 'No basis functions defined. Knot vector too short.'

      stdelems_i = []
      slices_i = []
      offsets = numpy.cumsum(m[:-1])-p
      if isperiodic:
        offsets = offsets[p:-p]
      offset0 = offsets[0]+npre

      for offset in offsets:
        start = max(offset0-offset,0) #Zero unless prepending influence
        stop  = p+1-max(offset-offsets[-1]+npost,0) #Zero unless appending influence
        slices_i.append(slice(offset-offset0+start,offset-offset0+stop))
        lknots  = km[offset:offset+2*p] - km[offset] #Copy operation required
        if p: #Normalize for optimized caching
          lknots /= lknots[-1]
        key = (tuple(numeric.round(lknots*numpy.iinfo(numpy.int32).max)), p)
        try:
          coeffs = cache[key]
        except KeyError:
          coeffs = cache[key] = self._localsplinebasis(lknots)
        stdelems_i.append(coeffs[start:stop])
      stdelems.append(stdelems_i)

      numbers = numpy.arange(nd)
      if isperiodic:
        numbers = numpy.concatenate([numbers,numbers[:p]])
      vertex_structure = vertex_structure[...,_]*nd+numbers
      dofshape.append(nd)
      slices.append(slices_i)

    #Cache effectivity
    log.debug('Local knot vector cache effectivity: {}'.format(100*(1.-len(cache)/float(sum(self.shape)))))

    # deduplicate stdelems and compute tensorial products `unique` with indices `index`
    # such that unique[index[i,j]] == poly_outer_product(stdelems[0][i], stdelems[1][j])
    index = numpy.array(0)
    for stdelems_i in stdelems:
      unique_i = tuple(set(stdelems_i))
      unique = unique_i if not index.ndim \
        else [numeric.poly_outer_product(a, b) for a in unique for b in unique_i]
      index = index[...,_] * len(unique_i) + tuple(map(unique_i.index, stdelems_i))

    coeffs = [unique[i] for i in index.flat]
    dofmap = [types.frozenarray(vertex_structure[S].ravel(), copy=False) for S in itertools.product(*slices)]
    return coeffs, dofmap, dofshape

  def basis_spline(self, degree, removedofs=None, knotvalues=None, knotmultiplicities=None, continuity=-1, periodic=None):
    'spline basis'

    if removedofs is None or isinstance(removedofs[0], int):
      removedofs = [removedofs] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = [degree]*self.ndims

    assert len(degree) == self.ndims

    if knotvalues is None or isinstance(knotvalues[0], (int,float)):
      knotvalues = [knotvalues] * self.ndims
    else:
      assert len(knotvalues) == self.ndims

    if knotmultiplicities is None or isinstance(knotmultiplicities[0], int):
      knotmultiplicities = [knotmultiplicities] * self.ndims
    else:
      assert len(knotmultiplicities) == self.ndims

    if not numpy.iterable(continuity):
      continuity = [continuity] * self.ndims
    else:
      assert len(continuity) == self.ndims

    start_dofs = []
    stop_dofs = []
    dofshape = []
    coeffs = []
    cache = {}
    for idim in range(self.ndims):
      p = degree[idim]
      n = self.shape[idim]
      isperiodic = idim in periodic

      c = continuity[idim]
      if c < 0:
        c += p
      assert -1 <= c < p

      k = knotvalues[idim]
      if k is None:
        k = numpy.arange(n+1) # default to uniform spacing
      else:
        k = numpy.array(k)
        while len(k) < n+1:
          k_ = numpy.empty(len(k)*2-1)
          k_[::2] = k
          k_[1::2] = (k[:-1] + k[1:]) / 2
          k = k_
        assert len(k) == n+1, 'knot values do not match the topology size'

      m = knotmultiplicities[idim]
      if m is None:
        m = numpy.repeat(p-c, n+1) # default to open spline without internal repetitions
      else:
        m = numpy.array(m)
        assert min(m) > 0 and max(m) <= p+1, 'incorrect multiplicity encountered'
        while len(m) < n+1:
          m_ = numpy.empty(len(m)*2-1, dtype=int)
          m_[::2] = m
          m_[1::2] = p-c
          m = m_
        assert len(m) == n+1, 'knot multiplicity do not match the topology size'

      if isperiodic:
        assert m[0] == m[n], 'periodic spline multiplicity expected'
        nd = m[:n].sum()
        while m[n:].sum() < 2*p - m[n]:
          k = numpy.concatenate([k, k[1:]+k[-1]-k[0]])
          m = numpy.concatenate([m, m[1:]])
      else:
        m[0] = m[-1] = p
        nd = m[:n].sum()+1

      km = numpy.array([ki for ki, mi in zip(k, m) for cnt in range(mi)], dtype=float)

      offsets = numpy.cumsum(m[:n]) - m[0]
      start_dofs.append(offsets)
      stop_dofs.append(offsets+p+1)
      dofshape.append(nd)

      coeffs_i = []
      for offset in offsets:
        lknots = km[offset:offset+2*p]
        key = tuple(numeric.round((lknots[1:-1]-lknots[0])/(lknots[-1]-lknots[0])*numpy.iinfo(numpy.int32).max)) if lknots.size else (), p
        try:
          local_coeffs = cache[key]
        except KeyError:
          local_coeffs = cache[key] = self._localsplinebasis(lknots)
        coeffs_i.append(local_coeffs)
      coeffs.append(tuple(coeffs_i))

    transforms_shape = tuple(axis.j-axis.i for axis in self.axes if axis.isdim)
    func = function.StructuredBasis(coeffs, start_dofs, stop_dofs, dofshape, self.transforms, transforms_shape)
    if not any(removedofs):
      return func

    mask = numpy.ones((), dtype=bool)
    for idofs, ndofs in zip(removedofs, dofshape):
      mask = mask[...,_].repeat(ndofs, axis=-1)
      if idofs:
        mask[..., [numeric.normdim(ndofs,idof) for idof in idofs]] = False
    assert mask.shape == tuple(dofshape)
    return func[mask.ravel()]

  @staticmethod
  def _localsplinebasis(lknots):

    assert numeric.isarray(lknots), 'Local knot vector should be numpy array'
    p, rem = divmod(len(lknots), 2)
    assert rem == 0

    #Based on Algorithm A2.2 Piegl and Tiller
    N    = [None]*(p+1)
    N[0] = numpy.poly1d([1.])

    if p > 0:

      assert numpy.less(lknots[:-1]-lknots[1:], numpy.spacing(1)).all(), 'Local knot vector should be non-decreasing'
      assert lknots[p]-lknots[p-1]>numpy.spacing(1), 'Element size should be positive'

      lknots = lknots.astype(float)

      xi = numpy.poly1d([lknots[p]-lknots[p-1],lknots[p-1]])

      left  = [None]*p
      right = [None]*p

      for i in range(p):
        left[i] = xi - lknots[p-i-1]
        right[i] = -xi + lknots[p+i]
        saved = 0.
        for r in range(i+1):
          temp = N[r]/(lknots[p+r]-lknots[p+r-i-1])
          N[r] = saved+right[r]*temp
          saved = left[i-r]*temp
        N[i+1] = saved

    assert all(Ni.order==p for Ni in N)

    return types.frozenarray([Ni.coeffs[::-1] for Ni in N])

  def basis_std(self, *args, **kwargs):
    return __class__.basis_spline(self, *args, continuity=0, **kwargs)

  @property
  def refined(self):
    'refine non-uniformly'

    axes = [transformseq.DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else transformseq.BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes]
    return StructuredTopology(self.root, axes, self.nrefine+1, bnames=self._bnames)

  def locate(self, geom, coords, *, eps=0, **kwargs):
    coords = numpy.asarray(coords, dtype=float)
    if geom.ndim == 0:
      geom = geom[_]
      coords = coords[...,_]
    if not geom.shape == coords.shape[1:] == (self.ndims,):
      raise Exception('invalid geometry or point shape for {}D topology'.format(self.ndims))
    index = function.rootcoords(len(self.axes))[[axis.isdim for axis in self.axes]]
    basis = function.concatenate([function.eye(self.ndims), function.diagonalize(index)], axis=0)
    A, b, f2 = self.integrate([(basis[:,_,:] * basis[_,:,:]).sum(-1), (basis * geom).sum(-1), (geom**2).sum(-1)], degree=3)
    x = A.solve(b)
    e = (f2 - b.dot(x)) / numpy.product(self.shape)
    if e > 1e-10:
      return super().locate(geom, coords, eps=eps, **kwargs)
    geom0 = x[:self.ndims]
    scale = x[self.ndims:]
    log.info('locate detected linear geometry: x = {} + {} xi ~{:+.1e}'.format(geom0, scale, e))
    mincoords, maxcoords = numpy.sort([geom0, geom0 + scale * self.shape], axis=0)
    outofbounds = numpy.less(coords, mincoords - eps) | numpy.greater(coords, maxcoords + eps)
    if outofbounds.any():
      raise LocateError('failed to locate {}/{} points'.format(outofbounds.sum(), len(coords)))
    xi = (coords - geom0) / scale
    ielem = numpy.minimum(numpy.maximum(xi.astype(int), 0), numpy.array(self.shape)-1)
    return self._sample(numpy.ravel_multi_index(ielem.T, self.shape), xi - ielem)

  def __str__(self):
    'string representation'

    return '{}({})'.format(self.__class__.__name__, 'x'.join(str(n) for n in self.shape))

class UnstructuredTopology(Topology):
  'unstructured topology'

  @types.aspreprocessor
  @types.apply_annotations
  def _preprocess_init(self, references, transforms, opposites=None, *, ndims:types.strictint=None):
    if not isinstance(references, elementseq.References):
      references = elementseq.asreferences(references, ndims)
    if not isinstance(transforms, transformseq.Transforms):
      transforms = transformseq.PlainTransforms(transforms, ndims)
    if opposites is None:
      opposites = transforms
    elif not isinstance(opposites, transformseq.Transforms):
      opposites = transformseq.PlainTransforms(opposites, ndims)
    if ndims is not None:
      assert references.ndims == ndims
      assert transforms.fromdims == ndims
      assert opposites.fromdims == ndims
    return (self, references, transforms, opposites), {}

  @_preprocess_init
  def __init__(self, references, transforms, opposites):
    super().__init__(references, transforms, opposites)

class ConnectedTopology(UnstructuredTopology):
  'unstructured topology with connectivity'

  __slots__ = 'connectivity',

  @types.apply_annotations
  def __init__(self, references:elementseq.strictreferences, transforms:transformseq.stricttransforms, opposites:transformseq.stricttransforms, connectivity):
    assert len(connectivity) == len(references) and all(len(c) == e.nedges for c, e in zip(connectivity, references))
    self.connectivity = connectivity
    super().__init__(references, transforms, opposites)

class SimplexTopology(Topology):
  'simpex topology'

  __slots__ = 'simplices', 'references', 'transforms', 'opposites'
  __cache__ = 'connectivity'

  def _renumber(simplices):
    simplices = numpy.asarray(simplices)
    keep = numpy.zeros(simplices.max()+1, dtype=bool)
    keep[simplices.flat] = True
    return types.frozenarray(simplices if keep.all() else (numpy.cumsum(keep)-1)[simplices], copy=False)

  @types.apply_annotations
  def __init__(self, simplices:_renumber, transforms:transformseq.stricttransforms, opposites:transformseq.stricttransforms):
    assert simplices.shape == (len(transforms), transforms.fromdims+1)
    assert numpy.greater(simplices[:,1:], simplices[:,:-1]).all(), 'nodes should be sorted'
    assert not numpy.equal(simplices[:,1:], simplices[:,:-1]).all(), 'duplicate nodes'
    self.simplices = simplices
    references = elementseq.asreferences([element.getsimplex(transforms.fromdims)], transforms.fromdims)*len(transforms)
    super().__init__(references, transforms, opposites)

  @property
  def connectivity(self):
    nverts = self.ndims + 1
    edge_vertices = numpy.arange(nverts).repeat(self.ndims).reshape(self.ndims, nverts)[:,::-1].T # nverts x ndims
    simplices_edges = self.simplices.take(edge_vertices, axis=1) # nelems x nverts x ndims
    elems, edges = divmod(numpy.lexsort(simplices_edges.reshape(-1, self.ndims).T), nverts)
    sorted_simplices_edges = simplices_edges[elems, edges] # (nelems x nverts) x ndims; matching edges are now adjacent
    i, = numpy.equal(sorted_simplices_edges[1:], sorted_simplices_edges[:-1]).all(axis=1).nonzero()
    j = i + 1
    assert numpy.greater(i[1:], j[:-1]).all(), 'single edge is shared by three or more simplices'
    connectivity = numpy.full((len(self.simplices), self.ndims+1), fill_value=-1, dtype=int)
    connectivity[elems[i],edges[i]] = elems[j]
    connectivity[elems[j],edges[j]] = elems[i]
    return types.frozenarray(connectivity, copy=False)

  def basis_std(self, degree):
    if degree == 1:
      coeffs = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
      return function.PlainBasis([coeffs] * len(self), self.simplices, self.simplices.max()+1, self.transforms)
    return super().basis_std(degree)

  def basis_bubble(self):
    'bubble from vertices'

    bernstein = element.getsimplex(self.ndims).get_poly_coeffs('bernstein', degree=1)
    bubble = functools.reduce(numeric.poly_mul, bernstein)
    coeffs = numpy.zeros((len(bernstein)+1,) + bubble.shape)
    coeffs[(slice(-1),)+(slice(2),)*self.ndims] = bernstein
    coeffs[-1] = bubble
    coeffs[:-1] -= bubble / (self.ndims+1)
    coeffs = types.frozenarray(coeffs, copy=False)
    nverts = self.simplices.max() + 1
    ndofs = nverts + len(self)
    nmap = [types.frozenarray(numpy.hstack([idofs, nverts+ielem]), copy=False) for ielem, idofs in enumerate(self.simplices)]
    return function.PlainBasis([coeffs] * len(self), nmap, ndofs, self.transforms)

class UnionTopology(Topology):
  'grouped topology'

  __slots__ = '_topos', '_names', 'references', 'transforms', 'opposites'

  @types.apply_annotations
  def __init__(self, topos:types.tuple[stricttopology], names:types.tuple[types.strictstr]=()):
    self._topos = topos
    self._names = tuple(names)[:len(self._topos)]
    assert len(set(self._names)) == len(self._names), 'duplicate name'
    ndims = self._topos[0].ndims
    assert all(topo.ndims == ndims for topo in self._topos)

    references = []
    transforms = []
    opposites =[]
    for trans, indices in util.gather((trans, (itopo, itrans)) for itopo, topo in enumerate(self._topos) for itrans, trans in enumerate(topo.transforms)):
      transforms.append(trans)
      if len(indices) == 1:
        (itopo, itrans), = indices
        references.append(self._topos[itopo].references[itrans])
        opposites.append(self._topos[itopo].opposites[itrans])
      else:
        refs = [self._topos[itopo].references[itrans] for itopo, itrans in indices]
        while len(refs) > 1: # sweep all possible unions until a single reference is left
          nrefs = len(refs)
          iref = 0
          while iref < len(refs)-1:
            for jref in range(iref+1, len(refs)):
              try:
                unionref = refs[iref] | refs[jref]
              except TypeError:
                pass
              else:
                refs[iref] = unionref
                del refs[jref]
                break
            iref += 1
          assert len(refs) < nrefs, 'incompatible elements in union'
        references.append(refs[0])
        opposite, = set(self._topos[itopo].opposites[itrans] for itopo, itrans in indices)
        opposites.append(opposite)

    super().__init__(elementseq.PlainReferences(references, ndims), transformseq.PlainTransforms(transforms, ndims), transformseq.PlainTransforms(opposites, ndims))

  def getitem(self, item):
    topos = [topo if name == item else topo.getitem(item) for topo, name in itertools.zip_longest(self._topos, self._names)]
    return functools.reduce(operator.or_, topos, EmptyTopology(self.ndims))

  def __or__(self, other):
    if not isinstance(other, UnionTopology):
      return UnionTopology(self._topos + (other,), self._names)
    return UnionTopology(self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names)

  @property
  def refined(self):
    return UnionTopology([topo.refined for topo in self._topos], self._names)

class DisjointUnionTopology(Topology):
  'grouped topology'

  __slots__ = '_topos', '_names'

  @types.apply_annotations
  def __init__(self, topos:types.tuple[stricttopology], names:types.tuple[types.strictstr]=()):
    self._topos = topos
    self._names = tuple(names)[:len(self._topos)]
    assert len(set(self._names)) == len(self._names), 'duplicate name'
    ndims = self._topos[0].ndims
    assert all(topo.ndims == ndims for topo in self._topos)
    super().__init__(
      elementseq.chain((topo.references for topo in self._topos), ndims),
      transformseq.chain((topo.transforms for topo in self._topos), ndims),
      transformseq.chain((topo.opposites for topo in self._topos), ndims))

  def getitem(self, item):
    topos = [topo if name == item else topo.getitem(item) for topo, name in itertools.zip_longest(self._topos, self._names)]
    topos = [topo for topo in topos if not isinstance(topo, EmptyTopology)]
    if len(topos) == 0:
      return EmptyTopology(self.ndims)
    elif len(topos) == 1:
      return topos[0]
    else:
      return DisjointUnionTopology(topos)

  @property
  def refined(self):
    return DisjointUnionTopology([topo.refined for topo in self._topos], self._names)

class SubsetTopology(Topology):
  'trimmed'

  __slots__ = 'refs', 'basetopo', 'newboundary', '_indices'
  __cache__ = 'connectivity', 'boundary', 'interfaces', 'refined'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, refs:types.tuple[element.strictreference], newboundary=None):
    if newboundary is not None:
      assert isinstance(newboundary, str) or isinstance(newboundary, Topology) and newboundary.ndims == basetopo.ndims-1
    assert len(refs) == len(basetopo)
    self.refs = refs
    self.basetopo = basetopo
    self.newboundary = newboundary

    self._indices = types.frozenarray(numpy.array([i for i, ref in enumerate(self.refs) if ref], dtype=int), copy=False)
    references = elementseq.asreferences(self.refs, self.basetopo.ndims)[self._indices]
    transforms = self.basetopo.transforms[self._indices]
    opposites = self.basetopo.opposites[self._indices]
    super().__init__(references, transforms, opposites)

  def getitem(self, item):
    return self.basetopo.getitem(item).subset(self, strict=False)

  def __rsub__(self, other):
    if self.basetopo == other:
      refs = [baseref - ref for baseref, ref in zip(self.basetopo.references, self.refs)]
      return SubsetTopology(self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary,Topology) else self.newboundary)
    return super().__rsub__(other)

  def __or__(self, other):
    if not isinstance(other, SubsetTopology) or self.basetopo != other.basetopo:
      return super().__or__(other)
    refs = [ref1 | ref2 for ref1, ref2 in zip(self.refs, other.refs)]
    if all(baseref == ref for baseref, ref in zip(self.basetopo.references, refs)):
      return self.basetopo
    return SubsetTopology(self.basetopo, refs) # TODO boundary

  @property
  def connectivity(self):
    mask = numpy.array([bool(ref) for ref in self.refs] + [False]) # trailing false serves to map -1 to -1
    renumber = numpy.cumsum(mask)-1
    renumber[~mask] = -1
    return tuple(types.frozenarray(renumber.take(ioppelems).tolist() + [-1] * (ref.nedges - len(ioppelems))) for ref, ioppelems in zip(self.refs, self.basetopo.connectivity) if ref)

  @property
  def refined(self):
    child_refs = self.references.children
    indices = types.frozenarray(numpy.array([i for i, ref in enumerate(child_refs) if ref], dtype=int), copy=False)
    refined_transforms = self.transforms.refined(self.references)[indices]
    self_refined = UnstructuredTopology(child_refs[indices], refined_transforms)
    return self.basetopo.refined.subset(self_refined, self.newboundary.refined if isinstance(self.newboundary,Topology) else self.newboundary, strict=True)

  @property
  def boundary(self):
    baseboundary = self.basetopo.boundary
    baseconnectivity = self.basetopo.connectivity
    brefs = [ref.empty for ref in baseboundary.references]
    trimmedreferences = []
    trimmedtransforms = []
    trimmedopposites = []
    for ielem, newref in enumerate(self.refs):
      if not newref:
        continue
      elemtrans = self.basetopo.transforms[ielem]
      # The first edges of newref by convention share location with the edges
      # of the original reference. We can therefore use baseconnectivity to
      # locate opposing edges.
      ioppelems = baseconnectivity[ielem]
      for (edgetrans, edgeref), ioppelem in zip(newref.edges, ioppelems):
        if not edgeref:
          continue
        if ioppelem == -1:
          # If the edge had no opposite in basetopology then it must already by
          # in baseboundary, so we can use index to locate it.
          brefs[baseboundary.transforms.index(elemtrans+(edgetrans,))] = edgeref
        else:
          # If the edge did have an opposite in basetopology then there is a
          # possibility this opposite (partially) disappeared, in which case
          # the exposed part is added to the trimmed group.
          ioppedge = baseconnectivity[ioppelem].index(ielem)
          oppref = self.refs[ioppelem]
          edgeref -= oppref.edge_refs[ioppedge]
          if edgeref:
            trimmedreferences.append(edgeref)
            trimmedtransforms.append(elemtrans+(edgetrans,))
            trimmedopposites.append(self.basetopo.transforms[ioppelem]+(oppref.edge_transforms[ioppedge],))
      # The last edges of newref (beyond the number of edges of the original)
      # cannot have opposites and are added to the trimmed group directly.
      for edgetrans, edgeref in newref.edges[len(ioppelems):]:
        trimmedreferences.append(edgeref)
        trimmedtransforms.append(elemtrans+(edgetrans,))
        trimmedopposites.append(elemtrans+(edgetrans.flipped,))
    origboundary = SubsetTopology(baseboundary, brefs)
    if isinstance(self.newboundary, Topology):
      trimmedbrefs = [ref.empty for ref in self.newboundary.references]
      for ref, trans in zip(trimmedreferences, trimmedtransforms):
        trimmedbrefs[self.newboundary.transforms.index(trans)] = ref
      trimboundary = SubsetTopology(self.newboundary, trimmedbrefs)
    else:
      trimboundary = UnstructuredTopology(trimmedreferences, trimmedtransforms, trimmedopposites, ndims=self.ndims-1)
    return DisjointUnionTopology([trimboundary, origboundary], names=[self.newboundary] if isinstance(self.newboundary,str) else [])

  @property
  def interfaces(self):
    baseinterfaces = self.basetopo.interfaces
    superinterfaces = super().interfaces
    irefs = [ref.empty for ref in baseinterfaces.references]
    for ref, trans, opp in zip(superinterfaces.references, superinterfaces.transforms, superinterfaces.opposites):
      try:
        iielem = baseinterfaces.transforms.index(trans)
      except ValueError:
        iielem = baseinterfaces.transforms.index(opp)
      irefs[iielem] = ref
    return SubsetTopology(baseinterfaces, irefs)

  @log.withcontext
  def basis(self, name, *args, **kwargs):
    if isinstance(self.basetopo, HierarchicalTopology):
      warnings.warn('basis may be linearly dependent; a linearly indepent basis is obtained by trimming first, then creating hierarchical refinements')
    basis = self.basetopo.basis(name, *args, **kwargs)
    return function.PrunedBasis(basis, self._indices)

class RefinedTopology(Topology):
  'refinement'

  __slots__ = 'basetopo',
  __cache__ = 'boundary', 'connectivity'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology):
    self.basetopo = basetopo
    super().__init__(
      self.basetopo.references.children,
      self.basetopo.transforms.refined(self.basetopo.references),
      self.basetopo.opposites.refined(self.basetopo.references))

  def getitem(self, item):
    return self.basetopo.getitem(item).refined

  @property
  def boundary(self):
    return self.basetopo.boundary.refined

  @property
  def connectivity(self):
    offsets = numpy.cumsum([0] + [ref.nchildren for ref in self.basetopo.references])
    connectivity = [offset + edges for offset, ref in zip(offsets, self.basetopo.references) for edges in ref.connectivity]
    for ielem, edges in enumerate(self.basetopo.connectivity):
      for iedge, jelem in enumerate(edges):
        if jelem == -1:
          for ichild, ichildedge in self.basetopo.references[ielem].edgechildren[iedge]:
            connectivity[offsets[ielem]+ichild][ichildedge] = -1
        elif jelem < ielem:
          jedge = self.basetopo.connectivity[jelem].index(ielem)
          for (ichild, ichildedge), (jchild, jchildedge) in zip(self.basetopo.references[ielem].edgechildren[iedge], self.basetopo.references[jelem].edgechildren[jedge]):
            connectivity[offsets[ielem]+ichild][ichildedge] = offsets[jelem]+jchild
            connectivity[offsets[jelem]+jchild][jchildedge] = offsets[ielem]+ichild
    return tuple(types.frozenarray(c, copy=False) for c in connectivity)

class HierarchicalTopology(Topology):
  'collection of nested topology elments'

  __slots__ = 'basetopo', 'levels', '_indices_per_level', '_offsets'
  __cache__ = 'refined', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, indices_per_level:types.tuple[types.frozenarray[types.strictint]]):
    'constructor'

    assert not isinstance(basetopo, HierarchicalTopology)
    self.basetopo = basetopo
    self._indices_per_level = indices_per_level
    self._offsets = numpy.cumsum([0, *map(len, self._indices_per_level)], dtype=int)

    level = None
    levels = []
    references = []
    transforms = []
    opposites = []
    for indices in indices_per_level:
      level = self.basetopo if level is None else level.refined
      levels.append(level)
      if len(indices):
        references.append(level.references[indices])
        transforms.append(level.transforms[indices])
        opposites.append(level.opposites[indices])
    self.levels = tuple(levels)

    super().__init__(elementseq.chain(references, basetopo.ndims), transformseq.chain(transforms, basetopo.ndims), transformseq.chain(opposites, basetopo.ndims))

  def getitem(self, item):
    itemtopo = self.basetopo.getitem(item)
    itemindices_per_level = []
    for baseindices, baselevel, itemlevel in zip(self._indices_per_level, self.basetopo.refine_iter, itemtopo.refine_iter):
      itemindices = []
      itemindex = itemlevel.transforms.index
      for basetrans in map(baselevel.transforms.__getitem__, baseindices):
        try:
          itemindices.append(itemindex(basetrans))
        except ValueError:
          pass
      itemindices_per_level.append(numpy.unique(numpy.array(itemindices, dtype=int)))
    return HierarchicalTopology(itemtopo, itemindices_per_level)

  def refined_by(self, refine):
    refine = tuple(refine)
    if not all(map(numeric.isint, refine)):
      refine = tuple(self.transforms.index_with_tail(item)[0] for item in refine)
    refine = numpy.unique(numpy.array(refine, dtype=int))
    splits = numpy.searchsorted(refine, self._offsets, side='left')
    indices_per_level = list(map(list, self._indices_per_level))+[[]]
    fine = self.basetopo
    for ilevel, (start, stop) in enumerate(zip(splits[:-1], splits[1:])):
      coarse, fine = fine, fine.refined
      coarse_indices = tuple(map(indices_per_level[ilevel].pop, reversed(refine[start:stop]-self._offsets[ilevel])))
      coarse_transforms = map(coarse.transforms.__getitem__, coarse_indices)
      coarse_references = map(coarse.references.__getitem__, coarse_indices)
      fine_transforms = (trans+(ctrans,) for trans, ref in zip(coarse_transforms, coarse_references) for ctrans, cref in ref.children if cref)
      indices_per_level[ilevel+1].extend(map(fine.transforms.index, fine_transforms))
    if not indices_per_level[-1]:
      indices_per_level.pop(-1)
    return HierarchicalTopology(self.basetopo, ([numpy.unique(numpy.array(i, dtype=int)) for i in indices_per_level]))

  @property
  def refined(self):
    refined_indices_per_level = [[]]
    fine = self.basetopo
    for coarse_indices in self._indices_per_level:
      coarse, fine = fine, fine.refined
      coarse_transforms = map(coarse.transforms.__getitem__, coarse_indices)
      coarse_references = map(coarse.references.__getitem__, coarse_indices)
      fine_transforms = (trans+(ctrans,) for trans, ref in zip(coarse_transforms, coarse_references) for ctrans, cref in ref.children if cref)
      refined_indices_per_level.append(numpy.unique(numpy.fromiter(map(fine.transforms.index, fine_transforms), dtype=int)))
    return HierarchicalTopology(self.basetopo, refined_indices_per_level)

  @property
  @log.withcontext
  def boundary(self):
    'boundary elements'

    basebtopo = self.basetopo.boundary
    bindices_per_level = []
    for indices, level, blevel in zip(self._indices_per_level, self.basetopo.refine_iter, basebtopo.refine_iter):
      bindex = blevel.transforms.index
      bindices = []
      for index in indices:
        for etrans, eref in level.references[index].edges:
          if eref:
            trans = level.transforms[index]+(etrans,)
            try:
              bindices.append(bindex(trans))
            except ValueError:
              pass
      bindices = numpy.array(bindices, dtype=int)
      if len(bindices) > 1:
        bindices.sort()
        assert not numpy.equal(bindices[1:], bindices[:-1]).any()
      bindices_per_level.append(bindices)
    return HierarchicalTopology(basebtopo, bindices_per_level)

  @property
  @log.withcontext
  def interfaces(self):
    'interfaces'

    hreferences = []
    htransforms = []
    hopposites = []
    for level, indices in zip(self.levels, self._indices_per_level):
      selection = []
      to = level.interfaces.transforms, level.interfaces.opposites
      for trans, ref in zip(map(level.transforms.__getitem__, indices), map(level.references.__getitem__, indices)):
        for etrans, eref in ref.edges:
          if not eref:
            continue
          for transforms, opposites in to, to[::-1]:
            try:
              i = transforms.index(trans+(etrans,))
            except ValueError:
              continue
            if self.transforms.contains_with_tail(opposites[i]):
              selection.append(i)
            break
      if selection:
        selection = types.frozenarray(numpy.unique(selection))
        hreferences.append(level.interfaces.references[selection])
        htransforms.append(level.interfaces.transforms[selection])
        hopposites.append(level.interfaces.opposites[selection])
    return UnstructuredTopology(elementseq.chain(hreferences, self.ndims-1), transformseq.chain(htransforms, self.ndims-1), transformseq.chain(hopposites, self.ndims-1))

  @log.withcontext
  def basis(self, name, *args, truncation_tolerance=1e-15, **kwargs):
    '''Create hierarchical basis.

    A hierarchical basis is constructed from bases on different levels of
    uniform refinement. Two different types of hierarchical bases are
    supported:

    1. Classical -- Starting from the set of all basis functions originating
    from all levels of uniform refinement, only those basis functions are
    selected for which at least one supporting element is part of the
    hierarchical topology.

    2. Truncated -- Like classical, but with basis functions modified such that
    the area of support is reduced. An additional effect of this procedure is
    that it restores partition of unity. The spanned function space remains
    unchanged.

    Truncation is based on linear combinations of basis functions, where fine
    level basis functions are used to reduce the support of coarser level basis
    functions. See `Giannelli et al. 2012`_ for more information on truncated
    hierarchical refinement.

    .. _`Giannelli et al. 2012`: https://pdfs.semanticscholar.org/a858/aa68da617ad9d41de021f6807cc422002258.pdf

    Args
    ----
    name : :class:`str`
      Type of basis function as provided by the base topology, with prefix
      ``h-`` (``h-std``, ``h-spline``) for a classical hierarchical basis and
      prefix ``th-`` (``th-std``, ``th-spline``) for a truncated hierarchical
      basis.
    truncation_tolerance : :class:`float` (default 1e-15)
      In order to benefit from the extra sparsity resulting from truncation,
      vanishing polynomials need to be actively identified and removed from the
      basis. The ``trunctation_tolerance`` offers control over this threshold.

    Returns
    -------
    basis : :class:`nutils.function.Array`
    '''

    if name.startswith('h-'):
      truncated = False
      name = name[2:]
    elif name.startswith('th-'):
      truncated = True
      name = name[3:]
    else:
      return super().basis(name, *args, **kwargs)

    # 1. identify active (supported) and passive (unsupported) basis functions
    ubases = []
    ubasis_active = []
    ubasis_passive = []
    prev_transforms = None
    prev_ielems = []
    map_indices = []
    with log.context('level'):
      for i, (topo, touchielems_i) in reversed(tuple(enumerate(zip(self.levels, self._indices_per_level)))):
        log.recontext('level {} ({:.0f}%)'.format(i, 100 * (i+.5) / len(self.levels)))

        topo_index_with_tail = topo.transforms.index_with_tail
        mapped_prev_ielems = [topo_index_with_tail(prev_transforms[j])[0] for j in prev_ielems]
        map_indices.insert(0, dict(zip(prev_ielems, mapped_prev_ielems)))
        nontouchielems_i = numpy.unique(numpy.array(mapped_prev_ielems, dtype=int))
        prev_ielems = ielems_i = numpy.unique(numpy.concatenate([numpy.asarray(touchielems_i, dtype=int), nontouchielems_i], axis=0))
        prev_transforms = topo.transforms

        basis_i = topo.basis(name, *args, **kwargs)
        assert isinstance(basis_i, function.Basis)
        ubases.insert(0, basis_i)
        # Basis functions that have at least one touchelem in their support.
        touchdofs_i = basis_i.get_dofs(touchielems_i)
        # Basis functions with (partial) support in this hierarchical topology.
        partsuppdofs_i = numpy.union1d(touchdofs_i, basis_i.get_dofs(numpy.setdiff1d(ielems_i, touchielems_i, assume_unique=True)))
        # Mask of basis functions in `partsuppdofs_i` with strict support in this hierarchical topology.
        partsuppdofs_supported_i = numpy.array([numeric.sorted_contains(ielems_i, basis_i.get_support(dof)).all() for dof in partsuppdofs_i], dtype=bool)
        ubasis_active.insert(0, numpy.intersect1d(touchdofs_i, partsuppdofs_i[partsuppdofs_supported_i], assume_unique=True))
        ubasis_passive.insert(0, partsuppdofs_i[~partsuppdofs_supported_i])

    *offsets, ndofs = numpy.cumsum([0, *map(len, ubasis_active)])

    # 2. construct hierarchical polynomials
    hbasis_dofs = []
    hbasis_coeffs = []
    projectcache = {}

    for ilevel, (level, indices) in enumerate(zip(self.levels, self._indices_per_level)):
      for ilocal in indices:

        hbasis_trans = transform.canonical(level.transforms[ilocal])
        tail = hbasis_trans[len(hbasis_trans)-ilevel:]
        trans_dofs = []
        trans_coeffs = []

        local_indices = [ilocal]
        for m in reversed(map_indices[:ilevel]):
          ilocal = m[ilocal]
          local_indices.insert(0, ilocal)

        if not truncated: # classical hierarchical basis

          for h, ilocal in enumerate(local_indices): # loop from coarse to fine
            mydofs = ubases[h].get_dofs(ilocal)

            imyactive = numeric.sorted_index(ubasis_active[h], mydofs, missing=-1)
            myactive = numpy.greater_equal(imyactive, 0)
            if myactive.any():
              trans_dofs.append(offsets[h]+imyactive[myactive])
              mypoly = ubases[h].get_coefficients(ilocal)
              trans_coeffs.append(mypoly[myactive])

            if h < len(tail):
              trans_coeffs = [tail[h].transform_poly(c) for c in trans_coeffs]

        else: # truncated hierarchical basis

          for h, ilocal in reversed(tuple(enumerate(local_indices))): # loop from fine to coarse
            mydofs = ubases[h].get_dofs(ilocal)
            mypoly = ubases[h].get_coefficients(ilocal)

            truncpoly = mypoly if h == len(tail) \
              else numpy.tensordot(numpy.tensordot(tail[h].transform_poly(mypoly), project[...,mypassive], self.ndims), truncpoly[mypassive], 1)

            imyactive = numeric.sorted_index(ubasis_active[h], mydofs, missing=-1)
            myactive = numpy.greater_equal(imyactive, 0) & numpy.greater(abs(truncpoly), truncation_tolerance).any(axis=tuple(range(1,truncpoly.ndim)))
            if myactive.any():
              trans_dofs.append(offsets[h]+imyactive[myactive])
              trans_coeffs.append(truncpoly[myactive])

            mypassive = numeric.sorted_contains(ubasis_passive[h], mydofs)
            if not mypassive.any():
              break

            try: # construct least-squares projection matrix
              project = projectcache[mypoly]
            except KeyError:
              P = mypoly.reshape(len(mypoly), -1)
              U, S, V = numpy.linalg.svd(P) # (U * S).dot(V[:len(S)]) == P
              project = (V.T[:,:len(S)] / S).dot(U.T).reshape(mypoly.shape[1:]+mypoly.shape[:1])
              projectcache[mypoly] = project

        # add the dofs and coefficients to the hierarchical basis
        hbasis_dofs.append(numpy.concatenate(trans_dofs))
        hbasis_coeffs.append(numeric.poly_concatenate(trans_coeffs))

    return function.PlainBasis(hbasis_coeffs, hbasis_dofs, ndofs, self.transforms)

class ProductTopology(Topology):
  'product topology'

  __slots__ = 'topo1', 'topo2'
  __cache__ = 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, topo1:stricttopology, topo2:stricttopology):
    assert not isinstance(topo1, ProductTopology)
    self.topo1 = topo1
    self.topo2 = topo2
    references = self.topo1.references * self.topo2.references
    transforms = transformseq.ProductTransforms(self.topo1.transforms, self.topo2.transforms)
    if (self.topo1.opposites != self.topo1.transforms) != (self.topo2.opposites != self.topo2.transforms):
      opposites = transformseq.ProductTransforms(self.topo1.opposites, self.topo2.opposites)
    else:
      opposites = transforms
    super().__init__(references, transforms, opposites)

  def __mul__(self, other):
    return ProductTopology(self.topo1, self.topo2 * other)

  @property
  def refined(self):
    return self.topo1.refined * self.topo2.refined

  def refine(self, n):
    if numpy.iterable(n):
      assert len(n) == self.ndims
    else:
      n = (n,)*self.ndims
    return self.topo1.refine(n[:self.topo1.ndims]) * self.topo2.refine(n[self.topo1.ndims:])

  def getitem(self, item):
    return self.topo1.getitem(item) * self.topo2 | self.topo1 * self.topo2.getitem(item) if isinstance(item, str) \
      else self.topo1[item[:self.topo1.ndims]] * self.topo2[item[self.topo1.ndims:]]

  def basis(self, name, *args, **kwargs):
    def _split(arg):
      if not numpy.iterable(arg):
        return arg, arg
      assert len(arg) == self.ndims
      return tuple(a[0] if all(ai == a[0] for ai in a[1:]) else a for a in (arg[:self.topo1.ndims], arg[self.topo1.ndims:]))
    splitargs = [_split(arg) for arg in args]
    splitkwargs = [(name,)+_split(arg) for name, arg in kwargs.items()]
    basis1, basis2 = function.bifurcate(
      self.topo1.basis(name, *[arg1 for arg1, arg2 in splitargs], **{name: arg1 for name, arg1, arg2 in splitkwargs}),
      self.topo2.basis(name, *[arg2 for arg1, arg2 in splitargs], **{name: arg2 for name, arg1, arg2 in splitkwargs}))
    return function.ravel(function.outer(basis1,basis2), axis=0)

  @property
  def boundary(self):
    return self.topo1 * self.topo2.boundary + self.topo1.boundary * self.topo2

  @property
  def interfaces(self):
    return self.topo1 * self.topo2.interfaces + self.topo1.interfaces * self.topo2

class RevolutionTopology(Topology):
  'topology consisting of a single revolution element'

  __slots__ = 'boundary', '_root'

  connectivity = numpy.empty([1,0], dtype=int)

  def __init__(self):
    self._root = transform.Identifier(1, 'angle')
    self.boundary = EmptyTopology(ndims=0)
    transforms = transformseq.PlainTransforms([(self._root,)], 1)
    references = elementseq.asreferences([element.RevolutionReference()], 1)
    super().__init__(references, transforms, transforms)

  @property
  def refined(self):
    return self

  def basis(self, name, *args, **kwargs):
    return function.asarray([1.])

class PatchBoundary(types.Singleton):

  __slots__ = 'id', 'dim', 'side', 'reverse', 'transpose'

  @types.apply_annotations
  def __init__(self, id:types.tuple[types.strictint], dim, side, reverse:types.tuple[bool], transpose:types.tuple[types.strictint]):
    super().__init__()
    self.id = id
    self.dim = dim
    self.side = side
    self.reverse = reverse
    self.transpose = transpose

  def apply_transform(self, array):
    return array[tuple(slice(None, None, -1) if i else slice(None) for i in self.reverse)].transpose(self.transpose)

class Patch(types.Singleton):

  __slots__ = 'topo', 'verts', 'boundaries'

  @types.apply_annotations
  def __init__(self, topo:stricttopology, verts:types.frozenarray, boundaries:types.tuple[types.strict[PatchBoundary]]):
    super().__init__()
    self.topo = topo
    self.verts = verts
    self.boundaries = boundaries

class MultipatchTopology(Topology):
  'multipatch topology'

  __slots__ = 'patches',
  __cache__ = '_patchinterfaces', 'boundary', 'interfaces', 'refined', 'connectivity'

  @staticmethod
  def build_boundarydata(connectivity):
    'build boundary data based on connectivity'

    boundarydata = []
    for patch in connectivity:
      ndims = len(patch.shape)
      patchboundarydata = []
      for dim, side in itertools.product(range(ndims), [-1, 0]):
        # ignore vertices at opposite face
        verts = numpy.array(patch)
        opposite = tuple({0:-1, -1:0}[side] if i == dim else slice(None) for i in range(ndims))
        verts[opposite] = verts.max()+1
        if len(set(verts.flat)) != 2**(ndims-1)+1:
          raise NotImplementedError('Cannot compute canonical boundary if vertices are used more than once.')
        # reverse axes such that lowest vertex index is at first position
        reverse = tuple(map(bool, numpy.unravel_index(verts.argmin(), verts.shape)))
        verts = verts[tuple(slice(None, None, -1) if i else slice(None) for i in reverse)]
        # transpose such that second lowest vertex connects to lowest vertex in first dimension, third in second dimension, et cetera
        k = [verts[tuple(1 if i == j else 0 for j in range(ndims))] for i in range(ndims)]
        transpose = tuple(sorted(range(ndims), key=k.__getitem__))
        verts = verts.transpose(transpose)
        # boundarid
        boundaryid = tuple(verts[...,0].flat)
        patchboundarydata.append(PatchBoundary(boundaryid,dim,side,reverse,transpose))
      boundarydata.append(tuple(patchboundarydata))

    # TODO: boundary sanity checks

    return boundarydata

  @types.apply_annotations
  def __init__(self, patches:types.tuple[types.strict[Patch]]):
    'constructor'

    self.patches = patches

    super().__init__(
      elementseq.chain([patch.topo.references for patch in self.patches], self.patches[0].topo.ndims),
      transformseq.chain([patch.topo.transforms for patch in self.patches], self.patches[0].topo.ndims),
      transformseq.chain([patch.topo.opposites for patch in self.patches], self.patches[0].topo.ndims))

  @property
  def _patchinterfaces(self):
    patchinterfaces = {}
    for patch in self.patches:
      for boundary in patch.boundaries:
        patchinterfaces.setdefault(boundary.id, []).append((patch.topo, boundary))
    return {
      boundaryid: tuple(data)
      for boundaryid, data in patchinterfaces.items()
      if len(data) > 1
    }

  def getitem(self, key):
    for i in range(len(self.patches)):
      if key == 'patch{}'.format(i):
        return self.patches[i].topo
    else:
      return DisjointUnionTopology(patch.topo.getitem(key) for patch in self.patches)

  def basis_spline(self, degree, patchcontinuous=True, knotvalues=None, knotmultiplicities=None, *, continuity=-1):
    '''spline from vertices

    Create a spline basis with degree ``degree`` per patch.  If
    ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
    interfaces.
    '''

    if knotvalues is None:
      knotvalues = {None: None}
    else:
      knotvalues, _knotvalues = {}, knotvalues
      for edge, k in _knotvalues.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotvalues[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotvalues
          assert (r,l) not in knotvalues
          knotvalues[(l,r)] = k
          knotvalues[(r,l)] = rk

    if knotmultiplicities is None:
      knotmultiplicities = {None: None}
    else:
      knotmultiplicities, _knotmultiplicities = {}, knotmultiplicities
      for edge, k in _knotmultiplicities.items():
        if k is None:
          rk = None
        else:
          k = tuple(k)
          rk = k[::-1]
        if edge is None:
          knotmultiplicities[edge] = k
        else:
          l, r = edge
          assert (l,r) not in knotmultiplicities
          assert (r,l) not in knotmultiplicities
          knotmultiplicities[(l,r)] = k
          knotmultiplicities[(r,l)] = rk

    missing = object()

    coeffs = []
    dofmap = []
    dofcount = 0
    commonboundarydofs = {}
    for ipatch, patch in enumerate(self.patches):
      # build structured spline basis on patch `patch.topo`
      patchknotvalues = []
      patchknotmultiplicities = []
      for idim in range(self.ndims):
        left = tuple(0 if j == idim else slice(None) for j in range(self.ndims))
        right = tuple(1 if j == idim else slice(None) for j in range(self.ndims))
        dimknotvalues = set()
        dimknotmultiplicities = set()
        for edge in zip(patch.verts[left].flat, patch.verts[right].flat):
          v = knotvalues.get(edge, knotvalues.get(None, missing))
          m = knotmultiplicities.get(edge, knotmultiplicities.get(None, missing))
          if v is missing:
            raise 'missing edge'
          dimknotvalues.add(v)
          if m is missing:
            raise 'missing edge'
          dimknotmultiplicities.add(m)
        if len(dimknotvalues) != 1:
          raise 'ambiguous knot values for patch {}, dimension {}'.format(ipatch, idim)
        if len(dimknotmultiplicities) != 1:
          raise 'ambiguous knot multiplicities for patch {}, dimension {}'.format(ipatch, idim)
        patchknotvalues.append(next(iter(dimknotvalues)))
        patchknotmultiplicities.append(next(iter(dimknotmultiplicities)))
      patchcoeffs, patchdofmap, patchdofcount = patch.topo._basis_spline(degree, knotvalues=patchknotvalues, knotmultiplicities=patchknotmultiplicities, continuity=continuity)
      coeffs.extend(patchcoeffs)
      dofmap.extend(types.frozenarray(dofs+dofcount, copy=False) for dofs in patchdofmap)
      if patchcontinuous:
        # reconstruct multidimensional dof structure
        dofs = dofcount + numpy.arange(numpy.prod(patchdofcount), dtype=int).reshape(patchdofcount)
        for boundary in patch.boundaries:
          # get patch boundary dofs and reorder to canonical form
          boundarydofs = boundary.apply_transform(dofs)[...,0].ravel()
          # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
          commonboundarydofs.setdefault(boundary.id, []).append(boundarydofs)
      dofcount += numpy.prod(patchdofcount)

    if patchcontinuous:
      # build merge mapping: merge common boundary dofs (from low to high)
      pairs = itertools.chain(*(zip(*dofs) for dofs in commonboundarydofs.values() if len(dofs) > 1))
      merge = {}
      for dofs in sorted(pairs):
        dst = merge.get(dofs[0], dofs[0])
        for src in dofs[1:]:
          merge[src] = dst
      # build renumber mapping: renumber remaining dofs consecutively, starting at 0
      remainder = set(merge.get(dof, dof) for dof in range(dofcount))
      renumber = dict(zip(sorted(remainder), range(len(remainder))))
      # apply mappings
      dofmap = tuple(types.frozenarray(tuple(renumber[merge.get(dof, dof)] for dof in v.flat), dtype=int).reshape(v.shape) for v in dofmap)
      dofcount = len(remainder)

    return function.PlainBasis(coeffs, dofmap, dofcount, self.transforms)

  def basis_patch(self):
    'degree zero patchwise discontinuous basis'

    return function.DiscontBasis(
      [types.frozenarray(1, dtype=int).reshape(1, *(1,)*self.ndims)]*len(self.patches),
      transformseq.PlainTransforms(tuple((patch.topo.root,) for patch in self.patches), self.ndims))

  @property
  def boundary(self):
    'boundary'

    subtopos = []
    subnames = []
    for i, patch in enumerate(self.patches):
      for boundary in patch.boundaries:
        if boundary.id in self._patchinterfaces:
          continue
        name = patch.topo._bnames[boundary.dim][boundary.side]
        subtopos.append(patch.topo.boundary[name])
        subnames.append('patch{}-{}'.format(i, name))
    if len(subtopos) == 0:
      return EmptyTopology(self.ndims-1)
    else:
      return DisjointUnionTopology(subtopos, subnames)

  @property
  def interfaces(self):
    '''interfaces

    Return a topology with all element interfaces.  The patch interfaces are
    accessible via the group ``'interpatch'`` and the interfaces *inside* a
    patch via ``'intrapatch'``.
    '''

    intrapatchtopo = EmptyTopology(self.ndims-1) if not self.patches else \
      DisjointUnionTopology(patch.topo.interfaces for patch in self.patches)

    btopos = []
    bconnectivity = []
    for boundaryid, patchdata in self._patchinterfaces.items():
      if len(patchdata) > 2:
        raise ValueError('Cannot create interfaces of multipatch topologies with more than two interface connections.')
      pairs = []
      references = None
      for topo, boundary in patchdata:
        btopo = topo.boundary[topo._bnames[boundary.dim][boundary.side]]
        if references is None:
          references = numeric.asobjvector(btopo.references).reshape(btopo.shape)
          references = references[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
          references = boundary.apply_transform(references)[..., 0]
          references = tuple(references.flat)
        transforms = numeric.asobjvector(btopo.transforms).reshape(btopo.shape)
        transforms = transforms[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
        transforms = boundary.apply_transform(transforms)[..., 0]
        pairs.append(tuple(transforms.flat))
      # create structured topology of joined element pairs
      transforms, opposites = pairs
      btopos.append(UnstructuredTopology(references, transforms, opposites, ndims=self.ndims-1))
      bconnectivity.append(numpy.array(boundaryid).reshape((2,)*(self.ndims-1)))
    # create multipatch topology of interpatch boundaries
    interpatchtopo = MultipatchTopology(tuple(map(Patch, btopos, bconnectivity, self.build_boundarydata(bconnectivity))))

    return DisjointUnionTopology((intrapatchtopo, interpatchtopo), ('intrapatch', 'interpatch'))

  @property
  def connectivity(self):
    connectivity = []
    patchinterfaces = {}
    for patch in self.patches: # len(connectivity) represents the element offset for the current patch
      ielems = numpy.arange(len(patch.topo)).reshape(patch.topo.shape) + len(connectivity)
      for boundary in patch.boundaries:
        patchinterfaces.setdefault(boundary.id, []).append((boundary.apply_transform(ielems)[...,0], boundary.dim * 2 + (boundary.side == 0)))
      connectivity.extend(patch.topo.connectivity + len(connectivity) * numpy.not_equal(patch.topo.connectivity, -1))
    connectivity = numpy.array(connectivity)
    for patchdata in patchinterfaces.values():
      if len(patchdata) > 2:
        raise ValueError('Cannot create connectivity of multipatch topologies with more than two interface connections.')
      if len(patchdata) == 2:
        (ielem, iedge), (jelem, jedge) = patchdata
        assert ielem.shape == jelem.shape
        assert numpy.equal(connectivity[ielem, iedge], -1).all()
        assert numpy.equal(connectivity[jelem, jedge], -1).all()
        connectivity[ielem, iedge] = jelem
        connectivity[jelem, jedge] = ielem
    return types.frozenarray(connectivity, copy=False)

  @property
  def refined(self):
    'refine'

    return MultipatchTopology(Patch(patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches)

# vim:sw=2:sts=2:et
