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

from . import element, function, util, numpy, parallel, log, config, numeric, cache, transform, warnings, matrix, types, _
import functools, collections.abc, itertools, functools, operator, numbers, pathlib

_identity = lambda x: x

class Topology(types.Singleton):
  'topology base class'

  __slots__ = 'ndims',
  __cache__ ='edict', 'border_transforms', 'simplex'

  # subclass needs to implement: .elements

  @types.apply_annotations
  def __init__(self, ndims:types.strictint):
    super().__init__()
    assert ndims >= 0
    self.ndims = ndims

  def __str__(self):
    'string representation'

    return '{}(#{})'.format(self.__class__.__name__, len(self))

  def __len__(self):
    return len(self.elements)

  def __iter__(self):
    return iter(self.elements)

  def getitem(self, item):
    return EmptyTopology(self.ndims)

  def __getitem__(self, item):
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
    # Strategy: loop over combined elements sorted by .transform while keeping
    # track of the origin (mine=True for self, mine=False for other), and
    # select an element if it is equal to or a refinement of the previous
    # (hold) element and it originates from the other topology (mine == need).
    # Hold is not updated in case of a match because it might match multiple
    # children.
    elems = []
    need = None
    for elem, mine in sorted([(elem, True) for elem in self] + [(elem, False) for elem in other], key=lambda v: v[0].transform):
      if mine == need and elem.transform[:len(hold.transform)] == hold.transform:
        assert elem.opposite[:len(hold.opposite)] == hold.opposite
        elems.append(elem)
      else:
        hold = elem
        need = not mine
    return UnstructuredTopology(self.ndims, elems)

  __rand__ = lambda self, other: self.__and__(other)

  def __add__(self, other):
    return self | other

  def __contains__(self, element):
    ielem = self.edict.get(element.transform)
    return ielem is not None and self.elements[ielem] == element

  def __sub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other.__rsub__(self)

  def __rsub__(self, other):
    assert isinstance(other, Topology) and other.ndims == self.ndims
    return other - other.subset(self, newboundary=getattr(self,'boundary',None))

  def __mul__(self, other):
    return ProductTopology(self, other)

  @property
  def edict(self):
    '''transform -> ielement mapping'''
    return {elem.transform: ielem for ielem, elem in enumerate(self)}

  @property
  def border_transforms(self):
    border_transforms = set()
    for belem in self.boundary:
      try:
        ielem, tail = transform.lookup_item(belem.transform, self.edict)
      except KeyError:
        pass
      else:
        border_transforms.add(self.elements[ielem].transform)
    return border_transforms

  @property
  def refine_iter(self):
    topo = self
    for irefine in log.count('refinement level'):
      yield topo
      topo = topo.refined

  stdfunc     = lambda self, *args, **kwargs: self.basis('std', *args, **kwargs)
  linearfunc  = lambda self, *args, **kwargs: self.basis('std', degree=1, *args, **kwargs)
  splinefunc  = lambda self, *args, **kwargs: self.basis('spline', *args, **kwargs)
  bubblefunc  = lambda self, *args, **kwargs: self.basis('bubble', *args, **kwargs)
  discontfunc = lambda self, *args, **kwargs: self.basis('discont', *args, **kwargs)

  def basis(self, name, *args, **kwargs):
    if self.ndims == 0:
      return function.asarray([1])
    f = getattr(self, 'basis_' + name)
    return f(*args, **kwargs)

  @log.title
  @util.single_or_multiple
  def elem_eval(self, funcs, ischeme, separate=False, geometry=None, asfunction=False, edit=_identity, *, arguments=None):
    'element-wise evaluation'

    fcache = cache.WrapperCache()

    assert not separate or not asfunction, '"separate" and "asfunction" are mutually exclusive'
    if geometry:
      iwscale = function.J(geometry, self.ndims)
      npoints = len(self)
      slices = range(npoints)
    else:
      iwscale = 1
      slices = []
      npoints = 0
      for elem in log.iter('elem', self):
        ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme](ischeme)
        np = len(ipoints)
        slices.append(slice(npoints,npoints+np))
        npoints += np

    nprocs = min(config.nprocs, len(self))
    zeros = parallel.shzeros if nprocs > 1 else numpy.zeros
    retvals = []
    idata = []
    for ifunc, func in enumerate(funcs):
      func = function.asarray(edit(func * iwscale))
      func = function.zero_argument_derivatives(func)
      retval = zeros((npoints,)+func.shape, dtype=func.dtype)
      idata.extend(function.Tuple([ifunc, function.Tuple(ind), f.simplified]) for ind, f in function.blocks(func))
      retvals.append(retval)
    idata = function.Tuple(idata)

    if config.dot:
      idata.graphviz()

    if arguments is None:
      arguments = {}

    for ielem, elem in parallel.pariter(log.enumerate('elem', self), nprocs=nprocs):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme](ischeme)
      s = slices[ielem],
      try:
        for ifunc, index, data in idata.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments):
          numpy.add.at(retvals[ifunc], s+numpy.ix_(*[ind for (ind,) in index]), numeric.dot(iweights,data) if geometry else data)
      except function.EvaluationError:
        warnings.warn('not all functions evaluated successfully')
        for ifunc, indexfunc, datafunc in idata:
          index = indexfunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)
          try:
            data = datafunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)
          except function.EvaluationError:
            retvals[ifunc][s+numpy.ix_(*[ind for (ind,) in index])] = numpy.nan
          else:
            numpy.add.at(retvals[ifunc], s+numpy.ix_(*[ind for (ind,) in index]), numeric.dot(iweights,data) if geometry else data)

    log.debug('cache', fcache.stats)
    log.info('created', ', '.join('{}({})'.format(retval.__class__.__name__, ','.join(str(n) for n in retval.shape)) for retval in retvals))

    if asfunction:
      if geometry:
        retvals = [function.elemwise({elem.transform: value for elem, value in zip(self, retval)}, shape=retval.shape[1:]) for retval in retvals]
      else:
        tsp = [(elem.transform, s, fcache[elem.reference.getischeme](ischeme)[0]) for elem, s in zip(self, slices)]
        retvals = [function.sampled({trans: (types.frozenarray(retval[s], copy=False), points) for trans, s, points in tsp}, self.ndims) for retval in retvals]
    elif separate:
      retvals = [[retval[s] for s in slices] for retval in retvals]

    return retvals

  @log.title
  @util.single_or_multiple
  def elem_mean(self, funcs, geometry, ischeme, *, arguments=None):
    'element-wise average'

    retvals = self.elem_eval((1,)+funcs, geometry=geometry, ischeme=ischeme, arguments=arguments)
    return [v / retvals[0][(slice(None),)+(_,)*(v.ndim-1)] for v in retvals[1:]]

  def _integrate(self, funcs, ischeme, fcache=None, arguments=None):

    if arguments is None:
      arguments = {}

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    blocks = [(ifunc, function.Tuple(ind), f.simplified)
      for ifunc, func in enumerate(funcs)
        for ind, f in function.blocks(function.zero_argument_derivatives(func))]

    block2func, indices, values = zip(*blocks) if blocks else ([],[],[])

    log.debug('integrating {} distinct blocks'.format('+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)))))

    if config.dot:
      function.Tuple(values).graphviz()

    if fcache is None:
      fcache = cache.WrapperCache()

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros((len(blocks), len(self)+1), dtype=int)
    if blocks:
      sizefunc = function.stack([f.size for ifunc, ind, f in blocks]).simplified
      for ielem, elem in enumerate(self):
        n, = sizefunc.eval(_transforms=(elem.transform, elem.opposite), _cache=fcache, **arguments)
        offsets[:,ielem+1] = offsets[:,ielem] + n

    # Since several blocks may belong to the same function, we post process the
    # offsets to form consecutive intervals in longer arrays. The length of
    # these arrays is captured in the nfuncs-array nvals.

    nvals = numpy.zeros(len(funcs), dtype=int)
    for iblock, ifunc in enumerate(block2func):
      offsets[iblock] += nvals[ifunc]
      nvals[ifunc] = offsets[iblock,-1]

    # The data_index list contains shared memory index and value arrays for
    # each function argument.

    nprocs = min(config.nprocs, len(self))
    empty = parallel.shempty if nprocs > 1 else numpy.empty
    data_index = [
      (empty(n, dtype=float),
        empty((funcs[ifunc].ndim,n), dtype=int))
            for ifunc, n in enumerate(nvals) ]

    # In a second, parallel element loop, valuefunc is evaluated to fill the
    # data part of data_index using the offsets array for location. Each
    # element has its own location so no locks are required. The index part of
    # data_index is filled in the same loop. It does not use valuefunc data but
    # benefits from parallel speedup.

    valueindexfunc = function.Tuple(function.Tuple([value]+list(index)) for value, index in zip(values, indices))
    for ielem, elem in parallel.pariter(log.enumerate('elem', self), nprocs=nprocs):
      ipoints, iweights = ischeme[elem] if isinstance(ischeme,collections.abc.Mapping) else fcache[elem.reference.getischeme](ischeme)
      assert iweights is not None, 'no integration weights found'
      for iblock, (intdata, *indices) in enumerate(valueindexfunc.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, _cache=fcache, **arguments)):
        s = slice(*offsets[iblock,ielem:ielem+2])
        data, index = data_index[block2func[iblock]]
        w_intdata = numeric.dot(iweights, intdata)
        data[s] = w_intdata.ravel()
        si = (slice(None),) + (_,) * (w_intdata.ndim-1)
        for idim, (ii,) in enumerate(indices):
          index[idim,s].reshape(w_intdata.shape)[...] = ii[si]
          si = si[:-1]

    log.debug('cache', fcache.stats)

    return data_index

  @log.title
  @util.single_or_multiple
  def integrate(self, funcs, ischeme='gauss', degree=None, geometry=None, fcache=None, edit=_identity, *, arguments=None):
    'integrate'

    if degree is not None:
      ischeme += str(degree)
    iwscale = function.J(geometry, self.ndims) if geometry else 1
    integrands = [function.asarray(edit(func * iwscale)) for func in funcs]
    data_index = self._integrate(integrands, ischeme, fcache, arguments)
    for integrand, (data,index) in zip(integrands, data_index):
      retval = matrix.assemble(data, index, integrand.shape)
      assert retval.shape == integrand.shape
      log.debug('assembled {}({})'.format(retval.__class__.__name__, ','.join(str(n) for n in retval.shape)))
      yield retval

  @log.title
  def integral(self, func, ischeme='gauss', degree=None, geometry=None, edit=_identity):
    'integral'

    if degree is not None:
      ischeme += str(degree)
    iwscale = function.J(geometry, self.ndims) if geometry else 1
    integrand = edit(func * iwscale)
    return Integral([((self, ischeme), integrand)])

  def projection(self, fun, onto, geometry, **kwargs):
    'project and return as function'

    weights = self.project(fun, onto, geometry, **kwargs)
    return onto.dot(weights)

  @log.title
  def project(self, fun, onto, geometry, ischeme='gauss', degree=None, droptol=1e-12, exact_boundaries=False, constrain=None, verify=None, ptype='lsqr', edit=_identity, *, arguments=None, **solverargs):
    'L2 projection of function onto function space'

    log.debug('projection type:', ptype)

    if degree is not None:
      ischeme += str(degree)
    if constrain is None:
      constrain = util.NanVec(onto.shape[0])
    else:
      constrain = constrain.copy()
    if exact_boundaries:
      constrain |= self.boundary.project(fun, onto, geometry, constrain=constrain, title='boundaries', ischeme=ischeme, droptol=droptol, ptype=ptype, edit=edit, arguments=arguments)
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
      A, b, f2, area = self.integrate([Afun,bfun,fun2,1], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments, title='building system')
      N = A.rowsupp(droptol)
      if numpy.equal(b, 0).all():
        constrain[~constrain.where&N] = 0
        avg_error = 0.
      else:
        solvecons = constrain.copy()
        solvecons[~(constrain.where|N)] = 0
        u = A.solve(b, constrain=solvecons, **solverargs)
        constrain[N] = u[N]
        err2 = f2 - numpy.dot(2*b-A.matvec(u), u) # can be negative ~zero due to rounding errors
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
      u, scale = self.integrate([ufun, afun], geometry=geometry, ischeme=ischeme, edit=edit, arguments=arguments)
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
      fun = function.zero_argument_derivatives(function.asarray(fun))
      data = function.Tuple(function.Tuple([fun, onto_f.simplified, function.Tuple(onto_ind)]) for onto_ind, onto_f in function.blocks(function.zero_argument_derivatives(onto)))
      for elem in self:
        ipoints, iweights = elem.getischeme('bezier2')
        for fun_, onto_f_, onto_ind_ in data.eval(_transforms=(elem.transform, elem.opposite), _points=ipoints, **arguments or {}):
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

  @property
  def simplex(self):
    simplices = [simplex for elem in self for simplex in elem.simplices]
    return UnstructuredTopology(self.ndims, simplices)

  def refined_by(self, refine):
    'create refined space by refining dofs in existing one'

    refine = set(item.transform if isinstance(item,element.Element) else item for item in refine)
    refined = []
    for elem in self:
      if elem.transform in refine:
        refined.extend(elem.children)
      else:
        refined.append(elem)
    return self.hierarchical(refined, precise=True)

  def hierarchical(self, refined, precise=False):
    return HierarchicalTopology(self, refined, precise)

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

  @log.title
  def trim(self, levelset, maxrefine, ndivisions=8, name='trimmed', leveltopo=None, *, arguments=None):
    'trim element along levelset'

    if arguments is None:
      arguments = {}

    fcache = cache.WrapperCache()
    levelset = function.zero_argument_derivatives(levelset).simplified
    if leveltopo is None:
      ischeme = 'vertex{}'.format(maxrefine)
      refs = [elem.reference.trim(levelset.eval(_transforms=(elem.transform, elem.opposite), _points=fcache[elem.reference.getischeme](ischeme)[0], _cache=fcache, **arguments), maxrefine=maxrefine, ndivisions=ndivisions) for elem in log.iter('elem', self)]
    else:
      log.info('collecting leveltopo elements')
      bins = [[] for ielem in range(len(self))]
      for elem in leveltopo:
        ielem, tail = transform.lookup_item(elem.transform, self.edict)
        bins[ielem].append(tail)
      refs = []
      for elem, ctransforms in log.zip('elem', self, bins):
        levels = numpy.empty(elem.reference.nvertices_by_level(maxrefine))
        cover = list(fcache[elem.reference.vertex_cover](tuple(sorted(ctransforms)), maxrefine))
        # confirm cover and greedily optimize order
        mask = numpy.ones(len(levels), dtype=bool)
        while mask.any():
          imax = numpy.argmax([mask[indices].sum() for trans, points, indices in cover])
          trans, points, indices = cover.pop(imax)
          levels[indices] = levelset.eval(_transforms=(elem.transform + trans,), _points=points, _cache=fcache, **arguments)
          mask[indices] = False
        refs.append(elem.reference.trim(levels, maxrefine=maxrefine, ndivisions=ndivisions))
    log.debug('cache', fcache.stats)
    return SubsetTopology(self, refs, newboundary=name)

  def subset(self, elements, newboundary=None, strict=False):
    'intersection'
    refs = [element.EmptyReference(self.ndims)] * len(self)
    for elem in elements:
      try:
        ielem = self.edict[elem.transform]
      except KeyError:
        assert not strict, 'elements do not form a strict subset'
      else:
        ref = self.elements[ielem].reference & elem.reference
        if strict:
          assert ref == elem.reference, 'elements do not form a strict subset'
        refs[ielem] = ref
    if not any(refs):
      return EmptyTopology(self.ndims)
    return SubsetTopology(self, refs, newboundary)

  def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
    return WithGroupsTopology(self, vgroups, bgroups, igroups, pgroups) if vgroups or bgroups or igroups or pgroups else self

  withsubdomain  = lambda self, **kwargs: self.withgroups(vgroups=kwargs)
  withboundary   = lambda self, **kwargs: self.withgroups(bgroups=kwargs)
  withinterfaces = lambda self, **kwargs: self.withgroups(igroups=kwargs)
  withpoints     = lambda self, **kwargs: self.withgroups(pgroups=kwargs)

  @log.title
  @util.single_or_multiple
  def elem_project(self, funcs, degree, ischeme=None, check_exact=False, *, arguments=None):

    if arguments is None:
      arguments = {}

    if ischeme is None:
      ischeme = 'gauss{}'.format(degree*2)

    blocks = function.Tuple([function.Tuple([function.Tuple((function.Tuple(ind), f.simplified))
      for ind, f in function.blocks(function.zero_argument_derivatives(func))])
        for func in funcs])

    bases = {}
    extractions = [[] for ifunc in range(len(funcs))]

    for elem in log.iter('elem', self):

      try:
        points, projector, basis = bases[elem.reference]
      except KeyError:
        points, weights = elem.reference.getischeme(ischeme)
        coeffs = elem.reference.get_poly_coeffs('bernstein', degree=degree)
        basis = numeric.poly_eval(coeffs[_], points)
        npoints, nfuncs = basis.shape
        A = numeric.dot(weights, basis[:,:,_] * basis[:,_,:])
        projector = numpy.linalg.solve(A, basis.T * weights)
        bases[elem.reference] = points, projector, basis

      for ifunc, ind_val in enumerate(blocks.eval(_transforms=(elem.transform, elem.opposite), _points=points, **arguments)):

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

  @log.title
  def volume(self, geometry, ischeme='gauss', degree=1, *, arguments=None):
    return self.integrate(1, geometry=geometry, ischeme=ischeme, degree=degree, arguments=arguments)

  @log.title
  def check_boundary(self, geometry, elemwise=False, ischeme='gauss', degree=1, tol=1e-15, print=print, *, arguments=None):
    if elemwise:
      for elem in self:
        elem.reference.check_edges(tol=tol, print=print)
    volume = self.volume(geometry, ischeme=ischeme, degree=degree, arguments=arguments)
    zeros, volumes = self.boundary.integrate([geometry.normal(), geometry * geometry.normal()], geometry=geometry, ischeme=ischeme, degree=degree, arguments=arguments)
    if numpy.greater(abs(zeros), tol).any():
      print('divergence check failed: {} != 0'.format(zeros))
    if numpy.greater(abs(volumes - volume), tol).any():
      print('divergence check failed: {} != {}'.format(volumes, volume))

  def volume_check(self, geometry, ischeme='gauss', degree=1, decimal=15, *, arguments=None):
    warnings.deprecation('volume_check will be removed in future, us check_boundary instead')
    self.check_boundary(geometry=geometry, ischeme=ischeme, degree=degree, tol=10**-decimal, arguments=arguments)

  def indicator(self, subtopo):
    if isinstance(subtopo, str):
      subtopo = self[subtopo]
    transforms = tuple(sorted(elem.transform for elem in self))
    values = types.frozenarray([int(trans in subtopo.edict) for trans in transforms])
    assert len(subtopo) == values.sum(0), '{} is not a proper subtopology of {}'.format(subtopo, self)
    return function.Get(values, axis=0, item=function.FindTransform(transforms, function.Promote(self.ndims, trans=function.TRANS)))

  def select(self, indicator, ischeme='bezier2', *, arguments=None):
    values = self.elem_eval(indicator, ischeme, separate=True, arguments=arguments)
    selected = [elem for elem, value in zip(self, values) if numpy.greater(value, 0).any()]
    return UnstructuredTopology(self.ndims, selected)

  def prune_basis(self, basis):
    used = numpy.zeros(len(basis), dtype=bool)
    for axes, func in function.blocks(basis):
      dofmap = axes[0]
      for elem in self:
        dofs = dofmap.eval(_transforms=(elem.transform, elem.opposite))
        used[dofs] = True
    return function.mask(basis, used)

  def locate(self, geom, points, ischeme='vertex', scale=1, tol=1e-12, eps=0, maxiter=100, *, arguments=None):
    nprocs = min(config.nprocs, len(self))
    if arguments is None:
      arguments = {}
    if geom.ndim == 0:
      geom = geom[_]
      points = points[...,_]
    assert geom.shape == (self.ndims,)
    points = numpy.asarray(points, dtype=float)
    assert points.ndim == 2 and points.shape[1] == self.ndims
    vertices = self.elem_eval(geom, ischeme=ischeme, separate=True, arguments=arguments)
    bboxes = numpy.array([numpy.mean(v,axis=0) * (1-scale) + numpy.array([numpy.min(v,axis=0), numpy.max(v,axis=0)]) * scale
      for v in vertices]) # nelems x {min,max} x ndims
    vref = element.getsimplex(0)
    ielems = parallel.shempty(len(points), dtype=int)
    xis = parallel.shempty((len(points),len(geom)), dtype=float)
    for ipoint, point in parallel.pariter(log.enumerate('point', points), nprocs=nprocs):
      ielemcandidates, = numpy.logical_and(numpy.greater_equal(point, bboxes[:,0,:]), numpy.less_equal(point, bboxes[:,1,:])).all(axis=-1).nonzero()
      for ielem in sorted(ielemcandidates, key=lambda i: numpy.linalg.norm(bboxes[i].mean(0)-point)):
        converged = False
        elem = self.elements[ielem]
        xi, w = elem.reference.getischeme('gauss1')
        xi = (numpy.dot(w,xi) / w.sum())[_] if len(xi) > 1 else xi.copy()
        J = function.localgradient(geom, self.ndims)
        geom_J = function.Tuple((function.zero_argument_derivatives(geom), function.zero_argument_derivatives(J))).simplified
        for iiter in range(maxiter):
          point_xi, J_xi = geom_J.eval(_transforms=(elem.transform, elem.opposite), _points=xi, **arguments)
          err = numpy.linalg.norm(point - point_xi)
          if err < tol:
            converged = True
            break
          if iiter and err > prev_err:
            break
          prev_err = err
          xi += numpy.linalg.solve(J_xi, point - point_xi)
        if converged and elem.reference.inside(xi[0], eps=eps):
          ielems[ipoint] = ielem
          xis[ipoint], = xi
          break
      else:
        raise LocateError('failed to locate point: {}'.format(point))

    pelems = []
    for ielem, xi in zip(ielems, xis):
      elem = self.elements[ielem]
      trans = transform.Matrix(numpy.empty((len(xi), 0)), xi)
      pelems.append(element.Element(vref, elem.transform + (trans,), elem.opposite and elem.opposite + (trans,), oriented=True))
    return UnstructuredTopology(0, pelems)

  def supp(self, basis, mask=None):
    if mask is None:
      mask = numpy.ones(len(basis), dtype=bool)
    elif isinstance(mask, list) or numeric.isarray(mask) and mask.dtype == int:
      tmp = numpy.zeros(len(basis), dtype=bool)
      tmp[mask] = True
      mask = tmp
    else:
      assert numeric.isarray(mask) and mask.dtype == bool and mask.shape == basis.shape[:1]
    indfunc = function.Tuple([ind[0] for ind, f in function.blocks(basis)])
    subset = []
    for elem in self:
      try:
        ind, = numpy.concatenate(indfunc.eval(_transforms=(elem.transform, elem.opposite)), axis=1)
      except function.EvaluationError:
        pass
      else:
        if mask[ind].any():
          subset.append(elem)
    if not subset:
      return EmptyTopology(self.ndims)
    return self.subset(subset, newboundary='supp', strict=True)

  def revolved(self, geom):
    assert geom.ndim == 1
    revdomain = self * RevolutionTopology()
    angle, = function.rootcoords(1)
    geom, angle = function.bifurcate(geom, angle)
    revgeom = function.concatenate([geom[0] * function.trignormal(angle), geom[1:]])
    simplify = cache.replace(lambda op: function.zeros(()) if op is angle else None)
    return revdomain, revgeom, simplify

  def extruded(self, geom, nelems, periodic=False, bnames=('front','back')):
    assert geom.ndim == 1
    root = transform.RootTrans('extrude', shape=[nelems if periodic else 0])
    extopo = self * StructuredLine(root, i=0, j=nelems, periodic=periodic, bnames=bnames)
    exgeom = function.concatenate(function.bifurcate(geom, function.rootcoords(1)))
    return extopo, exgeom

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
    super().__init__(basetopo.ndims)
    assert all(topo is Ellipsis or isinstance(topo, str) or isinstance(topo, Topology) and topo.ndims == basetopo.ndims and set(self.basetopo.edict).issuperset(topo.edict) for topo in self.vgroups.values())

  def withgroups(self, vgroups={}, bgroups={}, igroups={}, pgroups={}):
    args = []
    for groups, newgroups in (self.vgroups,vgroups), (self.bgroups,bgroups), (self.igroups,igroups), (self.pgroups,pgroups):
      groups = groups.copy()
      groups.update(newgroups)
      args.append(groups)
    return WithGroupsTopology(self.basetopo, *args)

  def __iter__(self):
    return iter(self.basetopo)

  def __len__(self):
    return len(self.basetopo)

  def getitem(self, item):
    if not isinstance(item, str):
      return self.basetopo.getitem(item)
    try:
      itemtopo = self.vgroups[item]
    except KeyError:
      return self.basetopo.getitem(item)
    else:
      return itemtopo if isinstance(itemtopo, Topology) else self.basetopo[itemtopo]

  @property
  def edict(self):
    return self.basetopo.edict

  @property
  def border_transforms(self):
    return self.basetopo.border_transforms

  @property
  def connectivity(self):
    return self.basetopo.connectivity

  @property
  def structure(self):
    return self.basetopo.structure

  @property
  def elements(self):
    return self.basetopo.elements

  @property
  def boundary(self):
    return self.basetopo.boundary.withgroups(self.bgroups)

  @property
  def interfaces(self):
    baseitopo = self.basetopo.interfaces
    # last minute orientation fix
    igroups = {name: UnstructuredTopology(self.ndims-1, [elem if elem.transform in baseitopo.edict else elem.flipped for elem in elems]) for name, elems in self.igroups.items()}
    return baseitopo.withgroups(igroups)

  @property
  def points(self):
    return self.basetopo.points.withgroups(self.pgroups)

  def basis(self, name, *args, **kwargs):
    return self.basetopo.basis(name, *args, **kwargs)

  @property
  def refined(self):
    groups = [{name: topo.refined if isinstance(topo,Topology) else topo for name, topo in groups.items()} for groups in (self.vgroups,self.bgroups,self.igroups,self.pgroups)]
    return self.basetopo.refined.withgroups(*groups)

class OppositeTopology(Topology):
  'opposite topology'

  __slots__ = 'basetopo',
  __cache__ = 'elements',

  def __init__(self, basetopo):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return ~(self.basetopo.getitem(item))

  def __iter__(self):
    return (elem.flipped for elem in self.basetopo)

  def __len__(self):
    return len(self.basetopo)

  @property
  def elements(self):
    return tuple(self)

  def __invert__(self):
    return self.basetopo

class EmptyTopology(Topology):
  'empty topology'

  __slots__ = ()

  def __iter__(self):
    return iter([])

  def __len__(self):
    return 0

  def __or__(self, other):
    assert self.ndims == other.ndims
    return other

  def __rsub__(self, other):
    return other

  @property
  def elements(self):
    return ()

class Point(Topology):
  'point'

  __slots__ = 'elem',

  @types.apply_annotations
  def __init__(self, trans:transform.stricttransform, opposite:transform.stricttransform=None):
    assert trans[-1].fromdims == 0
    self.elem = element.Element(element.getsimplex(0), trans, opposite, oriented=True)
    super().__init__(ndims=0)

  def __iter__(self):
    yield self.elem

  @property
  def elements(self):
    return self.elem,

class StructuredLine(Topology):
  'structured topology'

  __slots__ = 'root', 'i', 'j', 'periodic', 'bnames'
  __cache__ = '_transforms', 'elements', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, i:types.strictint, j:types.strictint, periodic:bool=False, bnames:types.tuple[types.strictstr]=None):
    'constructor'

    assert j > i
    assert not bnames or len(bnames) == 2
    assert isinstance(root, transform.TransformItem)
    self.root = root
    self.i = i
    self.j = j
    self.periodic = periodic
    self.bnames = bnames or ()
    super().__init__(ndims=1)

  @property
  def _transforms(self):
    # one extra left and right for opposites, even if periodic=True
    return tuple((self.root, transform.Shift([float(offset)])) for offset in range(self.i-1, self.j+1))

  def __iter__(self):
    reference = element.getsimplex(1)
    return (element.Element(reference, trans) for trans in self._transforms[1:-1])

  def __len__(self):
    return self.j - self.i

  @property
  def elements(self):
    return tuple(self)

  @property
  def boundary(self):
    if self.periodic:
      return EmptyTopology(ndims=0)
    transforms = self._transforms
    right, left = element.LineReference().edge_transforms
    bnd = Point(transforms[1] + (left,), transforms[0] + (right,)), Point(transforms[-2] + (right,), transforms[-1] + (left,))
    return UnionTopology(bnd, self.bnames)

  @property
  def interfaces(self):
    transforms = self._transforms
    right, left = element.LineReference().edge_transforms
    points = [Point(trans + (left,), opp + (right,)) for trans, opp in zip(transforms[2:-1], transforms[1:-2])]
    if self.periodic:
      points.append(Point(transforms[1] + (left,), transforms[-2] + (right,)))
    return UnionTopology(points)

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
        coeffs = [self._spline_coeffs(0, n)]
        dofs = types.frozenarray([[0]], copy=False)
      else:
        coeffs = list(self._spline_coeffs(p, n)[p-1:p]) * nelems
    else:
      coeffs = list(self._spline_coeffs(p, min(nelems,n)))
      if len(coeffs) < nelems:
        coeffs = coeffs[:p-1] + coeffs[p-1:p] * (nelems-2*(p-1)) + coeffs[p:]
    coeffs = types.frozenarray(coeffs, copy=False)

    func = function.polyfunc(coeffs, dofs, ndofs, self._transforms[1:-1], issorted=False)
    if not removedofs:
      return func

    mask = numpy.ones(ndofs, dtype=bool)
    mask[list(removedofs)] = False
    return function.mask(func, mask)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    ref = element.LineReference()
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = types.frozenarray(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), self._transforms[1:-1], issorted=False)

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
    func = function.polyfunc(coeffs, dofs, ndofs, self._transforms[1:-1], issorted=False)
    if not removedofs:
      return func

    mask = numpy.ones(ndofs, dtype=bool)
    mask[list(removedofs)] = False
    return function.mask(func, mask)

  def __str__(self):
    'string representation'

    return '{}({}:{})'.format(self.__class__.__name__, self.i, self.j)

class Axis(types.Singleton):
  __slots__ = ()

class DimAxis(Axis):
  __slots__ = 'i', 'j', 'isperiodic'
  isdim = True
  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, isperiodic:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.isperiodic = isperiodic

class BndAxis(Axis):
  __slots__ = 'i', 'j', 'ibound', 'side'
  isdim = False
  @types.apply_annotations
  def __init__(self, i:types.strictint, j:types.strictint, ibound:types.strictint, side:bool):
    super().__init__()
    self.i = i
    self.j = j
    self.ibound = ibound
    self.side = side

class StructuredTopology(Topology):
  'structured topology'

  __slots__ = 'root', 'axes', 'nrefine', 'shape', '_bnames'
  __cache__ = 'elements', '_transform', '_opposite', 'connectivity', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, root:transform.stricttransformitem, axes:types.tuple[types.strict[Axis]], nrefine:types.strictint=0, bnames:types.tuple[types.strictstr]=None):
    'constructor'

    self.root = root
    self.axes = axes
    self.nrefine = nrefine
    self.shape = tuple(axis.j - axis.i for axis in self.axes if axis.isdim)
    if bnames is None:
      assert len(self.axes) <= 3
      bnames = ('left', 'right'), ('bottom', 'top'), ('front', 'back')
      bnames = itertools.chain.from_iterable(n for axis, n in zip(self.axes, bnames) if axis.isdim and not axis.isperiodic)
    self._bnames = tuple(bnames)
    assert len(self._bnames) == sum(2 for axis in self.axes if axis.isdim and not axis.isperiodic)
    assert all(isinstance(bname,str) for bname in self._bnames)
    super().__init__(len(self.shape))

  def __iter__(self):
    reference = util.product(element.getsimplex(1 if axis.isdim else 0) for axis in self.axes)
    return (element.Element(reference, trans, opp, oriented=True) for trans, opp in zip(self._transform.flat, self._opposite.flat))

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
        start, stop, stride = s.indices(axis.j - axis.i)
        assert stride == 1
        assert stop > start
        if start > 0 or stop < axis.j - axis.i:
          axis = DimAxis(axis.i+start, axis.i+stop, isperiodic=False)
        idim += 1
      axes.append(axis)
    return StructuredTopology(self.root, axes, self.nrefine, bnames=self._bnames)

  @property
  def elements(self):
    return tuple(self)

  @property
  def periodic(self):
    dimaxes = (axis for axis in self.axes if axis.isdim)
    return tuple(idim for idim, axis in enumerate(dimaxes) if axis.isdim and axis.isperiodic)

  @staticmethod
  def mktransforms(axes, root, nrefine):
    assert nrefine >= 0

    updim = []
    rmdims = numpy.zeros(len(axes), dtype=bool)
    for order, side, idim in sorted((axis.ibound, axis.side, idim) for idim, axis in enumerate(axes) if not axis.isdim):
      ref = util.product(element.getsimplex(0 if rmdim else 1) for rmdim in rmdims)
      iedge = (idim - rmdims[:idim].sum()) * 2 + 1 - side
      updim.append(ref.edge_transforms[iedge])
      rmdims[idim] = True

    grid = [numpy.arange(axis.i>>nrefine, ((axis.j-1)>>nrefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>nrefine]) for axis in axes]
    indices = numeric.broadcast(*numeric.ix(grid))
    transforms = numeric.asobjvector([transform.Shift(numpy.array(index, dtype=float))] for index in log.iter('elem', indices, indices.size)).reshape(indices.shape)

    if nrefine:
      scales = numeric.asobjvector([trans] for trans in (element.LineReference()**len(axes)).child_transforms).reshape((2,)*len(axes))
      for irefine in log.range('level', nrefine-1, -1, -1):
        offsets = numpy.array([r[0] for r in grid])
        grid = [numpy.arange(axis.i>>irefine,((axis.j-1)>>irefine)+1) if axis.isdim else numpy.array([(axis.i-1 if axis.side else axis.j)>>irefine]) for axis in axes]
        A = transforms[numpy.broadcast_arrays(*numeric.ix(r//2-o for r, o in zip(grid, offsets)))]
        B = scales[numpy.broadcast_arrays(*numeric.ix(r%2 for r in grid))]
        transforms = A + B

    shape = tuple(axis.j - axis.i for axis in axes if axis.isdim)
    return numeric.asobjvector(transform.canonical([root] + trans + updim) for trans in log.iter('canonical', transforms.flat)).reshape(shape)

  @property
  @log.title
  def _transform(self):
    return self.mktransforms(self.axes, self.root, self.nrefine)

  @property
  @log.title
  def _opposite(self):
    nbounds = len(self.axes) - self.ndims
    if nbounds == 0:
      return self._transform
    axes = [BndAxis(axis.i, axis.j, axis.ibound, not axis.side) if not axis.isdim and axis.ibound==nbounds-1 else axis for axis in self.axes]
    return self.mktransforms(axes, self.root, self.nrefine)

  @property
  def structure(self):
    warnings.deprecation('topology.structure will be removed in future')
    reference = util.product(element.getsimplex(1 if axis.isdim else 0) for axis in self.axes)
    return numeric.asobjvector(element.Element(reference, trans, opp, oriented=True) for trans, opp in numpy.broadcast(self._transform, self._opposite)).reshape(self.shape)

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
    return connectivity.reshape(len(self), self.ndims*2)

  @property
  def boundary(self):
    'boundary'

    nbounds = len(self.axes) - self.ndims
    btopo = EmptyTopology(self.ndims-1)
    jdim = 0
    for idim, axis in enumerate(self.axes):
      if not axis.isdim or axis.isperiodic:
        continue
      btopos = [
        StructuredTopology(
          root=self.root,
          axes=self.axes[:idim] + (BndAxis(n,n if not axis.isperiodic else 0,nbounds,side),) + self.axes[idim+1:],
          nrefine=self.nrefine,
          bnames=self._bnames[:jdim*2]+self._bnames[jdim*2+2:])
        for side, n in enumerate((axis.i,axis.j)) ]
      btopo |= UnionTopology(btopos, self._bnames[jdim*2:jdim*2+2])
      jdim += 1
    return btopo

  @property
  def interfaces(self):
    'interfaces'

    assert self.ndims > 0, 'zero-D topology has no interfaces'
    itopos = []
    nbounds = len(self.axes) - self.ndims
    for idim, axis in enumerate(self.axes):
      if not axis.isdim:
        continue
      bndprops = [BndAxis(i, i, ibound=nbounds, side=True) for i in range(axis.i+1, axis.j)]
      if axis.isperiodic:
        assert axis.i == 0
        bndprops.append(BndAxis(axis.j, 0, ibound=nbounds, side=True))
      itopos.append(EmptyTopology(self.ndims-1) if not bndprops
                else UnionTopology(StructuredTopology(self.root, self.axes[:idim] + (axis,) + self.axes[idim+1:], self.nrefine) for axis in bndprops))
    assert len(itopos) == self.ndims
    return UnionTopology(itopos, names=['dir{}'.format(idim) for idim in range(self.ndims)])

  def _basis_spline(self, degree, knotvalues=None, knotmultiplicities=None, periodic=None):
    'spline with structure information'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = [degree]*self.ndims

    assert len(degree)==self.ndims

    if knotvalues is None:
      knotvalues = [None]*self.ndims

    if knotmultiplicities is None:
      knotmultiplicities = [None]*self.ndims

    vertex_structure = numpy.array(0)
    stdelems = []
    dofshape = []
    slices = []
    cache = {}
    for idim in range(self.ndims):
      p = degree[idim]
      n = self.shape[idim]
      isperiodic = idim in periodic

      k = knotvalues[idim]
      if k is None: #Defaults to uniform spacing
        k = numpy.arange(n+1)
      else:
        k = numpy.asarray(k)

      m = knotmultiplicities[idim]
      if m is None: #Defaults to open spline without internal repetitions
        m = numpy.array([p+1]+[1]*(n-1)+[p+1]) if not isperiodic else numpy.ones(n+1,dtype=int)
      else:
        m = numpy.array(m) #Make copy to prevent overwriting of data

      assert min(m)>0 and max(m)<=p+1, 'Incorrect multiplicity encountered'
      assert len(k)==len(m), 'Length mismatch between knots vector and knot multiplicities vector'
      assert len(k)==n+1, 'Knot vector size does not match the topology size'

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
          coeffs = types.frozenarray(self._localsplinebasis(lknots, p).T, copy=False)
          cache[key] = coeffs
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

  def basis_spline(self, degree, knotvalues=None, knotmultiplicities=None, periodic=None, removedofs=None):
    'spline basis'

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    coeffs, dofmap, dofshape = self._basis_spline(degree=degree, knotvalues=knotvalues, knotmultiplicities=knotmultiplicities, periodic=periodic)
    func = function.polyfunc(coeffs, dofmap, util.product(dofshape), (elem.transform for elem in self), issorted=False)
    if not any(removedofs):
      return func

    mask = numpy.ones((), dtype=bool)
    for idofs, ndofs in zip(removedofs, dofshape):
      mask = mask[...,_].repeat(ndofs, axis=-1)
      if idofs:
        mask[..., [numeric.normdim(ndofs,idof) for idof in idofs]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask(func, mask.ravel())

  @staticmethod
  def _localsplinebasis (lknots, p):

    assert numeric.isarray(lknots), 'Local knot vector should be numpy array'
    assert len(lknots)==2*p, 'Expected 2*p local knots'

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

    return types.frozenarray([Ni.coeffs for Ni in N]).T[::-1]

  def basis_discont(self, degree):
    'discontinuous shape functions'

    ref = util.product([element.LineReference()]*self.ndims)
    coeffs = [ref.get_poly_coeffs('bernstein', degree=degree)]*len(self)
    ndofs = ref.get_ndofs(degree)
    dofs = types.frozenarray(numpy.arange(ndofs*len(self), dtype=int).reshape(len(self), ndofs), copy=False)
    return function.polyfunc(coeffs, dofs, ndofs*len(self), (elem.transform for elem in self), issorted=False)

  def basis_std(self, degree, removedofs=None, periodic=None):
    'spline from vertices'

    if periodic is None:
      periodic = self.periodic

    if numeric.isint(degree):
      degree = (degree,) * self.ndims

    if removedofs == None:
      removedofs = [None] * self.ndims
    else:
      assert len(removedofs) == self.ndims

    dofshape = []
    slices = []
    vertex_structure = numpy.array(0)
    for idim in range(self.ndims):
      periodic_i = idim in periodic
      n = self.shape[idim]
      p = degree[idim]
      nd = n * p + 1
      numbers = numpy.arange(nd)
      if periodic_i and p > 0:
        numbers[-1] = numbers[0]
        nd -= 1
      vertex_structure = vertex_structure[...,_] * nd + numbers
      dofshape.append(nd)
      slices.append([slice(p*i,p*i+p+1) for i in range(n)])

    lineref = element.LineReference()
    coeffs = [functools.reduce(numeric.poly_outer_product, (lineref.get_poly_coeffs('bernstein', degree=p) for p in degree))]*len(self)
    dofs = [types.frozenarray(vertex_structure[S].ravel(), copy=False) for S in numpy.broadcast(*numpy.ix_(*slices))]
    func = function.polyfunc(coeffs, dofs, numpy.product(dofshape), self._transform.ravel(), issorted=False)
    if not any(removedofs):
      return func

    mask = numpy.ones((), dtype=bool)
    for idofs, ndofs in zip(removedofs, dofshape):
      mask = mask[...,_].repeat(ndofs, axis=-1)
      if idofs:
        mask[..., [numeric.normdim(ndofs,idof) for idof in idofs]] = False
    assert mask.shape == tuple(dofshape)
    return function.mask(func, mask.ravel())

  @property
  def refined(self):
    'refine non-uniformly'

    axes = [DimAxis(i=axis.i*2,j=axis.j*2,isperiodic=axis.isperiodic) if axis.isdim
        else BndAxis(i=axis.i*2,j=axis.j*2,ibound=axis.ibound,side=axis.side) for axis in self.axes]
    return StructuredTopology(self.root, axes, self.nrefine+1, bnames=self._bnames)

  def __str__(self):
    'string representation'

    return '{}({})'.format(self.__class__.__name__, 'x'.join(str(n) for n in self.shape))

class UnstructuredTopology(Topology):
  'unstructured topology'

  __slots__ = 'elements',
  __cache__ = 'connectivity', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, elements:types.tuple[element.strictelement]):
    self.elements = elements
    assert all(elem.ndims == ndims for elem in self.elements)
    super().__init__(ndims)

  @property
  @log.title
  def connectivity(self):
    edges = {}
    connectivity = []
    for ielem, elem in log.enumerate('elem', self):
      connectivity.append(-numpy.ones(elem.nedges, dtype=int))
      for iedge, belem in enumerate(elem.edges):
        belemcoords = belem.vertices
        edgekey = tuple(sorted(belemcoords))
        try:
          jelem, jedge = edges.pop(edgekey)
        except KeyError:
          edges[edgekey] = ielem, iedge
        else:
          # TODO assert transformation equivalence
          connectivity[ielem][iedge] = jelem
          connectivity[jelem][jedge] = ielem
    return tuple(connectivity)

  @property
  def boundary(self):
    elements = tuple(elem.edge(iedge) for elem, ioppelems in zip(self, self.connectivity) for iedge in numpy.where(numpy.equal(ioppelems, -1))[0])
    return UnstructuredTopology(self.ndims-1, elements)

  @property
  def interfaces(self):
    seen = set()
    elements = []
    for ielem, ioppelems in enumerate(self.connectivity):
      elem = self.elements[ielem]
      for iedge, ioppelem in enumerate(ioppelems):
        if ioppelem == -1:
          continue
        try:
          seen.remove((ielem, iedge))
        except KeyError:
          ioppedge = tuple(self.connectivity[ioppelem]).index(ielem)
          oppelem = self.elements[ioppelem]
          elements.append(elem.edge(iedge).withopposite(oppelem.edge(ioppedge), oriented=False))
          seen.add((ioppelem, ioppedge))
    assert not seen
    return UnstructuredTopology(self.ndims-1, elements)

  def basis_bubble(self):
    'bubble from vertices'

    assert self.ndims == 2
    coeffs = types.frozenarray([[[1,-1, 0, 0], [-1,  0,  0, 0], [0,  0, 0, 0], [0, 0, 0, 0]],
                                [[0, 0, 0, 0], [ 1,  0,  0, 0], [0,  0, 0, 0], [0, 0, 0, 0]],
                                [[0, 1, 0, 0], [ 0,  0,  0, 0], [0,  0, 0, 0], [0, 0, 0, 0]],
                                [[0, 0, 0, 0], [ 0, 27,-27, 0], [0,-27, 0, 0], [0, 0, 0, 0]]])

    nmap = []
    dofmap = {}
    for ielem, elem in enumerate(self):
      assert isinstance(elem.reference, element.TriangleReference)
      assert elem.nverts == 3
      dofs = numpy.empty(4, dtype=int)
      for i, v in enumerate(elem.vertices):
        dof = dofmap.get(v)
        if dof is None:
          dof = len(self) + len(dofmap)
          dofmap[v] = dof
        dofs[i] = dof
      dofs[3] = ielem
      nmap.append(types.frozenarray(dofs, copy=False))
    ndofs = len(self)+len(dofmap)

    return function.polyfunc([coeffs]*len(self), nmap, ndofs, (elem.transform for elem in self), issorted=False)

  def basis_spline(self, degree):
    assert degree == 1
    return self.basis('std', degree)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    assert numeric.isint(degree) and degree >= 0
    coeffs = []
    nmap = []
    ndofs = 0
    for elem in self:
      elemcoeffs = elem.reference.get_poly_coeffs('bernstein', degree=degree)
      coeffs.append(elemcoeffs)
      nmap.append(types.frozenarray(ndofs + numpy.arange(len(elemcoeffs)), copy=False))
      ndofs += len(elemcoeffs)
    degrees = set(n-1 for c in coeffs for n in c.shape[1:])
    return function.polyfunc(coeffs, nmap, ndofs, (elem.transform for elem in self), issorted=False)

  def _basis_c0_structured(self, name, degree):
    'C^0-continuous shape functions with lagrange stucture'

    assert numeric.isint(degree) and degree >= 0

    if degree == 0:
      raise ValueError('Cannot build a C^0-continuous basis of degree 0.  Use basis \'discont\' instead.')

    nlocaldofs = 0
    elem_slices = []
    coeffs = []
    for elem in self:
      elem_coeffs = elem.reference.get_poly_coeffs(name, degree=degree)
      coeffs.append(elem_coeffs)
      elem_slices.append(slice(nlocaldofs, nlocaldofs+len(elem_coeffs)))
      nlocaldofs += len(elem_coeffs)
    dofmap = -numpy.ones([nlocaldofs], dtype=int)

    for ielem, elem in enumerate(self):
      # Loop over all neighbors of elem and merge dofs.
      for iedge, jelem in enumerate(self.connectivity[ielem]):
        if jelem < 0:
          continue
        jedge = self.connectivity[jelem].tolist().index(ielem)
        #if ielem < jelem or ielem == jelem and iedge < jedge:
        #  continue
        idofs = elem.reference.get_edge_dofs(degree, iedge)
        verts = tuple(elem.edge(iedge).vertices)
        neighbor = self.elements[jelem]
        neighbor_verts = tuple(neighbor.edge(jedge).vertices)
        jdofs = neighbor.reference.get_edge_dofs(degree, jedge)
        #jdofs = jdofs[neighbor.reference.edges[jedge][1].get_dof_transpose_map(degree, tuple(map(neighbor_verts.index, verts)))]
        foo = neighbor.reference.edges[jedge][1].get_dof_transpose_map(degree, tuple(map(neighbor_verts.index, verts)))
        for idof, j in zip(idofs, foo):
          ridof = idof = elem_slices[ielem].start+idof
          # Resolve idof.
          while dofmap[ridof] >= 0:
            ridof = dofmap[ridof]
          # Resolve jdof.
          rjdof = jdof = elem_slices[jelem].start+jdofs[j]
          while dofmap[rjdof] >= 0:
            rjdof = dofmap[rjdof]
          if ridof != rjdof:
            dofmap[max(ridof,rjdof)] = min(ridof,rjdof)
          if ridof != idof:
            dofmap[idof] = min(ridof,rjdof)
    # Assign dof numbers.
    ndofs = 0
    for i in range(len(dofmap)):
      if dofmap[i] < 0:
        dofmap[i] = ndofs
        ndofs += 1
      else:
        dofmap[i] = dofmap[dofmap[i]]

    dofs = tuple(types.frozenarray(dofmap[s]) for s in elem_slices)
    return function.polyfunc(coeffs, dofs, ndofs, (elem.transform for elem in self), issorted=False)

  def basis_lagrange(self, degree):
    'lagrange shape functions'
    return self._basis_c0_structured('lagrange', degree)

  def basis_bernstein(self, degree):
    'bernstein shape functions'
    return self._basis_c0_structured('bernstein', degree)

  basis_std = basis_bernstein

class GmshTopology(UnstructuredTopology):
  'gmsh topology'

  __slots__ = '_connectivity', '_points', 'bgroups', 'igroups', 'pgroups', 'vgroups', 'geom'

  def _script_or_filename(arg):
    if isinstance(arg, pathlib.Path):
      with arg.open() as f:
        return f.read()
    elif isinstance(arg, str):
      if arg.startswith('$MeshFormat'):
        return arg
      else:
        with open(arg) as f:
          return f.read()
    else:
      raise ValueError("expected the contents of a Gmsh MSH file (as 'str') or a filename (as 'str' or 'pathlib.Path') but got {!r}".format(arg))

  @types.apply_annotations
  def __init__(self, script:_script_or_filename, name:types.strictstr=None):

    # split sections
    sections = {}
    lines = iter(script.splitlines())
    for line in lines:
      line = line.strip()
      assert line[0]=='$'
      sname = line[1:]
      slines = []
      for sline in lines:
        sline = sline.strip()
        if sline=='$End'+sname:
          break
        slines.append(sline)
      sections[sname] = slines

    # discard section MeshFormat
    sections.pop('MeshFormat', None)

    # parse section PhysicalNames
    PhysicalNames = sections.pop('PhysicalNames', [0])
    assert int(PhysicalNames[0]) == len(PhysicalNames)-1
    tagmapbydim = {}, {}, {}, {} # tagid->tagname dictionary
    for line in PhysicalNames[1:]:
      nd, tagid, tagname = line.split(' ', 2)
      nd = int(nd)
      tagmapbydim[nd][int(tagid)] = tagname.strip('"')

    # determine the dimension of the mesh
    ndims = 2 if len(tagmapbydim[3])==0 else 3
    if ndims==3 and len(tagmapbydim[1])>0:
      raise NotImplementedError('Physical line groups are not supported in volumetric meshes')

    # parse section Nodes
    Nodes = sections.pop('Nodes')
    assert int(Nodes[0]) == len(Nodes)-1
    nodes = numpy.empty((len(Nodes)-1,3))
    nodemap = {}
    for i, line in enumerate(Nodes[1:]):
      words = line.split()
      nodemap[int(words[0])] = i
      nodes[i] = [float(n) for n in words[1:]]
    assert not numpy.isnan(nodes).any()
    if ndims==2:
      assert numpy.all(nodes[:,2]) == 0, 'Non-zero z-coordinates found in 2D mesh.'
      nodes = nodes[:,:2]

    # parse section Elements
    Elements = sections.pop('Elements')
    assert int(Elements[0]) == len(Elements)-1
    inodesbydim = [],[],[],[] # nelems-list of 4-tuples of node numbers
    etypesbydim = [],[],[],[]
    tagnamesbydim = {},{},{},{} # tag->ielems dictionary
    etype2nd = {15:0, 1:1, 2:2, 4:3, 8:1, 9:2}
    etype2indices = {15:[0], 1:[0,1], 2:[0,1,2], 4:[0,1,2,3], 8:[0,2,1], 9:[0,3,1,5,4,2]}
    for line in Elements[1:]:
      words = line.split()
      etype = int(words[1])
      nd = etype2nd[etype]
      ntags = int(words[2])
      assert ntags >= 1
      tagname = tagmapbydim[nd][int(words[3])]
      inodes = tuple(nodemap[int(nodeid)] for nodeid in words[3+ntags:])
      if not inodesbydim[nd] or inodesbydim[nd][-1] != inodes: # multiple tags are repeated in consecutive lines
        inodesbydim[nd].append(inodes)
        etypesbydim[nd].append(etype )
      tagnamesbydim[nd].setdefault(tagname, []).append(len(inodesbydim[nd])-1)
    inodesbydim = [numpy.array(e) if e else numpy.empty((0,nd), dtype=int) for nd, e in enumerate(inodesbydim)]
    if tagnamesbydim[ndims]:
      log.info('topology groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagnamesbydim[ndims].items()))

    # check orientation
    vinodes = inodesbydim[ndims] # save for geom
    elemnodes = nodes[vinodes] # nelems x ndims+1 x ndims
    elemareas = numpy.linalg.det(elemnodes[:,1:ndims+1] - elemnodes[:,:1])
    assert numpy.all(elemareas > 0)

    vetype = etypesbydim[ndims][0]
    assert all(etype==vetype for etype in etypesbydim[ndims]), 'all interior elements should be of the same element type'

    # parse section Periodic
    Periodic = sections.pop('Periodic', [0])
    nperiodic = int(Periodic[0])
    renumber = numpy.arange(len(nodes))
    master = numpy.ones(len(nodes), dtype=bool)
    n = 0
    for line in Periodic[1:]:
      words = line.split()
      if len(words) == 1:
        n = int(words[0]) # initialize for counting backwards
      elif len(words) == 2:
        islave = nodemap[int(words[0])]
        imaster = nodemap[int(words[1])]
        renumber[islave] = renumber[imaster]
        master[islave] = False
        n -= 1
      else:
        assert len(words) == 3 # discard content
        assert n == 0 # check if batch of slave/master nodes matches announcement
        nperiodic -= 1
    assert nperiodic == 0 # check if number of periodic blocks matches announcement
    assert n == 0 # check if last batch of slave/master nodes matches announcement
    renumber = master.cumsum()[renumber]-1
    inodesbydim = [renumber[e] for e in inodesbydim]

    # warn about unused sections
    for section in sections:
      warnings.warn('section {!r} defined but not used'.format(section))

    # create elements
    simplexref = element.getsimplex(ndims)
    elements = [element.Element(simplexref, [transform.MapTrans(linear=[[-1,-1],[1,0],[0,1]] if ndims==2 else [[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]], offset=[1,0,0] if ndims==2 else [1,0,0,0], vertices=inodes[:ndims+1] if not name else [name+str(inode) for inode in inodes[:ndims+1]])])
      for ielem, inodes in log.enumerate('elem', inodesbydim[ndims])]
    # create connectivity matrix
    connectivity = -numpy.ones((len(inodesbydim[ndims]),ndims+1), dtype=int)
    edges = {} # binodes->(ielem,iedge) dictionary
    econn = [[1,2],[2,0],[0,1]] if ndims==2 else [[1,2,3],[0,3,2],[0,1,3],[0,2,1]] # consistent with simplex.edge_transforms
    for ielem, inodes in log.enumerate('elem', inodesbydim[ndims]):
      for iedge, binodes in enumerate([inodes[ec] for ec in econn]):
        key = tuple(sorted(binodes))
        try:
          jelem, jedge = edges[key]
        except KeyError:
          edges[key] = ielem, iedge
        else:
          connectivity[ielem][iedge] = jelem
          connectivity[jelem][jedge] = ielem

    # separate boundary and interface elements by tag
    tagsbelems = {}
    tagsielems = {}
    for name, ibelems in tagnamesbydim[ndims-1].items():
      for ibelem in ibelems:
        binodes = inodesbydim[ndims-1][ibelem][:ndims]
        ielem, iedge = edges[tuple(sorted(binodes))]
        elem = elements[ielem].edge(iedge)
        ioppelem = connectivity[ielem][iedge]
        if ioppelem == -1:
          tagsbelems.setdefault(name, []).append(elem)
        else:
          ioppedge = tuple(connectivity[ioppelem]).index(ielem)
          tagsielems.setdefault(name, []).append(elem.withopposite(elements[ioppelem].edge(ioppedge)))
    if tagsbelems:
      log.info('boundary groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsbelems.items()))
    if tagsielems:
      log.info('interface groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagsielems.items()))

    # create points topology and separate point elements by tag
    tagspelems = {}
    if tagnamesbydim[0]: # point gorups defined
      pelems = {inodes[0]: [] for inodes in inodesbydim[0]}
      pref = element.getsimplex(0)
      for inodes, elem in zip(inodesbydim[ndims], elements):
        for ivertex, inode in enumerate(inodes):
          if inode in pelems:
            offset = elem.reference.vertices[ivertex]
            trans = elem.transform + (transform.Matrix(linear=numpy.zeros(shape=(ndims,0)), offset=offset),)
            pelems[inode].append(element.Element(pref, trans))
      for name, ipelems in tagnamesbydim[0].items():
        tagspelems[name] = [pelem for ipelem in ipelems for inode in inodesbydim[0][ipelem] for pelem in pelems[inode]]
      self._points = UnstructuredTopology(0, sum(pelems.values(), []))
      log.info('points groups:', ', '.join('{} (#{})'.format(n,len(e)) for n, e in tagspelems.items()))
    else:
      self._points = None

    # create boundary, interface, point, volume groups
    self.bgroups={tagname: UnstructuredTopology(ndims-1, tagbelems) for tagname, tagbelems in tagsbelems.items()}
    self.igroups={tagname: UnstructuredTopology(ndims-1, tagielems) for tagname, tagielems in tagsielems.items()}
    self.pgroups={tagname: UnstructuredTopology(0, tagpelems) for tagname, tagpelems in tagspelems.items()}
    self.vgroups = {}
    for name, ielems in tagnamesbydim[ndims].items():
      if len(ielems) == len(elements):
        self.vgroups[name] = ...
      elif ielems:
        refs = numpy.array([None] * len(elements), dtype=object)
        refs[ielems] = simplexref
        self.vgroups[name] = tuple(refs)

    super().__init__(ndims, elements)
    self._connectivity = connectivity

    # create geometry
    dofs = tuple(map(types.frozenarray, vinodes[:,etype2indices[vetype]]))
    coeffs = [simplexref.get_poly_coeffs('lagrange', degree=2 if vetype in (8,9) else 1)] * len(dofs)
    basis = function.polyfunc(coeffs, dofs, len(nodes), (elem.transform for elem in elements), issorted=False)
    self.geom = (basis[:,_] * nodes).sum(0)

  @property
  def connectivity(self):
    return self._connectivity

  def getitem(self, item):
    if not isinstance(item, str):
      return super().getitem(item)
    try:
      itemrefs = self.vgroups[item]
    except KeyError:
      return super().getitem(item)
    else:
      return SubsetTopology(self, itemrefs) if itemrefs else super().getitem(item)

  @property
  def boundary(self):
    return super().boundary.withgroups(self.bgroups)

  @property
  def interfaces(self):
    return super().interfaces.withgroups(self.igroups)

  @property
  def points(self):
    return self._points.withgroups(self.pgroups)

class UnionTopology(Topology):
  'grouped topology'

  __slots__ = '_topos', '_names'
  __cache__ = 'elements',

  @types.apply_annotations
  def __init__(self, topos:types.tuple[stricttopology], names:types.tuple[types.strictstr]=()):
    self._topos = topos
    self._names = tuple(names)[:len(self._topos)]
    assert len(set(self._names)) == len(self._names), 'duplicate name'
    ndims = self._topos[0].ndims
    assert all(topo.ndims == ndims for topo in self._topos)
    super().__init__(ndims)

  def getitem(self, item):
    topos = [topo if name == item else topo.getitem(item) for topo, name in itertools.zip_longest(self._topos, self._names)]
    return functools.reduce(operator.or_, topos, EmptyTopology(self.ndims))

  def __or__(self, other):
    if not isinstance(other, UnionTopology):
      return UnionTopology(self._topos + (other,), self._names)
    return UnionTopology(self._topos[:len(self._names)] + other._topos + self._topos[len(self._names):], self._names + other._names)

  @property
  def elements(self):
    elements = []
    for trans, elems in util.gather((elem.transform, elem) for topo in self._topos for elem in topo):
      if len(elems) == 1:
        elements.append(elems[0])
      else:
        refs = [elem.reference for elem in elems]
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
        unionref, = refs
        opposite = elems[0].opposite
        assert all(elem.opposite == opposite for elem in elems[1:])
        elements.append(element.Element(unionref, trans, opposite, oriented=True))
    return elements

  @property
  def refined(self):
    return UnionTopology([topo.refined for topo in self._topos], self._names)

class SubsetTopology(Topology):
  'trimmed'

  __slots__ = 'refs', 'basetopo', 'newboundary'
  __cache__ = 'connectivity', 'elements', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, refs:types.tuple, newboundary=None):
    if newboundary is not None:
      assert isinstance(newboundary, str) or isinstance(newboundary, Topology) and newboundary.ndims == basetopo.ndims-1
    assert len(refs) == len(basetopo)
    self.refs = refs
    self.basetopo = basetopo
    self.newboundary = newboundary
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return self.basetopo.getitem(item).subset(self.elements, strict=False)

  def __rsub__(self, other):
    if self.basetopo == other:
      refs = [elem.reference - ref for elem, ref in zip(self.basetopo, self.refs)]
      return SubsetTopology(self.basetopo, refs, ~self.newboundary if isinstance(self.newboundary,Topology) else self.newboundary)
    return super().__rsub__(other)

  def __or__(self, other):
    if not isinstance(other, SubsetTopology) or self.basetopo != other.basetopo:
      return super().__or__(other)
    refs = [ref1 | ref2 for ref1, ref2 in zip(self.refs, other.refs)]
    if all(elem.reference == ref for elem, ref in zip(self.basetopo, refs)):
      return self.basetopo
    return SubsetTopology(self.basetopo, refs) # TODO boundary

  @property
  def connectivity(self):
    mask = numpy.array([bool(ref) for ref in self.refs] + [False]) # trailing false serves to map -1 to -1
    renumber = numpy.cumsum(mask)-1
    renumber[~mask] = -1
    connectivity = tuple(tuple(renumber[ioppelems[iedge]] if edgeref and iedge < len(ioppelems) else -1 for iedge, edgeref in enumerate(ref.edge_refs))
      for ref, ioppelems in zip(self.refs, self.basetopo.connectivity) if ref)
    return connectivity

  @property
  def elements(self):
    return tuple(element.Element(ref, elem.transform, elem.opposite) for elem, ref in zip(self.basetopo, self.refs) if ref)

  @property
  def refined(self):
    elems = [child for elem in self for child in elem.children if child]
    return self.basetopo.refined.subset(elems, self.newboundary.refined if isinstance(self.newboundary,Topology) else self.newboundary, strict=True)

  @property
  @log.title
  def boundary(self):
    brefs = [element.EmptyReference(self.ndims-1)] * len(self.basetopo.boundary) # subset of original boundary
    newbelems = [] # newly formed boundary elements
    connectivity = self.basetopo.connectivity
    elements = self.basetopo.elements
    baseboundary = self.basetopo.boundary
    for ielem, ioppelems in enumerate(connectivity):
      ref = self.refs[ielem]
      if not ref:
        continue
      elem = elements[ielem]
      newbelems.extend(element.Element(edge, elem.transform + (etrans,), elem.transform + (etrans.flipped,)) for etrans, edge in ref.edges[elem.reference.nedges:])
      for iedge, ioppelem in enumerate(ioppelems):
        bref = ref.edge_refs[iedge]
        if not bref:
          continue
        if ioppelem == -1:
          index = baseboundary.edict[transform.canonical(elem.transform + (ref.edge_transforms[iedge],))]
          brefs[index] = bref # by construction, bref must be equal or subset of original
        else:
          ioppedge = tuple(connectivity[ioppelem]).index(ielem)
          oppref = self.refs[ioppelem]
          if oppref:
            bref -= oppref.edge_refs[ioppedge]
          if bref:
            newbelems.append(element.Element(bref, elem.transform + (ref.edge_transforms[iedge],), elements[ioppelem].edge(ioppedge).transform))
    origboundary = SubsetTopology(baseboundary, brefs)
    trimboundary = OrientedGroupsTopology(self.newboundary if isinstance(self.newboundary,Topology) else self.basetopo.interfaces, newbelems)
    return UnionTopology([trimboundary, origboundary], names=[self.newboundary] if isinstance(self.newboundary,str) else [])

  @property
  @log.title
  def interfaces(self):
    irefs = [element.EmptyReference(self.ndims-1)] * len(self.basetopo.interfaces) # subset of original interfaces
    connectivity = self.basetopo.connectivity
    elements = self.basetopo.elements
    baseinterfaces = self.basetopo.interfaces
    for ielem, ioppelems in enumerate(connectivity):
      ref = self.refs[ielem]
      if not ref:
        continue # edge is empty
      elem = elements[ielem]
      for iedge, ioppelem in enumerate(ioppelems):
        if ioppelem == -1:
          continue # edge lies on the boundary
        oppref = self.refs[ioppelem]
        if not oppref:
          continue # edge is empty
        index = baseinterfaces.edict.get(transform.canonical(elem.transform + (ref.edge_transforms[iedge],)))
        if index is None:
          continue # edge is not oriented with an interface; rely on connectivity to also yield flipped element
        ioppedge = tuple(connectivity[ioppelem]).index(ielem)
        irefs[index] = ref.edge_refs[iedge] & oppref.edge_refs[ioppedge]
    return SubsetTopology(baseinterfaces, irefs)

  @log.title
  def basis(self, name, *args, **kwargs):
    if isinstance(self.basetopo, HierarchicalTopology):
      warnings.warn('basis may be linearly dependent; a linearly indepent basis is obtained by trimming first, then creating hierarchical refinements')
    basis = self.basetopo.basis(name, *args, **kwargs)
    return self.prune_basis(basis)

class OrientedGroupsTopology(UnstructuredTopology):
  'unstructured topology with undirected semi-overlapping basetopology'

  __slots__ = 'basetopo',

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, elems:types.tuple[element.strictelement]):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims, elems)

  def getitem(self, item):
    elements = []
    for elem in self.basetopo.getitem(item):
      try:
        ielem, tail = transform.lookup_item(elem.transform, self.edict)
      except KeyError:
        elem = elem.flipped
        try:
          ielem, tail = transform.lookup_item(elem.transform, self.edict)
        except KeyError:
          continue
      if tail:
        raise NotImplementedError
      ref = self.elements[ielem].reference & elem.reference
      elements.append(element.Element(ref, elem.transform, elem.opposite, oriented=True))
    return UnstructuredTopology(self.ndims, elements)

class RefinedTopology(Topology):
  'refinement'

  __slots__ = 'basetopo',
  __cache__ = 'elements', 'boundary'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology):
    self.basetopo = basetopo
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return self.basetopo.getitem(item).refined

  @property
  def elements(self):
    return tuple([child for elem in self.basetopo for child in elem.children])

  @property
  def boundary(self):
    return self.basetopo.boundary.refined

class TrimmedTopologyItem(Topology):
  'trimmed topology item'

  __slots__ = 'basetopo', 'refdict'
  __cache__ = 'elements',

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, refdict):
    self.basetopo = basetopo
    self.refdict = refdict
    super().__init__(basetopo.ndims)

  @property
  def elements(self):
    elements = []
    for elem in self.basetopo:
      ref = elem.reference & self.refdict[elem.transform]
      if ref:
        elements.append(element.Element(ref, elem.transform, elem.opposite, oriented=True))
    return elements

class TrimmedTopologyBoundaryItem(Topology):
  'trimmed topology boundary item'

  __slots__ = 'btopo', 'trimmed', 'othertopo'
  __cache__ = 'elements',

  @types.apply_annotations
  def __init__(self, btopo:stricttopology, trimmed, othertopo):
    self.btopo = btopo
    self.trimmed = trimmed
    self.othertopo = othertopo
    super().__init__(btopo.ndims)

  @property
  def elements(self):
    belems = [elem for elem in self.trimmed if elem.opposite in self.btopo.edict]
    if self.othertopo:
      belems.extend(self.othertopo)
    return belems

class HierarchicalTopology(Topology):
  'collection of nested topology elments'

  __slots__ = 'basetopo', 'allelements', '_precise'
  __cache__ = 'elements', 'levels', 'refined', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, basetopo:stricttopology, allelements:types.tuple[element.strictelement], precise:bool):
    'constructor'

    assert not isinstance(basetopo, HierarchicalTopology)
    self.basetopo = basetopo
    self.allelements = allelements
    self._precise = precise
    super().__init__(basetopo.ndims)

  def getitem(self, item):
    return self.basetopo.getitem(item).hierarchical(self.allelements, precise=False)

  def hierarchical(self, elements, precise=False):
    return self.basetopo.hierarchical(elements, precise)

  @property
  def elements(self):
    if self._precise:
      return self.allelements
    itemelems = []
    for elem in self.allelements:
      try:
        ielem, tail = transform.lookup_item(elem.transform, self.basetopo.edict)
      except KeyError:
        continue
      itemelem = self.basetopo.elements[ielem]
      ref = itemelem.reference
      for trans in tail:
        index = ref.child_transforms.index(trans)
        ref = ref.child_refs[index]
        if not ref:
          break
      else:
        ref &= elem.reference
        if ref:
          itemelems.append(element.Element(ref, elem.transform, elem.opposite, oriented=True))
    return itemelems

  @property
  @log.title
  def levels(self):
    levels = [self.basetopo]
    for elem in self:
      try:
        ielem, tail = transform.lookup_item(elem.transform, self.basetopo.edict)
      except KeyError:
        raise Exception('element is not a refinement of basetopo')
      else:
        nrefine = len(tail)
        while nrefine >= len(levels):
          levels.append(levels[-1].refined)
        assert elem.transform in levels[nrefine].edict, 'element is not a refinement of basetopo'
    return tuple(levels)

  @property
  def refined(self):
    elements = [child for elem in self for child in elem.children]
    return self.basetopo.hierarchical(elements, precise=True)

  @property
  @log.title
  def boundary(self):
    'boundary elements'

    basebtopo = self.basetopo.boundary
    edgepool = [edge for elem in self if transform.lookup(elem.transform, self.basetopo.border_transforms) for edge in elem.edges if edge is not None]
    belems = []
    for edge in edgepool: # superset of boundary elements
      try:
        iedge, tail = transform.lookup_item(edge.transform, basebtopo.edict)
      except KeyError:
        pass
      else:
        opptrans = basebtopo.elements[iedge].opposite + tail
        belems.append(element.Element(edge.reference, edge.transform, opptrans, oriented=True))
    return basebtopo.hierarchical(belems, precise=True)

  @property
  @log.title
  def interfaces(self):
    'interfaces'

    # Build a lookup table for level and element indices given elements in this
    # topology.
    elem_index_level = {
      elem: (ielem, ilevel)
      for ilevel, level in enumerate(self.levels)
      for ielem, elem in enumerate(level)
    }
    oriented = isinstance(self.basetopo, StructuredTopology)
    edict = self.edict
    interfaces = []
    for elem in log.iter('elem', self):
      # Get `level`, element number at `level` of `elem`.
      ielem, ilevel = elem_index_level[elem]
      level = self.levels[ilevel]
      # Loop over neighbours of `elem`.
      for ielemedge, ineighbor in enumerate(level.connectivity[ielem]):
        if ineighbor < 0:
          # Not an interface.
          continue
        neighbor = level.elements[ineighbor]
        # Lookup `neighbor` (from the same `level` as `elem`) in this topology.
        head, tail = transform.lookup(neighbor.transform, edict) or (None, None)
        if not head:
          # `neighbor` not found, hence refinements of `neighbor` are present.
          # The interface of this edge will be added when we encounter the
          # refined elements.
          continue
        # Find the edge of `neighbor` between `neighbor` and `elem`.
        ineighboredge = tuple(level.connectivity[ineighbor]).index(ielem)
        if not tail and (ielem, ielemedge) > (ineighbor, ineighboredge):
          # `neighbor` itself, not a parent of, exists in this topology (`tail`
          # is empty).  To make sure we add this interface only once we
          # continue here if the current element has a higher index (in
          # `level`) than the neighbor (or a higher edge number if the elements
          # are equal, which might occur when there is only one element in a
          # periodic dimension).
          continue
        # Create and add the interface between `elem` and `neighbor`.
        elemedge = elem.edges[ielemedge]
        neighboredge = neighbor.edges[ineighboredge]
        interfaces.append(element.Element(elemedge.reference, elemedge.transform, neighboredge.transform, oriented=oriented))
    return UnstructuredTopology(self.ndims-1, interfaces)

  @log.title
  def basis(self, name, *args, **kwargs):
    'build hierarchical function space'

    # The law: a basis function is retained if all elements of self can
    # evaluate it through cascade, and at least one element of self can
    # evaluate it directly.

    # Procedure: per refinement level, track which basis functions have at
    # least one supporting element coinsiding with self ('touched') and no
    # supporting element finer than self ('supported').

    dofs_coeffs = []
    renumber = []
    supports = []
    length = 0

    for topo in log.iter('level', self.levels):

      basis = topo.basis(name, *args, **kwargs) # shape functions for current level

      supported = numpy.ones(len(basis), dtype=bool) # True if dof is fully contained in self or parents
      touchtopo = numpy.zeros(len(basis), dtype=bool) # True if dof touches at least one elem in self

      (axes,func), = function.blocks(basis)
      dofmap, = axes
      if isinstance(func, function.Polyval):
        coeffs = func.coeffs
        assert coeffs.ndim == 1+self.ndims
      elif func.isconstant:
        assert func.ndim == 1
        coeffs = func[(slice(None),*(_,)*self.ndims)]
      else:
        raise ValueError

      for elem in topo:
        trans = elem.transform
        idofs, = dofmap.eval(_transforms=(elem.transform, elem.opposite))
        if trans in self.edict:
          touchtopo[idofs] = True
        elif transform.lookup(trans, self.edict):
          supported[idofs] = False

      support = supported & touchtopo
      supports.append(support)
      cumsum_support = numpy.cumsum(support)
      renumber.append(cumsum_support+(length-1))
      length += cumsum_support[-1]
      dofs_coeffs.append(function.Tuple((dofmap, coeffs)))

    dofs = []
    coeffs = []
    transforms = tuple(sorted(elem.transform for elem in self))
    for trans in transforms:
      hcoeffs = []
      hdofs = []
      ibase, tail = transform.lookup_item(trans, self.basetopo.edict)
      for ilevel in range(len(tail)+1):
        (idofs,), (icoeffs,) = dofs_coeffs[ilevel].eval(_transforms=(trans,))
        isupport = supports[ilevel][idofs]
        if not isupport.any():
          continue
        hdofs.extend(map(renumber[ilevel].__getitem__, idofs[isupport]))
        hcoeffs.extend(transform.transform_poly(tail[ilevel:], icoeffs[isupport]))
      dofs.append(types.frozenarray(hdofs))
      coeffs.append(numeric.poly_stack(hcoeffs))

    return function.polyfunc(coeffs, dofs, length, transforms, issorted=True)

class ProductTopology(Topology):
  'product topology'

  __slots__ = 'topo1', 'topo2'
  __cache__ = 'structure', 'elements', 'boundary', 'interfaces'

  @types.apply_annotations
  def __init__(self, topo1:stricttopology, topo2:stricttopology):
    self.topo1 = topo1
    self.topo2 = topo2
    super().__init__(topo1.ndims+topo2.ndims)

  def __len__(self):
    return len(self.topo1) * len(self.topo2)

  @property
  def structure(self):
    return self.topo1.structure[(...,)+(_,)*self.topo2.ndims] * self.topo2.structure

  @property
  def elements(self):
    return (numpy.array(self.topo1.elements, dtype=object)[:,_] * numpy.array(self.topo2.elements, dtype=object)[_,:]).ravel()

  def __iter__(self):
    return self.elements.flat

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
      if isinstance(arg, (list,tuple)):
        assert len(arg) == self.ndims
        return arg[:self.topo1.ndims], arg[self.topo1.ndims:]
      else:
        return arg, arg
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

  __slots__ = 'elements', 'boundary'

  def __init__(self):
    self.elements = element.Element(element.RevolutionReference(), [transform.RootTrans('angle',(1,))]),
    self.boundary = EmptyTopology(ndims=0)
    super().__init__(ndims=1)

  def basis(self, name, *args, **kwargs):
    return function.asarray([1])

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
  __cache__ = '_patchinterfaces', 'elements', 'boundary', 'interfaces', 'refined'

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

    super().__init__(self.patches[0].topo.ndims)

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

  @property
  def elements(self):
    return tuple(itertools.chain.from_iterable(patch.topo for patch in self.patches))

  def getitem(self, key):
    for i in range(len(self.patches)):
      if key == 'patch{}'.format(i):
        return self.patches[i].topo
    else:
      return UnionTopology(patch.topo.getitem(key) for patch in self.patches)

  def basis_spline(self, degree, patchcontinuous=True, knotvalues=None, knotmultiplicities=None):
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
    transforms = []
    dofcount = 0
    commonboundarydofs = {}
    for ipatch, patch in enumerate(self.patches):
      transforms.extend(elem.transform for elem in patch.topo)
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
      patchcoeffs, patchdofmap, patchdofcount = patch.topo._basis_spline(degree, knotvalues=patchknotvalues, knotmultiplicities=patchknotmultiplicities)
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

    return function.polyfunc(coeffs, dofmap, dofcount, transforms, issorted=False)

  def basis_discont(self, degree):
    'discontinuous shape functions'

    bases = [patch.topo.basis('discont', degree=degree) for patch in self.patches]
    coeffs = []
    dofs = []
    ndofs = 0
    for patch in self.patches:
      basis = patch.topo.basis('discont', degree=degree)
      (axes,func), = function.blocks(basis)
      patch_dofmap, = axes
      if isinstance(func, function.Polyval):
        patch_coeffs = func.coeffs
        assert patch_coeffs.ndim == 1+self.ndims
      elif func.isconstant:
        assert func.ndim == 1
        patch_coeffs = func[(slice(None),*(_,)*self.ndims)]
      else:
        raise ValueError
      patch_coeffs_dofs = function.Tuple((patch_coeffs, patch_dofmap))
      for elem in patch.topo:
        (elem_coeffs,), (elem_dofs,) = patch_coeffs_dofs.eval(_transforms=(elem.transform,))
        coeffs.append(elem_coeffs)
        dofs.append(types.frozenarray(ndofs+elem_dofs, copy=False))
      ndofs += len(basis)
    return function.polyfunc(coeffs, dofs, ndofs, (elem.transform for patch in self.patches for elem in patch.topo), issorted=False)

  def basis_patch(self):
    'degree zero patchwise discontinuous basis'

    npatches = len(self.patches)
    coeffs = [types.frozenarray(1, dtype=int).reshape(1, *(1,)*self.ndims)]*npatches
    dofs = types.frozenarray(range(npatches), dtype=int)[:,_]
    return function.polyfunc(coeffs, dofs, npatches, ((patch.topo.root,) for patch in self.patches), issorted=False)

  @property
  def boundary(self):
    'boundary'

    subtopos = []
    subnames = []
    for i, patch in enumerate(self.patches):
      names = dict(zip(itertools.product(range(self.ndims), [0,-1]), patch.topo._bnames))
      for boundary in patch.boundaries:
        if boundary.id in self._patchinterfaces:
          continue
        subtopos.append(patch.topo.boundary[names[boundary.dim,boundary.side]])
        subnames.append('patch{}-{}'.format(i, names[boundary.dim,boundary.side]))
    if len(subtopos) == 0:
      return EmptyTopology(self.ndims-1)
    else:
      return UnionTopology(subtopos, subnames)

  @property
  def interfaces(self):
    '''interfaces

    Return a topology with all element interfaces.  The patch interfaces are
    accessible via the group ``'interpatch'`` and the interfaces *inside* a
    patch via ``'intrapatch'``.
    '''

    intrapatchtopo = EmptyTopology(self.ndims-1) if not self.patches else \
      UnionTopology(patch.topo.interfaces for patch in self.patches)

    btopos = []
    bconnectivity = []
    for boundaryid, patchdata in self._patchinterfaces.items():
      if len(patchdata) > 2:
        raise ValueError('Cannot create interfaces of multipatch topologies with more than two interface connections.')
      pairs = []
      for topo, boundary in patchdata:
        names = dict(zip(itertools.product(range(self.ndims), [0,-1]), topo._bnames))
        # get structured set of boundary elements
        elems = topo.boundary[names[boundary.dim, boundary.side]].structure
        # add singleton axis
        elems = elems[tuple(_ if i == boundary.dim else slice(None) for i in range(self.ndims))]
        # apply canonical transformation
        elems = boundary.apply_transform(elems)[..., 0]
        shape = elems.shape
        pairs.append(elems.flat)
      # join element pairs
      elems = [
        element.Element(elem_a.reference, elem_a.transform, elem_b.transform, oriented=True)
        for elem_a, elem_b in zip(*pairs)
      ]
      # create structured topology of joined element pairs
      bpatch = numpy.array(boundaryid).reshape((2,)*(self.ndims-1))
      #btopos.append(StructuredTopology(numpy.array(elems).reshape(shape)))
      btopos.append(UnstructuredTopology(self.ndims-1, elems))
      bconnectivity.append(bpatch)
    # create multipatch topology of interpatch boundaries
    interpatchtopo = MultipatchTopology(tuple(map(Patch, btopos, bconnectivity, self.build_boundarydata(bconnectivity))))

    return UnionTopology((intrapatchtopo, interpatchtopo), ('intrapatch', 'interpatch'))

  @property
  def refined(self):
    'refine'

    return MultipatchTopology(Patch(patch.topo.refined, patch.verts, patch.boundaries) for patch in self.patches)

# INTEGRAL

def identity(value):
  return value

class Integral(types.Singleton):
  '''Postponed integral, used for derivative purposes'''

  __slots__ = '_integrands', 'shape'

  @types.apply_annotations
  def __init__(self, integrands:types.frozendict[identity, function.simplified]):
    self._integrands = integrands
    shapes = {integrand.shape for integrand in self._integrands.values()}
    assert len(shapes) == 1, 'incompatible shapes: {}'.format(' != '.join(str(shape) for shape in shapes))
    self.shape, = shapes

  def eval(self, **kwargs):
    retval, = eval_integrals(self, **kwargs)
    return retval

  def derivative(self, target):
    argshape = self._argshape(target)
    arg = function.Argument(target, argshape)
    seen = {}
    return Integral([di, function.derivative(integrand, var=arg, seen=seen)] for di, integrand in self._integrands.items())

  def replace(self, arguments):
    return Integral([di, function.replace_arguments(integrand, arguments)] for di, integrand in self._integrands.items())

  def contains(self, name):
    try:
      self._argshape(name)
    except KeyError:
      return False
    else:
      return True

  def __add__(self, other):
    if not isinstance(other, Integral):
      return NotImplemented
    assert self.shape == other.shape
    integrands = self._integrands.copy()
    for di, integrand in other._integrands.items():
      try:
        integrands[di] += integrand
      except KeyError:
        integrands[di] = integrand
    return Integral(integrands.items())

  def __neg__(self):
    return Integral([di, -integrand] for di, integrand in self._integrands.items())

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return Integral([di, integrand * other] for di, integrand in self._integrands.items())

  __rmul__ = __mul__

  def __truediv__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return self.__mul__(1/other)

  def _argshape(self, name):
    assert isinstance(name, str)
    shapes = {func.shape[:func.ndim-func._nderiv]
      for func in function.Tuple(self._integrands.values()).dependencies
        if isinstance(func, function.Argument) and func._name == name}
    if not shapes:
      raise KeyError(name)
    assert len(shapes) == 1, 'inconsistent shapes for argument {!r}'.format(name)
    shape, = shapes
    return shape

strictintegral = types.strict[Integral]

@types.apply_annotations
def eval_integrals(*integrals: types.tuple[strictintegral], fcache=None, arguments:types.frozendict[types.strictstr,types.frozenarray]=None):
  if fcache is None:
    fcache = cache.WrapperCache()
  gather = {}
  for iint, integral in enumerate(integrals):
    for di in integral._integrands:
      gather.setdefault(di, []).append(iint)
  retvals = [None] * len(integrals)
  for (domain, ischeme), iints in gather.items():
    for iint, retval in zip(iints, domain.integrate([integrals[iint]._integrands[domain, ischeme] for iint in iints], ischeme=ischeme, fcache=fcache, arguments=arguments)):
      if retvals[iint] is None:
        retvals[iint] = retval
      else:
        retvals[iint] += retval
  return retvals

# UTILITY FUNCTIONS

def common_refine(topo1, topo2):
  warnings.deprecation('common_refine(a, b) will be removed in future; use a & b instead')
  return topo1 & topo2

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
