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

'''
The sample module defines the :class:`Sample` class, which represents a
collection of discrete points on a topology and is typically formed via
:func:`nutils.topology.Topology.sample`. Any function evaluation starts from
this sampling step, which drops element information and other topological
properties such as boundaries and groups, but retains point positions and
(optionally) integration weights. Evaluation is performed by subsequent calls
to :func:`Sample.integrate`, :func:`Sample.integral` or :func:`Sample.eval`.

Besides the location of points, :class:`Sample` also keeps track of point
connectivity through its :attr:`Sample.tri` and :attr:`Sample.hull`
properties, representing a (n-dimensional) triangulation of the interior and
boundary, respectively. Availability of these properties depends on the
selected sample points, and is typically used in combination with the "bezier"
set.

In addition to :class:`Sample`, the sample module defines the :class:`Integral`
class which represents postponed integration. Integrals are internally
represented as pairs of :class:`Sample` and :class:`nutils.function.Array`
objects. Evaluation proceeds via either the :func:`Integral.eval` method, or
the :func:`eval_integrals` function. The latter can also be used to evaluate
multiple integrals simultaneously, which has the advantage that it can
efficiently combine common substructures.
'''

from . import types, points, util, function, parallel, numeric, matrix, transformseq, sparse, warnings
import numpy, numbers, collections.abc, os, treelog as log, operator, functools

graphviz = os.environ.get('NUTILS_GRAPHVIZ')

def argdict(arguments):
  if len(arguments) == 1 and 'arguments' in arguments and isinstance(arguments['arguments'], collections.abc.Mapping):
    arguments = arguments['arguments']
  return types.frozendict[types.strictstr,types.frozenarray](arguments)

class Sample(types.Singleton):
  '''Collection of points on a topology.

  The :class:`Sample` class represents a collection of discrete points on a
  topology and is typically formed via :func:`nutils.topology.Topology.sample`.
  Any function evaluation starts from this sampling step, which drops element
  information and other topological properties such as boundaries and groups,
  but retains point positions and (optionally) integration weights. Evaluation
  is performed by subsequent calls to :func:`integrate`, :func:`integral` or
  :func:`eval`.

  Besides the location of points, ``Sample`` also keeps track of point
  connectivity through its :attr:`tri` and :attr:`hull` properties,
  representing a (n-dimensional) triangulation of the interior and boundary,
  respectively. Availability of these properties depends on the selected sample
  points, and is typically used in combination with the "bezier" set.

  Parameters
  ----------
  roots : :class:`tuple` of :class:`~nutils.function.Root`
      The roots of this sample.
  ndims : :class:`int`
      The dimension of the :class:`~nutils.topology.Topology` from which this
      sample is created.
  npoints : :class:`int`
      The number of points in this sample.
  transforms : :class:`tuple` or transformation chains
      List of transformation chains leading to local coordinate systems that
      contain points.
  '''

  __cache__ = 'allcoords', 'index', 'subsamplemetas'

  @types.apply_annotations
  def __init__(self, roots:types.tuple[function.strictroot], ndims:types.strictint, npoints:types.strictint, transforms:types.tuple[transformseq.stricttransforms]):
    self.roots = roots
    self.ndims = ndims
    self.npoints = npoints
    self.transforms = transforms
    self.nelems = len(transforms[0])

  def __repr__(self):
    return '{}<{}D, {} elems, {} points>'.format(type(self).__qualname__, self.ndims, self.nelems, self.npoints)

  def _prepare_funcs(self, funcs):
    return [function.asarray(func).prepare_eval(subsamples=self.subsamplemetas) for func in funcs]

  @property
  def index(self):
    warnings.deprecation('`Sample.index` is deprecated; replace `Sample.index[ielem]` with `Sample.getindex(ielem)`')
    return tuple(self.getindex(ielem) for ielem in range(self.nelems))

  @property
  def points(self):
    warnings.deprecation('`Sample.points` is deprecated; replace `Sample.points[ielem]` with `Sample.getpoints(ielem)`')
    return tuple(self.getpoints(ielem) for ielem in range(self.nelems))

  @util.positional_only
  @util.single_or_multiple
  @types.apply_annotations
  def integrate(self, funcs, arguments:argdict=...):
    '''Integrate functions.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    datas = self.integrate_sparse(funcs, arguments)
    with log.iter.fraction('assembling', datas) as items:
      return [sparse.convert(data, inplace=True) for data in items]

  @util.single_or_multiple
  @types.apply_annotations
  def integrate_sparse(self, funcs:types.tuple[function.asarray], arguments:types.frozendict[str,types.frozenarray]=None):
    '''Integrate functions into sparse data.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    if arguments is None:
      arguments = {}

    detJ = 1
    for isubsample, subsample in enumerate(self.subsamplemetas):
      J = function.rootbasis(self.subsamplemetas, isubsample)[:,:subsample.ndimsmanifold]
      if J.shape[0] == J.shape[1]:
        detJ *= abs(function.determinant(J))
      else:
        detJ *= abs(function.determinant((J[:,:,None] * J[:,None,:]).sum(0)))**.5
    funcs = [func * detJ for func in funcs]

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    funcs = self._prepare_funcs(funcs)
    weights = function.Weights().prepare_eval(subsamples=self.subsamplemetas)
    blocks = [(ifunc, function.Tuple(ind), function.DotWeights(f, weights).simplified.optimized_for_numpy) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func)]
    block2func, indices, values = zip(*blocks) if blocks else ([],[],[])

    log.debug('integrating {} distinct blocks'.format('+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)))))

    if graphviz:
      function.Tuple(values).graphviz(graphviz)

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros((len(blocks), self.nelems+1), dtype=int)
    if blocks:
      sizefunc = function.stack([f.size for ifunc, ind, f in blocks]).simplified
      for ielem in range(self.nelems):
        n, = sizefunc.eval(*self.getsubsamples(ielem), **arguments)
        offsets[:,ielem+1] = offsets[:,ielem] + n

    # Since several blocks may belong to the same function, we post process the
    # offsets to form consecutive intervals in longer arrays. The length of
    # these arrays is captured in the nfuncs-array nvals.

    nvals = numpy.zeros(len(funcs), dtype=int)
    for iblock, ifunc in enumerate(block2func):
      offsets[iblock] += nvals[ifunc]
      nvals[ifunc] = offsets[iblock,-1]

    # In a second, parallel element loop, value and index are evaluated and
    # stored in shared memory using the offsets array for location. Each
    # element has its own location so no locks are required.

    datas = [parallel.shempty(n, dtype=sparse.dtype(funcs[ifunc].shape)) for ifunc, n in enumerate(nvals)]
    valueindexfunc = function.Tuple(function.Tuple([value]+list(index)) for value, index in zip(values, indices))
    with parallel.ctxrange('integrating', self.nelems) as ielems:
      for ielem in ielems:
        subsamples = self.getsubsamples(ielem)
        for iblock, ((intdata,), *indices) in enumerate(valueindexfunc.eval(*subsamples, **arguments)):
          data = datas[block2func[iblock]][offsets[iblock,ielem]:offsets[iblock,ielem+1]].reshape(intdata.shape)
          data['value'] = intdata
          for idim, ii in enumerate(indices):
            data['index']['i'+str(idim)] = ii.reshape([-1]+[1]*(data.ndim-1-idim))

    return datas

  def integral(self, func):
    '''Create Integral object for postponed integration.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    '''

    return Integral([(self, func)], shape=func.shape)

  @util.positional_only
  @util.single_or_multiple
  @types.apply_annotations
  def eval(self, funcs, arguments:argdict=...):
    '''Evaluate function.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    funcs = self._prepare_funcs(funcs)
    retvals = [parallel.shzeros((self.npoints,)+func.shape, dtype=func.dtype) for func in funcs]
    idata = function.Tuple(function.Tuple([ifunc, function.Tuple(ind), f.simplified.optimized_for_numpy]) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func))

    if graphviz:
      idata.graphviz(graphviz)

    with parallel.ctxrange('evaluating', self.nelems) as ielems:
      for ielem in ielems:
        for ifunc, inds, data in idata.eval(*self.getsubsamples(ielem), **arguments):
          numpy.add.at(retvals[ifunc], numpy.ix_(self.getindex(ielem), *[ind for (ind,) in inds]), data)

    return retvals

  @property
  def allcoords(self):
    coords = numpy.empty([self.npoints, self.ndims])
    for ielem in range(self.nelems):
      coords[self.getindex(ielem)] = self.getpoints(ielem).coords
    return types.frozenarray(coords, copy=False)

  def basis(self):
    '''Basis-like function that for every point in the sample evaluates to the
    unit vector corresponding to its index.'''

    index, tail, linear = function.TransformsIndexWithTail(self.transforms[0], self.ndims, function.SelectChain(self.roots))
    I = function.Elemwise(self.index, index, dtype=int)
    B = function.Sampled(function.ApplyTransforms(tail, linear), expect=function.take(self.allcoords, I, axis=0))
    return function.Inflate(func=B, dofmap=I, length=self.npoints, axis=0)

  def asfunction(self, array):
    '''Convert sampled data to evaluable array.

    Using the result of :func:`Sample.eval`, create a
    :class:`nutils.function.Sampled` array that upon evaluation recovers the
    original function in the set of points matching the original sampling.

    >>> from nutils import mesh
    >>> domain, geom = mesh.rectilinear([1,2])
    >>> gauss = domain.sample('gauss', 2)
    >>> data = gauss.eval(geom)
    >>> sampled = gauss.asfunction(data)
    >>> domain.integrate(sampled, degree=2)
    array([ 1.,  2.])

    Args
    ----
    array :
        The sampled data.
    '''

    return function.matmat(self.basis(), array)

  @property
  def tri(self):
    '''Triangulation of interior.

    A two-dimensional integer array with ``ndims+1`` columns, of which every
    row defines a simplex by mapping vertices into the list of points.
    '''

    return types.frozenarray(numpy.concatenate([self.getindex(ielem).take(self.getpoints(ielem).tri) for ielem in range(self.nelems)]), copy=False)

  @property
  def hull(self):
    '''Triangulation of the exterior hull.

    A two-dimensional integer array with ``ndims`` columns, of which every row
    defines a simplex by mapping vertices into the list of points. Note that
    the hull often does contain internal element boundaries as the
    triangulations originating from separate elements are disconnected.
    '''

    return types.frozenarray(numpy.concatenate([self.getindex(ielem).take(self.getpoints(ielem).hull) for ielem in range(self.nelems)]), copy=False)

  def subset(self, mask):
    '''Reduce the number of points.

    Simple selection mechanism that returns a reduced Sample based on a
    selection mask. Points that are marked True will still be part of the new
    subset; points marked False may be dropped but this is not guaranteed. The
    point order of the original Sample is preserved.

    Args
    ----
    mask : :class:`bool` array.
        Boolean mask that selects all points that should remain. The resulting
        Sample may contain more points than this, but not less.

    Returns
    -------
    subset : :class:`Sample`
    '''

    selection = types.frozenarray([ielem for ielem in range(self.nelems) if mask[self.getindex(ielem)].any()])
    transforms = tuple(transform[selection] for transform in self.transforms)
    points = [self.getpoints(ielem) for ielem in selection]
    offset = numpy.cumsum([0] + [p.npoints for p in points])
    return PlainSample(self.roots, self.ndims, transforms, points, map(numpy.arange, offset[:-1], offset[1:]))

  def getsubsamples(self, ielem):
    return function.Subsample(roots=self.roots, transforms=tuple(t[ielem] for t in self.transforms), points=self.getpoints(ielem), ielem=ielem),

  @property
  def subsamplemetas(self):
    if self.nelems:
      ndimspoints = self.getpoints(0).ndims
      if not all(self.getpoints(ielem).ndims == ndimspoints for ielem in range(self.nelems)):
        ndimspoints = None
    else:
      ndimspoints = None
    return function.SubsampleMeta(roots=self.roots, ndimsnormal=sum(root.ndims for root in self.roots)-self.ndims, transforms=self.transforms, ndimspoints=ndimspoints),

strictsample = types.strict[Sample]

class PlainSample(Sample):
  '''A general purpose implementation of :class:`Sample`.

  Parameters
  ----------
  roots : :class:`tuple` of :class:`~nutils.function.Root`
      The roots of this sample.
  ndims : :class:`int`
      The dimension of the :class:`~nutils.topology.Topology` from which this
      sample is created.
  transforms : :class:`tuple` or transformation chains
      List of transformation chains leading to local coordinate systems that
      contain points.
  points : :class:`tuple` of point sets
      List of point sets matching ``transforms``.
  index : :class:`tuple` of integer arrays
      List of indices matching ``transforms``, defining the order on which
      points show up in the evaluation.
  '''

  @types.apply_annotations
  def __init__(self, roots:types.tuple[function.strictroot], ndims:types.strictint, transforms:types.tuple[transformseq.stricttransforms], points:types.tuple[points.strictpoints], index:types.tuple[types.frozenarray[types.strictint]]):
    assert len(points) == len(index)
    assert len(transforms) >= 1
    assert all(len(t) == len(points) for t in transforms)
    self._points = points
    self._index = index
    npoints = sum(p.npoints for p in points)
    super().__init__(roots, ndims, npoints, transforms)

  def getpoints(self, ielem):
    return self._points[ielem]

  def getindex(self, ielem):
    return self._index[ielem]

class UniformSample(Sample):
  '''A sample with uniform points.

  Parameters
  ----------
  roots : :class:`tuple` of :class:`~nutils.function.Root`
      The roots of this sample.
  ndims : :class:`int`
      The dimension of the :class:`~nutils.topology.Topology` from which this
      sample is created.
  transforms : :class:`tuple` or transformation chains
      List of transformation chains leading to local coordinate systems that
      contain points.
  points : :class:`~nutils.points.Points`
      Point set.
  '''

  __cache__ = 'tri', 'hull', 'subsamplemetas'

  @types.apply_annotations
  def __init__(self, roots:types.tuple[function.strictroot], ndims:types.strictint, transforms:types.tuple[transformseq.stricttransforms], points:points.strictpoints):
    assert len(transforms) >= 1
    self._points = points
    super().__init__(roots, ndims, points.npoints*len(transforms[0]), transforms)

  def getpoints(self, ielem):
    return self._points

  def getindex(self, ielem):
    return numpy.arange(ielem*self._points.npoints, (ielem+1)*self._points.npoints)

  @property
  def tri(self):
    tri = self._points.tri
    return types.frozenarray((numpy.arange(0, self.nelems*self._points.npoints, self._points.npoints)[:,None,None] + tri).reshape(-1,tri.shape[-1]), copy=False)

  @property
  def hull(self):
    hull = self._points.hull
    return types.frozenarray((numpy.arange(0, self.nelems*self._points.npoints, self._points.npoints)[:,None,None] + hull).reshape(-1,hull.shape[-1]), copy=False)

  @property
  def subsamplemetas(self):
    return function.SubsampleMeta(roots=self.roots, ndimsnormal=sum(root.ndims for root in self.roots)-self.ndims, transforms=self.transforms, points=self._points, ndimspoints=self._points.ndims),

class ProductSample(Sample):

  __cache__ = 'subsamplemetas', 'tri', 'hull'

  @types.apply_annotations
  def __init__(self, sample1:strictsample, sample2:strictsample, transforms:types.tuple[transformseq.stricttransforms]):
    self._sample1 = sample1
    self._sample2 = sample2
    super().__init__(sample1.roots+sample2.roots,
                     sample1.ndims+sample2.ndims,
                     sample1.npoints*sample2.npoints,
                     transforms)

  def getpoints(self, ielem):
    ielem1, ielem2 = divmod(ielem, self._sample2.nelems)
    return points.TensorPoints(self._sample1.getpoints(ielem1), self._sample2.getpoints(ielem2))

  def getindex(self, ielem):
    ielem1, ielem2 = divmod(ielem, self._sample2.nelems)
    return (self._sample1.getindex(ielem1)[:,numpy.newaxis]*self._sample2.npoints + self._sample2.getindex(ielem2)[numpy.newaxis,:]).ravel()

  @property
  def tri(self):
    if self._sample1.ndims == 1:
      tri12 = self._sample1.tri[:,None,:,None] * self._sample2.npoints + self._sample2.tri[None,:,None,:] # ntri1 x ntri2 x 2 x ndims
      return types.frozenarray(numeric.overlapping(tri12.reshape(-1, 2*self.ndims), n=self.ndims+1).reshape(-1, self.ndims+1), copy=False)
    return super().tri

  @property
  def hull(self):
    # NOTE: the order differs from `super().hull`
    if self._sample1.ndims == 1:
      hull1 = self._sample1.hull[:,None,:,None] * self._sample2.npoints + self._sample2.tri[None,:,None,:] # 2 x ntri2 x 1 x ndims
      hull2 = self._sample1.tri[:,None,:,None] * self._sample2.npoints + self._sample2.hull[None,:,None,:] # ntri1 x nhull2 x 2 x ndims-1
      # The subdivision of hull2 into simplices follows identical logic to that
      # used in the construction of self.tri.
      hull = numpy.concatenate([hull1.reshape(-1, self.ndims), numeric.overlapping(hull2.reshape(-1, 2*(self.ndims-1)), n=self.ndims).reshape(-1, self.ndims)])
      return types.frozenarray(hull, copy=False)
    return super().hull

  def getsubsamples(self, ielem):
    ielem1, ielem2 = divmod(ielem, self._sample2.nelems)
    return self._sample1.getsubsamples(ielem1) + self._sample2.getsubsamples(ielem2)

  @property
  def subsamplemetas(self):
    return self._sample1.subsamplemetas + self._sample2.subsamplemetas

class ChainedSample(Sample):

  __cache__ = 'tri', 'hull'

  @types.apply_annotations
  def __init__(self, samples:types.tuple[strictsample], transforms:types.tuple[transformseq.stricttransforms]):
    if not len(samples):
      raise ValueError('cannot chain zero samples')
    roots = samples[0].roots
    ndims = samples[0].ndims
    if not all(sample.roots == roots for sample in samples):
      raise ValueError('all samples to be chained should have the same (order of) roots')
    if not all(sample.ndims == ndims for sample in samples):
      raise ValueError('all samples to be chained should have the same dimension')
    todims = tuple(root.ndims for root in roots)
    self._samples = samples
    self._elemoffsets = numpy.cumsum([0, *(sample.nelems for sample in samples[:-1])])
    self._pointsoffsets = numpy.cumsum([0, *(sample.npoints for sample in samples[:-1])])
    super().__init__(roots, ndims, sum(sample.npoints for sample in samples), transforms)

  def _findelem(self, ielem):
    if ielem < 0 or ielem >= self.nelems:
      raise IndexError('element index out of range')
    isample = numpy.searchsorted(self._elemoffsets[1:], ielem, side='right')
    return isample, ielem - self._elemoffsets[isample]

  def getpoints(self, ielem):
    isample, ielem = self._findelem(ielem)
    return self._samples[isample].getpoints(ielem)

  def getindex(self, ielem):
    isample, ielem = self._findelem(ielem)
    return self._samples[isample].getindex(ielem) + self._pointsoffsets[isample]

  def integral(self, func):
    return functools.reduce(operator.add, (sample.integral(func) for sample in self._samples))

  @property
  def tri(self):
    offsets = util.cumsum(sample.npoints for sample in self._samples)
    return types.frozenarray(numpy.concatenate([sample.tri+offset for sample, offset in zip(self._samples, offsets)], axis=0), copy=False)

  @property
  def hull(self):
    offsets = util.cumsum(sample.npoints for sample in self._samples)
    return types.frozenarray(numpy.concatenate([sample.hull+offset for sample, offset in zip(self._samples, offsets)], axis=0), copy=False)

class Integral(types.Singleton):
  '''Postponed integration.

  The :class:`Integral` class represents postponed integration. Integrals are
  internally represented as pairs of :class:`Sample` and
  :class:`nutils.function.Array` objects. Evaluation proceeds via either the
  :func:`eval` method, or the :func:`eval_integrals` function. The latter can
  also be used to evaluate multiple integrals simultaneously, which has the
  advantage that it can efficiently combine common substructures.

  Integrals support basic arithmetic such as summation, subtraction, and scalar
  multiplication and division. It also supports differentiation via the
  :func:`derivative` method. This makes Integral particularly well suited for
  use in combination with the :mod:`nutils.solver` module which provides linear
  and non-linear solvers.

  Args
  ----
  integrands : :class:`dict`
      Dictionary representing a sum of integrals, where every key-value pair
      binds together the sample set and the integrand.
  shape : :class:`tuple`
      Array dimensions of the integral.
  '''

  __slots__ = '_integrands', 'shape'
  __cache__ = 'derivative'

  @types.apply_annotations
  def __init__(self, integrands:types.frozendict[strictsample, function.simplified], shape:types.tuple[int]):
    assert all(ig.shape == shape for ig in integrands.values()), 'incompatible shapes: expected {}, got {}'.format(shape, ', '.join({str(ig.shape) for ig in integrands.values()}))
    self._integrands = {topo: func for topo, func in integrands.items() if not function.iszero(func)}
    self.shape = shape

  @property
  def ndim(self):
    return len(self.shape)

  def __repr__(self):
    return 'Integral<{}>'.format(','.join(map(str, self.shape)))

  def eval(self, **kwargs):
    '''Evaluate integral.

    Equivalent to :func:`eval_integrals` (self, ...).
    '''

    retval, = eval_integrals(self, **kwargs)
    return retval

  def derivative(self, target):
    '''Differentiate integral.

    Return an Integral in which all integrands are differentiated with respect
    to a target. This is typically used in combination with
    :class:`nutils.function.Namespace`, in which targets are denoted with a
    question mark (e.g. ``'?dofs_n'`` corresponds to target ``'dofs'``).

    Args
    ----
    target : :class:`str`
        Name of the derivative target.

    Returns
    -------
    derivative : :class:`Integral`
    '''

    argshape = self._argshape(target)
    arg = function.Argument(target, argshape)
    seen = {}
    return Integral({di: function.derivative(integrand, var=arg, seen=seen) for di, integrand in self._integrands.items()}, shape=self.shape+argshape)

  def replace(self, arguments):
    '''Return copy with arguments applied.

    Return a copy of self in which all all arguments are edited into the
    integrands. The effect is that ``self.eval(..., arguments=args)`` is
    equivalent to ``self.replace(args).eval(...)``. Note, however, that after
    the replacement it is no longer possible to take derivatives against any of
    the targets in ``arguments``.

    Args
    ----
    arguments : :class:`dict`
        Arguments for function evaluation.

    Returns
    -------
    replaced : :class:`Integral`
    '''

    return Integral({di: function.replace_arguments(integrand, arguments) for di, integrand in self._integrands.items()}, shape=self.shape)

  def contains(self, name):
    '''Test if target occurs in any of the integrands.

    Args
    ----
    name : :class:`str`
        Target name.

    Returns
    _______
    iscontained : :class:`bool`
    '''

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
    return Integral(integrands.items(), shape=self.shape)

  def __neg__(self):
    return Integral({di: -integrand for di, integrand in self._integrands.items()}, shape=self.shape)

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, numbers.Number):
      return NotImplemented
    return Integral({di: integrand * other for di, integrand in self._integrands.items()}, shape=self.shape)

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

  @property
  def T(self):
    return Integral({sample: func.T for sample, func in self._integrands.items()}, shape=self.shape[::-1])

strictintegral = types.strict[Integral]

@types.apply_annotations
def eval_integrals(*integrals: types.tuple[strictintegral], **arguments:argdict):
  '''Evaluate integrals.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :func:`Integral.eval` on each integral
  individually, integrations will be grouped per Sample and jointly executed,
  potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  with log.iter.fraction('assembling', eval_integrals_sparse(*integrals, **arguments)) as retvals:
    return [sparse.convert(retval, inplace=True) for retval in retvals]

@types.apply_annotations
def eval_integrals_sparse(*integrals: types.tuple[strictintegral], **arguments: argdict):
  '''Evaluate integrals into sparse data.

  Evaluate one or several postponed integrals. By evaluating them
  simultaneously, rather than using :func:`Integral.eval` on each integral
  individually, integrations will be grouped per Sample and jointly executed,
  potentially increasing efficiency.

  Args
  ----
  integrals : :class:`tuple` of integrals
      Integrals to be evaluated.
  arguments : :class:`dict` (default: None)
      Optional arguments for function evaluation.

  Returns
  -------
  results : :class:`tuple` of arrays and/or :class:`nutils.matrix.Matrix` objects.
  '''

  if arguments is None:
    arguments = types.frozendict({})

  retvals = [[sparse.empty(integral.shape)] for integral in integrals] # initialize with zeros to set shape and avoid empty addition
  with log.iter.fraction('topology', util.gather((di, iint) for iint, integral in enumerate(integrals) for di in integral._integrands)) as gathered:
    for sample, iints in gathered:
      for iint, retval in zip(iints, sample.integrate_sparse([integrals[iint]._integrands[sample] for iint in iints], arguments)):
        retvals[iint].append(retval)
      del retval

  return [sparse.add(retval) for retval in retvals]

# vim:sw=2:sts=2:et
