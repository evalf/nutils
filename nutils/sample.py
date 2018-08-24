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

from . import types, points, log, util, function, config, parallel, numeric, cache, matrix
import numpy, numbers, collections.abc

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

  Args
  ----
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
  def __init__(self, transforms:tuple, points:types.tuple[points.strictpoints], index:types.tuple[types.frozenarray[types.strictint]]):
    assert len(transforms) == len(points) == len(index)
    self.nelems = len(transforms)
    self.transforms = transforms
    self.points = points
    self.index = index
    self.npoints = sum(p.npoints for p in points)
    self.ndims = points[0].ndims

  def __repr__(self):
    return '{}<{}D, {} elems, {} points>'.format(type(self).__qualname__, self.ndims, self.nelems, self.npoints)

  @log.withcontext
  @util.positional_only('self', 'funcs')
  @util.single_or_multiple
  @types.apply_annotations
  @cache.function
  def integrate(*args, **arguments:argdict):
    '''Integrate functions.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    self, funcs = args

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    funcs = [function.asarray(func).prepare_eval(ndims=self.ndims) for func in funcs]
    blocks = [(ifunc, function.Tuple(ind), f.simplified) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func)]
    block2func, indices, values = zip(*blocks) if blocks else ([],[],[])

    log.debug('integrating {} distinct blocks'.format('+'.join(
      str(block2func.count(ifunc)) for ifunc in range(len(funcs)))))

    if config.dot:
      function.Tuple(values).graphviz()

    # To allocate (shared) memory for all block data we evaluate indexfunc to
    # build an nblocks x nelems+1 offset array, and nblocks index lists of
    # length nelems.

    offsets = numpy.zeros((len(blocks), self.nelems+1), dtype=int)
    if blocks:
      sizefunc = function.stack([f.size for ifunc, ind, f in blocks]).simplified
      for ielem, transforms in enumerate(self.transforms):
        n, = sizefunc.eval(_transforms=transforms, **arguments)
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

    nprocs = min(config.nprocs, self.nelems)
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
    for ielem in parallel.pariter(log.range('elem', self.nelems), nprocs=nprocs):
      points = self.points[ielem]
      for iblock, (intdata, *indices) in enumerate(valueindexfunc.eval(_transforms=self.transforms[ielem], _points=points.coords, **arguments)):
        s = slice(*offsets[iblock,ielem:ielem+2])
        data, index = data_index[block2func[iblock]]
        w_intdata = numeric.dot(points.weights, intdata)
        data[s] = w_intdata.ravel()
        si = (slice(None),) + (numpy.newaxis,) * (w_intdata.ndim-1)
        for idim, (ii,) in enumerate(indices):
          index[idim,s].reshape(w_intdata.shape)[...] = ii[si]
          si = si[:-1]

    return [matrix.assemble(data, index, func.shape) for func, (data,index) in log.zip('assembling', funcs, data_index)]

  def integral(self, func):
    '''Create Integral object for postponed integration.

    Args
    ----
    func : :class:`nutils.function.Array`
        Integrand.
    '''

    return Integral([(self, func)])

  @log.withcontext
  @util.positional_only('self', 'funcs')
  @util.single_or_multiple
  @types.apply_annotations
  def eval(*args, **arguments:argdict):
    '''Evaluate function.

    Args
    ----
    funcs : :class:`nutils.function.Array` object or :class:`tuple` thereof.
        The integrand(s).
    arguments : :class:`dict` (default: None)
        Optional arguments for function evaluation.
    '''

    self, funcs = args

    nprocs = min(config.nprocs, self.nelems)
    zeros = parallel.shzeros if nprocs > 1 else numpy.zeros
    funcs = [function.asarray(func) for func in funcs]
    retvals = [zeros((self.npoints,)+func.shape, dtype=func.dtype) for func in funcs]
    idata = function.Tuple(function.Tuple([ifunc, function.Tuple(ind), f.simplified]) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(func.prepare_eval(ndims=self.ndims)))
    fcache = cache.WrapperCache()

    if config.dot:
      idata.graphviz()

    for transforms, points, index in parallel.pariter(log.zip('elem', self.transforms, self.points, self.index), nprocs=nprocs):
      for ifunc, inds, data in idata.eval(_transforms=transforms, _points=points.coords, _cache=fcache, **arguments):
        numpy.add.at(retvals[ifunc], numpy.ix_(index, *[ind for (ind,) in inds]), data)

    return retvals

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

    return function.Sampled(self, array)

  @property
  def tri(self):
    '''Triangulation of interior.

    A two-dimensional integer array with ``ndims+1`` columns, of which every
    row defines a simplex by mapping vertices into the list of points.
    '''

    return types.frozenarray(numpy.concatenate([index.take(points.tri) for points, index in zip(self.points, self.index)]), copy=False)

  @property
  def hull(self):
    '''Triangulation of the exterior hull.

    A two-dimensional integer array with ``ndims`` columns, of which every row
    defines a simplex by mapping vertices into the list of points. Note that
    the hull often does contain internal element boundaries as the
    triangulations originating from separate elements are disconnected.
    '''

    return types.frozenarray(numpy.concatenate([index.take(points.hull) for points, index in zip(self.points, self.index)]), copy=False)

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

    selection = [ielem for ielem in range(self.nelems) if mask[self.index[ielem]].any()]
    transforms = [self.transforms[ielem] for ielem in selection]
    points = [self.points[ielem] for ielem in selection]
    offset = numpy.cumsum([0] + [p.npoints for p in points])
    return Sample(transforms, points, map(numpy.arange, offset[:-1], offset[1:]))

strictsample = types.strict[Sample]

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
  '''

  __slots__ = '_integrands', 'shape'

  @types.apply_annotations
  def __init__(self, integrands:types.frozendict[strictsample, function.simplified]):
    self._integrands = integrands
    shapes = {integrand.shape for integrand in self._integrands.values()}
    assert len(shapes) == 1, 'incompatible shapes: {}'.format(' != '.join(str(shape) for shape in shapes))
    self.shape, = shapes

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
    return Integral([di, function.derivative(integrand, var=arg, seen=seen)] for di, integrand in self._integrands.items())

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

    return Integral([di, function.replace_arguments(integrand, arguments)] for di, integrand in self._integrands.items())

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
@cache.function
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

  retvals = [None] * len(integrals)
  for sample, iints in util.gather((di, iint) for iint, integral in enumerate(integrals) for di in integral._integrands):
    for iint, retval in zip(iints, sample.integrate([integrals[iint]._integrands[sample] for iint in iints], **arguments)):
      if retvals[iint] is None:
        retvals[iint] = retval
      else:
        retvals[iint] += retval
  return retvals

# vim:sw=2:sts=2:et
