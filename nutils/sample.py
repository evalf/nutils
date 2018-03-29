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

from . import types, points, log, util, function, config, parallel, numeric, cache, matrix
import numpy

class Sample(types.Singleton):

  @types.apply_annotations
  def __init__(self, transforms:tuple, points:types.tuple[points.strictpoints], index:types.tuple[types.frozenarray[types.strictint]]):
    assert len(transforms) == len(points) == len(index)
    self.nelems = len(transforms)
    self.transforms = transforms
    self.points = points
    self.index = index
    self.npoints = sum(p.npoints for p in points)
    self.ndims = points[0].ndims

  @log.title
  @util.single_or_multiple
  @cache.function
  def integrate(self, funcs, *, arguments=None):
    'integrate functions'

    if arguments is None:
      arguments = {}

    # Functions may consist of several blocks, such as originating from
    # chaining. Here we make a list of all blocks consisting of triplets of
    # argument id, evaluable index, and evaluable values.

    funcs = [function.zero_argument_derivatives(function.asarray(func)) for func in funcs]
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

    for func, (data,index) in zip(funcs, data_index):
      retval = matrix.assemble(data, index, func.shape)
      assert retval.shape == func.shape
      log.debug('assembled {}({})'.format(retval.__class__.__name__, ','.join(str(n) for n in retval.shape)))
      yield retval

  @log.title
  @util.single_or_multiple
  def eval(self, funcs, *, arguments=None):
    'sample function in discrete points'

    if arguments is None:
      arguments = {}

    nprocs = min(config.nprocs, self.nelems)
    zeros = parallel.shzeros if nprocs > 1 else numpy.zeros
    funcs = [function.asarray(func) for func in funcs]
    retvals = [zeros((self.npoints,)+func.shape, dtype=func.dtype) for func in funcs]
    idata = function.Tuple(function.Tuple([ifunc, function.Tuple(ind), f.simplified]) for ifunc, func in enumerate(funcs) for ind, f in function.blocks(function.zero_argument_derivatives(func)))
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


# vim:sw=2:sts=2:et
