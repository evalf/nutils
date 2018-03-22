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

from . import types, points, log, util, function, config, parallel, numeric, cache
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
