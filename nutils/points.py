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

from . import types, transform, numeric
import numpy, functools, itertools, warnings
_ = numpy.newaxis

class Points(types.Singleton):

  @types.apply_annotations
  def __init__(self, npoints:types.strictint, ndims:types.strictint):
    self.npoints = npoints
    self.ndims = ndims

strictpoints = types.strict[Points]

class CoordsPoints(Points):

  @types.apply_annotations
  def __init__(self, coords:types.frozenarray[float]):
    self.coords = coords
    super().__init__(*coords.shape)

class CoordsWeightsPoints(CoordsPoints):

  @types.apply_annotations
  def __init__(self, coords:types.frozenarray[float], weights:types.frozenarray[float]):
    self.weights = weights
    super().__init__(coords)

class CoordsUniformPoints(CoordsPoints):

  @types.apply_annotations
  def __init__(self, coords:types.frozenarray[float], volume:float):
    self.weights = types.frozenarray.full([len(coords)], volume/len(coords))
    super().__init__(coords)

class TensorPoints(Points):

  __cache__ = 'coords', 'weights'

  @types.apply_annotations
  def __init__(self, points1:strictpoints, points2:strictpoints):
    self.points1 = points1
    self.points2 = points2
    super().__init__(points1.npoints * points2.npoints, points1.ndims + points2.ndims)

  @property
  def coords(self):
    coords = numpy.empty((self.points1.npoints, self.points2.npoints, self.ndims))
    coords[:,:,:self.points1.ndims] = self.points1.coords[:,_,:]
    coords[:,:,self.points1.ndims:] = self.points2.coords[_,:,:]
    return types.frozenarray(coords.reshape(self.npoints, self.ndims), copy=False)

  @property
  def weights(self):
    return types.frozenarray((self.points1.weights[:,_] * self.points2.weights[_,:]).ravel(), copy=False)

class SimplexGaussPoints(CoordsWeightsPoints):

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, degree:types.strictint):
    super().__init__(*gaussn[ndims](degree))

class SimplexBezierPoints(CoordsPoints):

  @types.apply_annotations
  def __init__(self, ndims:types.strictint, npoints:types.strictint):
    linspace = numpy.linspace(0, 1, npoints)
    super().__init__([linspace[list(index)[::-1]] for index in numpy.ndindex(*[npoints] * ndims) if sum(index) < npoints])

class TransformPoints(Points):

  __cache__ = 'coords', 'weights'

  @types.apply_annotations
  def __init__(self, points:strictpoints, trans:transform.stricttransformitem):
    self.points = points
    self.trans = trans
    super().__init__(points.npoints, points.ndims)

  @property
  def coords(self):
    return self.trans.apply(self.points.coords)

  @property
  def weights(self):
    return self.points.weights * abs(float(self.trans.det))

class ConcatPoints(Points):

  __cache__ = 'coords', 'weights'

  @types.apply_annotations
  def __init__(self, allpoints:types.tuple[strictpoints]):
    self.allpoints = allpoints
    super().__init__(sum(points.npoints for points in allpoints), allpoints[0].ndims)

  @property
  def coords(self):
    return types.frozenarray(numpy.concatenate([points.coords for points in self.allpoints]), copy=False)

  @property
  def weights(self):
    return types.frozenarray(numpy.concatenate([points.weights for points in self.allpoints]), copy=False)

## UTILITY FUNCTIONS

@functools.lru_cache(8)
def gauss(n):
  k = numpy.arange(n) + 1
  d = k / numpy.sqrt(4*k**2-1)
  x, w = numpy.linalg.eigh(numpy.diagflat(d,-1)) # eigh operates (by default) on lower triangle
  return types.frozenarray((x+1) * .5, copy=False), types.frozenarray(w[0]**2, copy=False)

def gauss1(degree):
  '''Gauss quadrature for line.'''

  x, w = gauss(degree//2)
  return x[:,_], w

@functools.lru_cache(8)
def gauss2(degree):
  '''Gauss quadrature for triangle.

  Reference: http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''

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

  return types.frozenarray(numpy.concatenate([numpy.take(c,i) for i, c, w in icw]), copy=False), \
         types.frozenarray(numpy.concatenate([[w/2] * len(i) for i, c, w in icw]), copy=False)

@functools.lru_cache(8)
def gauss3(degree):
  '''Gauss quadrature for tetrahedron.

  Reference http://www.cs.rpi.edu/~flaherje/pdf/fea6.pdf'''

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

  return types.frozenarray(numpy.concatenate([numpy.take(c,i) for i, c, w in icw]), copy=False), \
         types.frozenarray(numpy.concatenate([[w/6] * len(i) for i, c, w in icw]), copy=False)

gaussn = None, gauss1, gauss2, gauss3

# vim:sw=2:sts=2:et
