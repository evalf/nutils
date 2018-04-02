from nutils import *
from . import *

@parametrize
class tri(TestCase):

  # Triangles and node numbering:
  #
  #   2/4-(5)
  #    | \ |
  #   (0)-1/3

  def setUp(self):
    self.x = numpy.array([[0,0],[1,0],[0,1],[1,0],[0,1],[1,1]], dtype=float)
    self.tri = numpy.array([[0,1,2],[3,4,5]])

  def test_merge(self):
    tri_merged = util.tri_merge(self.tri, self.x, mergetol=self.mergetol).tolist()
    tri_expected = self.tri.tolist() if self.mergetol < 0 else [[0,1,2],[1,2,5]] if self.mergetol < 1 else [[0,0,0],[0,0,0]]
    self.assertEqual(tri_merged, tri_expected)

  def test_interpolate(self):
    interpolate = util.tri_interpolator(self.tri, self.x, mergetol=self.mergetol)
    x = [.1, .9],
    if self.mergetol < 0:
      with self.assertRaises(RuntimeError):
        interpolate[x]
    else:
      f = interpolate[x]
      vtri = [0,0], [1,0], [0,1], [10,10], [10,10], [1,1]
      vx = f(vtri) # note: v[3] and v[4] should be ignored, leaving a linear ramp
      self.assertEqual(vx.shape, (1,2))
      if self.mergetol < 1:
        self.assertEqual(vx.tolist(), list(x))
      else:
        self.assertTrue(numpy.isnan(vx).all())

  @parametrize.enable_if(lambda mergetol: 0 <= mergetol < 1)
  def test_outofbounds(self):
    interpolate = util.tri_interpolator(self.tri, self.x, mergetol=self.mergetol)
    x = [.5, .5], [1.5, .5]
    vtri = 0, 1, 0, 10, 10, 1
    vx = interpolate[x](vtri)
    self.assertEqual(vx.shape, (2,))
    self.assertEqual(vx[0], .5)
    self.assertTrue(numpy.isnan(vx[1]))

tri(mergetol=-1)
tri(mergetol=0)
tri(mergetol=.1)
tri(mergetol=2)
