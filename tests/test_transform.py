from nutils import *
from nutils.testing import *

class specialcases(TestCase):

  def test_tensoredge_swapup_identifier(self):
    lineedge = transform.SimplexEdge(1, 0, False)
    for edge in transform.TensorEdge1(lineedge, 1), transform.TensorEdge2(1, lineedge):
      with self.subTest(type(edge).__name__):
        idnt = transform.Identifier(1, 'test')
        self.assertEqual(edge.swapup(idnt), None)
