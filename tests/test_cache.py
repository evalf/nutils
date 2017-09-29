from nutils import *
from . import *
import sys

class refcount(TestCase):

  def setUp(self):
    self.x = object()
    self.d = {'referenced': self.x, 'dangling': object()}

  def test_noremove(self):
    keep = set(k for k, v in self.d.items() if sys.getrefcount(v) > 3)
    assert keep == {'referenced', 'dangling'}

  def test_remove(self):
    keep = set(k for k, v in self.d.items() if sys.getrefcount(v) > 4)
    assert keep == {'referenced'}
