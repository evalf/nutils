import nutils.config
from nutils.testing import *

class Config(TestCase):

  def assertNotHasattr(self, obj, attr):
    if hasattr(obj, attr):
      self.fail('{!r} unexpectedly has an attribute {!r}'.format(obj, attr))

  def test_context_new(self):
    c = type(nutils.config)('config', eggs=1)
    self.assertNotHasattr(c, 'spam')
    self.assertEqual(c.eggs, 1)
    with c(spam=2):
      self.assertEqual(c.spam, 2)
      self.assertEqual(c.eggs, 1)
    self.assertNotHasattr(c, 'spam')
    self.assertEqual(c.eggs, 1)

  def test_context_update(self):
    c = type(nutils.config)('config', spam=1, eggs=1)
    self.assertEqual(c.spam, 1)
    self.assertEqual(c.eggs, 1)
    with c(spam=2):
      self.assertEqual(c.spam, 2)
      self.assertEqual(c.eggs, 1)
      with c(spam=3, eggs=2):
        self.assertEqual(c.spam, 3)
        self.assertEqual(c.eggs, 2)
      self.assertEqual(c.spam, 2)
      self.assertEqual(c.eggs, 1)
    self.assertEqual(c.spam, 1)
    self.assertEqual(c.eggs, 1)

  def test_delattr(self):
    c = type(nutils.config)('config', spam=1)
    with self.assertRaises(AttributeError):
      del c.spam

  def test_setattr(self):
    c = type(nutils.config)('config', spam=1)
    with self.assertRaises(AttributeError):
      c.spam = 2

  def test_str(self):
    c = type(nutils.config)('config', spam=1, eggs='test')
    self.assertEqual(str(c), "configuration: eggs='test', spam=1")
