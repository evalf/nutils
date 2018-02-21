from . import *
import nutils.types
import inspect

class apply_annotations(TestCase):

  def test_without_annotations(self):
    @nutils.types.apply_annotations
    def f(a, b):
      return a, b
    a, b = f(1, 2)
    self.assertEqual(a, 1)
    self.assertEqual(b, 2)

  def test_pos_or_kw(self):
    @nutils.types.apply_annotations
    def f(a:int, b, c:str):
      return a, b, c
    a, b, c = f(1, 2, 3)
    self.assertEqual(a, 1)
    self.assertEqual(b, 2)
    self.assertEqual(c, '3')

  def test_with_signature(self):
    def f(a):
      return a
    f.__signature__ = inspect.Signature([inspect.Parameter('a', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)])
    f = nutils.types.apply_annotations(f)
    self.assertEqual(f(1), '1')

  def test_posonly(self):
    def f(a):
      return a
    f.__signature__ = inspect.Signature([inspect.Parameter('a', inspect.Parameter.POSITIONAL_ONLY, annotation=str)])
    f = nutils.types.apply_annotations(f)
    self.assertEqual(f(1), '1')

  def test_kwonly(self):
    @nutils.types.apply_annotations
    def f(a:str, *, b:int, c:bool):
      return a, b, c
    self.assertEqual(f(1, b='2', c=3), ('1', 2, True))

  def test_varpos(self):
    @nutils.types.apply_annotations
    def f(a:str, *args):
      return a, args
    self.assertEqual(f(1, 2, 3), ('1', (2, 3)))

  def test_varpos_annotated(self):
    @nutils.types.apply_annotations
    def f(a:str, *args:lambda args: map(str, args)):
      return a, args
    self.assertEqual(f(1, 2, 3), ('1', ('2', '3')))

  def test_varkw(self):
    @nutils.types.apply_annotations
    def f(a:str, **kwargs):
      return a, kwargs
    self.assertEqual(f(1, b=2, c=3), ('1', dict(b=2, c=3)))

  def test_varkw_annotated(self):
    @nutils.types.apply_annotations
    def f(a:str, **kwargs:lambda kwargs: {k: str(v) for k, v in kwargs.items()}):
      return a, kwargs
    self.assertEqual(f(1, b=2, c=3), ('1', dict(b='2', c='3')))

  def test_posonly_varkw(self):
    def f(a, b, **c):
      return a, b, c
    f.__signature__ = inspect.Signature([inspect.Parameter('a', inspect.Parameter.POSITIONAL_ONLY, annotation=str),
                                         inspect.Parameter('b', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
                                         inspect.Parameter('c', inspect.Parameter.VAR_KEYWORD)])
    f = nutils.types.apply_annotations(f)
    self.assertEqual(f(1, c=2, d=3), ('1', None, dict(c=2, d=3)))
    self.assertEqual(f(1, None, c=2, d=3), ('1', None, dict(c=2, d=3)))
    self.assertEqual(f(1, b=None, c=2, d=3), ('1', None, dict(c=2, d=3)))
    self.assertEqual(f(1, b=4, c=2, d=3), ('1', '4', dict(c=2, d=3)))

  def test_default_none(self):
    @nutils.types.apply_annotations
    def f(a:str=None):
      return a
    self.assertEqual(f(), None)
    self.assertEqual(f(None), None)
    self.assertEqual(f(1), '1')

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
