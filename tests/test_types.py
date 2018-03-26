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

class nutils_hash(TestCase):

  def test_ellipsis(self):
    self.assertEqual(nutils.types.nutils_hash(...).hex(), '0c8bce06e451e4d5c49f60da0abf2ccbadf80600')

  def test_None(self):
    self.assertEqual(nutils.types.nutils_hash(None).hex(), 'bdfcbd663476b2db5b2b2e59a6d93882a908dc76')

  def test_bool(self):
    self.assertEqual(nutils.types.nutils_hash(False).hex(), '04a5e8f73dcea55dcd7482a476cf2e7b53d6dc50')
    self.assertEqual(nutils.types.nutils_hash(True).hex(), '3fe990437e1624c831729f2866979254437bb7e9')

  def test_int(self):
    self.assertEqual(nutils.types.nutils_hash(1).hex(), '00ec7dea895ebd921e56bbc554688d8b3a1e4dfc')
    self.assertEqual(nutils.types.nutils_hash(2).hex(), '8ae88fa39407cf75e46f9e0aba8c971de2256b14')

  def test_float(self):
    self.assertEqual(nutils.types.nutils_hash(1.).hex(), 'def4bae4f2a3e29f6ddac537d3fa7c72195e5d8b')
    self.assertEqual(nutils.types.nutils_hash(2.5).hex(), '5216c2bf3c16d8b8ff4d9b79f482e5cea0a4cb95')

  def test_complex(self):
    self.assertEqual(nutils.types.nutils_hash(1+0j).hex(), 'cf7a0d933b7bb8d3ca252683b137534a1ecae073')
    self.assertEqual(nutils.types.nutils_hash(2+1j).hex(), 'ee088890528f941a80aa842dad36591b05253e55')

  def test_inequality_numbers(self):
    self.assertNotEqual(nutils.types.nutils_hash(1).hex(), nutils.types.nutils_hash(1.).hex())
    self.assertNotEqual(nutils.types.nutils_hash(1).hex(), nutils.types.nutils_hash(1+0j).hex())
    self.assertNotEqual(nutils.types.nutils_hash(1).hex(), nutils.types.nutils_hash(True).hex())

  def test_str(self):
    self.assertEqual(nutils.types.nutils_hash('spam').hex(), '3ca1023ab75a68dc7b0f83b43ec624704a7aef61')
    self.assertEqual(nutils.types.nutils_hash('eggs').hex(), '124b0a7b3984e08125c380f7454896c1cad22e2c')

  def test_bytes(self):
    self.assertEqual(nutils.types.nutils_hash(b'spam').hex(), '5e717ec15aace7c25610c1dea340f2173f2df014')
    self.assertEqual(nutils.types.nutils_hash(b'eggs').hex(), '98f2061978497751cac94f982fd96d9b015b74c3')

  def test_tuple(self):
    self.assertEqual(nutils.types.nutils_hash(()).hex(), '15d44755bf0731b2a3e9a5c5c8e0807b61881a1f')
    self.assertEqual(nutils.types.nutils_hash((1,)).hex(), '328b16ebbc1815cf579ae038a35c4d68ebb022af')
    self.assertNotEqual(nutils.types.nutils_hash((1,'spam')).hex(), nutils.types.nutils_hash(('spam',1)).hex())

  def test_frozenset(self):
    self.assertEqual(nutils.types.nutils_hash(frozenset([1,2])).hex(), '3862dc7e5321bc8a576c385ed2c12c71b96a375a')
    self.assertEqual(nutils.types.nutils_hash(frozenset(['spam','eggs'])).hex(), '2c75fd3db57f5e505e1425ae9ff6dcbbc77fd123')

  def test_type_bool(self):
    self.assertEqual(nutils.types.nutils_hash(bool).hex(), 'feb912889d52d45fcd1e778c427b093a19a1ea78')

  def test_type_int(self):
    self.assertEqual(nutils.types.nutils_hash(int).hex(), 'aa8cb9975f7161b1f7ceb88b4b8585b49946b31e')

  def test_type_float(self):
    self.assertEqual(nutils.types.nutils_hash(float).hex(), '6d5079a53075f4b6f7710377838d8183730f1388')

  def test_type_complex(self):
    self.assertEqual(nutils.types.nutils_hash(complex).hex(), '6b00f6b9c6522742fd3f8054af6f10a24a671fff')

  def test_type_str(self):
    self.assertEqual(nutils.types.nutils_hash(str).hex(), '2349e11586163208d2581fe736630f4e4b680a7b')

  def test_type_bytes(self):
    self.assertEqual(nutils.types.nutils_hash(bytes).hex(), 'b0826ca666a48739e6f8b968d191adcefaa39670')

  def test_type_tuple(self):
    self.assertEqual(nutils.types.nutils_hash(tuple).hex(), '07cb4a24ca8ac53c820f20721432b4726e2ad1af')

  def test_type_frozenset(self):
    self.assertEqual(nutils.types.nutils_hash(frozenset).hex(), '48dc7cd0fbd54924498deb7c68dd363b4049f5e2')

  def test_custom(self):
    class custom:
      @property
      def __nutils_hash__(self):
        return b'01234567890123456789'
    self.assertEqual(nutils.types.nutils_hash(custom()).hex(), b'01234567890123456789'.hex())

  def test_unhashable(self):
    with self.assertRaises(TypeError):
      nutils.types.nutils_hash([])

class CacheMeta(TestCase):

  def test_property(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = 'x',
          @property
          def x(self):
            nonlocal ncalls
            ncalls += 1
            return 1

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.x, 1)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x, 1)
        self.assertEqual(ncalls, 1)

  def test_set_property(self):

    class T(metaclass=nutils.types.CacheMeta):
      __cache__ = 'x',
      @property
      def x(self):
        return 1

    t = T()
    with self.assertRaises(AttributeError):
      t.x = 1

  def test_del_property(self):

    class T(metaclass=nutils.types.CacheMeta):
      __cache__ = 'x',
      @property
      def x(self):
        return 1

    t = T()
    with self.assertRaises(AttributeError):
      del t.x

  def test_method_without_args(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = 'x',
          def x(self):
            nonlocal ncalls
            ncalls += 1
            return 1

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.x(), 1)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(), 1)
        self.assertEqual(ncalls, 1)

  def test_method_with_args(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = 'x',
          def x(self, a, b):
            nonlocal ncalls
            ncalls += 1
            return a + b

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.x(1, 2), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(a=1, b=2), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(2, 2), 4)
        self.assertEqual(ncalls, 2)
        self.assertEqual(t.x(a=2, b=2), 4)
        self.assertEqual(ncalls, 2)
        self.assertEqual(t.x(1, 2), 3)
        self.assertEqual(ncalls, 3)

  def test_method_with_args_and_preprocessors(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = 'x',
          @nutils.types.apply_annotations
          def x(self, a:int, b:int):
            nonlocal ncalls
            ncalls += 1
            return a + b

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.x(1, 2), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(a='1', b='2'), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x('2', '2'), 4)
        self.assertEqual(ncalls, 2)
        self.assertEqual(t.x(a=2, b=2), 4)
        self.assertEqual(ncalls, 2)
        self.assertEqual(t.x('1', 2), 3)
        self.assertEqual(ncalls, 3)

  def test_method_with_kwargs(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = 'x',
          def x(self, a, **kwargs):
            nonlocal ncalls
            ncalls += 1
            return a + sum(kwargs.values())

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.x(1, b=2), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(a=1, b=2), 3)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.x(1, b=2, c=3), 6)
        self.assertEqual(ncalls, 2)
        self.assertEqual(t.x(a=1, b=2, c=3), 6)
        self.assertEqual(ncalls, 2)

  def test_subclass_redefined_property(self):

    class T(metaclass=nutils.types.CacheMeta):
      __cache__ = 'x',
      @property
      def x(self):
        return 1

    class U(T):
      __cache__ = 'x',
      @property
      def x(self):
        return super().x + 1
      @property
      def y(self):
        return super().x

    u1 = U()
    self.assertEqual(u1.x, 2)
    self.assertEqual(u1.y, 1)

    u2 = U()
    self.assertEqual(u2.y, 1)
    self.assertEqual(u2.x, 2)

  def test_missing_attribute(self):

    with self.assertRaisesRegex(TypeError, 'Attribute listed in __cache__ is undefined: x'):
      class T(metaclass=nutils.types.CacheMeta):
        __cache__ = 'x',

  def test_invalid_attribute(self):

    with self.assertRaisesRegex(TypeError, "Don't know how to cache attribute x: None"):
      class T(metaclass=nutils.types.CacheMeta):
        __cache__ = 'x',
        x = None

  def test_name_mangling(self):

    for withslots in False, True:
      with self.subTest(withslots=withslots):

        class T(metaclass=nutils.types.CacheMeta):
          if withslots:
            __slots__ = ()
          __cache__ = '__x',
          @property
          def __x(self):
            nonlocal ncalls
            ncalls += 1
            return 1
          @property
          def y(self):
            return self.__x

        ncalls = 0
        t = T()
        self.assertEqual(ncalls, 0)
        self.assertEqual(t.y, 1)
        self.assertEqual(ncalls, 1)
        self.assertEqual(t.y, 1)
        self.assertEqual(ncalls, 1)

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
