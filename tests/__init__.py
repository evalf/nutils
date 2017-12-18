import unittest, sys, types, operator, contextlib
import nutils.core


class _ParametrizedCollection(type):

  def __new__(mcls, name, bases, namespace, base):
    return super().__new__(mcls, name, bases, namespace)

  def __init__(cls, name, bases, namespace, base):
    super().__init__(name, bases, namespace)
    cls.__base = base
    cls.__test_cases = []
    for attr in '__module__', '__qualname__', '__doc__':
      if hasattr(base, attr):
        setattr(cls, attr, getattr(base, attr))

  def __call__(*args, **params):
    assert 1 <= len(args) <= 2
    cls = args[0]

    if len(args) == 1 and not params:
      loader = unittest.defaultTestLoader
      ts = unittest.TestSuite()
      for test_case in sorted(cls.__test_cases, key=operator.attrgetter('__name__')):
        ts.addTest(loader.loadTestsFromTestCase(test_case))
      return ts

    name = args[1] if len(args) == 2 else None
    if name is None:
      name = ','.join('{}={}'.format(k, v) for k, v in sorted(params.items()))
      name = name.replace('%', '%{}'.format(ord('%'))).replace('.', '%{}'.format(ord('.')))
    assert '.' not in name
    assert not hasattr(cls, name), 'duplicate test name'

    def setUp(self):
      for k, v in params.items():
        setattr(self, k, v)
      return cls.__base.setUp(self)
    def populate(ns):
      for k, v in vars(cls.__base).items():
        enable_if = getattr(v, '_parametrize_enable_if', None)
        if enable_if and not enable_if(**params):
          ns[k] = None
        else:
          for skip_if, reason in getattr(v, '_parametrize_skip_if', []):
            if skip_if(**params):
              ns[k] = unittest.skip(reason)(v)
              break
      ns.update(setUp=setUp, __qualname__=cls.__qualname__+':'+name, __module__=cls.__module__, __doc__=cls.__doc__)
      return ns
    TestCase = types.new_class(name, (cls.__base,), exec_body=populate)

    cls.__test_cases.append(TestCase)
    # Add `TestCase` as `name` to this collection.
    setattr(cls, name, TestCase)
    # Trick `unittest.loader.TestLoader.loadTestsFromModule` into finding
    # this test case.
    setattr(sys.modules[cls.__module__], cls.__qualname__+':'+name, TestCase)


def parametrize(TestCase):
  '''Parametrize a :class:`unittest.TestCase`.

  >>> @parametrize
  ... class TestSomething(unittest.TestCase):
  ...   def test_equality(self):
  ...     self.assertEqual(self.x, self.y)
  >>> TestSomething(x=1, y=1)
  >>> TestSomething(x=2, y=2)
  '''
  return types.new_class(TestCase.__name__, (), dict(metaclass=_ParametrizedCollection, base=TestCase))

def _parametrize_enable_if(test):
  def wrapper(func):
    func._parametrize_enable_if = test
    return func
  return wrapper

def _parametrize_skip_if(test, reason):
  def wrapper(func):
    if not hasattr(func, '_parametrize_skip_if'):
      func._parametrize_skip_if = []
    func._parametrize_skip_if.append((test, reason))
    return func
  return wrapper

parametrize.enable_if = _parametrize_enable_if
parametrize.skip_if = _parametrize_skip_if

TestCase = unittest.TestCase

class ContextTestCase(TestCase):

  def setUpContext(self, stack):
    pass

  def setUp(self):
    super().setUp()
    stack = contextlib.ExitStack()
    stack.__enter__()
    try:
      self.setUpContext(stack)
    except:
      stack.__exit__(None, None, None)
      raise
    else:
      self.addCleanup(stack.__exit__, None, None, None)


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
