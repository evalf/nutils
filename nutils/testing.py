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

'''
Extensions of the :mod:`unittest` module.
'''

import unittest, sys, types as builtin_types, operator, contextlib, treelog, functools, importlib, doctest, re, zlib, binascii
import numpy
from nutils import warnings, numeric


def _not_has_module(module):
  try:
    importlib.import_module(module)
  except ImportError:
    return True
  else:
    return False

def requires(*modules):
  missing = tuple(filter(_not_has_module, modules))
  if missing:
    return unittest.skip('missing module{}: {}'.format('s' if len(missing) > 1 else '', ','.join(missing)))
  else:
    return lambda func: func


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
    TestCase = builtin_types.new_class(name, (cls.__base,), exec_body=populate)

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
  return builtin_types.new_class(TestCase.__name__, (), dict(metaclass=_ParametrizedCollection, base=TestCase))

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


class TestCase(unittest.TestCase):
  '''A class whose instances are single test cases.'''

  def setUpContext(self, stack):
    stack.enter_context(treelog.set(treelog.TeeLog(treelog.StdoutLog(), treelog.LoggingLog())))

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

  def assertAllEqual(self, actual, desired):
    for args in numpy.broadcast(actual, desired):
      self.assertEqual(*args)

  def assertAllAlmostEqual(self, actual, desired, **kwargs):
    for args in numpy.broadcast(actual, desired):
      self.assertAlmostEqual(*args, **kwargs)

  def assertAlmostEqual64(self, actual, desired, atol=2e-15, rtol=2e-3):
    '''Assert numerical equivalence with packed data.

    Test closeness of ``actual`` to ``desired`` data, where the latter are
    specified as a base64 packed data string (see :func:`nutils.numeric.pack`
    and :func:`nutils.numeric.unpack` for details on packing). The primary use
    case is embedded regression testing.

    The ``atol`` and ``rtol`` arguments are used for both unpacking and
    equivalence testing and cannot be changed independently of the base64 string.
    Doing so will raise an exception with a suggested update.

    Args
    ----
    actual : :class:`float` array
      The obtained data.
    desired : :class:`str`
      The desired data in the form of a base64 string.
    atol : :class:`float`
      Absolute tolerance
    rtol : :class:`float`
      Relative tolerance
    '''

    try:
      desired = numeric.unpack(numpy.frombuffer(zlib.decompress(binascii.a2b_base64(desired)), dtype=numpy.int16), atol, rtol).reshape(actual.shape)
    except Exception as e:
      status = ['failed to decode data: {}'.format(e)]
    else:
      error = abs(actual - desired)
      spacing = numpy.sqrt(atol**2 + (desired*rtol)**2)
      fail = numpy.logical_xor(numpy.isnan(actual), numpy.isnan(desired))
      numpy.greater(error, spacing, where=~numpy.isnan(error), out=fail)
      nfail = fail.sum()
      if not nfail:
        return
      status = ['{}/{} values do not match up to atol={:.2e}, rtol={:.2e}:'.format(nfail, fail.size, atol, rtol)]
      status.extend('{} desired: {:+.4e}, actual: {:+.4e}, spacing: {:.1e}'.format(list(index), desired[index], actual[index], spacing[index]) for index in zip(*fail.nonzero()))
      if nfail > 10:
        status[6:-5] = '...',
    status.append('If this is expected, update the base64 string to:')
    with warnings.via(status.append):
      s = binascii.b2a_base64(zlib.compress(numeric.pack(actual, atol, rtol, numpy.int16).tobytes(), 9)).decode().rstrip()
    status.extend(s[i:i+80] for i in range(0, len(s), 80))
    self.fail('\n'.join(status))


ContextTestCase = TestCase


class FloatNeighborhoodOutputChecker(doctest.OutputChecker):

  posnum = '(?:[0-9]+|[0-9]+[.]|[.][0-9]+)(?:e[+-]?[0-9]+)?'
  re_spread = re.compile('\\b((?:-?{posnum}|array[(][^()]*[)])±{posnum})\\b'.format(posnum=posnum))

  def check_output(self, want, got, optionflags):
    if want == got:
      return True
    elif '±' in want and self._check_plus_minus(want, got, optionflags):
      return True
    return super().check_output(want, got, optionflags)

  @classmethod
  def _check_plus_minus(cls, want, got, optionflags):
    if optionflags & doctest.NORMALIZE_WHITESPACE:
      want = re.sub(r'\s+', ' ', want, flags=re.MULTILINE)
      got = re.sub(r'\s+', ' ', got, flags=re.MULTILINE)
    for i, part in enumerate(cls.re_spread.split(want)):
      if i % 2 == 0:
        if got[:len(part)] != part:
          return False
        got = got[len(part):]
      elif part.startswith('array('):
        match = re.search('^array[(]([^()]*)[)]'.format(posnum=cls.posnum), got)
        if not match:
          return False
        got, got_array = got[len(match.group(0)):], cls._parse_array(match.group(1))
        want_array, want_spread = part.split('±')
        want_array = cls._parse_array(want_array[6:-1])
        if want_array.shape != got_array.shape:
          return False
        if (numpy.isnan(want_array) != numpy.isnan(got_array)).any():
          return False
        mask = numpy.isnan(want_array)
        if numpy.greater(abs(want_array - got_array)[~mask], float(want_spread)).any():
          return False
      else:
        match = re.search('^(-?{posnum})\\b'.format(posnum=cls.posnum), got)
        if not match:
          return False
        got, got_number = got[len(match.group(0)):], float(match.group(1))
        want_number, want_spread = map(float, part.split('±'))
        if not (abs(got_number - want_number) <= want_spread):
          return False
    return True

  @classmethod
  def _parse_array(cls, s):
    return numpy.array(cls._parse_array_tokens(filter(None, re.split(r'\s*([\[\],])\s*', s, flags=re.MULTILINE))), dtype=float)

  @classmethod
  def _parse_array_tokens(cls, tokens):
    token = next(tokens)
    if token == '[':
      data = [cls._parse_array_tokens(tokens)]
      for token in tokens:
        if token == ',':
          data.append(cls._parse_array_tokens(tokens))
        elif token == ']':
          return data
        else:
          raise ValueError('unexpected token: {}'.format(token))
    else:
      return float(token)

# vim:sw=2:sts=2:et
