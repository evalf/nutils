import doctest as _doctest, unittest, importlib, os, tempfile, pathlib, functools, warnings, subprocess, sys, treelog
import nutils.testing

info = treelog.proto.Level.info if hasattr(treelog, 'proto') else 1
_doctestlog = treelog.FilterLog(treelog.StdoutLog(), minlevel=info)

class DocTestCase(nutils.testing.ContextTestCase, _doctest.DocTestCase):

  def __init__(self, test, *, requires=None, **kwargs):
    self.__test = test
    self.__requires = tuple(requires) if requires else ()
    super().__init__(test, **kwargs)

  def setUp(self):
    lines = self.__test.docstring.splitlines()
    indent = min((len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()), default=0)
    blank = True
    requires = list(self.__requires)
    for line in lines:
      if blank and line[indent:].startswith('.. requires:: '):
        requires.extend(name.strip() for name in line[indent+13:].split(','))
      blank = not line.strip()
    missing = tuple(filter(nutils.testing._not_has_module, requires))
    if missing:
      self.skipTest('missing module{}: {}'.format('s' if len(missing) > 1 else '', ','.join(missing)))

    if 'matplotlib' in requires:
      import matplotlib.testing
      matplotlib.testing.setup()

    super().setUp()
    self.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    self.enter_context(treelog.set(_doctestlog))
    import numpy
    printoptions = numpy.get_printoptions()
    if 'legacy' in printoptions:
      self.addCleanup(numpy.set_printoptions, **printoptions)
      numpy.set_printoptions(legacy='1.13')

  def shortDescription(self):
    return None

  def __repr__(self):
    return '{} ({}.doctest)'.format(self.id(), __name__)

  __str__ = __repr__

doctest = unittest.TestSuite()
parser = _doctest.DocTestParser()
finder = _doctest.DocTestFinder(parser=parser)
checker = nutils.testing.FloatNeighborhoodOutputChecker()
root = pathlib.Path(__file__).parent.parent
for name in nutils.__all__:
  module = importlib.import_module('.'+name, 'nutils')
  for test in sorted(finder.find(module)):
    if len(test.examples) == 0:
      continue
    if not test.filename:
      test.filename = module.__file__
    doctest.addTest(DocTestCase(test, optionflags=_doctest.ELLIPSIS, checker=checker))
for path in sorted((root/'docs').glob('**/*.rst')):
  name = str(path.relative_to(root))
  with path.open(encoding='utf-8') as f:
    doc = f.read()
  test = parser.get_doctest(doc, globs={}, name=name, filename=str(path), lineno=0)
  if test.examples:
    doctest.addTest(DocTestCase(test, optionflags=_doctest.ELLIPSIS, checker=checker, requires=['matplotlib']))

del DocTestCase

def load_tests(loader, suite, pattern):
  # Ignore default suite (containing `DocTestCase`).
  return doctest
