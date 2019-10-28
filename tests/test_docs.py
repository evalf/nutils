import doctest as _doctest, unittest, importlib, os, tempfile, pathlib, functools, warnings, subprocess, sys, treelog
import nutils.testing

info = treelog.proto.Level.info if hasattr(treelog, 'proto') else 1
_doctestlog = treelog.FilterLog(treelog.StdoutLog(), minlevel=info)

class DocTestCase(nutils.testing.ContextTestCase, _doctest.DocTestCase):

  def __init__(self, test, *, requires=None, **kwargs):
    self.__test = test
    self.__requires = tuple(requires) if requires else ()
    super().__init__(test, **kwargs)

  def setUpContext(self, stack):
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

    super().setUpContext(stack)
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    stack.enter_context(treelog.set(_doctestlog))
    import numpy
    printoptions = numpy.get_printoptions()
    if 'legacy' in printoptions:
      stack.callback(numpy.set_printoptions, **printoptions)
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
for path in sorted((root/'nutils').glob('**/*.py')):
  name = '.'.join(path.relative_to(root).parts)[:-3]
  if name.endswith('.__init__'):
    name = name[:-9]
  module = importlib.import_module(name)
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


class sphinx(nutils.testing.TestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.tmpdir = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='nutils')))

  @nutils.testing.requires('sphinx', 'matplotlib', 'scipy')
  def test(self):
    from sphinx.application import Sphinx
    app = Sphinx(srcdir=str(root/'docs'),
                 confdir=str(root/'docs'),
                 outdir=str(self.tmpdir/'html'),
                 doctreedir=str(self.tmpdir/'doctree'),
                 buildername='html',
                 freshenv=True,
                 warningiserror=True,
                 confoverrides=dict(nitpicky=True))
    app.build()
    if app.statuscode:
      self.fail('sphinx build failed with code {}'.format(app.statuscode))


def load_tests(loader, suite, pattern):
  # Ignore default suite (containing `DocTestCase`).
  suite = unittest.TestSuite()
  suite.addTest(doctest)
  suite.addTests(loader.loadTestsFromTestCase(sphinx))
  return suite
