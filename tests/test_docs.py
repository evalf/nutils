import doctest as _doctest, importlib, os, tempfile, pathlib, functools, warnings, subprocess, sys, treelog
from nutils.testing import *

_doctestlog = treelog.FilterLog(treelog.StdoutLog(), minlevel=1)

@parametrize
class module(TestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    stack.enter_context(treelog.set(_doctestlog))
    import numpy
    printoptions = numpy.get_printoptions()
    if 'legacy' in printoptions:
      stack.callback(numpy.set_printoptions, **printoptions)
      numpy.set_printoptions(legacy='1.13')

  def test(self):
    failcnt, testcnt = _doctest.testmod(importlib.import_module(self.name), optionflags=_doctest.ELLIPSIS)
    self.assertEqual(failcnt, 0)

@parametrize
class file(TestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    stack.enter_context(treelog.set(_doctestlog))

  def test(self):
    failcnt, testcnt = _doctest.testfile(str(self.path), module_relative=False, optionflags=_doctest.ELLIPSIS)
    self.assertEqual(failcnt, 0)

root = pathlib.Path(__file__).parent.parent
for path in sorted((root / 'nutils').glob('**/*.py')):
  name = '.'.join(path.relative_to(root).parts)[:-3]
  if name.endswith('.__init__'):
    name = name[:-9]
  module(name.replace('.', '/'), name=name)

for path in sorted((root / 'docs').glob('**/*.rst')):
  if path == root / 'docs' / 'tutorial.rst':
    continue
  name = str(path.relative_to(root))
  file(name[:-4], name=name, path=path)


class sphinx(TestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.tmpdir = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='nutils')))

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
