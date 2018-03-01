import doctest as _doctest, importlib, os, tempfile, pathlib, functools, warnings, subprocess, sys
import nutils.log
from . import *


class DocTestLog(nutils.log.ContextLog):
  '''Output plain text to sys.stdout.'''

  def _mkstr(self, level, text):
    return ' > '.join(self._context + ([text] if text is not None else []))

  def write(self, level, text, endl=True):
    verbose = nutils.config.verbose
    if level not in nutils.log.LEVELS[verbose:]:
      s = self._mkstr(level, text)
      print(s, end='\n' if endl else '')


@parametrize
class module(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    stack.enter_context(nutils.config(log=stack.enter_context(DocTestLog())))

  def test(self):
    with DocTestLog():
      failcnt, testcnt = _doctest.testmod(importlib.import_module(self.name), optionflags=_doctest.ELLIPSIS)
      self.assertEqual(failcnt, 0)

@parametrize
class file(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter('ignore')
    stack.enter_context(nutils.config(log=stack.enter_context(DocTestLog())))

  def test(self):
    with DocTestLog():
      failcnt, testcnt = _doctest.testfile(str(self.path), module_relative=False, optionflags=_doctest.ELLIPSIS)
      self.assertEqual(failcnt, 0)

root = pathlib.Path(__file__).parent.parent
for path in sorted((root / 'nutils').glob('**/*.py')):
  name = '.'.join(path.relative_to(root).parts)[:-3]
  if name.endswith('.__init__'):
    name = name[:-9]
  module(name.replace('.', '/'), name=name)

for path in sorted((root / 'docs').glob('**/*.rst')):
  name = str(path.relative_to(root))
  file(name[:-4], name=name, path=path)


class sphinx(ContextTestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.tmpdir = stack.enter_context(tempfile.TemporaryDirectory(prefix='nutils'))

  def test(self):
    process = subprocess.run([sys.executable, '-m', 'sphinx', '-W', '-b', 'html', 'docs', self.tmpdir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(process.stdout)
    self.assertEqual(process.returncode, 0)
