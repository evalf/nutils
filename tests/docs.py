import doctest as _doctest, importlib, os, tempfile, multiprocessing, pathlib, functools
import nutils.log, nutils.core
from . import register, unittest


class DocTestLog(nutils.log.ContextLog):
  '''Output plain text to sys.stdout.'''

  def _mkstr(self, level, text):
    return ' > '.join(self._context + ([text] if text is not None else []))

  def write(self, level, text, endl=True):
    verbose = nutils.core.getprop( 'verbose', len(nutils.log.LEVELS) )
    if level not in nutils.log.LEVELS[verbose:]:
      s = self._mkstr(level, text)
      print(s, end='\n' if endl else '')


@register
def doctest():

  root = pathlib.Path(__file__).parent.parent

  for f in sorted((root / 'nutils').glob('**/*.py')):
    name = '.'.join(f.relative_to(root).parts)[:-3]
    @unittest(name=name)
    def test():
      mod = importlib.import_module(name)
      __log__ = DocTestLog()
      failcnt, testcnt = _doctest.testmod(mod)
      assert failcnt == 0

  for f in sorted((root / 'docs').glob('**/*.rst')):
    name = str(f.relative_to(root))
    @unittest(name=name)
    def test():
      finder = _doctest.DocTestFinder(recurse=False)
      runner = _doctest.DocTestRunner()
      __log__ = DocTestLog()
      with f.open() as g:
        for t in finder.find(g.read(), name, globs={}):
          runner.run(t)
          assert not runner.failures


@register
def sphinx():

  @unittest
  def test():
    import sphinx
    with tempfile.TemporaryDirectory(prefix='nutils') as tmpdir:
      process = multiprocessing.Process(target=sphinx.main, args=[['sphinx', '-W', '-b', 'html', 'docs', tmpdir]])
      process.start()
      process.join()
      assert not process.exitcode, 'sphinx failed'
