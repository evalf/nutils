import pathlib, importlib

def load_tests(loader, suite, pattern):
  root = pathlib.Path(__file__).parent.parent
  for example in sorted((root / 'examples').glob('*.py')):
    if example.name == '__init__.py':
      continue
    mod_name = '.'.join(example.with_suffix('').relative_to(root).parts)
    mod = importlib.import_module(mod_name)
    suite.addTest(loader.loadTestsFromModule(mod))
  return suite
