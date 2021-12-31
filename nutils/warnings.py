import warnings

class NutilsWarning(Warning):
  'Base class for warnings from Nutils.'

class NutilsDeprecationWarning(NutilsWarning):
  'Warning about deprecated Nutils features.'

class NutilsInefficiencyWarning(NutilsWarning):
  'Warning about inefficient runtime.'

def warn(message, category=NutilsWarning, stacklevel=1):
  warnings.warn(message, category, stacklevel=stacklevel)

def deprecation(message):
  warnings.warn(message, NutilsDeprecationWarning, stacklevel=2)

class via:
  '''context manager to set/reset warnings.showwarning'''

  def __init__(self, print):
    self.print = print

  def __enter__(self):
    self.oldshowwarning = warnings.showwarning
    warnings.showwarning = self.showwarning

  def __exit__(self, *args):
    warnings.showwarning = self.oldshowwarning

  def showwarning(self, message, category, filename, lineno, *args):
    self.print('{}: {}\n  In {}:{}'.format(category.__name__, message, filename, lineno))

# vim:sw=2:sts=2:et
