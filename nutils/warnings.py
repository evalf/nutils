import warnings, contextlib


class NutilsWarning(Warning):
    'Base class for warnings from Nutils.'


class NutilsDeprecationWarning(NutilsWarning):
    'Warning about deprecated Nutils features.'


class NutilsInefficiencyWarning(NutilsWarning):
    'Warning about inefficient runtime.'


def warn(message, category=NutilsWarning, stacklevel=2):
    warnings.warn(message, category, stacklevel=stacklevel+1)


def deprecation(message, stacklevel=2):
    warnings.warn(message, NutilsDeprecationWarning, stacklevel=stacklevel+1)


@contextlib.contextmanager
def via(print):
    '''context manager to set/reset warnings.showwarning'''

    oldshowwarning = warnings.showwarning
    warnings.showwarning = lambda message, category, filename, lineno, *args: print(f'{category.__name__}: {message}\n  In {filename}:{lineno}')
    yield
    warnings.showwarning = oldshowwarning


# vim:sw=4:sts=4:et
