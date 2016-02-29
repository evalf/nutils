import sys, numpy
from distutils.version import LooseVersion

assert sys.version_info >= (3, 3)
assert LooseVersion(numpy.version.version) >= LooseVersion('1.8'), 'nutils requires numpy 1.8 or higher, got %s' % numpy.version.version

version = '3beta'

_ = numpy.newaxis
__all__ = [ '_', 'numpy', 'core', 'numeric', 'element', 'function',
  'mesh', 'plot', 'library', 'topology', 'util', 'matrix', 'parallel', 'log',
  'debug', 'cache', 'transform', 'rational' ]
