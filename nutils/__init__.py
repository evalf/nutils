import numpy
from distutils.version import LooseVersion

assert LooseVersion(numpy.version.version) >= LooseVersion('1.8'), 'nutils requires numpy 1.8 or higher, got %s' % numpy.version.version

version = '2.0'

_ = numpy.newaxis
__all__ = [ '_', 'numpy', 'core', 'numeric', 'element', 'function',
  'mesh', 'plot', 'library', 'topology', 'util', 'matrix', 'parallel', 'log',
  'debug', 'cache', 'transform', 'rational' ]
