import numpy

assert [ int(n) for n in numpy.__version__.split('.') ] >= [1,8], \
  'nutils requires numpy 1.8 or higher, got %s' % numpy.__version__

version = '1.dev'

_ = numpy.newaxis
__all__ = [ '_', 'numpy', 'core', 'numeric', 'element', 'function',
  'mesh', 'plot', 'library', 'topology', 'util', 'matrix', 'parallel', 'log',
  'debug', 'cache', 'transform', 'rational' ]
