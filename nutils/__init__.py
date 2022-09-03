import sys
import numpy
from distutils.version import LooseVersion

assert sys.version_info >= (3, 5)
assert LooseVersion(numpy.version.version) >= LooseVersion('1.16'), 'nutils requires numpy 1.16 or higher, got {}'.format(numpy.version.version)

version = '7.1'
version_name = 'hiyamugi'
long_version = ('{} "{}"' if version_name else '{}').format(version, version_name)

__all__ = [
    'cache',
    'cli',
    'element',
    'elementseq',
    'evaluable',
    'export',
    'expression_v1',
    'expression_v2',
    'function',
    'matrix',
    'mesh',
    'numeric',
    'parallel',
    'points',
    'pointsseq',
    'sample',
    'solver',
    'sparse',
    'testing',
    'topology',
    'transform',
    'transformseq',
    'types',
    'unit',
    'util',
    'warnings',
]

# vim:sw=2:sts=2:et
