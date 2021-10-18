# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys, numpy
from distutils.version import LooseVersion

assert sys.version_info >= (3, 5)
assert LooseVersion(numpy.version.version) >= LooseVersion('1.16'), 'nutils requires numpy 1.16 or higher, got {}'.format(numpy.version.version)

version = '7.0a0'
version_name = None
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
