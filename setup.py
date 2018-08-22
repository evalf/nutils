from setuptools import setup

long_description = """
Nutils is a Free and Open Source Python programming library for Finite Element
Method computations, developed by `Evalf Computing <http://evalf.com>`_ and
distributed under the permissive MIT license. Key features are a readable, math
centric syntax, an object oriented design, strict separation of topology and
geometry, and high level function manipulations with support for automatic
differentiation.

Nutils provides the tools required to construct a typical simulation workflow
in just a few lines of Python code, while at the same time leaving full
flexibility to build novel workflows or interact with third party tools. With
native support for Isogeometric Analysis (IGA), the Finite Cell method (FCM),
multi-physics, mixed methods, and hierarchical refinement, Nutils is at the
forefront of numerical discretization science. Efficient under-the-hood
vectorization and built-in parallellisation provide for an effortless
transition from academic research projects to full scale, real world
applications.
"""

import os, re
with open(os.path.join('nutils', '__init__.py')) as f:
  version = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)

tests_require = ['Sphinx>=1.6','pillow>2.6']

setup(
  name = 'nutils',
  version = version,
  description = 'Numerical Utilities for Finite Element Analysis',
  author = 'Evalf',
  author_email = 'info@nutils.org',
  url = 'http://nutils.org',
  packages = [ 'nutils' ],
  package_data = { 'nutils': ['_log/*'] },
  long_description = long_description,
  license = 'MIT',
  python_requires = '>=3.5',
  install_requires = ['numpy>=1.12', 'matplotlib>=1.3', 'scipy>=0.13'],
  tests_require = tests_require,
  extras_require = dict(
    test=tests_require,
    docs=['Sphinx>=1.6'],
    mkl=['mkl'],
    readthedocs=['pillow>2.6'],
  ),
  command_options = dict(
    test=dict(test_loader=('setup.py', 'unittest:TestLoader')),
  ),
)
