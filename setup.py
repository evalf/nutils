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

setup(
  name = 'nutils',
  version = version,
  description = 'Numerical Utilities for Finite Element Analysis',
  author = 'Evalf',
  author_email = 'info@nutils.org',
  url = 'http://nutils.org',
  download_url = 'https://github.com/nutils/nutils/releases',
  packages = ['nutils'],
  long_description = long_description,
  license = 'MIT',
  python_requires = '>=3.5',
  install_requires = ['numpy>=1.12', 'treelog>=1.0b5', 'stringly'],
  extras_require = dict(
    docs=['Sphinx>=1.6','scipy>=0.13','matplotlib>=1.3'],
    matrix_scipy=['scipy>=0.13'],
    matrix_mkl=['mkl'],
    export_mpl=['matplotlib>=1.3','pillow>2.6'],
    import_gmsh=['meshio'],
  ),
  command_options = dict(
    test=dict(test_loader=('setup.py', 'unittest:TestLoader')),
  ),
)
