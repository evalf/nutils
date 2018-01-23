extra = {}
try:
  from setuptools import setup, Extension
except:
  from distutils.core import setup, Extension
else:
  extra['install_requires'] = ['numpy>=1.12', 'matplotlib>=1.3', 'scipy>=0.13', 'sphinx']

long_description = """
The nutils project is a collaborative programming effort aimed at the creation
of a general purpose python programming library for setting up finite element
computations. Identifying features are a heavily object oriented design, strict
separation of topology and geometry, and CAS-like function arithmetic such as
found in maple and mathematica. Primary design goals are:

  * Readability. Finite element scripts built on top of nutils should focus
    on work flow and maths, unobscured by finite element infrastructure.
  * Flexibility. The nutils are tools; they do not enforce a strict work
    flow. Missing components can be added locally without loosing
    interoperability.
  * Compatibility. Exposed objects are of native python type or allow for
    easy conversion to leverage third party tools.
  * Speed. Nutils are self-optimizing and support parallel computation.
    Typical scripting inefficiencies are discouraged by design.

The nutils are under active development, and are presently in use for academic
research by Phd and MSc students.
"""

import os, re
with open(os.path.join('nutils', '__init__.py')) as f:
  version = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)

setup(
  name = 'nutils',
  version = version,
  description = 'Numerical Utilities',
  author = 'Evalf',
  author_email = 'info@nutils.org',
  url = 'http://nutils.org',
  packages = [ 'nutils' ],
  package_data = { 'nutils': ['_log/*'] },
  long_description = long_description,
  **extra
)
