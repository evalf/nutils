#! /usr/bin/env python

import numpy

extra = {}
try:
  from setuptools import setup, Extension
except:
  from distutils.core import setup, Extension
else:
  extra['install_requires'] = [ 'numpy>=1.8', 'matplotlib>=1.3', 'scipy>=0.13' ]

setup(
  name = 'nutils',
  version = '0.99',
  include_dirs = [ numpy.get_include() ],
  ext_modules = [ Extension( 'nutils._numeric', sources = ['nutils/_numeric.c'] ) ],
  description = 'Numerical Utilities',
  author = 'Gertjan van Zwieten and others',
  author_email = 'info@nutils.org',
  url = 'http://nutils.org',
  packages = [ 'nutils' ],
  package_data = { 'nutils': ['_log/*'] },
  long_description = open('README.md').read(),
  **extra
)
