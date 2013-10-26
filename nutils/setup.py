from distutils.core import setup, Extension
import numpy

module = Extension( '_numeric', sources = [ '_numeric.c' ] )

setup( name = 'NumpyExtraModule',
       version = '1.0',
       description = 'This is the numeric module',
       include_dirs = [ numpy.get_include() ],
       ext_modules = [ module ] )
