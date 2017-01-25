import os, importlib
from . import register, unittest

@register
def test_examples():
  for name in os.listdir( 'examples' ):
    if not name.endswith( '.py' ):
      continue
    example = importlib.import_module( 'examples.'+name[:-3] )
    for __nprocs__ in 1,2:
      unittest( example.unittest, name='{}_np{}'.format( name[:-3], __nprocs__ ) )
