from . import element, numeric, cache
import re, warnings

class Pointset( cache.Immutable ):

  def __init__( self, name, *args ):
    self.name = name
    self.args = args

  def __call__( self, elem ):
    func = getattr( elem, 'pointset_' + self.name, None )
    return func( *self.args ) if func else elem.pointset( self )

vtk = Pointset( 'vtk' )

_pattern = re.compile( '(^[a-zA-Z]+)(.*)$' )
def aspointset( p ):
  if isinstance( p, str ):
    match = _pattern.match( p )
    assert match
    name, args = match.groups()
    args = args and eval( args+',' ) or ()
    p = Pointset( name, *args )
  assert isinstance( p, Pointset )
  return p

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
