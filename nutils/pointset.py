from . import element, numeric
import re

class Pointset( object ):

  def __init__( self, *args ):
    self.name = 'pointset_' + self.__class__.__name__.lower()
    self.cache = {}
    self.args = args

  def __call__( self, head ):
    try:
      return self.cache[head]
    except:
      func = getattr( head, self.name, None )
      pointset = func( *self.args ) if func else head.pointset( self )
      self.cache[head] = pointset
      return pointset
    
  def __getitem__( self, elem ):
    return self( elem[-1] )

class Gauss( Pointset ):
  pass

class Uniform( Pointset ):
  pass

class Vertex( Pointset ):
  pass

class Vtk( Pointset ):
  pass

vtk = Vtk()

_pattern = re.compile( '(^[a-zA-Z]+)(.*)$' )
def aspointset( p ):
  if isinstance( p, str ):
    match = _pattern.match( p )
    assert match
    P = eval( match.group(1).title() )
    args = match.group(2) and eval( match.group(2)+',' ) or ()
    p = P( *args )
  assert isinstance( p, Pointset )
  return p

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
