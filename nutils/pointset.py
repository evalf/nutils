from . import element, numeric

class DynamicPointset( object ):

  def __init__( self, *args ):
    self.name = self.__class__.__name__.lower()
    self.cache = {}
    self.args = args

  def __getitem__( self, elem ):
    return elem.pointset( self.name, *self.args )
    reference = elem.reference
    try:
      return self.cache[ reference ]
    except:
      pointset = reference.pointset( self.name, *self.args )
      self.cache[ reference ] = pointset
      return pointset

class Gauss( DynamicPointset ):
  pass

class Uniform( DynamicPointset ):
  pass

class Vertex( DynamicPointset ):
  pass

class StaticPointset( object ):

  def __init__( self, refmap ):
    self.refmap = refmap

  def __getitem__( self, elem ):
    return self.refmap[ elem.reference ]


_lin = element.Simplex(1)
_tri = element.Simplex(2)
_tet = element.Simplex(3)
_qua = _lin**2
_hex = _lin**3

vtk = StaticPointset({
  _tri: _tri.vertices,
  _tet: _tet.vertices,
  _qua: numeric.array( [[0,0],[1,0],[1,1],[0,1]] ),
  _hex: numeric.array( [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]] ),
})

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
