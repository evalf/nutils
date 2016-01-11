from nutils import *
from . import register, unittest

@register( 'point', 0 )
@register( 'line', 1 )
@register( 'triangle', 2 )
@register( 'tetrahedron', 3 )
@register( 'square', 1, 1 )
@register( 'hexagon', 1, 1, 1 )
@register( 'prism1', 2, 1 )
@register( 'prism2', 1, 2 )
def elem( *ndims ):

  ref = element.getsimplex( ndims[0] )
  for ndim in ndims[1:]:
    ref *= element.getsimplex( ndim )
  assert ref.ndims == sum(ndims)

  if ref.ndims > 0 and not isinstance( ref, element.TetrahedronReference ):
    @unittest
    def children():
      childvol = sum( abs(trans.det) * child.volume for trans, child in ref.children )
      numpy.testing.assert_almost_equal( childvol, ref.volume )
