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
