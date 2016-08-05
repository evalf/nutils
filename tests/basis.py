from nutils import *
from . import register, unittest

@register( 'tri:discont0', btype='discont', degree=0 )
@register( 'tri:discont1', btype='discont', degree=1 )
@register( 'tri:std1', btype='std', degree=1 )
@register( '1D:discont0', btype='discont', degree=0, ndims=1 )
@register( '1D:discont1', btype='discont', degree=1, ndims=1 )
@register( '1D:discont2', btype='discont', degree=2, ndims=1 )
@register( '1D:std1', btype='std', degree=1, ndims=1 )
@register( '1D:std2', btype='std', degree=2, ndims=1 )
@register( '1D:std3', btype='std', degree=3, ndims=1 )
@register( '1D:spline2', btype='spline', degree=2, ndims=1 )
@register( '1D:spline3', btype='spline', degree=3, ndims=1 )
@register( '2D:discont0', btype='discont', degree=0, ndims=2 )
@register( '2D:discont1', btype='discont', degree=1, ndims=2 )
@register( '2D:discont2', btype='discont', degree=2, ndims=2 )
@register( '2D:std1', btype='std', degree=1, ndims=2 )
@register( '2D:std2', btype='std', degree=2, ndims=2 )
@register( '2D:std3', btype='std', degree=3, ndims=2 )
@register( '2D:spline2', btype='spline', degree=2, ndims=2 )
@register( '2D:spline3', btype='spline', degree=3, ndims=2 )
@register( '3D:discont0', btype='discont', degree=0, ndims=3 )
@register( '3D:discont1', btype='discont', degree=1, ndims=3 )
@register( '3D:discont2', btype='discont', degree=2, ndims=3 )
@register( '3D:std1', btype='std', degree=1, ndims=3 )
@register( '3D:std2', btype='std', degree=2, ndims=3 )
@register( '3D:std3', btype='std', degree=3, ndims=3 )
@register( '3D:spline2', btype='spline', degree=2, ndims=3 )
@register( '3D:spline3', btype='spline', degree=3, ndims=3 )
def basis( btype, degree, ndims=None ):

  domain, geom = mesh.rectilinear( [[0,1,2]]*ndims ) if ndims else mesh.demo()
  basis = domain.basis( btype, degree=degree )
  gauss = 'gauss{}'.format(2*degree)

  @unittest
  def pum():
    error = numpy.sqrt( domain.integrate( (1-basis.sum(0))**2, geometry=geom, ischeme=gauss ) )
    numpy.testing.assert_almost_equal( error, 0, decimal=14 )

  @unittest
  def poly():
    target = (geom**degree).sum(-1)
    projection = domain.projection( target, onto=basis, geometry=geom, ischeme=gauss, droptol=0 )
    error = numpy.sqrt( domain.integrate( (target-projection)**2, geometry=geom, ischeme=gauss ) )
    numpy.testing.assert_almost_equal( error, 0, decimal=12 )
