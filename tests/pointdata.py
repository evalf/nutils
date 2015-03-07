from nutils import *
from . import register, unittest

grid = numpy.linspace( 0., 1., 5 )

@register
def check():

  @unittest
  def pointdata1d():

    domain, geom = mesh.rectilinear( 1*(grid,) )
    ischeme = 'gauss4'

    #Create point data for the function "x"
    pdat = function.pointdata( domain, ischeme, geom )

    res = domain.integrate( pdat, geometry=geom, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.5, decimal=15 )

    #Update the point data with the function "1-x"
    pdat = pdat.update_max( 1-geom )

    res = domain.integrate( pdat, geometry=geom, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.75, decimal=15 )

    #Update the point data with the function "x-1"
    pdat = pdat.update_max( geom-1 )

    res = domain.integrate( pdat, geometry=geom, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.75, decimal=15 )

    #Update the point data with the function "0.25+x"
    pdat = pdat.update_max( 0.5+geom )

    res = domain.integrate( pdat, geometry=geom, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 17./16., decimal=15 )
