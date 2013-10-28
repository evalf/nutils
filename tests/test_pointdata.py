from nutils import *
import numpy

grid = numpy.linspace( 0., 1., 5 )

class TestPointdata ( object ):

  def test_1Dpointdata ( self ):

    domain, coords = mesh.rectilinear( 1*(grid,) )
    ischeme = 'gauss4'

    #Create point data for the function "x"
    pdat = function.pointdata( domain, ischeme, coords )

    res = domain.integrate( pdat, coords=coords, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.5, decimal=15 )

    #Update the point data with the function "1-x"
    pdat = pdat.update_max( 1-coords )

    res = domain.integrate( pdat, coords=coords, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.75, decimal=15 )

    #Update the point data with the function "x-1"
    pdat = pdat.update_max( coords-1 )

    res = domain.integrate( pdat, coords=coords, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 0.75, decimal=15 )

    #Update the point data with the function "0.25+x"
    pdat = pdat.update_max( 0.5+coords )

    res = domain.integrate( pdat, coords=coords, ischeme=ischeme )[0]

    numpy.testing.assert_almost_equal( res, 17./16., decimal=15 )
