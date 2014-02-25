#!/usr/bin/env python

from nutils import mesh, topology, pointset, log
from test import unittest, testgroup
import numpy

def surface_volume( ndims ):
  volume = 1.
  surf0 = 2.
  for idim in range( ndims ):
    surface = surf0
    surf0 = volume * (2*numpy.pi)
    volume = surface / (idim+1)
  return surface, volume

@testgroup
def teststuff( ndims, maxrefine, errorbound ):

  nelems = 3
  verts = numpy.linspace( 0, 2**(1./ndims), nelems+1 )
  wholedomain, geom = mesh.rectilinear( [verts]*ndims )
  levelset = ( geom**2 ).sum() - 1.
  domain, complement = wholedomain.trim( levelset, maxrefine=maxrefine )

  @unittest
  def volume():
    volume = domain.integrate( 1., geometry=geom, ischeme='gauss1' )
    ball_surface, ball_volume = surface_volume( ndims )
    exact_volume = 2 - .5**ndims * ball_volume
    error = ( volume - exact_volume ) / exact_volume
    log.info( 'volume %f, expected %f, error %.3f%%' % ( volume, exact_volume, error ) )
    assert error < errorbound

#   topo = topology.Topology( domain.get_trimmededges( maxrefine ), ndims=ndims-1 )

#   vol_gauss = (1./float(ndims))*topo.integrate( sum(geom*geom.normal()), geometry=geom, ischeme='gauss1' )
#   print 'Volume (Gauss)=', vol_gauss, '(%5.4f)' % vol

#   numpy.testing.assert_almost_equal( vol_gauss, vol, decimal=14 )

# @unittest
# def surfacearea():
#   topo = topology.Topology( domain.get_trimmededges( maxrefine ), ndims=ndims-1 )
#   surf = topo.integrate( 1., geometry=geom, ischeme='gauss1' )
#   print 'Surface area =', surf, '(%5.4f)' % surf_exact
#   numpy.testing.assert_almost_equal( surf, surf_exact, decimal=surf_decimal )

teststuff( ndims=2, maxrefine=3, errorbound=0.00073 )
teststuff( ndims=3, maxrefine=2, errorbound=0.00293 )
