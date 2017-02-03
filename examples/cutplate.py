#! /usr/bin/env python3

from nutils import *


@log.title
def makeplots( domain, geom, sigma, index ):

  sigma_dev = sigma - (function.trace(sigma)/domain.ndims) * function.eye(domain.ndims)
  vonmises = function.sqrt( ( sigma_dev**2 ).sum([0,1]) * 3./2 ) # TODO check fix for 2D

  points, colors = domain.simplex.elem_eval( [ geom, vonmises ], ischeme='bezier5', separate=True )
  with plot.PyPlot( 'solution', index=index ) as plt:
    plt.mesh( points, colors )
    plt.colorbar()
    plt.xlim( 0, 1.3 )
    plt.ylim( 0, 1.3 )


def main(
    nelems: 'number of elements, 0 for triangulation' = 0,
    maxrefine: 'maxrefine level for trimming' = 2,
    radius: 'cut-out radius' = .5,
    degree: 'polynomial degree' = 1,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    solvetol: 'solver tolerance' = 1e-5,
    plots: 'create plots' = True,
  ):

  if nelems > 0:
    verts = numpy.linspace( 0, 1, nelems+1 )
    wholedomain, geom = mesh.rectilinear( [verts,verts] )
  else:
    wholedomain, geom = mesh.demo()
    if degree != 1:
      log.warning( 'setting degree=1 for triangular mesh' )
      degree = 1

  stress = library.Hooke( lmbda=lmbda, mu=mu )

  # plane strain case (see e.g. http://en.wikiversity.org/wiki/Introduction_to_Elasticity/Plate_with_hole_in_tension)
  x, y = geom / radius
  r2 = x**2 + y**2
  uexact = .2 * geom * ( [1-stress.nu,-stress.nu] + [2-2*stress.nu,2*stress.nu-1]/r2 + (.5-.5/r2)*[x**2-3*y**2,3*x**2-y**2]/r2**2 )

  levelset = function.norm2( geom ) - radius
  domain = wholedomain.trim( levelset, maxrefine=maxrefine )
  complement = wholedomain - domain
  dbasis = domain.basis( 'spline', degree=degree ).vector( 2 )

  cons = domain.boundary['left'].project( 0, geometry=geom, ischeme='gauss6', onto=dbasis[:,0] )
  cons |= domain.boundary['bottom'].project( 0, geometry=geom, ischeme='gauss6', onto=dbasis[:,1] )
  cons |= domain.boundary['top,right'].project( uexact, geometry=geom, ischeme='gauss6', onto=dbasis )

  elasticity = function.outer( dbasis.grad(geom), stress(dbasis.symgrad(geom)) ).sum([2,3])
  matrix = domain.integrate( elasticity, geometry=geom, ischeme='gauss6' )
  lhs = matrix.solve( constrain=cons, tol=solvetol, symmetric=True, precon='diag' )
  disp = dbasis.dot( lhs )

  if plots:
    makeplots( domain, geom+disp, stress(disp.symgrad(geom)), index=nelems )

  error = disp - uexact
  err = numpy.sqrt( domain.integrate( [ (error**2).sum(-1), ( error.grad(geom)**2 ).sum([-2,-1]) ], geometry=geom, ischeme='gauss7' ) )
  log.user( 'errors: l2={}, h1={}'.format(*err) )

  return err, cons, lhs


def conv( degree=1, nrefine=4 ):

  l2err = []
  h1err = []

  for irefine in log.range( 'refine', nrefine ):
    err, cons, lhs = main( nelems=2**(1+irefine), degree=degree )
    l2err.append( err[0] )
    h1err.append( err[1] )

  h = .5**numpy.arange(nrefine)

  with plot.PyPlot( 'convergence' ) as plt:
    plt.subplot( 211 )
    plt.loglog( h, l2err, 'k*--' )
    plt.slope_triangle( h, l2err )
    plt.ylabel( 'L2 error' )
    plt.grid( True )
    plt.subplot( 212 )
    plt.loglog( h, h1err, 'k*--' )
    plt.slope_triangle( h, h1err )
    plt.ylabel( 'H1 error' )
    plt.grid( True )


def unittest():

  retvals = main( degree=1, maxrefine=2, plots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNplUEGOBCEI/E53IhtAQHnOHPo6/z+uiHZmehJJFcZCqqgcUkjPchzcrV9gRVHoAi3j7v16F2rGF4xH
    VCUw7oziqU6Of3EWC3gWS6MQMjfeA8jdNo8CJdKYCCaiezR0rn43bii7+azv3weA16rhZMrYCLdsWCLx
    tPOw5b1yPNvWFJsFpr1W1VdHplPIhG0iUiJr2lwDn7apx1JyW6dqNRCEcP78mwHo8DzJDgKaZwh3GtAF
    LVUuPVXMvojsrYEx1k1ukvl8ZgW8jN+BQbWUnOX8B+7/gYY=''' )

  retvals = main( nelems=4, degree=2, maxrefine=2, plots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqdUkluwzAM/E4CmAU54iI9p4dc8/9jZVJO2sK+BLDBZUSKmqFsN93E7tvtZgJ+UGzCzR9k28w9v58b
    f72/Pf7/g4fsx88wV7MrTOY9D9JzzCwxaOvLWlqYpGXJWDoSl3A/60WjRVwNMEEbO3jyMOo8cIW5WVxh
    pn7UkQzM/j4d5kwStHM6ikjSyKR6TbJFdObaH8IpmmF/mbaRLxy6s2YbAlkHRMVLBaBYBVyTGe0Zd4++
    49PkBIcyQB/rPBejXHWtJ9OS8+lLLTTlZXMeiGApUYq4ZP5TBUX2udskJkYOTIM1loPi7BCVBL0OH9NQ
    SBTB3UtbGgJdVS+1SVkL1ZUkGyiJwtSrw3sDDG127fNC5bq5mS/9Qqru11IQJEr44fB04FjCm5Xz+aLc
    fwA+deIa''' )

  retvals = main( nelems=4, degree=2, maxrefine=3, plots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqdU0luwzAM/E4CmAWH4iI9p4dc8/9jJVJp0iK+BLBBisPFmqFxXPSAXY/LxVjlRnGApd/Ijhm7f98P
    /no+6/z/FR5Y6e8wV7MzDOB2I32PmSUm2vq2llYMaRl5RpfEEe7vetFoEWcfMEEbC3xzMeo85AxzszjD
    TP1RRxgy+/t0mDNIop3TUYkkjQzVa5INrFj7QziF97yxquSNhyI7SLSWdjpptwoixapIMQLtWde9Z55H
    z7yHMiKDdz6KUS5mW81Ffp/+qiVNsW32F0C2ElXnqLkfKgiMWCSQRaQ4NFh1O1KcPUQltEXwTJbmuqop
    UPRTd/NKhuzyp9qk7NVBm9cUG5J8UJhW4csGNLQZ63Ogcv4a1My3foEa+LIUJFiiTOGHI9KZ7NacSXtt
    wOeLcv0BRKniIA==''' )


if __name__ == '__main__':
  cli.choose( main, conv, unittest )
