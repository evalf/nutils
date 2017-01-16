#! /usr/bin/python3

from nutils import mesh, cli, log, function, plot, debug, _
import numpy


class MakePlots( object ):

  def __init__( self, domain, geom, video ):
    self.domain = domain
    self.geom = geom
    self.plt = video and plot.PyPlotVideo( 'solution' )
    self.index = 0

  def __call__( self, u ):
    self.index += 1
    xp, up = self.domain.elem_eval( [ self.geom, u ], ischeme='bezier7', separate=True )
    with self.plt if self.plt else plot.PyPlot( 'solution', index=self.index ) as plt:
      plt.mesh( xp, up )
      plt.clim( 0, 1 )
      plt.colorbar()


def main( nelems=20, degree=1, timescale=.5, tol=1e-5, ndims=1, endtime=0, withplots=True ):

  # construct mesh
  domain, geom = mesh.rectilinear( [numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims) )
  basis = domain.basis( 'discont', degree=degree )

  # construct initial condition (centered gaussian)
  u = function.exp( -( ((geom-.5)*5)**2 ).sum(-1) )
  lhs = domain.project( u, onto=basis, geometry=geom, ischeme='gauss5' )

  # prepare matrix
  timestep = timescale/nelems
  At = (1/timestep) * function.outer( basis )
  matrix0 = domain.integrate( At, geometry=geom, ischeme='gauss5' )

  # prepare plotting
  makeplots = MakePlots( domain, geom, video=withplots=='video' ) if withplots else lambda *args: None

  # start time stepping
  for itime in log.count( 'timestep' ):
    makeplots( u )
    if endtime and itime * timestep >= endtime:
      break
    rhs = matrix0.matvec(lhs)
    for ipicard in log.count( 'picard' ):
      u = basis.dot( lhs )
      beta = function.repeat( u[_], ndims, axis=0 )
      Am = -function.outer( ( basis[:,_] * beta ).div(geom), basis )
      dmatrix = domain.integrate( Am, geometry=geom, ischeme='gauss5' )
      alpha = .5 * function.sign( function.mean(beta).dotnorm(geom) )
      Ai = function.outer( function.jump(basis), ( alpha * function.jump(basis[:,_]*beta) - function.mean(basis[:,_]*beta) ).dotnorm(geom) )
      dmatrix += domain.interfaces.integrate( Ai, geometry=geom, ischeme='gauss5' )
      lhs, info = ( matrix0 + dmatrix ).solve( rhs, lhs0=lhs, tol=tol, restart=999, precon='spilu', info=True )
      if info.niter == 0:
        break

  return rhs, lhs


def unittest():

  retvals = main( ndims=1, nelems=10, timescale=.1, degree=1, endtime=.01, withplots=False )
  assert debug.checkdata( retvals, '''
    eNpVjssNwzAMQ9dJAAXQ39JA3X+F2kx16OnBhEw+octJ4qbrUm/5PEHBqw6lar+dTLsOawUfiol+HiMN
    rkNTtkN3dbBY//jL527+Tc/0zs7sjsd4bcdHmnfX2saGrDV3R1Kb47bivTXnRqcnOveEgSsUGywNB3aG
    gwU2S/11WZFgWiAvZex5Ghi9zu72QP5Iah6xm+4vsgVJqA==''' )

  retvals = main( ndims=1, nelems=10, timescale=.1, degree=2, endtime=.01, withplots=False )
  assert debug.checkdata( retvals, '''
    eNpVkcsNxDAIBdtJJFsyYH4Fuf8W1kA47Gkk5DzeEBjPHsDveB4goDN5ILEFydyDsBjOvK+EPYhbLMgg
    ElSQnDtTzoFZz6QBvjCIyEVTDtJySyLTH795v+vvOqdze0/v7R7dq3t27/Zor/a8znh3nSkDBZKgKvnE
    oU4AUSVOgF8kFdUxo5mlSLmKN62sdjslt2GdxnfSq6q7UuVBqtmynBtg5XipkbqWQuWBaf2KpViUUoRQ
    jd5bk9nvKvP29Lu3Db7j/QGOWGzS''' )

  retvals = main( ndims=2, nelems=4, timescale=.1, degree=1, endtime=.01, withplots=False )
  assert debug.checkdata( retvals, '''
    eNplUllywyAMvU4yAzNoRT5Q7n+FGuRHU/VLWMhvE9Re2sje7fUSV/302YhnfLqfqhq86nS5sk/06dbY
    57Wqsvqq6GMOfcwBDzzARR///c4n3sH3cX9rY5P4rjpYvr8xV/vArfrAU/mqnspX9fzjq3oKX9VT8677
    qHnXffzPu+yj5F33cb8BEh6r1Y3d/hyYxgbr0+ZG6xKXrI4MTjozXTa7XD5yNIZvIouxboSI94UJz30h
    6gnO+lx48oMF4IcNLOoxt1niDPupNOzWK0tvcoakOlrLuPthc8/PDKu7yDZIM1K+RqQY+ID84wc+wAFs
    cIHjdsW7P3Q/gFP1eXChe64bjfxxzkzOOYM5SSEhJHaSgo8jH36OD7AccLAdFibxJ/58EedAYhvs3d4/
    YD7oyg==''' )


if __name__ == '__main__':
  cli.run( main, unittest )
