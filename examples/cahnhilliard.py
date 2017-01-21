#! /usr/bin/env python3

from nutils import mesh, plot, util, log, library, function, numeric, debug
import numpy


class MakePlots( object ):

  def __init__( self, domain, geom, nsteps, video ):
    self.domain = domain
    self.geom = geom
    self.energies = numpy.empty( (nsteps,3) )
    self.plt = video and plot.PyPlotVideo( 'concentration' )
    self.index = 0

  def __call__( self, c, mu, *energies ):
    self.energies[self.index] = energies
    self.index += 1
    xpnt, cpnt = self.domain.elem_eval( [ self.geom, c ], ischeme='bezier4', separate=True )
    I = numpy.arange(self.index-1,-1,-1)
    E = self.energies[self.index-1::-1].T
    with self.plt if self.plt else plot.PyPlot( 'flow', index=self.index ) as plt:
      plt.axes( yticks=[], xticks=[] )
      plt.mesh( xpnt, cpnt )
      plt.colorbar()
      plt.clim( -1, 1 )
      plt.axes( [.07,.05,.35,.25], yticks=[], xticks=[], axisbg='w' ).patch.set_alpha( .8 )
      for i, name in enumerate(['mixture','interface','wall','total']):
        plt.plot( I, E[i] if i < len(E) else E.sum(0), '-o', markevery=self.index, label=name )
      plt.legend( numpoints=1, frameon=False, fontsize=8 )
      plt.xlim( 0, len(self.energies) )
      plt.ylim( 0, self.energies[0].sum() )
      plt.xlabel( 'time' )
      plt.ylabel( 'energy' )


def main( nelems=20, epsilon=None, timestep=.01, maxtime=1., theta=90, init='random', withplots=True ):

  mineps = 1./nelems
  if epsilon is None:
    log.info( 'setting epsilon=%f' % mineps )
    epsilon = mineps
  elif epsilon < mineps:
    log.warning( 'epsilon under crititical threshold: %f < %f' % ( epsilon, mineps ) )

  ewall = .5 * numpy.cos( theta * numpy.pi / 180 )

  # construct mesh
  xnodes = ynodes = numpy.linspace(0,1,nelems+1)
  domain, geom = mesh.rectilinear( [ xnodes, ynodes ] )

  # prepare bases
  cbasis, mubasis = function.chain([ domain.basis( 'spline', degree=2 ), domain.basis( 'spline', degree=2 ) ])

  # define mixing energy and splitting: F' = f_p - f_n
  F = lambda c_: (.5/epsilon**2) * (c_**2-1)**2
  f_p = lambda c_: (1./epsilon**2) * 4*c_
  f_n = lambda c_: (1./epsilon**2) * ( 6*c_ - 2*c_**3 )

  # prepare matrix
  A = function.outer( cbasis ) \
    + (timestep*epsilon**2) * function.outer( cbasis.grad(geom), mubasis.grad(geom) ).sum(-1) \
    + function.outer( mubasis, mubasis - f_p(cbasis) ) \
    - function.outer( mubasis.grad(geom), cbasis.grad(geom) ).sum(-1)
  matrix = domain.integrate( A, geometry=geom, ischeme='gauss4' )

  # prepare wall energy right hand side
  rhs0 = domain.boundary.integrate( mubasis * ewall, geometry=geom, ischeme='gauss4' )

  # construct initial condition
  if init == 'random':
    numpy.random.seed( 0 )
    c = cbasis.dot( numpy.random.normal(0,.5,cbasis.shape) )
  elif init == 'bubbles':
    R1 = .25
    R2 = numpy.sqrt(.5) * R1 # area2 = .5 * area1
    c = 1 + function.tanh( (R1-function.norm2(geom-(.5+R2/numpy.sqrt(2)+.8*epsilon)))/epsilon ) \
          + function.tanh( (R2-function.norm2(geom-(.5-R1/numpy.sqrt(2)-.8*epsilon)))/epsilon )
  else:
    raise Exception( 'unknown init %r' % init )

  # prepare plotting
  nsteps = numeric.round(maxtime/timestep)
  makeplots = MakePlots( domain, geom, nsteps, video=withplots=='video' ) if withplots else lambda *args: None

  # start time stepping
  for istep in log.range( 'timestep', nsteps ):

    Emix = F(c)
    Eiface = .5 * (c.grad(geom)**2).sum(-1)
    Ewall = ( abs(ewall) + ewall * c )

    b = cbasis * c - mubasis * f_n(c)
    rhs, total, energy_mix, energy_iface = domain.integrate( [ b, c, Emix, Eiface ], geometry=geom, ischeme='gauss4' )
    energy_wall = domain.boundary.integrate( Ewall, geometry=geom, ischeme='gauss4' )
    log.user( 'concentration {}, energy {}'.format( total, energy_mix + energy_iface + energy_wall ) )

    lhs = matrix.solve( rhs0 + rhs, tol=1e-12, restart=999 )
    c = cbasis.dot( lhs )
    mu = mubasis.dot( lhs )

    makeplots( c, mu, energy_mix, energy_iface, energy_wall )

  return lhs, energy_mix, energy_iface, energy_wall


def unittest():

  retvals = main( nelems=8, init='bubbles', maxtime=.1, withplots=False )
  assert debug.checkdata( retvals, '''
    eNptk9lxwzAMRNuxZ6QMARJXQem/hVC4bDn+AihRj7sLCo7HOoCex+OhJPJ7rkNJ6aqCYFelBfp70nHK
    WPN6cKpN33HahHhiC0c2jLemaEoGQV1+CsOCpE4o6ioqF5W+U0tb0UhQfQPr4MCiYWIh+IYkyWdL2pRq
    +N1qi2vcEgynohritI7cutP74GxgYTVhoKNrt63vBayvhCyCYBbLZsZZAgrV6H0WHV8bboUNnBBAsIgP
    JSqQaNSJH4PtWVR6bbfUFUxk+Sm6yNeKEYcirs9LUnPtSVR05bSUFcwGuSm1hFvm+O9qvE21A83UymYr
    K5iESeW+GF+g/DHSnEBlVjZLWcGU0d8TgL8nJv+OyHadB4Dufbg5QFEvzq4Lwesc4hUvBV59X3EYbCZX
    gmecvOQiJw+Tpx88r6WnOYPjfEp9AiPW13nOSw5HBbZ7ze9aT67VslL4Yg5dK/Xug7yOYVn1ng/c9RRH
    Z+xjTZ/i+01oXn7YBnpV9XXnXPmUr9JTHMq1D37nJBL5SPD2fvJa86qcK5/ylXqa47/N3D/lWHGVF8nw
    J4shJNboe2QVdURU1krSC7W1u7gTgNzV1di3sfe4Mu6Kqey9VL1QEk53M+H9Sn5hut6OPKNqg62qUbtZ
    3jz318sHTDOUjJ/j+QdkFXSp''' )


if __name__ == '__main__':
  util.run( main, unittest )
