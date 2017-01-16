#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, library, function, debug
import numpy


@log.title
def makeplots( domain, geom, stress ):

  points, colors = domain.elem_eval( [ geom, stress[0,1] ], ischeme='bezier3', separate=True )
  with plot.PyPlot( 'stress', ndigits=0 ) as plt:
    plt.mesh( points, colors, tight=False )
    plt.colorbar()


def main( nelems=12, stress=library.Hooke(lmbda=1,mu=1), degree=2, withplots=True, solvetol=1e-10 ):

  # construct mesh
  verts = numpy.linspace( 0, 1, nelems+1 )
  domain, geom = mesh.rectilinear( [verts,verts] )

  # prepare basis
  dbasis = domain.basis( 'spline', degree=degree ).vector( 2 )

  # construct matrix
  elasticity = function.outer( dbasis.grad(geom), stress(dbasis.symgrad(geom)) ).sum([2,3])
  matrix = domain.integrate( elasticity, geometry=geom, ischeme='gauss2' )

  # construct dirichlet boundary constraints
  cons = domain.boundary['left'].project( 0, geometry=geom, onto=dbasis, ischeme='gauss2' ) \
       | domain.boundary['right'].project( .5, geometry=geom, onto=dbasis.dotnorm(geom), ischeme='gauss2' )

  # solve system
  lhs = matrix.solve( constrain=cons, tol=solvetol, symmetric=True, precon='diag' )

  # construct solution function
  disp = dbasis.dot( lhs )

  # plot solution
  if withplots:
    makeplots( domain, geom+disp, stress(disp.symgrad(geom)) )

  return lhs, cons


def unittest():

  retvals = main( nelems=4, degree=1, withplots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqlkEsKwzAMRK8Tg1ysnz/H6aLb3H9Zx7LaOhQKKVjMSG/AQgibAEqAbUs3+HzInB+xQxQxZVn6yUmZ
    hgrrUG649JNz0anYTFNae+OabP5LT+vmKn2sQKXUQ/souo8OKyc8VIjUQ+6jw5pLGyGh9gpNHx3WkthC
    zO+Q+eiwX/W05X7fL9fFw/zz5bcKEJ7x0YpY''' )

  retvals = main( nelems=4, degree=2, withplots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNq1ksEOwyAIhl+nTXQRENDH2aHXvv9xFqRJ1+ywLEtqvr/wi0gLaakJ6pqWpTzS29NA+5Y5cSlqpE4X
    znj4oGPd8qjXjvdBZb4w4tNHQMUJziJyYcSnr5LSJDpZroy4+0Z/5RveJ8C92M1QBe2mDKOypHyKyOSw
    aod2UKhWG4oqmOEUkclhbQLoW6TYaQRuOEVkclgbixUTISMCiW8JEZkc1ibkjdVmROR5SojI5LCO3+I+
    k/25/3X9/tX+3eGntab1Bd2X0og=''' )


if __name__ == '__main__':
  cli.run( main, unittest )
