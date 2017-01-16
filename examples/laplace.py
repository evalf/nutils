#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, debug
import numpy


def main( nelems=None, degree=1, basistype='spline', solvetol=1e-10, withplots=True ):

  # construct mesh
  if nelems: # rectilinear
    verts = numpy.linspace( 0, .5*numpy.pi, nelems+1 )
    domain, geom = mesh.rectilinear( [verts,verts] )
  else: # triangulated demo mesh
    domain, geom = mesh.demo( xmax=.5*numpy.pi, ymax=.5*numpy.pi )
    assert degree == 1, 'degree={} unsupported for triangular mesh'.format( degree )

  # construct target solution
  x, y = geom
  exact = function.sin(x) * function.exp(y)
  flux = exact.ngrad( geom )

  # prepare basis
  basis = domain.basis( basistype, degree=degree )

  # construct matrix
  laplace = function.outer( basis.grad(geom) ).sum(-1)
  matrix = domain.integrate( laplace, geometry=geom, ischeme='gauss4' )

  # construct right hand side vector from neumann boundary
  rhs = domain.boundary['right,top'].integrate( basis * flux, geometry=geom, ischeme='gauss7' )

  # construct dirichlet boundary constraints
  cons = domain.boundary['left,bottom'].project( exact, ischeme='gauss7', geometry=geom, onto=basis )

  # solve system
  lhs = matrix.solve( rhs, constrain=cons, tol=solvetol, solver='cg' )

  # construct solution function
  sol = basis.dot(lhs)

  # plot solution
  if withplots:
    points, colors = domain.elem_eval( [ geom, sol ], ischeme='bezier9', separate=True )
    with plot.PyPlot( 'solution', index=nelems ) as plt:
      plt.mesh( points, colors )
      plt.colorbar()

  # evaluate approximation error
  error = sol - exact
  err = numpy.sqrt( domain.integrate( [ error**2, ( error.grad(geom)**2 ).sum(-1) ], geometry=geom, ischeme='gauss7' ) )
  log.user( 'errors: l2={}, h1={}'.format(*err) )

  return err, rhs, cons, lhs


def conv( degree=1, nrefine=4 ):

  l2err = []
  h1err = []

  for irefine in log.range( 'refine', nrefine ):
    err, rhs, cons, lhs = main( nelems=2**irefine, degree=degree )
    l2err.append( err[0] )
    h1err.append( err[1] )

  h = (.25*numpy.pi) * .5**numpy.arange(nrefine)

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

  retvals = main( degree=1, withplots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqNkEEOAyEIRa8zk2AjHxA9Thdu5/7LChm7aLtoonnwRfjKdCixnXQcTbrNYtQrdBalpdUHsQCzCKFa
    MiSOwgxhnLUr/HOtrtfzol+7rMZtFieII5jiwOCdCMNnaWSqNcaG1pu0HXPFCGefnddQDNU4Uo/yxapJ
    qOQVGDJnWA9+mWld8qlvQ6Y+Qvg0NRwS3MZYPL9rm+PbAWrjZLvPTTx40vkCzKNdVA==''' )

  retvals = main( nelems=4, degree=1, withplots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqdkM1uAzEIhF9nV2IjDz8GHqeHveb9jzWkWzVpTpFsDfoww8igTQm207ax5jgPI3VdqrTYuNHvAbMX
    /ssc8o9BkOchz8xZXlkK288aS5nn4XTAElUox3ofCzBrFRNlmiThPXT/uj9dx5R3PMXiHcdAB3rlH4Wx
    6SgN8Q4BVi5zhM7SKxyGRnNTL2VOKxUFSq+wkNk/yIPbhzN7TtXa7woPjUcf0X1hbb+Fu7/T/g3BeW4p''' )

  retvals = main( nelems=4, degree=2, withplots=False, solvetol=0 )
  assert debug.checkdata( retvals, '''
    eNqlUUtuhTAMvA5ISeW/4+N0wfbdf1nbFKni8VaVQBNmxsFj49hkoO5j21DCjmmDhPyYOpKDr/H3MVQo
    5UZz8seUO41AcUx+o1XwicbgB9pIou7OZiYFZ6UPZI7CqQh5WNnxiRNNuA6unhFiYARU9ev79faqGX/S
    llT9sxYR65OGAFAhnrT/JSD27pYVrVAJukN3xMJQ6+8rlbk1nx31tFFWj5ewlpzopIVXUgSk0xe9NEpj
    IdNqv4Cvwiv9tS70+m/6DVvn5ad/QddfE0m//PrXze+nH9u3j/0HiSeZcA==''' )


if __name__ == '__main__':
  cli.run( main, conv, unittest )
