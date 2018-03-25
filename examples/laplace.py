#! /usr/bin/env python3

from nutils import mesh, util, cli, function, log, numeric, solver, export
import numpy, unittest
from matplotlib import collections

def main(
    nelems: 'number of elements' = 10,
    degree: 'polynomial degree' = 1,
    basistype: 'basis function' = 'spline',
    solvetol: 'solver tolerance' = 1e-10,
    figures: 'create figures' = True,
  ):

  # construct mesh
  verts = numpy.linspace(0, numpy.pi/2, nelems+1)
  domain, geom = mesh.rectilinear([verts, verts])

  # create namespace
  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(basistype, degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.fx = 'sin(x_0)'
  ns.fy = 'exp(x_1)'

  # construct residual
  res = domain.integral('-basis_n,i u_,i' @ ns, geometry=ns.x, degree=degree*2)
  res += domain.boundary['top'].integral('basis_n fx fy' @ ns, geometry=ns.x, degree=degree*2)

  # construct dirichlet constraints
  sqr = domain.boundary['left'].integral('u^2' @ ns, geometry=ns.x, degree=degree*2)
  sqr += domain.boundary['bottom'].integral('(u - fx)^2' @ ns, geometry=ns.x, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # find lhs such that res == 0 and substitute this lhs in the namespace
  lhs = solver.solve_linear('lhs', res, constrain=cons)
  ns = ns(lhs=lhs)

  # plot solution
  if figures:
    bezier = domain.sample('bezier', 9)
    x, u = bezier.eval([ns.x, ns.u])
    with export.mplfigure('solution') as fig:
      ax = fig.add_subplot(111, aspect='equal')
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', cmap='jet')
      ax.add_collection(collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1))
      ax.autoscale(enable=True, axis='both', tight=True)
      fig.colorbar(im)

  # evaluate error against exact solution fx fy
  err = domain.integrate('(u - fx fy)^2' @ ns, geometry=ns.x, degree=degree*2)**.5
  log.user('L2 error: {:.2e}'.format(err))

  return cons, lhs, err


class test(unittest.TestCase):

  def test_p1(self):
    cons, lhs, err = main(nelems=4, degree=1, figures=False, solvetol=0)
    numeric.assert_allclose64(cons, 'eNor1ZC9Fawsf79NvsKUoQECV5vBWIbmMFYknAUAgVMONA==')
    numeric.assert_allclose64(lhs, 'eNor1ZC9Fawsf79NvsLUzOyn2T7zVovVZpnmOhYfLXZYGpq/N99'
      'kUW5paxVpLm5xy2K+ZaoVAOTZEq4=')
    numpy.testing.assert_almost_equal(err, 6.850e-3, decimal=6)

  def test_p2(self):
    cons, lhs, err = main(nelems=4, degree=2, figures=False, solvetol=0)
    numeric.assert_allclose64(cons, 'eNqbeOO56u/bvUpG97wVtE0YGmDQzAzB/o/EDjLHzgYAhtsWCA==')
    numeric.assert_allclose64(lhs, 'eNqbeOO56u/bvUpG97wVtE2aTdxNecwumNmYm5n1mQWbi1vctnC'
      '3/G8WYS5j8dBiqaWgVZD5KvN8C2PL75bJQPZqMPsHkA0A82saTw==')
    numpy.testing.assert_almost_equal(err, 1.268e-3, decimal=6)


if __name__ == '__main__':
  cli.run(main)
