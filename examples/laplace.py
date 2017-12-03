#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, numeric, solver
import numpy, unittest


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
  ns |= dict(lhs=lhs)

  # plot solution
  if figures:
    points, colors = domain.elem_eval([ns.x, ns.u], ischeme='bezier9', separate=True)
    with plot.PyPlot('solution', index=nelems) as plt:
      plt.mesh(points, colors, cmap='jet')
      plt.colorbar()

  # evaluate error against exact solution fx fy
  err = domain.integrate('(u - fx fy)^2' @ ns, geometry=ns.x, degree=degree*2)**.5
  log.user('L2 error: {:.2e}'.format(err))

  return cons, lhs, err


class test(unittest.TestCase):

  def test_p1(self):
    cons, lhs, err = main(nelems=4, degree=1, figures=False, solvetol=0)
    numeric.assert_allclose64(cons,
      'eNptyjkKwDAMRNELSeDREklnCSlSuPX9S9tlgouBz2OM4HRbeXQOYqjUDvVYkAua6A4HWucizbDORuMd'
      'nwUuOXmp58nRsP7692cCdHUoSw==')
    numeric.assert_allclose64(lhs,
      'eNoljkESBUEEQy+kq5pE4yy//rLvf4UZZvVICBR1+bE87gpZCqsu4PEK+Qrb0IWr7rtKkMG7KH5iNxOB'
      'phr1LogmvRl6bPTNGN0/mhWbYOdBCp4zh5Pjb5scq5qe9DP7W23I/HzN8WHfPWb/if8DoOMtyg==')
    numpy.testing.assert_almost_equal(err, 6.850e-3, decimal=6)

  def test_p2(self):
    cons, lhs, err = main(nelems=4, degree=2, figures=False, solvetol=0)
    numeric.assert_allclose64(cons,
      'eNp9yjEOwDAIA8APgRQCBPyWqkMH1vx/TDNX6WDZ0tlInC7u0F4cJKrYzS7SipO6NexmGWZ7hEcWgwR4'
      'D0bzmZ/4GHqytIiTAcgfw8HuBTunOOw=')
    numeric.assert_allclose64(lhs,
      'eNp1z7sRAzEIRdGG0Iwef2rxOFT/LezCbuLA0UVwEinB6LO4hM8Kgkh1lwH7rCTWXd0FV+0hLPKsIlTd'
      'QIklpCsG7xrv6EZg7mU+b3Mf5+Gzx944Swia3GWUTYO1mxrxOLyucu437Aqnd3W3E6rq++1RMj589uyY'
      'Ssbjc9fr64/PX9//kO8F+0tBmQ==')
    numpy.testing.assert_almost_equal(err, 1.268e-3, decimal=6)


if __name__ == '__main__':
  cli.run(main)
