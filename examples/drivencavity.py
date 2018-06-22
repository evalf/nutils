#! /usr/bin/env python3

from nutils import *
import numpy, unittest
from matplotlib import collections


@log.title
def postprocess(name, domain, ns, every=.05, spacing=.01, **arguments):

  # confirm that velocity is pointwise divergence-free
  div = domain.integrate('(u_k,k)^2' @ ns, geometry=ns.x, degree=9, arguments=arguments)**.5
  log.info('velocity divergence: {:.2e}'.format(div))

  # compute streamlines
  ns = ns.copy_()
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n'
  sqr = domain.integral('(u_0 - stream_,1)^2 + (u_1 + stream_,0)^2' @ ns, geometry=ns.x, degree=4)
  arguments['streamdofs'] = solver.optimize('streamdofs', sqr, arguments=arguments)

  # plot velocity as field, pressure as contours, streamlines as dashed
  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval([ns.x, function.norm2(ns.u), ns.p, ns.stream], arguments=arguments)
  with export.mplfigure(name) as fig:
    ax = fig.add_axes([.1,.1,.8,.8], yticks=[], aspect='equal')
    ax.add_collection(collections.LineCollection(x[bezier.hull], colors='w', linewidths=.5, alpha=.2))
    ax.tricontour(x[:,0], x[:,1], bezier.tri, stream, 16, colors='k', linestyles='dotted', linewidths=.5, zorder=9)
    caxu = fig.add_axes([.1,.1,.03,.8], title='velocity')
    imu = ax.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', cmap='jet')
    fig.colorbar(imu, cax=caxu)
    caxu.yaxis.set_ticks_position('left')
    caxp = fig.add_axes([.87,.1,.03,.8], title='pressure')
    imp = ax.tricontour(x[:,0], x[:,1], bezier.tri, p, 16, cmap='gray', linestyles='solid')
    fig.colorbar(imp, cax=caxp)


def main(
    nelems: 'number of elements' = 12,
    viscosity: 'fluid viscosity' = 1e-3,
    density: 'fluid density' = 1,
    degree: 'polynomial degree' = 2,
    warp: 'warp domain (downward bend)' = False,
  ):

  log.user( 'reynolds number: {:.1f}'.format(density / viscosity) ) # based on unit length and velocity

  # create namespace
  ns = function.Namespace()
  ns.viscosity = viscosity
  ns.density = density

  # construct mesh
  verts = numpy.linspace( 0, 1, nelems+1 )
  domain, ns.x0 = mesh.rectilinear( [verts,verts] )

  # construct bases
  ns.uxbasis, ns.uybasis, ns.pbasis, ns.lbasis = function.chain([
    domain.basis( 'spline', degree=(degree+1,degree), removedofs=((0,-1),None) ),
    domain.basis( 'spline', degree=(degree,degree+1), removedofs=(None,(0,-1)) ),
    domain.basis( 'spline', degree=degree ),
    [1], # lagrange multiplier
  ])
  ns.ubasis_ni = '<uxbasis_n, uybasis_n>_i'

  # construct geometry
  if not warp:
    ns.x = ns.x0
  else:
    xi, eta = ns.x0
    ns.x = (eta+2) * function.rotmat(xi*.4)[:,1] - (0,2) # slight downward bend
    ns.J_ij = 'x_i,x0_j'
    ns.detJ = function.determinant(ns.J)
    ns.ubasis_ni = 'ubasis_nj J_ij / detJ' # piola transform
    ns.pbasis_n = 'pbasis_n / detJ'

  # populate namespace
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.l = 'lbasis_n ?lhs_n'
  ns.sigma_ij = 'viscosity (u_i,j + u_j,i) - p δ_ij'
  ns.c = 5 * (degree+1) / domain.boundary.integrate_elementwise(1, geometry=ns.x, degree=2, asfunction=True)
  ns.nietzsche_ni = 'viscosity (c ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j)'
  ns.top = domain.boundary.indicator('top')
  ns.utop_i = 'top <n_1, -n_0>_i'

  # solve stokes flow
  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n (u_k,k + l) + lbasis_n p' @ ns, geometry=ns.x, degree=2*(degree+1))
  res += domain.boundary.integral('nietzsche_ni (u_i - utop_i)' @ ns, geometry=ns.x, degree=2*(degree+1))
  lhs0 = solver.solve_linear('lhs', res)
  postprocess('stokes', domain, ns, lhs=lhs0)

  # solve navier-stokes flow
  res += domain.integral('density ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*(degree+1))
  lhs1 = solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
  postprocess('navierstokes', domain, ns, lhs=lhs1)

  return lhs0, lhs1


class test(unittest.TestCase):

  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=False)
    numeric.assert_allclose64(lhs0, 'eNpTvPBI3/o0t1mzds/pltM65opQ/n196QvcZh4XO03MTHbolZ'
      '8+dVrxwlP9rycVL03Xjbm45tQfrZc37M/LGLBcFVc/aPDk/H3dzEtL9EJMGRgAJt4mPA==')
    numeric.assert_allclose64(lhs1, 'eNoBUgCt/6nOuTGJy4M1SCzJy4zLCjcsLk3PCst/Nlcx9M2DNe'
      'DPgDR+NB7UG8wVzSwuPc6ByezUQiudMKTL/y4AL73NLS6jLUov8s4zzXoscdMJMSo2AABO+yTF')

  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=2, warp=False)
    numeric.assert_allclose64(lhs0, 'eNp7ZmB71sY46VSq2dLzludvnMo20jFHsJ7BZaXObzbedDrVbJ'
      'nBjPM1ZkuNGaAg6nyGQcvJ6DPPDHzP+JnMPsltwKl1/DyrYcPJUxf0LuXqvDkzzYgBDsz0L+lOvixinH'
      'X26/nvVy0Nfp9rMGNgAADUrDbX')
    numeric.assert_allclose64(lhs1, 'eNoBhAB7/3Axm8zRM23KHDbJzyrMAs7DzOY2yM/vLvfJ8TQ/N8'
      'AvSc5FMkjKwTaQzlo0K8scNuwwLDKfNWQzcCLOzCs1jTEA0FcxA8kLzcAvU81jMz/JVTELMUjOLDL+ye'
      'MsaS6lLkLOajM9LDgwWNBzzOvOMTBCMHnXnDHFzcDTYDCgKo0vLzcAACOlOuU=')

  def test_p1_warped(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=True)
    numeric.assert_allclose64(lhs0, 'eNozv9CjZ35a2axMx/P0jdPq5uZQ/kn9zVeVzewubjHhNjmk53'
      'P662nzC75ad0/evZSv+/1846n3WluvK51PNhC86q1xz2DueWXdiZc4DepNGRgALu0l4g==')
    numeric.assert_allclose64(lhs1, 'eNoBUgCt/67OazGNy5M1fy2Oy+XL+ja0LqzO8sqmNlIxfM6TNc'
      'fPoTRQNCrU98sHzrQul81aycHY6dfjL4PLpC5YLsfN7S+BLMEu3s74zF0pHdKIMWQ2AAB5wCwp')


if __name__ == '__main__':
  cli.run(main)
