#! /usr/bin/env python3

import nutils, numpy, unittest, matplotlib.collections

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  # confirm that velocity is pointwise divergence-free
  div = domain.integrate('(u_k,k)^2' @ ns, geometry=ns.x, degree=1, arguments=arguments)**.5
  nutils.log.info('velocity divergence: {:.2e}'.format(div))

  # compute streamlines
  ns = ns.copy_()
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n'
  sqr = domain.integral('(u_0 - stream_,1)^2 + (u_1 + stream_,0)^2' @ ns, geometry=ns.x, degree=4)
  arguments['streamdofs'] = nutils.solver.optimize('streamdofs', sqr, arguments=arguments)

  # plot velocity as field, pressure as contours, streamlines as dashed
  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval([ns.x, nutils.function.norm2(ns.u), ns.p, ns.stream], arguments=arguments)
  with nutils.export.mplfigure('flow.jpg') as fig:
    ax = fig.add_axes([.1,.1,.8,.8], yticks=[], aspect='equal')
    ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='w', linewidths=.5, alpha=.2))
    ax.tricontour(x[:,0], x[:,1], bezier.tri, stream, 16, colors='k', linestyles='dotted', linewidths=.5, zorder=9)
    caxu = fig.add_axes([.1,.1,.03,.8], title='velocity')
    imu = ax.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', cmap='jet')
    fig.colorbar(imu, cax=caxu)
    caxu.yaxis.set_ticks_position('left')
    caxp = fig.add_axes([.87,.1,.03,.8], title='pressure')
    imp = ax.tricontour(x[:,0], x[:,1], bezier.tri, p, 16, cmap='gray', linestyles='solid')
    fig.colorbar(imp, cax=caxp)


def main(nelems: 'number of elements' = 12,
         degree: 'polynomial degree for velocity' = 2,
         reynolds: 'reynolds number' = 1e-3):

  # construct mesh
  verts = numpy.linspace(0, 1, nelems+1)
  domain, geom = nutils.mesh.rectilinear([verts, verts])

  # create namespace
  ns = nutils.function.Namespace()
  ns.x = geom
  ns.Re = reynolds
  ns.uxbasis, ns.uybasis, ns.pbasis, ns.lbasis = nutils.function.chain([
    domain.basis('spline', degree=(degree,degree-1), removedofs=((0,-1),None)),
    domain.basis('spline', degree=(degree-1,degree), removedofs=(None,(0,-1))),
    domain.basis('spline', degree=degree-1),
    [1], # lagrange multiplier
  ])
  ns.ubasis_ni = '<uxbasis_n, uybasis_n>_i'
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.l = 'lbasis_n ?lhs_n'
  ns.stress_ij = 'Re (u_i,j + u_j,i) - p δ_ij'
  ns.uwall = domain.boundary.indicator('top'), 0
  ns.k = 5 * degree * nelems * reynolds # nietzsche constant

  res = domain.integral('ubasis_ni,j stress_ij + pbasis_n (u_k,k + l) + lbasis_n p' @ ns, geometry=ns.x, degree=2*degree)
  res += domain.boundary.integral('(k ubasis_ni - Re (ubasis_ni,j + ubasis_nj,i) n_j) (u_i - uwall_i)' @ ns, geometry=ns.x, degree=2*degree)
  with nutils.log.context('stokes'):
    lhs0 = nutils.solver.solve_linear('lhs', res)
    postprocess(domain, ns, lhs=lhs0)

  res += domain.integral('ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*degree)
  with nutils.log.context('navierstokes'):
    lhs1 = nutils.solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
    postprocess(domain, ns, lhs=lhs1)

  return lhs0, lhs1


class test(unittest.TestCase):

  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, reynolds=1e-2, degree=2)
    nutils.numeric.assert_allclose64(lhs0, 'eNpTvPBI3/o0t1mzds/pltM65opQ/n196QvcZh4XO03MTHbolZ'
      '8+dVrxwlP9rycVL03Xjbm45tQfrZc37M/LGLBcFVc/aPDk/H3dzEtL9EJMGRgAJt4mPA==')
    nutils.numeric.assert_allclose64(lhs1, 'eNoBUgCt/6nOuTGJy4M1SCzJy4zLCjcsLk3PCst/Nlcx9M2DNe'
      'DPgDR+NB7UG8wVzSwuPc6ByezUQiudMKTL/y4AL73NLS6jLUov8s4zzXoscdMJMSo2AABO+yTF')

  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, reynolds=1e-2, degree=3)
    nutils.numeric.assert_allclose64(lhs0, 'eNp7ZmB71sY46VSq2dLzludvnMo20jFHsJ7BZaXObzbedDrVbJ'
      'nBjPM1ZkuNGaAg6nyGQcvJ6DPPDHzP+JnMPsltwKl1/DyrYcPJUxf0LuXqvDkzzYgBDsz0L+lOvixinH'
      'X26/nvVy0Nfp9rMGNgAADUrDbX')
    nutils.numeric.assert_allclose64(lhs1, 'eNoBhAB7/3Axm8zRM23KHDbJzyrMAs7DzOY2yM/vLvfJ8TQ/N8'
      'AvSc5FMkjKwTaQzlo0K8scNuwwLDKfNWQzcCLOzCs1jTEA0FcxA8kLzcAvU81jMz/JVTELMUjOLDL+ye'
      'MsaS6lLkLOajM9LDgwWNBzzOvOMTBCMHnXnDHFzcDTYDCgKo0vLzcAACOlOuU=')


if __name__ == '__main__':
  nutils.cli.run(main)
