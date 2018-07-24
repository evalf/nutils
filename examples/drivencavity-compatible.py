#! /usr/bin/env python3
#
# In this script we solve the lid driven cavity problem for stationary Stokes
# and Navier-Stokes flow. That is, a unit square domain, with no-slip left,
# bottom and right boundaries and a top boundary that is moving at unit
# velocity in positive x-direction.
#
# The script is identical :ref:`examples/drivencavity.py` except that it uses
# the Raviart-Thomas discretization providing compatible velocity and pressure
# spaces resulting in a pointwise divergence-free velocity field.

import nutils, numpy, matplotlib.collections

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# polynomial degree, and Reynolds number.

def main(nelems: 'number of elements' = 12,
         degree: 'polynomial degree for velocity' = 2,
         reynolds: 'reynolds number' = 1000.):

  verts = numpy.linspace(0, 1, nelems+1)
  domain, geom = nutils.mesh.rectilinear([verts, verts])

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
  ns.stress_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
  ns.uwall = domain.boundary.indicator('top'), 0
  ns.N = 5 * degree * nelems # nietzsche constant

  res = domain.integral('ubasis_ni,j stress_ij + pbasis_n (u_k,k + l) + lbasis_n p' @ ns, geometry=ns.x, degree=2*degree)
  res += domain.boundary.integral('(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) (u_i - uwall_i) / Re' @ ns, geometry=ns.x, degree=2*degree)
  with nutils.log.context('stokes'):
    lhs0 = nutils.solver.solve_linear('lhs', res)
    postprocess(domain, ns, lhs=lhs0)

  res += domain.integral('ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*degree)
  with nutils.log.context('navierstokes'):
    lhs1 = nutils.solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
    postprocess(domain, ns, lhs=lhs1)

  return lhs0, lhs1

# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  div = domain.integrate('(u_k,k)^2' @ ns, geometry=ns.x, degree=1, arguments=arguments)**.5
  nutils.log.info('velocity divergence: {:.2e}'.format(div)) # confirm that velocity is pointwise divergence-free

  ns = ns.copy_() # copy namespace so that we don't modify the calling argument
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n' # stream function
  sqr = domain.integral('(u_0 - stream_,1)^2 + (u_1 + stream_,0)^2' @ ns, geometry=ns.x, degree=4)
  arguments['streamdofs'] = nutils.solver.optimize('streamdofs', sqr, arguments=arguments) # compute streamlines

  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval([ns.x, nutils.function.norm2(ns.u), ns.p, ns.stream], arguments=arguments)
  with nutils.export.mplfigure('flow.jpg') as fig: # plot velocity as field, pressure as contours, streamlines as dashed
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

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. To
# keep with the default arguments simply run :sh:`python3
# drivencavity-compatible.py`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, reynolds=100, degree=2)
    nutils.numeric.assert_allclose64(lhs0, 'eNpTvPBI3/o0t1mzds/pltM65opQ/n196QvcZh4XO03MTHbolZ'
      '8+dVrxwlP9rycVL03Xjbm45tQfrZc37M/LGLBcFVc/aPDk/H3dzEtL9EJMGRgAJt4mPA==')
    nutils.numeric.assert_allclose64(lhs1, 'eNoBUgCt/6nOuTGJy4M1SCzJy4zLCjcsLk3PCst/Nlcx9M2DNe'
      'DPgDR+NB7UG8wVzSwuPc6ByezUQiudMKTL/y4AL73NLS6jLUov8s4zzXoscdMJMSo2AABO+yTF')

  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, reynolds=100, degree=3)
    nutils.numeric.assert_allclose64(lhs0, 'eNp7ZmB71sY46VSq2dLzludvnMo20jFHsJ7BZaXObzbedDrVbJ'
      'nBjPM1ZkuNGaAg6nyGQcvJ6DPPDHzP+JnMPsltwKl1/DyrYcPJUxf0LuXqvDkzzYgBDsz0L+lOvixinH'
      'X26/nvVy0Nfp9rMGNgAADUrDbX')
    nutils.numeric.assert_allclose64(lhs1, 'eNoBhAB7/3Axm8zRM23KHDbJzyrMAs7DzOY2yM/vLvfJ8TQ/N8'
      'AvSc5FMkjKwTaQzlo0K8scNuwwLDKfNWQzcCLOzCs1jTEA0FcxA8kLzcAvU81jMz/JVTELMUjOLDL+ye'
      'MsaS6lLkLOajM9LDgwWNBzzOvOMTBCMHnXnDHFzcDTYDCgKo0vLzcAACOlOuU=')
