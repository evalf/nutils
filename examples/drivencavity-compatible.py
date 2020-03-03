#! /usr/bin/env python3
#
# In this script we solve the lid driven cavity problem for stationary Stokes
# and Navier-Stokes flow. That is, a unit square domain, with no-slip left,
# bottom and right boundaries and a top boundary that is moving at unit
# velocity in positive x-direction.
#
# The script is identical to :ref:`examples/drivencavity.py` except that it
# uses the Raviart-Thomas discretization providing compatible velocity and
# pressure spaces resulting in a pointwise divergence-free velocity field.

from nutils import mesh, function, solver, export, cli, testing
import numpy, treelog

def main(nelems:int, degree:int, reynolds:float):
  '''
  Driven cavity benchmark problem using compatible spaces.

  .. arguments::

     nelems [12]
       Number of elements along edge.
     degree [2]
       Polynomial degree for velocity; the pressure space is one degree less.
     reynolds [1000]
       Reynolds number, taking the domain size as characteristic length.
  '''

  verts = numpy.linspace(0, 1, nelems+1)
  domain, geom = mesh.rectilinear([verts, verts])

  ns = function.Namespace()
  ns.x = geom
  ns.Re = reynolds
  ns.uxbasis, ns.uybasis, ns.pbasis, ns.lbasis = function.chain([
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
  ns.N = 5 * degree * nelems # nitsche constant based on element size = 1/nelems
  ns.nitsche_ni = '(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) / Re'

  res = domain.integral('(ubasis_ni,j stress_ij + pbasis_n (u_k,k + l) + lbasis_n p) d:x' @ ns, degree=2*degree)
  res += domain.boundary.integral('(nitsche_ni (u_i - uwall_i) - ubasis_ni stress_ij n_j) d:x' @ ns, degree=2*degree)
  with treelog.context('stokes'):
    lhs0 = solver.solve_linear('lhs', res)
    postprocess(domain, ns, lhs=lhs0)

  res += domain.integral('ubasis_ni u_i,j u_j d:x' @ ns, degree=3*degree)
  with treelog.context('navierstokes'):
    lhs1 = solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
    postprocess(domain, ns, lhs=lhs1)

  return lhs0, lhs1

# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  div = domain.integral('(u_k,k)^2 d:x' @ ns, degree=1).eval(**arguments)**.5
  treelog.info('velocity divergence: {:.2e}'.format(div)) # confirm that velocity is pointwise divergence-free

  ns = ns.copy_() # copy namespace so that we don't modify the calling argument
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = 'streambasis_n ?streamdofs_n' # stream function
  sqr = domain.integral('((u_0 - stream_,1)^2 + (u_1 + stream_,0)^2) d:x' @ ns, degree=4)
  arguments['streamdofs'] = solver.optimize('streamdofs', sqr, arguments=arguments) # compute streamlines

  bezier = domain.sample('bezier', 9)
  x, u, p, stream = bezier.eval(['x_i', 'sqrt(u_i u_i)', 'p', 'stream'] @ ns, **arguments)
  with export.mplfigure('flow.png') as fig: # plot velocity as field, pressure as contours, streamlines as dashed
    ax = fig.add_axes([.1,.1,.8,.8], yticks=[], aspect='equal')
    import matplotlib.collections
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
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, reynolds=100, degree=2)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNrzu9Bt8OuUndkD/eTTSqezzP2g/E3698/ZmZlf2GjSaHJS3/90/Wm/C4qGh066XzLQ47846VSPpoWK
      3vnD+iXXTty+ZGB7YafuhYsf9fJMGRgAkFIn4A==''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNoBUgCt/2XOWjJSy5k1jS+yyzvLODfgL1rO0MrINpsxHM2ZNSrPqDTANCPVQsxCzeAvcc04yaUmYysm
      MbLLAi9YL6TN+y3eLcgvM87NzOUrTNY9MWA2AABnnyYn''')

  @testing.requires('matplotlib')
  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, reynolds=100, degree=3)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNo7aLjtjIjJxZN7zVgvZJ9jOv3lfK05gnUQLmt/Ttlk5qm9ZgKGQeeXmj0zZoCCD+fWGUSflDpz0PDu
      6XRT55OL9dt11pwvNYw5+f7ClYv2Oq/O7DBigANBfR29g5fFjD3Oxl6ovBxi0H1uiRkDAwD+ITkl''')
    with self.subTest('navier-stokes'): self.assertAlmostEqual64(lhs1, '''
      eNoBhAB7/14yGcxyNPbJYTahLj/LSDE7yy43SM9WMsXJoDR+N3Iw8s1hM5zJODeizcE0X8phNrQwUDOO
      NbMzJi+ty4s1oDFqzxIzysjWzXIwFM3tNMjIKjG8MeLNoTLzyQMuCi+IK3jOcjMzLuMvudNEzOrOEDAF
      MD8sTTDpzNjYZDCg0RgwcTcAAJCyOzM=''')
