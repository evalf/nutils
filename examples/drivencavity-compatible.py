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

  uxbasis = domain.basis('spline', degree=(degree,degree-1), removedofs=((0,-1),None))
  uybasis = domain.basis('spline', degree=(degree-1,degree), removedofs=(None,(0,-1)))

  #ns.ubasis = function.vectorize([uxbasis, uybasis]) # FAILS
  #ns.ubasis = function.concatenate([function.kronecker(uxbasis, axis=1, length=2, pos=0), function.kronecker(uybasis, axis=1, length=2, pos=1)]) # FAILS
  ns.Ubasis = function.stack([function.concatenate([uxbasis, function.zeros_like(uybasis)]), function.concatenate([function.zeros_like(uxbasis), uybasis])], axis=1) # OK
  ns.ubasis = function.kronecker(function.concatenate([uxbasis, function.zeros_like(uybasis)]), 1, 2, 0) + function.kronecker(function.concatenate([function.zeros_like(uxbasis), uybasis]), 1, 2, 1) # FAILS

  treelog.info('cmp1:', domain.integrate(((ns.Ubasis - ns.ubasis)**2).sum(1), degree=9).sum())
  treelog.info('cmp2:', domain.integrate(((ns.Ubasis - ns.ubasis).grad(ns.x)**2).sum([1,2]), degree=9).sum())
  treelog.info('cmp3:', domain.boundary.integrate(((ns.Ubasis - ns.ubasis)**2).sum(1), degree=9).sum())
  treelog.info('cmp4:', domain.boundary.integrate(((ns.Ubasis - ns.ubasis).grad(ns.x)**2).sum([1,2]), degree=9).sum())

  ns.pbasis = domain.basis('spline', degree=degree-1)
  ns.u_i = 'ubasis_ni ?u_n'
  ns.p = 'pbasis_n ?p_n'
  ns.stress_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
  ns.uwall = domain.boundary.indicator('top'), 0
  ns.N = 5 * degree * nelems # nitsche constant based on element size = 1/nelems
  ns.nitsche_ni = '(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) / Re'

  ures = domain.integral('ubasis_ni,j stress_ij d:x' @ ns, degree=2*degree)
  ures += domain.boundary.integral('(nitsche_ni (u_i - uwall_i) - ubasis_ni stress_ij n_j) d:x' @ ns, degree=2*degree)
  pres = domain.integral('pbasis_n (u_k,k + ?lm) d:x' @ ns, degree=2*degree)
  lres = domain.integral('p d:x' @ ns, degree=2*degree)

  state0 = solver.solve_linear(['u', 'p', 'lm'], [ures, pres, lres])

  Ures = domain.integral('Ubasis_ni,j stress_ij d:x' @ ns, degree=2*degree)
  Ures += domain.boundary.integral('(nitsche_ni (u_i - uwall_i) - Ubasis_ni stress_ij n_j) d:x' @ ns, degree=2*degree)

  state1 = solver.solve_linear(['u', 'p', 'lm'], [Ures, pres, lres])

  treelog.info('ures:', numpy.linalg.norm(ures.eval(**state0)), numpy.linalg.norm(ures.eval(**state1)))
  treelog.info('pres:', numpy.linalg.norm(pres.eval(**state0)), numpy.linalg.norm(pres.eval(**state1)))
  treelog.info('lres:', lres.eval(**state0), lres.eval(**state1))
  treelog.info('Ures:', numpy.linalg.norm(Ures.eval(**state0)), numpy.linalg.norm(Ures.eval(**state1)))
  treelog.info('pres:', numpy.linalg.norm(pres.eval(**state0)), numpy.linalg.norm(pres.eval(**state1)))

  return numpy.hstack([state0['u'], state0['p'], state0['lm']])

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
    lhs0 = main(nelems=3, reynolds=100, degree=2)
    with self.subTest('stokes'): self.assertAlmostEqual64(lhs0, '''
      eNrzu9Bt8OuUndkD/eTTSqezzP2g/E3698/ZmZlf2GjSaHJS3/90/Wm/C4qGh066XzLQ47846VSPpoWK
      3vnD+iXXTty+ZGB7YafuhYsf9fJMGRgAkFIn4A==''')
