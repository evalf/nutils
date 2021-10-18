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
from nutils.expression_v2 import Namespace
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

  ns = Namespace()
  ns.δ = function.eye(domain.ndims)
  ns.Σ = function.ones([domain.ndims])
  ns.x = geom
  ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
  ns.Re = reynolds
  ns.ubasis = function.vectorize([
    domain.basis('spline', degree=(degree,degree-1), removedofs=((0,-1),None)),
    domain.basis('spline', degree=(degree-1,degree), removedofs=(None,(0,-1)))])
  ns.pbasis = domain.basis('spline', degree=degree-1)
  ns.u = function.dotarg('u', ns.ubasis)
  ns.p = function.dotarg('p', ns.pbasis)
  ns.lm = function.Argument('lm', ())
  ns.stress_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'
  ns.uwall = function.stack([domain.boundary.indicator('top'), 0])
  ns.N = 5 * degree * nelems # nitsche constant based on element size = 1/nelems
  ns.nitsche_ni = '(N ubasis_ni - (∇_j(ubasis_ni) + ∇_i(ubasis_nj)) n_j) / Re'

  ures = domain.integral('∇_j(ubasis_ni) stress_ij dV' @ ns, degree=2*degree)
  ures += domain.boundary.integral('(nitsche_ni (u_i - uwall_i) - ubasis_ni stress_ij n_j) dS' @ ns, degree=2*degree)
  pres = domain.integral('pbasis_n (∇_k(u_k) + lm) dV' @ ns, degree=2*degree)
  lres = domain.integral('p dV' @ ns, degree=2*degree)

  with treelog.context('stokes'):
    state0 = solver.solve_linear(['u', 'p', 'lm'], [ures, pres, lres])
    postprocess(domain, ns, **state0)

  ures += domain.integral('ubasis_ni ∇_j(u_i) u_j dV' @ ns, degree=3*degree)
  with treelog.context('navierstokes'):
    state1 = solver.newton(('u', 'p', 'lm'), (ures, pres, lres), arguments=state0).solve(tol=1e-10)
    postprocess(domain, ns, **state1)

  return state0, state1

# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, every=.05, spacing=.01, **arguments):

  div = domain.integral('∇_k(u_k)^2 dV' @ ns, degree=1).eval(**arguments)**.5
  treelog.info('velocity divergence: {:.2e}'.format(div)) # confirm that velocity is pointwise divergence-free

  ns = ns.copy_() # copy namespace so that we don't modify the calling argument
  ns.streambasis = domain.basis('std', degree=2)[1:] # remove first dof to obtain non-singular system
  ns.stream = function.dotarg('streamdofs', ns.streambasis) # stream function
  ns.ε = function.levicivita(2)
  sqr = domain.integral('Σ_i (u_i - ε_ij ∇_j(stream))^2 dV' @ ns, degree=4)
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
  def test_p2(self):
    state0, state1 = main(nelems=3, reynolds=100, degree=2)
    with self.subTest('stokes-velocity'): self.assertAlmostEqual64(state0['u'], '''
      eNrzu9Bt8OuUndkD/eTTSqezzP2g/E3698/ZmZlf2GjSaHJS3/90/Wm/C4qGh04CAErOF+g=''')
    with self.subTest('stokes-pressure'): self.assertAlmostEqual64(state0['p'], '''
      eNoBIADf/0fSMC4P0ZLKjCk4JC7Pwy901sjb0jA90Lkt0NHxLm41+KgP+Q==''')
    with self.subTest('stokes-multiplier'): self.assertAlmostEqual(state0['lm'], 0)
    with self.subTest('navier-stokes-velocity'): self.assertAlmostEqual64(state1['u'], '''
      eNoBMADP/2XOWjJSy5k1jS+yyzvLODfgL1rO0MrINpsxHM2ZNSrPqDTANCPVQsxCzeAvcc04yT3AF9c=''')
    with self.subTest('navier-stokes-pressure'): self.assertAlmostEqual64(state1['p'], '''
      eNpbqpasrWa46TSTfoT+krO/de/pntA3Pnf2zFNtn2u2hglmAOKVDlE=''')
    with self.subTest('navier-stokes-multiplier'): self.assertAlmostEqual(state1['lm'], 0)

  @testing.requires('matplotlib')
  def test_p3(self):
    state0, state1 = main(nelems=3, reynolds=100, degree=3)
    with self.subTest('stokes-velocity'): self.assertAlmostEqual64(state0['u'], '''
      eNqbo3f7/AeDb2dmGGtd7r+odk7icoQJgjUHLpty8b/BvrMzjF/rll9qMD5kwAAFopc6dRvO2J2fo8d4
      3sko4wwAjL4lyw==''', atol=1e-14)
    with self.subTest('stokes-pressure'): self.assertAlmostEqual64(state0['p'], '''
      eNpbrN+us+Z8qWHMyfcXrly013l1ZocRAxwI6uvoHbwsZuxxNvZC5eUQg+5zS8wAElAT9w==''')
    with self.subTest('stokes-multiplier'): self.assertAlmostEqual(state0['lm'], 0)
    with self.subTest('navier-stokes-velocity'): self.assertAlmostEqual64(state1['u'], '''
      eNoBUACv/14yGcxyNPbJYTahLj/LSDE7yy43SM9WMsXJoDR+N3Iw8s1hM5zJODeizcE0X8phNrQwUDOO
      NbMzJi+ty4s1oDFqzxIzysjWzXIwFM3tNMjIpDMlJw==''')
    with self.subTest('navier-stokes-pressure'): self.assertAlmostEqual64(state1['p'], '''
      eNoBMgDN/yoxvDHizaEy88kDLgoviCt4znIzMy7jL7nTRMzqzhAwBTA/LE0w6czY2GQwoNEYMHE3NDUW
      DQ==''')
    with self.subTest('navier-stokes-multiplier'): self.assertAlmostEqual(state1['lm'], 0)
