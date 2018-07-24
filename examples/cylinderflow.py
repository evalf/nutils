#! /usr/bin/env python3
#
# In this script we solve the Navier-Stokes equations around a cylinder, using
# the same Raviart-Thomas discretization as in
# :ref:`examples/drivencavity-compatible.py` but in curvilinear coordinates.
# The mesh is constructed such that all elements are shape similar, growing
# exponentially with radius such that the artificial exterior boundary is
# placed at large (configurable) distance.

import nutils, numpy, matplotlib.collections

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along the cylinder
# wall), polynomial degree, Reynolds number, rotational velocity of the
# cylinder, time step, exterior radius, and the end time of the simulation (by
# default it will run forever).

def main(nelems: 'number of elements' = 24,
         degree: 'polynomial degree' = 3,
         reynolds: 'reynolds number' = 1000.,
         rotation: 'cylinder rotation speed' = 0.,
         timestep: 'time step' = .1,
         maxradius: 'approximate domain size' = 50.,
         endtime: 'end time' = numpy.inf):

  elemangle = 2 * numpy.pi / nelems
  melems = int(numpy.log(2*maxradius) / elemangle + .5)
  nutils.log.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, nelems, .5*numpy.exp(elemangle*melems)))
  domain, geom = nutils.mesh.rectilinear([melems, nelems], periodic=(1,))
  domain = domain.withboundary(inner='left', outer='right')

  ns = nutils.function.Namespace()
  ns.uinf = 1, 0
  ns.r = .5 * nutils.function.exp(elemangle * geom[0])
  ns.Re = reynolds
  ns.phi = (geom[1] + 1/3) * elemangle # add small angle to break element symmetry
  ns.x_i = 'r <-cos(phi), -sin(phi)>_i'
  ns.J = ns.x.grad(geom)
  ns.unbasis, ns.utbasis, ns.pbasis = nutils.function.chain([ # compatible spaces
    domain.basis('spline', degree=(degree,degree-1), removedofs=((0,),None)),
    domain.basis('spline', degree=(degree-1,degree)),
    domain.basis('spline', degree=degree-1),
  ]) / nutils.function.determinant(ns.J)
  ns.ubasis_ni = 'unbasis_n J_i0 + utbasis_n J_i1' # piola transformation
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.sigma_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
  ns.h = .5 * elemangle
  ns.N = 5 * degree / ns.h
  ns.rotation = rotation
  ns.uwall_i = '0.5 rotation <sin(phi), -cos(phi)>_i'

  inflow = domain.boundary['outer'].select(-ns.uinf.dotnorm(ns.x), ischeme='gauss1') # upstream half of the exterior boundary
  sqr = inflow.integral('(u_i - uinf_i) (u_i - uinf_i)' @ ns, degree=degree*2)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15) # constrain inflow semicircle to uinf

  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n u_k,k' @ ns, geometry=ns.x, degree=9)
  res += domain.boundary['inner'].integral('(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) (u_i - uwall_i) / Re' @ ns, geometry=ns.x, degree=9)
  lhs0 = nutils.solver.solve_linear('lhs', res, constrain=cons) # use stokes flow as initial condition

  res += domain.integral('ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=9) # add convective term for Navier-Stokes
  inertia = domain.integral('ubasis_ni u_i' @ ns, geometry=ns.x, degree=9)

  bbox = numpy.array([[-2,6],[-3,3]]) # bounding box for plots
  bezier0 = domain.sample('bezier', 5)
  bezier = bezier0.subset((bezier0.eval((ns.x-bbox[:,0]) * (bbox[:,1]-ns.x)) > 0).all(axis=1))
  interpolate = nutils.util.tri_interpolator(bezier.tri, bezier.eval(ns.x), mergetol=1e-5) # interpolator for quivers
  spacing = .075 # initial quiver spacing
  xgrd = nutils.util.regularize(bbox, spacing)

  for istep, lhs in nutils.log.enumerate('timestep', nutils.solver.impliciteuler('lhs', residual=res, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)):

    t = istep * timestep
    x, u, normu, p = bezier.eval([ns.x, ns.u, nutils.function.norm2(ns.u), ns.p], arguments=dict(lhs=lhs))
    ugrd = interpolate[xgrd](u)

    with nutils.export.mplfigure('flow.jpg') as fig:
      ax = fig.add_axes([0,0,1,1], yticks=[], xticks=[], frame_on=False, xlim=bbox[0], ylim=bbox[1])
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, p, shading='gouraud', cmap='jet')
      ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1, alpha=.5))
      ax.quiver(xgrd[:,0], xgrd[:,1], ugrd[:,0], ugrd[:,1], angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9)
      ax.plot(0, 0, 'k', marker=(3,2,t*rotation*180/numpy.pi-90), markersize=20)
      cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
      fig.colorbar(im, cax=cax)

    if t >= endtime:
      break

    xgrd = nutils.util.regularize(bbox, spacing, xgrd + ugrd * timestep)

  return lhs0, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_rot0(self):
    lhs0, lhs = main(nelems=6, reynolds=100, timestep=.1, endtime=.05, rotation=0)
    nutils.numeric.assert_allclose64(lhs0, 'eNoB2AAn/9PRqdIcLCkuWy3o02bNRc56MJYyv'
      'jGLz7jHjsg4NkQ4dzfMyYvEZsVhOW47oTqnxl7CL8OnO5Q93zxmxIjBVsKSPF8+vj2Mw4LQQC5'
      'RMHsvx9Guz6fNHDEmM1Yy6c7ZzKjIGTYrOFU37snUx83F9jgDOy06Fsf9xC7DlDurPcY8icRTw'
      'kfCezyWPpU9ysNqwYAstCvn1arTidSMKngtsCza1KzSgtOJK4Qvxy6t0/bQ99HFLQQyczGx1EH'
      'Ph9C1MOs0djROMkjPUTD7M903hTfgNFTM6DARN18Jbag=')
    nutils.numeric.assert_allclose64(lhs, 'eNoB2AAn/yvR7tI+LWIthC2A2O7MaM59MfYx5D'
      'Gl0bDHkMhLNjs4eTfgyYfEZ8VqOWo7oTqwxl7CL8OnO5Q93zxmxIjBVsKSPF8+vj2Mw+bP/i52'
      'MMcus9LUzzzNmjFFM8cxnc/1zKjIGjYrOFQ378nUx8vF+TgDOyo6Gcf9xC/DkjurPcg8h8RSwk'
      'fCezyWPpU9ysNqwajQIc94zpLP7M6LztvNs82UzaPNWc3EzRrNMcpjyQPMEspxyR/LuMbqxRHL'
      'j8b5xW072cR2w4I7N8U7w0/BMcOoPYLEycNPO0LNduM=')

  def test_rot1(self):
    lhs0, lhs = main(nelems=6, reynolds=100, timestep=.1, endtime=.05, rotation=1)
    nutils.numeric.assert_allclose64(lhs0, 'eNoB2AAn/9PRqdIcLCkuWy3o02bNRc56MJYyv'
      'jGLz7jHjsg4NkQ4dzfMyYvEZsVhOW47oTqnxl7CL8OnO5Q93zxmxIjBVsKSPF8+vj2Mwz80fjS'
      'jNI80UzQnNKg0aDXFNZY16zRTNA/JrjZkOKo3ycoVyO/FLzkXO0w6WMcSxTPDnTuuPcs8k8RWw'
      'kfCezyWPpU9ysNqwYAstCvn1arTidSMKngtsCza1KzSgtOJK4Qvxy6t0/bQ99HFLQQyczGx1EH'
      'Ph9C1MOs0djROMkjPUTD7M903hTfgNFTM6DARNxFpaOI=')
    nutils.numeric.assert_allclose64(lhs, 'eNoB2AAn/wLR7dWDLrQtmSvT0urMkM68MQMysz'
      'HF0LDHkchONjs4eDfdyYfEZ8VqOWo7oTqwxl7CL8OnO5Q93zxmxIjBVsKSPF8+vj2Mw0w0pzSt'
      'NGk0NjQkNIM0eTXONXw1/DRhNAzJrjZkOKo3z8oVyOzFMTkYO0o6XMcTxTTDmzuuPc08kcRWwk'
      'fCezyWPpU9ysNqwcPO5cyazNXNHM8zz2bN0Mucy5HMqs5V0r7NY8m7yG3LE8tSyivMQsaKxSTK'
      'EMdsxn47mcRSw3s7msVnw1bB88J5PXrEQMQDPARdb3A=')
