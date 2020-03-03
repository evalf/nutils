#! /usr/bin/env python3
#
# In this script we solve the Navier-Stokes equations around a cylinder, using
# the same Raviart-Thomas discretization as in
# :ref:`examples/drivencavity-compatible.py` but in curvilinear coordinates.
# The mesh is constructed such that all elements are shape similar, growing
# exponentially with radius such that the artificial exterior boundary is
# placed at large (configurable) distance.

from nutils import mesh, function, solver, util, export, cli, testing
import numpy, treelog

def main(nelems:int, degree:int, reynolds:float, rotation:float, timestep:float, maxradius:float, seed:int, endtime:float):
  '''
  Flow around a cylinder.

  .. arguments::

     nelems [24]
       Element size expressed in number of elements along the cylinder wall.
       All elements have similar shape with approximately unit aspect ratio,
       with elements away from the cylinder wall growing exponentially.
     degree [3]
       Polynomial degree for velocity space; the pressure space is one degree
       less.
     reynolds [1000]
       Reynolds number, taking the cylinder radius as characteristic length.
     rotation [0]
       Cylinder rotation speed.
     timestep [.04]
       Time step
     maxradius [25]
       Target exterior radius; the actual domain size is subject to integer
       multiples of the configured element size.
     seed [0]
       Random seed for small velocity noise in the intial condition.
     endtime [inf]
       Stopping time.
  '''

  elemangle = 2 * numpy.pi / nelems
  melems = int(numpy.log(2*maxradius) / elemangle + .5)
  treelog.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, nelems, .5*numpy.exp(elemangle*melems)))
  domain, geom = mesh.rectilinear([melems, nelems], periodic=(1,))
  domain = domain.withboundary(inner='left', outer='right')

  ns = function.Namespace()
  ns.uinf = 1, 0
  ns.r = .5 * function.exp(elemangle * geom[0])
  ns.Re = reynolds
  ns.phi = geom[1] * elemangle # add small angle to break element symmetry
  ns.x_i = 'r <cos(phi), sin(phi)>_i'
  ns.J = ns.x.grad(geom)
  ns.unbasis, ns.utbasis, ns.pbasis = function.chain([ # compatible spaces
    domain.basis('spline', degree=(degree,degree-1), removedofs=((0,),None)),
    domain.basis('spline', degree=(degree-1,degree)),
    domain.basis('spline', degree=degree-1),
  ]) / function.determinant(ns.J)
  ns.ubasis_ni = 'unbasis_n J_i0 + utbasis_n J_i1' # piola transformation
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.sigma_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
  ns.N = 10 * degree / elemangle # Nitsche constant based on element size = elemangle/2
  ns.nitsche_ni = '(N ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j) / Re'
  ns.rotation = rotation
  ns.uwall_i = '0.5 rotation <-sin(phi), cos(phi)>_i'

  inflow = domain.boundary['outer'].select(-ns.uinf.dotnorm(ns.x), ischeme='gauss1') # upstream half of the exterior boundary
  sqr = inflow.integral('(u_i - uinf_i) (u_i - uinf_i)' @ ns, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15) # constrain inflow semicircle to uinf

  sqr = domain.integral('(u_i - uinf_i) (u_i - uinf_i) + p^2' @ ns, degree=degree*2)
  lhs0 = solver.optimize('lhs', sqr) # set initial condition to u=uinf, p=0

  numpy.random.seed(seed)
  lhs0 *= numpy.random.normal(1, .1, lhs0.shape) # add small velocity noise

  res = domain.integral('(ubasis_ni u_i,j u_j + ubasis_ni,j sigma_ij + pbasis_n u_k,k) d:x' @ ns, degree=9)
  res += domain.boundary['inner'].integral('(nitsche_ni (u_i - uwall_i) - ubasis_ni sigma_ij n_j) d:x' @ ns, degree=9)
  inertia = domain.integral('ubasis_ni u_i d:x' @ ns, degree=9)

  bbox = numpy.array([[-2,46/9],[-2,2]]) # bounding box for figure based on 16x9 aspect ratio
  bezier0 = domain.sample('bezier', 5)
  bezier = bezier0.subset((bezier0.eval((ns.x-bbox[:,0]) * (bbox[:,1]-ns.x)) > 0).all(axis=1))
  interpolate = util.tri_interpolator(bezier.tri, bezier.eval(ns.x), mergetol=1e-5) # interpolator for quivers
  spacing = .05 # initial quiver spacing
  xgrd = util.regularize(bbox, spacing)

  with treelog.iter.plain('timestep', solver.impliciteuler('lhs', residual=res, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)) as steps:
    for istep, lhs in enumerate(steps):

      t = istep * timestep
      x, u, normu, p = bezier.eval(['x_i', 'u_i', 'sqrt(u_k u_k)', 'p'] @ ns, lhs=lhs)
      ugrd = interpolate[xgrd](u)

      with export.mplfigure('flow.png', figsize=(12.8,7.2)) as fig:
        ax = fig.add_axes([0,0,1,1], yticks=[], xticks=[], frame_on=False, xlim=bbox[0], ylim=bbox[1])
        im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, p, shading='gouraud', cmap='jet')
        import matplotlib.collections
        ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1, alpha=.5))
        ax.quiver(xgrd[:,0], xgrd[:,1], ugrd[:,0], ugrd[:,1], angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9, alpha=.5)
        ax.plot(0, 0, 'k', marker=(3,2,t*rotation*180/numpy.pi-90), markersize=20)
        cax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
        cax.tick_params(labelsize='large')
        fig.colorbar(im, cax=cax)

      if t >= endtime:
        break

      xgrd = util.regularize(bbox, spacing, xgrd + ugrd * timestep)

  return lhs0, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib', 'scipy')
  def test_rot0(self):
    lhs0, lhs = main(nelems=6, degree=3, reynolds=100, rotation=0, timestep=.1, maxradius=25, seed=0, endtime=.05)
    with self.subTest('initial condition'): self.assertAlmostEqual64(lhs0, '''
      eNpT1n+qx8Bw8sLNCwwM6bpGugwMmy7tv8TA4GmoZcjAcObctHMMDOuNio0YGBzPmp9lYHhuYmTCwNB5
      2uI0A4OFqbMpA4Pd6YenGBhSgDpfXXoG1HlXpwXItrxkCmSz683WZ2CwvvDrPAPDVv3fQBMZzn0FmvLK
      8LkxA4PCmZAzDAzfjL8ATXx0agPQlBCgedQBAOgCMhE=''', atol=2e-13)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNoB2AAn/4Y0pjUwMHTKhMrmMoI4Qzcpz4TI78egy545+Dm7MwPGEsa+NVY8pjtVNSzE18OoyXI9VD02
      M5zCnsJazE0+Hj76NsPByMH/yhQ30DN6yFjIAjCrN5Y4FcooyE3I8ssCOGk4QjXXxrPGNzILOXo7AMj3
      xOjEM8k3O8Y85DcZwyTDAzjaPFY+sMfJwavBhDNPPvbFX8cuOKI3/zpFOFI87TqmN9k8C8hkNFnCgcXV
      Pds7VT/qPdZBbEF5QUZD7UEJQYi527ziROVETEeVRfZIfrfuRKZKr7s6SRCVaAA=''')

  @testing.requires('matplotlib', 'scipy')
  def test_rot1(self):
    lhs0, lhs = main(nelems=6, degree=3, reynolds=100, rotation=1, timestep=.1, maxradius=25, seed=0, endtime=.05)
    with self.subTest('initial condition'): self.assertAlmostEqual64(lhs0, '''
      eNpT1n+qx8Bw8sLNCwwM6bpGugwMmy7tv8TA4GmoZcjAcObctHMMDOuNio0YGBzPmp9lYHhuYmTCwNB5
      2uI0A4OFqbMpA4Pd6YenGBhSgDpfXXoG1HlXpwXItrxkCmSz683WZ2CwvvDrPAPDVv3fQBMZzn0FmvLK
      8LkxA4PCmZAzDAzfjL8ATXx0agPQlBCgedQBAOgCMhE=''', atol=2e-13)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNoB2AAn/380qTWFMHXKgsrUMoI4RDdJz4XI78eZy545+Dm8MwPGEsa+NVY8pjtWNSzE18OoyXI9VD02
      M5zCnsJazE0+Hj76NsPByMH/yjM3ejSWyGzI/TG+N5I4A8oiyEjIzsv9N2o4RTXYxrTGajIMOXo7AMj3
      xOjEMsk3O8Y85TcZwyTDAzjaPFY+sMfJwavBhDNPPvPFZMc4OKg3/jo7OFI87jqtN9k8Ccg6NFjChcXW
      Pd07VT/oPdZBbEF5QUZD7EEIQYe527ziROVETEeURfZIfrfuRKZKrrs6SVFLajU=''')
