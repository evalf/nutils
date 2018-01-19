#! /usr/bin/env python3

from nutils import mesh, util, cli, log, function, plot, numeric, solver, _
import numpy, unittest


class MakePlots:

  def __init__(self, domain, ns, timestep, rotation, bbox=((-2,6),(-3,3))):
    self.bbox = numpy.asarray(bbox)
    self.plotdomain = domain.select(function.min(*(ns.x-self.bbox[:,0])*(self.bbox[:,1]-ns.x)), 'bezier3')
    self.ns = ns
    self.every = .01
    self.index = 0
    self.timestep = timestep
    self.rotation = rotation
    self.spacing = .075
    self.xy = util.regularize(self.bbox, self.spacing)
    x = domain.elem_eval(ns.x, ischeme='bezier5', separate=True)
    inflow = domain.boundary['inflow'].elem_eval(ns.x, ischeme='bezier5', separate=True)
    with plot.PyPlot('mesh', ndigits=0) as plt:
      plt.mesh(x)
      plt.rectangle(self.bbox[:,0], *(self.bbox[:,1] - self.bbox[:,0]), ec='green')
      plt.segments(inflow, colors='red')

  def __call__(self, lhs):
    angle = self.index * self.timestep * self.rotation
    ns = self.ns(lhs=lhs)
    x, u, normu, p = self.plotdomain.elem_eval([ns.x, ns.u, function.norm2(ns.u), ns.p], ischeme='bezier9', separate=True)
    with plot.PyPlot('flow', index=self.index) as plt:
      plt.axes([0,0,1,1], yticks=[], xticks=[], frame_on=False)
      tri = plt.mesh(x, normu, mergetol=1e-5, cmap='jet')
      plt.clim(0, 1.5)
      plt.tricontour(tri, p, every=self.every, cmap='gray', linestyles='solid', alpha=.8)
      uv = plot.interpolate(tri, self.xy, u)
      plt.vectors(self.xy, uv, zorder=9, pivot='mid', stems=False)
      plt.plot(0, 0, 'k', marker=(3,2,angle*180/numpy.pi-90), markersize=20)
      plt.xlim(self.bbox[0])
      plt.ylim(self.bbox[1])
    self.xy = util.regularize(self.bbox, self.spacing, self.xy + uv * self.timestep)
    self.index += 1


def main(
    nelems: 'number of elements' = 12,
    viscosity: 'fluid viscosity' = 1e-2,
    density: 'fluid density' = 1,
    tol: 'solver tolerance' = 1e-12,
    rotation: 'cylinder rotation speed' = 0,
    timestep: 'time step' = 1/24,
    maxradius: 'approximate domain size' = 25,
    tmax: 'end time' = numpy.inf,
    degree: 'polynomial degree' = 2,
    figures: 'create figures' = True,
  ):

  log.user('reynolds number: {:.1f}'.format(density / viscosity)) # based on unit length and velocity

  # create namespace
  ns = function.Namespace()
  ns.uinf = 1, 0
  ns.density = density
  ns.viscosity = viscosity

  # construct mesh
  rscale = numpy.pi / nelems
  melems = numpy.ceil(numpy.log(2*maxradius) / rscale).astype(int)
  log.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, 2*nelems, .5*numpy.exp(rscale*melems)))
  domain, x0 = mesh.rectilinear([range(melems+1),numpy.linspace(0,2*numpy.pi,2*nelems+1)], periodic=(1,))
  rho, phi = x0
  phi += 1e-3 # tiny nudge (0.057 deg) to break element symmetry
  radius = .5 * function.exp(rscale * rho)
  ns.x = radius * function.trigtangent(phi)
  domain = domain.withboundary(inner='left', inflow=domain.boundary['right'].select(-ns.uinf.dotnorm(ns.x), ischeme='gauss1'))

  # prepare bases (using piola transformation to maintain u/p compatibility)
  J = ns.x.grad(x0)
  detJ = function.determinant(J)
  ns.unbasis, ns.utbasis, ns.pbasis = function.chain([ # compatible spaces using piola transformation
    domain.basis('spline', degree=(degree+1,degree), removedofs=((0,),None))[:,_] * J[:,0] / detJ,
    domain.basis('spline', degree=(degree,degree+1))[:,_] * J[:,1] / detJ,
    domain.basis('spline', degree=degree) / detJ,
  ])
  ns.ubasis_ni = 'unbasis_ni + utbasis_ni'

  # populate namespace
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.sigma_ij = 'viscosity (u_i,j + u_j,i) - p δ_ij'
  ns.hinner = 2 * numpy.pi / nelems
  ns.c = 5 * (degree+1) / ns.hinner
  ns.nietzsche_ni = 'viscosity (c ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j)'
  ns.ucyl = -.5 * rotation * function.trignormal(phi)

  # create residual vector components
  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n u_k,k' @ ns, geometry=ns.x, degree=2*(degree+1))
  res += domain.boundary['inner'].integral('nietzsche_ni (u_i - ucyl_i)' @ ns, geometry=ns.x, degree=2*(degree+1))
  oseen = domain.integral('density ubasis_ni u_i,j uinf_j' @ ns, geometry=ns.x, degree=2*(degree+1))
  convec = domain.integral('density ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*(degree+1))
  inertia = domain.integral('density ubasis_ni u_i' @ ns, geometry=ns.x, degree=2*(degree+1))

  # constrain full velocity vector at inflow
  sqr = domain.boundary['inflow'].integral('(u_i - uinf_i) (u_i - uinf_i)' @ ns, geometry=ns.x, degree=9)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # solve unsteady navier-stokes equations, starting from stationary oseen flow
  lhs0 = solver.solve_linear('lhs', res+oseen, constrain=cons)
  makeplots = MakePlots(domain, ns, timestep=timestep, rotation=rotation) if figures else lambda *args: None
  for istep, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', residual=res+convec, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)):
    makeplots(lhs)
    if istep * timestep >= tmax:
      break

  return lhs0, lhs


class test(unittest.TestCase):

  def test_rot0(self):
    lhs0, lhs = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=0, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNo1k8ltwEAIRRuyJfallijH9N9CZj74BIIPHh7YHvbn52Uu+nv1YdK49mWn3kBN4FNkMwJNrX+vPV1S'
      '175cpVNCOgpxvxlORalpGhKmJdO0fCPUDgkTun8JtRilau3DUCun5LwEJdLIsAYSLPcrR8BiDufY+iK8'
      'EgREq2GJegQtCkczvgjnSDiuVd/PS92EPlWXyxlfNPCez77VBkBHiZms7XzVb3OLKWlCrXRCqS4A9lpk'
      'QBqgfRwPAggOngCr5nVcBeQC+7klEqD9utLADbVaCXqEJBRBgdI3szHLK6JDqtiww0qfoTiAodRoN5e5'
      'a6gpEfIYUrOHk4CCa1EamWA6lXHCLv4J5MxNAif9ngEyMnQPM6BJ/jLJ67CDprdMN3fXpegoPvP3BipX'
      'krjZKJoM59zumWtYeZRtxvdWYyThtu27ly/3/iMac3qH7N515fLijkl5D47Eag8fkilK0rlXP5RBEG+R'
      '339o6Mpq')
    numeric.assert_allclose64(lhs,
      'eNo1UslxADEIa2h3hvuoJZNn+m8hWHhfMELCIGMP+/PzMkn8vfowBZ34crBdgBcgaQBuDqCJ8u+1YZAu'
      'o6lWIrYMDgMjGs1Nc6WmtQmX73NGDa2NGD3K0VTtPq9a25S1piIj4UxIpNGMNeQUWNJAYDmTTjKxPoQv'
      'BYBoNSJRL6Flm2vGh3AuheNE9ZLbVDBxtsGGV1yx5cT1o+gw7Blm7uhcS1V2WY2EgqLH9ol9Rh7HPmbZ'
      'Wc6f14NhxEgDgPQB/JnXYEzW2WUKIfcLXdfLN9ShDV4Ph4HRgwITv5m9E8o4gu2KDUtUOqjCARtK7Z5D'
      'Zd5vqJUIeaxT+w+GHcbs+qz0M6EdT713W75AnUFO4gwf2JoBmDoMev06pdwLWPdaZLRiNlqrgmnFUor7'
      'G5f3HWmCvUORTTjPnn5G2Budw3fcefs3nOW97xQk6cH7Jxr3noOvWHRvz/T7BFoDp5Sw5w0XQpK+ZBHi'
      'raTtbXk0KlZ2vP39B5jEykU=')

  def test_rot1(self):
    lhs0, lhs = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=1, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNo9UkluBEEI+1C3xL68Jcox///CFKYmfQGBy40N9rA/P2+m5N9rT0Qjvux8En3YuCe+HCQoFBEKbkF4'
      'wiH7pBod1tJ9osQopMZESwPna+myiIpNNPd3mrIcFXiqFXQBxZd02GUqYQvZDmugwZIGAIs5khPrW+EL'
      'QUG0GpGoF9DyT/6tzGAD4ZioXrKNWmOCyle+NVSqRO2k3LoDKrT5+QbJ3TZRShErSSe2JIy0i/P0APMh'
      'BmO4AhCiWJL3OpAUO4LnKB6DizfxUhh89rrTpl2qWl9TavcrxxwIy1CM3eLQI7KK68o4yhOWvMa1vgp5'
      'rEkB6GkAcba4LkrbSfw5lIjhvDcTmYwki6AhfU05x7RmxV5XsqwZzY6zewuTjwuOF33tYjJIC73U3Hkl'
      'Uq8ZtS/Ed0HiAoDDpJmK92DfoD2/8JZdVcGD9Ix7jXGPrVfynPRNuLclaveo3OAXk3wh82M+CTFNYjU+'
      'ye8H5sTITg==')
    numeric.assert_allclose64(lhs,
      'eNo9ksuNIDEIRBNqS+YPsYzmOPmnsHZV955AULbxo/yReH5WzM6/5U9JxI1LMudv2SPu+8YllsFCFgqR'
      'I1eamsojY0mFNY+oI5FsNLxC0DiJU9G8dFkJJFaG56QptX6VVm3vHDfRW4mmpBVHLNEQLYfgPB9ITuyv'
      'Iq8EBbUexL2HgtHv8vwqUpTcAfWxuK/dhnGy2NL8fjnAmA//JNswmGmgYFvBVrNxwjcBtrYjmgyAeqEf'
      'HdjJ8mncfBo4EKUUTAFWYaYjjP+AW5icDiBl7c3omDqjyfVcRa4qSlqVhs2POqIqf9zt+i6tgGS5NLnq'
      'jiSkhPQ0oDjrBUVJxQ9lD+Lx1InxrMSa7+Tu8/rOCjBy9pXYvm/EWdl+jWmEsUQqmeSAU1byFaEhlvbn'
      'zCKoO+iQC7xxdlHJs0akEozLcpPgWfO8pHibxuAy3ymvzStpCRl5fZafBT+z5vX1ZVa02nEaJSNhL3cu'
      'wB0+/v0H0NrIrg==')


if __name__ == '__main__':
  cli.run(main)
