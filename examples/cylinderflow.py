#! /usr/bin/env python3

from nutils import mesh, util, cli, log, function, plot, debug, solver, _
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
    ns = self.ns | dict(lhs=lhs)
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
    withplots: 'create plots' = True,
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
  makeplots = MakePlots(domain, ns, timestep=timestep, rotation=rotation) if withplots else lambda *args: None
  for istep, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', residual=res+convec, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)):
    makeplots(lhs)
    if istep * timestep >= tmax:
      break

  return lhs0, lhs


class test(unittest.TestCase):

  def test_rot0(self):
    retvals = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=0, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNrFlTmOHDEMRa8zBroA7suBfP8rWOJnOZjJnDiSmotK/ORT8+fLPuy/Pl9fD3P270c/TJp3fdip1tAw
      cNEYsmUMTXdjny7puz5cJUghQ4S4Xw+nTqpp+jhMi3FoOUKNelzGNJ7XoRb4mmohl/V+RU7KXtmkY1I0
      xsGSNgEsho2Y12uhDem7ivb8FmJCQIvORjN6LRwbMqv6fl6K57NVt5ZTvmjMfd71qbYR6ERODdZ2ruH3
      cHOkNNFEdOpd1QUOi6vUCY1sno3HChP3ML86aNyNq8w9YvpzU8TxfdftQ6jlhsxFQnL6ERST+mRCwkfk
      tv9UV2w61SUOE45xlBpt5zK3DYWNkK9SMRq+jrNAXCPo8ahgE9b0GhJ10y3GPun0hjKG62hm45kBGQNv
      MmPKvHcO3V1XRZ+QUKzHULEhWeMp2tnN0q0LWnmUrcd3VmPVc9vju5ET3AuNBiYuRoM715WrF3fA5Y1h
      PKJjGOk2/RpIke2FjZXNEB5CHya0mSn2wGBdAwg5ESDWLRZQGlX/8sj9Mi1g6cz2FMmBAg6g8QJqL6D0
      ApoLqCyg+h1Q/QFoLqD+PwA16JGD3aXNdao7K25aJCPQG3nfuHxnaqGWmGYfqBmkAt0TiUNPi3wB5VHq
      pOYYpJkAqE75WdU/AJUFFCofQB2ARn0HNH8AKgtoLaC1gPK/Asq+RXF7oVqhfdoD4Jy/hQGGDY//mRTH
      xiVmo9z24hELdUCzs4EAL8NSCnXnQcKDiOTzyr342b53tW8UB+Rr36eBLZeAlGlseiDnNHz/cIL330MU
      82O6HDn5wivpSyhjtNIRLHLvez1zmUto9DdCf/0BdaOUcQ=='''))

  def test_rot1(self):
    retvals = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=1, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNrFlU2OHCEMha8zLVVJ+N8caO5/hRg/M4qSXTZZQYMx+Pl91fR86UP2eb6+3giO71cf993jS0b+/cpD
      yuuML/miXsiFBVPbfYSccSR375Ck4ojMkZA4o4Zmb2gYISLP2ZpIUIdI4Lq7IelrAnKOSMr3y2fFFSHY
      IfHeIA7tAGLFhNXyrqwJ2Wdk2f2bFy0EbP5Jvmfl6NAhPYolYyNpn2t9paF83a2YsKNKqmx4IMo3Mz+R
      tLeekVP6ZMbq35tDzqgVeEYLQy8qcWd0kw5wluyA7XbWY3knfi0UymuSoUkpLU71Fa+dFrgldA3O3ni5
      xOnCws8tUs+xfjYzKs4poyqP6EitGQReNiKhBXejBujKW+sWeyrlOqMbdSWvR1BPIiFCGEQpM0Es944M
      YtS4yVZPsl9+VIBce+SipShZJjXtQEN4oSGeOMG2u1I2bg3qcUjt5aU+4Qv28448rcpOVVf5uNHHbDt9
      fOEQpdo/9hOFZcQUsiweP9K5mGqyVk80rQUrJl/by1G5jcEcNiBV+u3ys+DojPkmGOQiucUQIehyNblD
      yXOQNL5IykVSL5Lwc8ggaWNnuwEpfyJpOUjK/0BSUIMt2ig/FCXoBg60gIOw9YIshhM8myhdELC81W4s
      evCN0wBx+YNkdubaYLDKg2QsGCRlkJxPXSFJgyQHUMTXsZC0QdIukvIXknuQXIOkAEnlf0SyPtoKWnYr
      oHr4s7J8Dl+mmuM76aKz3WUl2hHLqmUw6On/fBnoYuGg18NBM92S0gfKgFDnobCmsaIX4fgrEm7tyTC+
      Mv9Ah6M9SiFbgdzJdPmQwWEXQhqfXVD5mtUZftJYPSmnIaQ+MTK6c6dRxeHP8/kFoXSQwA=='''))


if __name__ == '__main__':
  cli.run(main)
