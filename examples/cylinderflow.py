#! /usr/bin/env python3

from nutils import *
import numpy, unittest
from matplotlib import collections, patches, ticker


class MakePlots:

  def __init__(self, domain, ns, timestep, rotation, bbox=((-2,6),(-3,3))):
    self.bbox = numpy.asarray(bbox)
    self.ns = ns
    self.locator = ticker.MultipleLocator(.01)
    self.index = 0
    self.timestep = timestep
    self.rotation = rotation
    self.spacing = .075
    self.xgrd = util.regularize(self.bbox, self.spacing)
    bezier = domain.sample('bezier', 5)
    x = bezier.eval(ns.x)
    inflow = domain.boundary['inflow'].sample('bezier', 5)
    xin = inflow.eval(ns.x)
    with export.mplfigure('mesh.png') as fig:
      ax = fig.add_subplot(111, aspect='equal')
      ax.add_collection(collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1))
      ax.add_patch(patches.Rectangle(self.bbox[:,0], *(self.bbox[:,1] - self.bbox[:,0]), fc='none', ec='green'))
      ax.add_collection(collections.LineCollection(xin.take(inflow.tri,0), colors='r', linewidths=1))
      ax.autoscale(enable=True, axis='both', tight=True)
    self.bezier = bezier.subset((x > self.bbox[:,0]).all(axis=1) & (x < self.bbox[:,1]).all(axis=1))
    self.interpolate = util.tri_interpolator(self.bezier.tri, self.bezier.eval(ns.x), mergetol=1e-5)

  def __call__(self, **arguments):
    angle = self.index * self.timestep * self.rotation
    x, u, normu, p = self.bezier.eval([self.ns.x, self.ns.u, function.norm2(self.ns.u), self.ns.p], arguments=arguments)
    ugrd = numeric.normalize(self.interpolate[self.xgrd](u), axis=1)
    with export.mplfigure('flow.png') as fig:
      ax = fig.add_axes([0,0,1,1], yticks=[], xticks=[], frame_on=False, xlim=self.bbox[0], ylim=self.bbox[1])
      im = ax.tripcolor(x[:,0], x[:,1], self.bezier.tri, normu, shading='gouraud', cmap='jet')
      im.set_clim(0, 1.5)
      ax.add_collection(collections.LineCollection(x[self.bezier.hull], colors='k', linewidths=.5, alpha=.1))
      ax.tricontour(x[:,0], x[:,1], self.bezier.tri, p, locator=self.locator, cmap='gray', linestyles='solid')
      ax.quiver(self.xgrd[:,0], self.xgrd[:,1], ugrd[:,0], ugrd[:,1], angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9)
      ax.plot(0, 0, 'k', marker=(3,2,angle*180/numpy.pi-90), markersize=20)
    self.xgrd = util.regularize(self.bbox, self.spacing, self.xgrd + ugrd * self.timestep)
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
    domain.basis('spline', degree=(degree+1,degree), removedofs=((0,),None))[:,numpy.newaxis] * J[:,0] / detJ,
    domain.basis('spline', degree=(degree,degree+1))[:,numpy.newaxis] * J[:,1] / detJ,
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
  makeplots = MakePlots(domain, ns, timestep=timestep, rotation=rotation)
  for istep, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', residual=res+convec, inertia=inertia, lhs0=lhs0, timestep=timestep, constrain=cons, newtontol=1e-10)):
    makeplots(lhs=lhs)
    if istep * timestep >= tmax:
      break

  return lhs0, lhs


class test(unittest.TestCase):

  def test_rot0(self):
    lhs0, lhs = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=0)
    numeric.assert_allclose64(lhs0, 'eNoB2AAn/1zIZDfgx2U3W8h2ON3ISTdyx1Q31chkODQ6y8V5x+'
      '3FFDqHOL85TMaNw+PFKzpsPEs8wcNtwsPDTDyFPS496MKXwenCMD1PPqjGFzf/xgE55shXOXMxmcoZy/'
      'Q0STWpzu/NUMVCLxzRljriMRnFbMUSxeE66DrXOqvEnMKVxI87VD13O/TCY8HGwjs9nT4KPXPK4cp6Nu'
      'DKdcrCNuXKmcqnNp7K78pNNtTJLcpvNinK0smSNo3He8c4NpbHqsd+NorFHsWUwynFlsXVw3/ALMDeO1'
      '/AvcDnPisGcyQ=')
    numeric.assert_allclose64(lhs, 'eNoB2AAn/6HIbze/x3A3oMgsOOLIVDdrx1432shaODQ6y8V6x+3'
      'FFDqHOL85TcaNw+PFKzpsPEs8wcNtwsPDTDyFPS496MKXwenCMD1PPvLG3zbdxiM5HMkOOdjM78qVy4A'
      '07zQpM3/NUMWSMJbPlzpcMhnFbMUSxeE66DrXOqvEnMKVxI87VD13O/TCY8HGwjs9nT4KPUbM3cvrNPX'
      'LX8ybM07K78kXNQnKcMr+LqXJJMttNBjLo8mX0G7Hscc5N+3HnsdHMmbFscXGw87Fg8W1w4XAPcBcPXT'
      'AycDnPrEddBk=')

  def test_rot1(self):
    lhs0, lhs = main(nelems=3, viscosity=1e-2, timestep=.1, tmax=.05, rotation=1)
    numeric.assert_allclose64(lhs0, 'eNoB2AAn/y/JijbexwE4w8d4ON3JYDZwx/Q3KshmOF06o8V3xx'
      '7G4zmJOPU5FcaNwxfG+DlsPEs8wcNtwsPDTDyFPS496MKXwenCMD1PPoPHWjblx6I5VsjyOSw2LDTcNA'
      '03LDfJNT42jMV9NmY2zzqkNkTFnMU8xQk7DTsAO83EtsK2xKs7bj2UO/TCY8HGwjs9nT4KPZDPKzB8Nn'
      'HJKsnFNhM00TOpNtrIDMlPNkY3ZjdxNlnHPceVNi85Ijk5NiPFKcWCNpE7RzuUw9jC98LWw4XCtsHfO3'
      'O/q7/nPotaZqU=')
    numeric.assert_allclose64(lhs, 'eNoB2AAn/7XJqzauxwk4E8gSOMjJaDZkxwg4P8hYOF06o8V2xx7'
      'G5DmGOPU5FcaNwxjG+DlsPEs8wcNtwsPDTDyFPS496MKXwenCMD1PPhfI+TWXx8Q5n8igOfw0wTSTNfQ'
      '29jZ5Nkc2i8V5Njw2zzrINkPFnMU9xQk7DDsAO83EtsK1xKo7bj2UO/TCY8HGwjs9nT4KPcUzADM5MWv'
      'J3sk3zI0yhzAwMwrIXcglzI82pjcgy5HHRcfxydA0bTanMy3GNcZOyS45Dzq1w9fDusOyw6rBKMEZPdy'
      '/N8C9PtY3aIA=')


if __name__ == '__main__':
  cli.run(main)
