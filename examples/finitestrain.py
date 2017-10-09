#! /usr/bin/env python3

from nutils import *
import unittest


def makeplots(name, domain, ns):
  from matplotlib import colors
  X, energy = domain.simplex.elem_eval([ns.X, ns.energy], ischeme='bezier3', separate=True)
  with plot.PyPlot(name, ndigits=0) as plt:
    plt.mesh(X, energy, triangulate='bezier', cmap='jet', norm=colors.LogNorm())
    plt.colorbar()
    plt.axis('equal')


def main(
    nelems: 'number of elements' = 12,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    angle: 'bend angle (degrees)' = 20,
    restol: 'residual tolerance' = 1e-10,
    trim: 'create circular-shaped hole' = False,
    plots: 'create plots' = True,
 ):

  ns = function.Namespace()
  ns.angle = angle * numpy.pi / 180
  ns.lmbda = lmbda
  ns.mu = mu

  verts = numpy.linspace(0, 1, nelems+1)
  domain, ns.x = mesh.rectilinear([verts,verts])
  if trim:
    levelset = function.norm2(ns.x - (.5,.5)) - .2
    domain = domain.trim(levelset, maxrefine=2)

  ns.ubasis = domain.basis('spline', degree=2).vector(domain.ndims)
  ns.u_i = 'ubasis_ki ?lhs_k'
  ns.X_i = 'x_i + u_i'

  sqr = domain.boundary['left'].integral('u_0^2 + u_1^2' @ ns, geometry=ns.x, degree=6)
  sqr += domain.boundary['right'].integral('(u_0 - x_1 sin(2 angle) - cos(angle) + 1)^2 + (u_1 - x_1 (cos(2 angle) - 1) + sin(angle))^2' @ ns, geometry=ns.x, degree=6)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  ns.strain_ij = '.5 (u_i,j + u_j,i)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  energy = domain.integral('energy' @ ns, geometry=ns.x, degree=7)
  lhs0 = solver.optimize('lhs', energy, constrain=cons)
  if plots:
    makeplots('linear', domain, ns | dict(lhs=lhs0))

  ns.strain_ij = '.5 (u_i,j + u_j,i + u_k,i u_k,j)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  energy = domain.integral('energy' @ ns, geometry=ns.x, degree=7)
  lhs1 = solver.optimize('lhs', energy, lhs0=lhs0, constrain=cons, newtontol=restol)
  if plots:
    makeplots('nonlinear', domain, ns | dict(lhs=lhs1))

  return lhs0, lhs1


class test(unittest.TestCase):

  def test(self):
    retvals = main(nelems=4, angle=10, plots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqtU1tu5DAMu84UiAu9Hwea+1+htuj5aT93gQBiHEmWSIaflz3sX8/rRd/Pr2dxl75XPJxZE1npvfwR
      bjtRSfJE55Pnu4DdDwjxPAVaMQlJJicyE7/XvlG5T1ycXPNBfRIiT8J+F0VikZ0oLIGC0LmaO6awU1CQ
      mpPop+GOTYICP1ftmdMDI9zOXdNRyuZdJabwLw2aPts7y0RNxMVGDSA60y8VchAxXQ+IxomIBXI0hsTl
      ngCRl77IRp8oAUi1IXBV1m1IJ/ksJtj0jMEAWnRP4n4KNoBiB2i/IDU+ny5THegsmyMACZSL2nTePvnL
      jxBc4m4+LjGILtazsTLP6FYCBSr9uipwnkNRpg0Ne8MZiMXnduujXB0Fp0+Roo9eJbMNijZMVZSjizLs
      yQQ6tpkIHsGikpb/ySNm+EOMdIy/JM4MtRUsuybJhKbGV2WOj0nSGib5qGzW8EY6AfRocbxBDEuECqoy
      r0m6Eid7/iuuXU72/3O9oXKXtrpOiPyYpPMawPhfTPL1Ayv17i8='''))


if __name__ == '__main__':
  cli.run(main)
