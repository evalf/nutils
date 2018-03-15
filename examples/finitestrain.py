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
    figures: 'create figures' = True,
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
  if figures:
    makeplots('linear', domain, ns(lhs=lhs0))

  ns.strain_ij = '.5 (u_i,j + u_j,i + u_k,i u_k,j)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  energy = domain.integral('energy' @ ns, geometry=ns.x, degree=7)
  lhs1 = solver.optimize('lhs', energy, lhs0=lhs0, constrain=cons, newtontol=restol)
  if figures:
    makeplots('nonlinear', domain, ns(lhs=lhs1))

  return lhs0, lhs1


class test(unittest.TestCase):

  def test(self):
    lhs0, lhs1 = main(nelems=4, angle=10, figures=False)
    numeric.assert_allclose64(lhs0, 'eNpjYECAa1e+aE3Qu6Nfa9BlmHoxU/eHgbIRs3Gs8bwLr/S4ja'
      'yNfxn7mGy/sEz/qNFz4wUmL0xuX/AzEDDWMrlromyKZAxDlg6bbppOw1WXi2nnqy8svSBxwf980Ln3Z9'
      '+ffXP2+Nm8s6xnT59pOdNzJveM3RnmM/dOS55hOXPn9PbTU0+3nAYAZeQ9sA==')
    numeric.assert_allclose64(lhs1, 'eNpjYEAAZ21dXWF9WYNug3RDPu1i/RzDYKNfRi7G35R9DVKNko'
      'y/G+uaiF/qM/hi9NN4pckZk9sX/AwEjLVM7poomyIZw3BIp0/H7q7apb4LD85tvTD1wtrz+87tPRt8Vv'
      'uc0Lm1Z43PLjmTfGbXmQVn0s/onHl7euNpyTMsZ+6c3n566umW0wB3/jrU')


if __name__ == '__main__':
  cli.run(main)
