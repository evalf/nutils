#! /usr/bin/env python3
#
# In this script we solve the nonlinear Saint Venant-Kichhoff problem on a unit
# square domain (optionally with a circular cutout), clamped at both the left
# and right boundary in such a way that an arc is formed over a specified
# angle. The configuration is constructed such that a symmetric solution is
# expected.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), type of basis function (std or
# spline, with availability depending on element type), polynomial degree,
# Poisson's ratio, wedge angle, Newton tolerance, and a boolean flag for a
# circular cutout.

def main(nelems: 'number of elements along edge' = 10,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (std/spline)' = 'std',
         degree: 'polynomial degree' = 1,
         poisson: 'poisson ratio < 0.5' = .25,
         angle: 'bend angle (degrees)' = 20,
         restol: 'residual tolerance' = 1e-10,
         trim: 'create circular-shaped hole' = False):

  domain, geom = nutils.mesh.unitsquare(nelems, etype)
  if trim:
    domain = domain.trim(nutils.function.norm2(geom-.5)-.2, maxrefine=2)
  bezier = domain.sample('bezier', 5)

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.angle = angle * numpy.pi / 180
  ns.lmbda = 2 * poisson
  ns.mu = 1 - 2 * poisson
  ns.ubasis = domain.basis(btype, degree=degree).vector(2)
  ns.u_i = 'ubasis_ki ?lhs_k'
  ns.X_i = 'x_i + u_i'
  ns.strain_ij = '.5 (u_i,j + u_j,i)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  sqr = domain.boundary['left'].integral('u_k u_k' @ ns, geometry=ns.x, degree=degree*2)
  sqr += domain.boundary['right'].integral('(u_0 - x_1 sin(2 angle) - cos(angle) + 1)^2 + (u_1 - x_1 (cos(2 angle) - 1) + sin(angle))^2' @ ns, geometry=ns.x, degree=degree*2)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  energy = domain.integral('energy' @ ns, geometry=ns.x, degree=degree*2)
  lhs0 = nutils.solver.optimize('lhs', energy, constrain=cons)
  X, energy = bezier.eval([ns.X, ns.energy], arguments=dict(lhs=lhs0))
  nutils.export.triplot('linear.jpg', X, energy, tri=bezier.tri, hull=bezier.hull)

  ns.strain_ij = '.5 (u_i,j + u_j,i + u_k,i u_k,j)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  energy = domain.integral('energy' @ ns, geometry=ns.x, degree=degree*2)
  lhs1 = nutils.solver.optimize('lhs', energy, lhs0=lhs0, constrain=cons, newtontol=restol)
  X, energy = bezier.eval([ns.X, ns.energy], arguments=dict(lhs=lhs1))
  nutils.export.triplot('nonlinear.jpg', X, energy, tri=bezier.tri, hull=bezier.hull)

  return lhs0, lhs1

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# finitestrain.py`. To select quadratic splines and a cutout add :sh:`python3
# finitestrain.py btype=spline degree=2 trim`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_default(self):
    lhs0, lhs1 = main(nelems=4, angle=10)
    nutils.numeric.assert_allclose64(lhs0, 'eNpjYICB8ku8+icMthvOM+K42G1ga6Rv/Mh42'
      'YVcQwnj/8bzTW5fUDbaaNxtomwK18CQfCnxkuPFL+f7zt06d/Rc1rnbZ73Pyp4VPvvwzOwz7mc'
      'kz3w4ffL0stMtpwGSOirA')
    nutils.numeric.assert_allclose64(lhs1, 'eNpjYICBMu1b+jKGFw2bjdy1LICkk/Fx4+bLj'
      'wxdjAVM2k1uX1A22mjcbaJsCtfAoHz53sXiC27nGc6pnD94Tutc5dlLZyLOSpw9fab4DOsZyTM'
      'fTp88vex0y2kA6e4nVQ==')

  def test_mixed(self):
    lhs0, lhs1 = main(nelems=4, angle=10, etype='mixed')
    nutils.numeric.assert_allclose64(lhs0, 'eNoBZACb/wAAAADV0WwvAAChMAAAtjEAAKgyX'
      'jBl0UUyMjPeMyXQjzESM/ozqDQjMtvQsTOLNCM1AAAAAIfS7NEAAM/RAADQzwAAmc7czsvOU87'
      '1zUrNMs0NzenMk8xQzPDLGczJy6bLhMsZ2Sx5')
    nutils.numeric.assert_allclose64(lhs1, 'eNoBZACb/wAAAAAr3xowAAD9MAAA1DEAAI8yH'
      'zGKLIEySDPKM6fS9TFCMwM0mzQjMtvQsTOLNCM1AAAAAD/TYNEAAN7QAAA3zwAACc7SzgnPEc6'
      'TzdjMZ80TzdHMa8wXzPDLGczJy6bLhMthnih2')

  def test_spline(self):
    lhs0, lhs1 = main(nelems=4, angle=10, degree=2, btype='spline')
    nutils.numeric.assert_allclose64(lhs0, 'eNpjYECAa1e+aE3Qu6Nfa9BlmHoxU/eHgbIRs'
      '3Gs8bwLr/S4jayNfxn7mGy/sEz/qNFz4wUmL0xuX/AzEDDWMrlromyKZAxDlg6bbppOw1WXi2n'
      'nqy8svSBxwf980Ln3Z9+ffXP2+Nm8s6xnT59pOdNzJveM3RnmM/dOS55hOXPn9PbTU0+3nAYAZ'
      'eQ9sA==')
    nutils.numeric.assert_allclose64(lhs1, 'eNpjYEAAZ21dXWF9WYNug3RDPu1i/RzDYKNfR'
      'i7Gn5V9DVKNkoy/G+uaiF/qM/hi9NN4pckZk9sX/AwEjLVM7poomyIZw3BIp0/H/a7qpf4LD85'
      'tvTD1wtrz+87tPRt8Vvuc0Lm1Z43PLjmTfGbXmQVn0s/onHl7euNpyTMsZ+6c3n566umW0wB4s'
      'Dra')
