#! /usr/bin/env python3
#
# In this script we solve the nonlinear Saint Venant-Kichhoff problem on a unit
# square domain (optionally with a circular cutout), clamped at both the left
# and right boundary in such a way that an arc is formed over a specified
# angle. The configuration is constructed such that a symmetric solution is
# expected.

from nutils import mesh, function, solver, export, cli, testing
import numpy

def main(nelems:int, etype:str, btype:str, degree:int, poisson:float, angle:float, restol:float, trim:bool):
  '''
  Deformed hyperelastic plate.

  .. arguments::

     nelems [10]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline).
     degree [1]
       Polynomial degree.
     poisson [.25]
       Poisson's ratio, nonnegative and stricly smaller than 1/2.
     angle [20]
       Rotation angle for right clamp (degrees).
     restol [1e-10]
       Newton tolerance.
     trim [no]
       Create circular-shaped hole.
  '''

  domain, geom = mesh.unitsquare(nelems, etype)
  if trim:
    domain = domain.trim(function.norm2(geom-.5)-.2, maxrefine=2)
  bezier = domain.sample('bezier', 5)

  ns = function.Namespace()
  ns.x = geom
  ns.angle = angle * numpy.pi / 180
  ns.lmbda = 2 * poisson
  ns.mu = 1 - 2 * poisson
  ns.ubasis = domain.basis(btype, degree=degree).vector(2)
  ns.u_i = 'ubasis_ki ?lhs_k'
  ns.X_i = 'x_i + u_i'
  ns.strain_ij = '.5 (u_i,j + u_j,i)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  sqr = domain.boundary['left'].integral('u_k u_k d:x' @ ns, degree=degree*2)
  sqr += domain.boundary['right'].integral('((u_0 - x_1 sin(2 angle) - cos(angle) + 1)^2 + (u_1 - x_1 (cos(2 angle) - 1) + sin(angle))^2) d:x' @ ns, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  energy = domain.integral('energy d:x' @ ns, degree=degree*2)
  lhs0 = solver.optimize('lhs', energy, constrain=cons)
  X, energy = bezier.eval(['X_i', 'energy'] @ ns, lhs=lhs0)
  export.triplot('linear.png', X, energy, tri=bezier.tri, hull=bezier.hull)

  ns.strain_ij = '.5 (u_i,j + u_j,i + u_k,i u_k,j)'
  ns.energy = 'lmbda strain_ii strain_jj + 2 mu strain_ij strain_ij'

  energy = domain.integral('energy d:x' @ ns, degree=degree*2)
  lhs1 = solver.minimize('lhs', energy, lhs0=lhs0, constrain=cons).solve(restol)
  X, energy = bezier.eval(['X_i', 'energy'] @ ns, lhs=lhs1)
  export.triplot('nonlinear.png', X, energy, tri=bezier.tri, hull=bezier.hull)

  return lhs0, lhs1

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# finitestrain.py`. To select quadratic splines and a cutout add :sh:`python3
# finitestrain.py btype=spline degree=2 trim=yes`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  @testing.requires('matplotlib')
  def test_default(self):
    lhs0, lhs1 = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25, angle=10, restol=1e-10, trim=False)
    with self.subTest('linear'): self.assertAlmostEqual64(lhs0, '''
      eNpjYICB8ku8+icMthvOM+K42G1ga6Rv/Mh42YVcQwnj/8bzTW5fUDbaaNxtomwK18CQfCnxkuPFL+f7
      zt06d/Rc1rnbZ73Pyp4VPvvwzOwz7mckz3w4ffL0stMtpwGSOirA''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNpjYICBMu1b+jKGFw2bjdy1LICkk/Fx4+bLjwxdjAVM2k1uX1A22mjcbaJsCtfAoHz53sXiC27nGc6p
      nD94Tutc5dlLZyLOSpw9fab4DOsZyTMfTp88vex0y2kA6e4nVQ==''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    lhs0, lhs1 = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25, angle=10, restol=1e-10, trim=False)
    with self.subTest('linear'): self.assertAlmostEqual64(lhs0, '''
      eNoBZACb/wAAAADV0WwvAAChMAAAtjEAAKgyXjBl0UUyMjPeMyXQjzESM/ozqDQjMtvQsTOLNCM1AAAA
      AIfS7NEAAM/RAADQzwAAmc7czsvOU871zUrNMs0NzenMk8xQzPDLGczJy6bLhMsZ2Sx5''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNoBZACb/wAAAAAr3xowAAD9MAAA1DEAAI8yHzGKLIEySDPKM6fS9TFCMwM0mzQjMtvQsTOLNCM1AAAA
      AD/TYNEAAN7QAAA3zwAACc7SzgnPEc6TzdjMZ80TzdHMa8wXzPDLGczJy6bLhMthnih2''')

  @testing.requires('matplotlib')
  def test_spline(self):
    lhs0, lhs1 = main(nelems=4, etype='square', btype='spline', degree=2, poisson=.25, angle=10, restol=1e-10, trim=False)
    with self.subTest('linear'): self.assertAlmostEqual64(lhs0, '''
      eNpjYECAa1e+aE3Qu6Nfa9BlmHoxU/eHgbIRs3Gs8bwLr/S4jayNfxn7mGy/sEz/qNFz4wUmL0xuX/Az
      EDDWMrlromyKZAxDlg6bbppOw1WXi2nnqy8svSBxwf980Ln3Z9+ffXP2+Nm8s6xnT59pOdNzJveM3Rnm
      M/dOS55hOXPn9PbTU0+3nAYAZeQ9sA==''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNpjYEAAZ21dXWF9WYNug3RDPu1i/RzDYKNfRi7Gn5V9DVKNkoy/G+uaiF/qM/hi9NN4pckZk9sX/AwE
      jLVM7poomyIZw3BIp0/H/a7qpf4LD85tvTD1wtrz+87tPRt8Vvuc0Lm1Z43PLjmTfGbXmQVn0s/onHl7
      euNpyTMsZ+6c3n566umW0wB4sDra''')
