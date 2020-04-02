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
      eNpjYMAE5ZeSL/HqJ146YeB4cbvhl/PzjPrOcVy8da7b4Og5W6Osc/rGt88+MvY+u+yC7NlcQ+GzEsYP
      z/w3nn1mvon7mdsXJM8oG304vdH45Oluk2WnlU1bTgMAv04qwA==''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNpjYMAEZdrKl2/p37soY1h84aKh2/lmI4Zz7loq5y0MD55rNtI652Rcefa48aUzzZcjzj4ylDjrYnz6
      jIBJ8Zl2E9Yzty9InlE2+nB6o/HJ090my04rm7acBgAKcSdV''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    lhs0, lhs1 = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25, angle=10, restol=1e-10, trim=False)
    with self.subTest('linear'): self.assertAlmostEqual64(lhs0, '''
      eNpjYICAqxfbL+Xov7kIYi80OA+mtxleOA+iVxjNPBdncOdc6sXT51yNgs8ZGX89e8/Y66zqBaOz/Ya8
      Z4WMX575ZTz5zAqTgDPKRh9O374geWaj8cnT3SbLTiubtpwGAJ6hLHk=''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNpjYIAA7fv2l6UMEi6C2H8N7l0A0VcMzc+D6H4jznPyhpfOdelwnm80EjznYTz57CnjG2eWX0o/+9VQ
      +KyT8cUzzCbZZ2abiJ9RNvpw+vYFyTMbjU+e7jZZdlrZtOU0AJN4KHY=''')

  @testing.requires('matplotlib')
  def test_spline(self):
    lhs0, lhs1 = main(nelems=4, etype='square', btype='spline', degree=2, poisson=.25, angle=10, restol=1e-10, trim=False)
    with self.subTest('linear'): self.assertAlmostEqual64(lhs0, '''
      eNpjYMAOrl3J0vmixaY7QS9N545+w9VaA5eLXYZp51MvVl/I1F164YeBxAVlI//zzMZB52KN35+dd+H9
      2Vd6b85yGx0/a22cd/aXMetZH5PTZ7ZfaDmzTL/nzFGj3DPPje3OLDBhPvPC5N7p2xckz/gZsJwRML5z
      Wstk++m7JlNPK5u2nAYATqg9sA==''')
    with self.subTest('non-linear'): self.assertAlmostEqual64(lhs1, '''
      eNpjYMAOnLUP6ejq9ukI67vflTVQvdRt0H8h3fDBOT7trReK9adeyDFcez7YaN+5X0Z7z7oYB5/9rKx9
      ztdA6Fyq0dqzScbGZ78bLzmja5J8RvzSrjN9BgvOfDFKP/PTWOfMSpO3p8+YbDx9+4LkGT8DljMCxndO
      a5lsP33XZOppZdOW0wApLzra''')
