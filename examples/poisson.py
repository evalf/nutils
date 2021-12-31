#! /usr/bin/env python3
#
# In this script we solve Poisson's equation Δu = 1 subject to boundary
# constraints, using the fact that the solution to the strong form minimizes
# the functional ∫ .5 ‖∇u‖² - u. The domain is a unit square, and the solution
# is constrained to zero along the entire boundary.

from nutils import mesh, function, solver, export, cli, testing

def main(nelems: int):
  '''
  Poisson's equation on a unit square.

  .. arguments::

     nelems [10]
       Number of elements along edge.
  '''

  domain, x = mesh.unitsquare(nelems, etype='square')
  u = function.dotarg('udofs', domain.basis('std', degree=1))
  g = u.grad(x)
  J = function.J(x)
  cons = solver.optimize('udofs',
    domain.boundary.integral(u**2 * J, degree=2), droptol=1e-12)
  udofs = solver.optimize('udofs',
    domain.integral((g @ g / 2 - u) * J, degree=1), constrain=cons)
  bezier = domain.sample('bezier', 3)
  x, u = bezier.eval([x, u], udofs=udofs)
  export.triplot('u.png', x, u, tri=bezier.tri, hull=bezier.hull)

  return udofs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. To
# keep with the default arguments simply run :sh:`python3 poisson.py`.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  def test_default(self):
    udofs = main(nelems=10)
    self.assertAlmostEqual64(udofs, '''
      eNp9zrENwCAMBEBGYQJ444o2ozAAYgFmYhLEFqxAmye1FUtf+PSy7Jw9J6yoKGiMYsUTrq44kaVKZ7JM
      +lWlDdlymEFXXC2o3H1C8mmzXz5t6OwhPfTDO+2na9+1f7D/teYFdsk5vQ==''')
