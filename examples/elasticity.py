#! /usr/bin/env python3
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the left boundary, and stretched at the right boundary
# while keeping vertical displacements free.

from nutils import mesh, function, solver, export, cli, testing

def main(nelems:int, etype:str, btype:str, degree:int, poisson:float):
  '''
  Horizontally loaded linear elastic plate.

  .. arguments::

     nelems [10]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), with availability depending on the
       configured element type.
     degree [1]
       Polynomial degree.
     poisson [.25]
       Poisson's ratio, nonnegative and strictly smaller than 1/2.
  '''

  domain, geom = mesh.unitsquare(nelems, etype)

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree).vector(2)
  ns.u_i = 'basis_ni ?lhs_n'
  ns.X_i = 'x_i + u_i'
  ns.lmbda = 2 * poisson
  ns.mu = 1 - 2 * poisson
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'

  sqr = domain.boundary['left'].integral('u_k u_k d:x' @ ns, degree=degree*2)
  sqr += domain.boundary['right'].integral('(u_0 - .5)^2 d:x' @ ns, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  res = domain.integral('basis_ni,j stress_ij d:x' @ ns, degree=degree*2)
  lhs = solver.solve_linear('lhs', res, constrain=cons)

  bezier = domain.sample('bezier', 5)
  X, sxy = bezier.eval(['X_i', 'stress_01'] @ ns, lhs=lhs)
  export.triplot('shear.png', X, sxy, tri=bezier.tri, hull=bezier.hull)

  return cons, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# elasticity.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 elasticity.py etype=mixed degree=2`.

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
    cons, lhs = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYMACGsiHP0wxMQBKlBdi''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYMAEKcaiRmLGQQZCxgwMYsbrzqcYvz672KTMaIKJimG7CQPDBJM75xabdJ3NMO0xSjG1MUw0Beox
      PXIuw7Tk7A/TXqMfQLEfQLEfQLEfpsVnAUzzHtI=''')

  @testing.requires('matplotlib')
  def test_mixed(self):
    cons, lhs = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYICCBiiEsdFpIuEPU0wMAG6UF2I=''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYICAJGMOI3ljcQMwx3i/JohSMr51HkQnGP8422eiYrjcJM+o3aToWq/Jy3PLTKafzTDtM0oxtTRM
      MF2okmJ67lyGacnZH6aOhj9Mu41+mMZq/DA9dO6HaflZAAMdIls=''')

  @testing.requires('matplotlib')
  def test_quadratic(self):
    cons, lhs = main(nelems=4, etype='square', btype='std', degree=2, poisson=.25)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYCACNIxc+MOUMAYA/+NOFg==''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNqFzL9KA0EQx/HlLI5wprBJCol/rtfN7MxobZEXOQIJQdBCwfgAItwVStQmZSAvcOmtVW6z5wP4D2yE
      aKOwEhTnDRz4VvPhp9T/1zeP0ILF5hhSnUK5cQlKpaDvx3DoWvA57Zt128PIMO5CjHvNOn5s1lCpOi6V
      MZ5PGS/k/1U0qGcqVMIcQ5jhmX4XM8N9N8dvWyFtG3RVjOjADOkNBrQMGV3rlJTKaMcN6NUOqWZHlBVV
      PjER/0DIDAE/6ICVCjh2Id/ZiBdslY+LrpiOmLaYhJ90IibhNdcW0xHTFTPhUzPhX8h5W3rRuZicV1zO
      N3bCgXRUeDFedjxvSc/ai/G86jzfWi87Xswfg5Nx3Q==''')

  @testing.requires('matplotlib')
  def test_poisson(self):
    cons, lhs = main(nelems=4, etype='square', btype='std', degree=1, poisson=.4)
    with self.subTest('constraints'): self.assertAlmostEqual64(cons, '''
      eNpjYMACGsiHP0wxMQBKlBdi''')
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNpjYMAEFsaTjdcYvTFcasTAsMZI5JyFce6ZKSavjbNMFhhFmPz/n2WScHaKieiZRFMmk3DTrUaBpv//
      h5t6n000/Xf6hymLyQ/TbUY/gGI/TL3O/jD9cxoASiglXw==''')
