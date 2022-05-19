#! /usr/bin/env python3
#
# In this script we solve the linear elasticity problem on a unit square
# domain, clamped at the left boundary, and stretched at the right boundary
# while keeping vertical displacements free.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace


def main(nelems: int, etype: str, btype: str, degree: int, poisson: float):
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

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.X = geom
    ns.define_for('X', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('u', 'v'), domain.basis(btype, degree=degree), shape=(2,))
    ns.x_i = 'X_i + u_i'
    ns.λ = 2 * poisson
    ns.μ = 1 - 2 * poisson
    ns.ε_ij = '(∇_j(u_i) + ∇_i(u_j)) / 2'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'

    sqr = domain.boundary['left'].integral('u_k u_k dS' @ ns, degree=degree*2)
    sqr += domain.boundary['right'].integral('(u_0 - .5)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15)

    res = domain.integral('∇_j(v_i) σ_ij dV' @ ns, degree=degree*2)
    args = solver.solve_linear('u:v', res, constrain=cons)

    bezier = domain.sample('bezier', 5)
    x, sxy = bezier.eval(['x_i', 'σ_01'] @ ns, **args)
    export.triplot('shear.png', x, sxy, tri=bezier.tri, hull=bezier.hull)

    return cons['u'], args['u']

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

    def test_default(self):
        cons, u = main(nelems=4, etype='square', btype='std', degree=1, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYMACGsiHP0wxMQBKlBdi''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpjYMAEKcaiRmLGQQZCxgwMYsbrzqcYvz672KTMaIKJimG7CQPDBJM75xabdJ3NMO0xSjG1MUw0Beox
                PXIuw7Tk7A/TXqMfQLEfQLEfQLEfpsVnAUzzHtI=''')

    def test_mixed(self):
        cons, u = main(nelems=4, etype='mixed', btype='std', degree=1, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYICCBiiEsdFpIuEPU0wMAG6UF2I=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpjYICAJGMOI3ljcQMwx3i/JohSMr51HkQnGP8422eiYrjcJM+o3aToWq/Jy3PLTKafzTDtM0oxtTRM
                MF2okmJ67lyGacnZH6aOhj9Mu41+mMZq/DA9dO6HaflZAAMdIls=''')

    def test_quadratic(self):
        cons, u = main(nelems=4, etype='square', btype='std', degree=2, poisson=.25)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYCACNIxc+MOUMAYA/+NOFg==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNqFzL9KA0EQx/HlLI5wprBJCol/rtfN7MxobZEXOQIJQdBCwfgAItwVStQmZSAvcOmtVW6z5wP4D2yE
                aKOwEhTnDRz4VvPhp9T/1zeP0ILF5hhSnUK5cQlKpaDvx3DoWvA57Zt128PIMO5CjHvNOn5s1lCpOi6V
                MZ5PGS/k/1U0qGcqVMIcQ5jhmX4XM8N9N8dvWyFtG3RVjOjADOkNBrQMGV3rlJTKaMcN6NUOqWZHlBVV
                PjER/0DIDAE/6ICVCjh2Id/ZiBdslY+LrpiOmLaYhJ90IibhNdcW0xHTFTPhUzPhX8h5W3rRuZicV1zO
                N3bCgXRUeDFedjxvSc/ai/G86jzfWi87Xswfg5Nx3Q==''')

    def test_poisson(self):
        cons, u = main(nelems=4, etype='square', btype='std', degree=1, poisson=.4)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons, '''
                eNpjYMACGsiHP0wxMQBKlBdi''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNpjYMAEFsaTjdcYvTFcasTAsMZI5JyFce6ZKSavjbNMFhhFmDAwZJkknJ1iInom0ZTJJNx0q1GgKQND
                uKn32UTTf6d/mLKY/DDdZvQDKPbD1OvsD9M/pwGZyh9l''')
