from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import treelog

# This is a heavily commented example that strives to introduce the main
# building blocks of a typical Nutils script.


def main(nelems: int = 10,
         etype: str = 'square',
         btype: str = 'std',
         degree: int = 1):

    '''Laplace problem on a unit square
    
    Solves Laplace's equation `Δu = 0` on a unit square domain `Ω` with
    boundary `Γ`, subject to boundary conditions:
    have a known exact solution in `uexact = sin(x) cosh(y)`.

    Parameters
    ----------
    nelems
        Number of elements along edge.
    etype
        Type of elements (square/triangle/mixed).
    btype
        Type of basis function (std/spline), availability depending on the
        selected element type.
    degree
        Polynomial degree.
    '''

    # A unit square domain is created by calling the nutils.mesh.unitsquare
    # mesh generator, with the number of elements along an edge as the first
    # argument, and the type of elements ("square", "triangle", or "mixed") as
    # the second. The result is a topology object `domain` and a vectored
    # valued geometry function `geom`.

    domain, geom = mesh.unitsquare(nelems, etype)

    # To be able to write index based tensor contractions, we need to bundle
    # all relevant functions together in a namespace. Here we add the geometry
    # `x`, a test function `v`, and the solution `u`. The latter two are formed
    # by contracting a basis with function arguments of the same name.

    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

    basis = domain.basis(btype, degree=degree)
    ns.add_field(('u', 'v'), basis)

    mask = domain.boundary['left,top'].integrate(basis, degree=1).astype(bool)
    ns.add_field(('p', 'q'), basis[mask])

    # We are now ready to implement the Laplace equation. In weak form, the
    # solution is a scalar field `u` for which ∫_Ω ∇v·∇u - ∫_Γ v n·∇u = 0 ∀ v.

    res = domain.integral('∇_i(v) ∇_i(u) dV' @ ns, degree=degree*2)

    # Substitute n·∇u for the Neumann boundaries

    res -= domain.boundary['right'].integral('v cos(1) cosh(x_1) dS' @ ns, degree=degree*2)

    # Substitute n·∇u = -p for the Dirichlet boundaries, where the new unknown
    # p serves as Lagrange multiplier, paired with a new test function q that
    # weakly imposes the constraints.

    res += domain.boundary['left'].integral('(q u + v p) dS' @ ns, degree=degree*2)
    res += domain.boundary['top'].integral('(q (u - cosh(1) sin(x_0)) + v p) dS' @ ns, degree=degree*2)

    # Solve the linear system for u and p

    args = solver.solve_linear('u:v,p:q', res)

    # Once all arguments are establised, the corresponding solution can be
    # vizualised by sampling values of `ns.u` along with physical coordinates
    # `ns.x`, with the solution vector provided via keyword arguments. The
    # sample members `tri` and `hull` provide additional inter-point
    # information required for drawing the mesh and element outlines.

    bezier = domain.sample('bezier', 9)
    xsmp, usmp = bezier.eval(['x_i', 'u'] @ ns, **args)
    export.triplot('solution.png', xsmp, usmp, tri=bezier.tri, hull=bezier.hull)

    # To confirm that our computation is correct, we use our knowledge of the
    # analytical solution to evaluate the L2-error of the discrete result.

    err = domain.integral('(u - sin(x_0) cosh(x_1))^2 dV' @ ns, degree=degree*2).eval(**args)**.5
    treelog.user('L2 error: {:.2e}'.format(err))

    return args['u'], err


# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

    def test_simple(self):
        u, err = main(nelems=4)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNoBMgDN/7Ed9eB+IfLboCaXNKc01DQaNXM14jXyNR82ZTa+NpI2oTbPNhU3bjf7Ngo3ODd+N9c3SNEU
                1g==''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 1.63e-3, places=5)

    def test_spline(self):
        u, err = main(nelems=4, btype='spline', degree=2)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNqrkmN+sEfhzF0xleRbrsauxsnGc43fGMuZJJgmmNaZ7jBlN7M08wLCDLNFZh/NlM0vmV0y+2CmZV5p
                vtr8j9kfMynzEPPF5lfNAcuhGvs=''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 8.04e-5, places=7)

    def test_mixed(self):
        u, err = main(nelems=4, etype='mixed', degree=2)
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(u, '''
                eNorfLZF2ueJq7GrcYjxDJPpJstNbsq9fOBr3Gh8xWS7iYdSxd19xseMP5hImu5UZbv1xljOxM600DTW
                NN/0k2mC6SPTx6Z1pnNMGc3kzdaaPjRNMbMyEzWzNOsy223mBYRRZpPNJpktMks1azM7Z7bRbIXZabNX
                ZiLmH82UzS3Ns80vmj004za/ZPYHCD+Y8ZlLmVuYq5kHm9eahwDxavPF5lfNAWFyPdk=''')
        with self.subTest('L2-error'):
            self.assertAlmostEqual(err, 1.25e-4, places=6)


# If the script is executed (as opposed to imported), `nutils.cli.run` calls
# the main function with arguments provided from the command line. For example,
# to keep with the default arguments simply run `python3 laplace.py`. To select
# mixed elements and quadratic basis functions add `python3 laplace.py
# etype=mixed degree=2`.

if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Laplace
