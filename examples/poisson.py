from nutils import mesh, function, solver, export, testing

# This script demonstrates direct function manipulation without the aid of
# namespace expressions.


def main(nelems: int = 32):
    '''Poisson's equation on a unit square

    Solves Poisson's equation `Δu = 1` subject to boundary constraints, using
    the fact that the solution to the strong form minimizes the functional `∫
    .5 ‖∇u‖² - u`. The domain is a unit square, and the solution is constrained
    to zero along the entire boundary.

    Parameters
    ----------
    nelems
        Number of elements along edge.
    '''

    topo, x = mesh.unitsquare(nelems, etype='square')
    u = function.dotarg('u', topo.basis('std', degree=1))
    g = u.grad(x)
    J = function.J(x)

    sqr = topo.boundary.integral(u**2 * J, degree=2)
    cons = solver.optimize(('u',), sqr, droptol=1e-12)

    energy = topo.integral((g @ g / 2 - u) * J, degree=1)
    args = solver.optimize(('u',), energy, constrain=cons)

    bezier = topo.sample('bezier', 3)
    x, u = bezier.eval([x, u], **args)
    export.triplot('u.png', x, u, tri=bezier.tri, cmap='jet')

    return args


class test(testing.TestCase):

    def test_simple(self):
        args = main(nelems=10)
        self.assertAlmostEqual64(args['u'], '''
            eNp9zrENwCAMBEBGYQJ444o2ozAAYgFmYhLEFqxAmye1FUtf+PSy7Jw9J6yoKGiMYsUTrq44kaVKZ7JM
            +lWlDdlymEFXXC2o3H1C8mmzXz5t6OwhPfTDO+2na9+1f7D/teYFdsk5vQ==''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Poisson's equation
