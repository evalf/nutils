from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy


def main(nelems: int = 24,
         etype: str = 'triangle',
         btype: str = 'std',
         degree: int = 2,
         poisson: float = .3,
         energy: bool = False):

    '''Plane strain plate under gravitational pull
    
    Solves the linear elasticity problem on a unit square domain, clamped at
    the top boundary, and stretched under the influence of a vertical
    distributed load.

    Parameters
    ----------
    nelems
        Number of elements along edge.
    etype
        Type of elements (square/triangle/mixed).
    btype
        Type of basis function (std/spline), with availability depending on the
        configured element type.
    degree
        Polynomial degree.
    poisson
        Poisson's ratio, nonnegative and strictly smaller than 1/2.
    energy
        Use energy formulation.
    '''

    domain, geom = mesh.unitsquare(nelems, etype)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

    basis = domain.basis(btype, degree=degree)
    ns.add_field(('u', 'v'), basis, shape=(2,))

    mask = domain.boundary['top'].integrate(basis, degree=1).astype(bool)
    ns.add_field(('t', 's'), basis[mask], shape=(2,))

    ns.X_i = 'x_i + u_i'
    ns.λ = 1
    ns.μ = .5/poisson - 1
    ns.ε_ij = '.5 (∇_i(u_j) + ∇_j(u_i))'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.q_i = '-.5 δ_i1'
    ns.E = '.5 ε_ij σ_ij'

    if energy:
        nrg = domain.integral('(E - u_i q_i) dV' @ ns, degree=degree*2)
        nrg -= domain.boundary['top'].integral('u_k t_k dS' @ ns, degree=degree*2)
        args = solver.optimize('u,t', nrg)
    else:
        res = domain.integral('(∇_j(v_i) σ_ij - v_i q_i) dV' @ ns, degree=degree*2)
        res -= domain.boundary['top'].integral('(s_k u_k + v_k t_k) dS' @ ns, degree=degree*2)
        args = solver.solve_linear('u:v,t:s', res)

    F = domain.boundary['top'].integrate('t_i dS' @ ns, degree=degree*2, arguments=args)
    log.user('total clamping force:', F)

    # visualize solution
    bezier = domain.sample('bezier', 3)
    X, E = bezier.eval(['X_i', 'E'] @ ns, **args)
    Xt, t = domain.boundary['top'].sample('bezier', 2).eval(['X_i', 't_i'] @ ns, **args)
    with export.mplfigure('energy.png') as fig:
        ax = fig.add_subplot(111, ylim=(-.2,1), aspect='equal')
        im = ax.tripcolor(*X.T, bezier.tri, E, shading='gouraud', rasterized=True, cmap='turbo')
        export.plotlines_(ax, X.T, bezier.hull, colors='k', linewidths=.1, alpha=.5)
        ax.quiver(*Xt.T, *t.T, clip_on=False)
        fig.colorbar(im)

    return args


class test(testing.TestCase):

    def test_simple(self):
        args = main(nelems=4, etype='square', degree=1, poisson=.25)
        with self.subTest('displacement'):
            self.assertAlmostEqual64(args['u'], '''
                eNqT1yk8K6o35ay2PsO5ev3v5xiA4ItW1NlnOrVnX+l+PrtZV+Y8AxiEnGVgqATir0AsARbjuRp1Vupy
                7VmxS5/P+l6CqHt4ufDs64tTzl69wHCu8QLEPADBQyml''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNpLOh1ntl5vlykDw0/TwIu7TOeZxJkBAFi/CDM=''')

    def test_mixed(self):
        args = main(nelems=4, etype='mixed', degree=1, poisson=.25)
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNoz1c0466vXfrZeJ+ystm7TWVl9lnPPdd+erdf/fG66rvR5Bijg0Ko4e0or+uwjjT9nHTVEweKOt2PO
                rrrcdjbj0uezTpdkwGK2l6afnXW14SznRZ5z+y5wgcUAJX0p8A==''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNqbfzrJ7N25zaacRpxmwmdnmuablpsBAGCbCCA=''')

    def test_quadratic(self):
        args = main(nelems=4, etype='square', degree=2, poisson=.25)
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNolkL1KA0EUhcdI+sQnEFPtZma3tlBEQSwCJmskLFisvoAKu40hEAjaqkWaCKaIYCGSOhYiWlidmZ2Z
                /FQRXyBuirSim+yFr7jwcTn3MNPHkPr4ZU28WD0Ydorf2BV+bs/4o/0sSDzbpocT6qHB6ti0OniyIvxZ
                BT620uKL3YVz59tw0c67+KBV1FgbYzbFulXkI5YRJbazcLKGgyPTQTUfwKYtXNEJ3miZl2hW3Ob3Fw4h
                Dgg5iAliWjE/850TshJn2Vs40dDBw8DBZT+A1C1c6Am2dJl3dVac9pM7q0MXx30XG7qKhmojp6b4lEVu
                qIzoqiTP68DDvfZQU3W8yw4OZYQlWeA5mRZrKvlLD3yY2seyamJX9jAKU/wsrPDrcMbdMOnnH2mlmk4=''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNp7fOqM2STjJSZFZ7nNzPR3mzIwfDU9dWG3aZ8Rt1nemSUmsqZnzAAr4A89''')

    def test_poisson(self):
        args = main(nelems=4, etype='square', degree=1, poisson=.4)
        with self.subTest('solution'):
            self.assertAlmostEqual64(args['u'], '''
                eNqTNig6vcVwwekjRuJn5Iy1zzIAwQs999MdBmWn+w0Zz7QYpoPFGBisTzMw5AMx6xkGhniwmMRF99MV
                58tOF55jPFNzDqLu6fmi0z7nFpy2OSt+5tEZiHkAKRAl5A==''')
        with self.subTest('traction'):
            self.assertAlmostEqual64(args['t'], '''
                eNp7eXK6WZ1hhykDA7NZ07kOU3Gz6WYAVpgHTA==''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=elasticity
