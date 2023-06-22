from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy


def main(nelems: int = 32,
         etype: str = 'square',
         degree: int = 3,
         reynolds: float = 1000.,
         compatible: bool = False,
         strongbc: bool = False):

    '''Lid-driven cavity flow

    Solves the lid driven cavity problem for stationary Stokes and
    Navier-Stokes flow. This benchmark problem consists of a square domain with
    fixed left, bottom and right boundaries and a top boundary that is moving
    at unit velocity in positive x-direction. Reference results can be found
    for instance at https://www.acenumerics.com/the-benchmarks.html.

    The general conservation laws are:

        Dρ/Dt = 0 (mass)
        ρ Du_i/Dt = ∇_j(σ_ij) (momentum)

    where we used the material derivative D·/Dt := ∂·/∂t + ∇_j(· u_j). The stress
    tensor is σ_ij := μ (∇_j(u_i) + ∇_i(u_j) - 2 ∇_k(u_k) δ_ij / δ_nn) - p δ_ij,
    with pressure p and dynamic viscosity μ, and ρ is the density.

    Assuming steady, incompressible flow, we take density to be constant. Further
    introducing a reference length L and reference velocity U, we make the
    equations dimensionless by taking spatial coordinates relative to L, velocity
    relative to U, and pressure relative to ρ U^2. This reduces the conservation
    laws to:

        ∇_k(u_k) = 0 (mass)
        Du_i/Dt = ∇_j(σ_ij) (momentum)

    where the material derivative simplifies to D·/Dt := ∇_j(·) u_j, and the
    stress tensor becomes σ_ij := (∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij, with
    Reynolds number Re = ρ U L / μ.

    The weak form is obtained by multiplication of a test function and partial
    integration of the right hand side of the momentum balance.

        ∀ q: ∫_Ω q ∇_k(u_k) = 0 (mass)
        ∀ v: ∫_Ω (v_i Du_i/Dt + ∇_j(v_i) σ_ij) = ∫_Γ v_i σ_ij n_j (momentum)

    A remaining issue with this system is that its solution is not unique due to
    its strictly kinematic boundary conditions: since ∫_Ω ∇_k(u_k) = 0 for any u
    that satisfies the non-penetrating boundary conditions, any pressure space
    that contains unity results in linear dependence. Furthermore, the strong
    form shows that constant pressure shifts do not affect the momentum balance.
    Both issues are solved by arbitrarily removing one of the basis functions.

    The normal velocity components at the wall are strongly constrained. The
    tangential components are either strongly or weakly constrained depending on
    the `strongbc` parameter. Since the tangential components at the top are
    incompatible with the normal components at the left and right boundary, the
    constraints are constructed in two steps to avoid Gibbs type oscillations,
    and to make sure that the non-penetrating condition takes precedence.

    Depending on the `compatible` parameter, the script uses either a Taylor-Hood
    (False) or a Raviart-Thomas (True) discretization. In case of TH, the system
    is consistently modified by adding ∫_Ω .5 u_i v_i ∇_j(u_j) to yield the skew-
    symmetric advective term ∫_Ω .5 (v_i ∇_j(u_i) - u_i ∇_j(v_i)) u_j. In case of
    RT, the discretization guarantees a pointwise divergence-free velocity field,
    and skew symmetry is implied.

    Parameters
    ----------
    nelems
        Number of elements along edge.
    etype
        Element type (square/triangle/mixed).
    degree
        Polynomial degree for velocity; the pressure space is one degree less.
    reynolds
        Reynolds number, taking the domain size as characteristic length.
    strongbc
        Use strong boundary constraints
    compatible
        Use compatible spaces and weakly imposed boundary conditions; requires
        etype='square' and strongbc=False
    '''

    if compatible and (strongbc or etype != 'square'):
        raise Exception(f'compatible mode requires square elements and weak boundary conditions')

    domain, geom = mesh.unitsquare(nelems, etype)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.ε = function.levicivita(2)
    ns.Σ = function.ones([domain.ndims])
    ns.Re = reynolds
    ns.uwall = numpy.stack([domain.boundary.indicator('top'), 0])
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    if not compatible:
        ns.add_field(('u', 'v'), domain.basis('std', degree=degree), shape=(domain.ndims,))
        ns.add_field(('p', 'q'), domain.basis('std', degree=degree-1)[1:])
        ns.add_field('ψ', domain.basis('std', degree=2)[1:])
    else:
        ns.add_field(('u', 'v'), function.vectorize([
            domain.basis('spline', degree=(degree, degree-1)),
            domain.basis('spline', degree=(degree-1, degree))]))
        ns.add_field(('p', 'q'), domain.basis('spline', degree=degree-1)[1:])
        ns.add_field('ψ', domain.basis('spline', degree=degree)[1:])
    ns.σ_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'

    # weak formulation for Stokes flow, over-integrating for improved
    # efficiency when we turn this to Navier-Stokes later on
    res = domain.integral('∇_j(v_i) σ_ij dV' @ ns, degree=degree*3)
    res += domain.integral('q ∇_k(u_k) dV' @ ns, degree=degree*3)

    # strong enforcement of non-penetrating boundary conditions
    sqr = domain.boundary.integral('(u_k n_k)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15)

    if strongbc:
        # strong enforcement of tangential boundary conditions
        sqr = domain.boundary.integral('(ε_ij n_i (u_j - uwall_j))^2 dS' @ ns, degree=degree*2)
        tcons = solver.optimize('u,', sqr, droptol=1e-15)
        cons['u'] = numpy.choose(numpy.isnan(cons['u']), [cons['u'], tcons['u']])
    else:
        # weak enforcement of tangential boundary conditions via Nitsche's method
        ns.N = 5 * degree * nelems # Nitsche constant based on element size = 1/nelems
        ns.nitsche_i = '(N v_i - (∇_j(v_i) + ∇_i(v_j)) n_j) / Re'
        res += domain.boundary.integral('(nitsche_i (u_i - uwall_i) - v_i σ_ij n_j) dS' @ ns, degree=2*degree)

    with log.context('stokes'):
        args0 = solver.solve_linear('u:v,p:q', res, constrain=cons)
        postprocess(domain, ns, **args0)

    # change to Navier-Stokes by adding convection
    res += domain.integral('v_i ∇_j(u_i) u_j dV' @ ns, degree=degree*3)
    if not compatible:
        # add consistent term for skew-symmetry
        res += domain.integral('.5 u_i v_i ∇_j(u_j) dV' @ ns, degree=degree*3)

    with log.context('navier-stokes'):
        args1 = solver.newton('u:v,p:q', res, arguments=args0, constrain=cons).solve(tol=1e-10)
        postprocess(domain, ns, **args1)

    return args0, args1


# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, **arguments):

    # reconstruct velocity streamlines
    sqr = domain.integral('Σ_i (u_i - ε_ij ∇_j(ψ))^2 dV' @ ns, degree=4)
    arguments = solver.optimize('ψ,', sqr, arguments=arguments)

    bezier = domain.sample('bezier', 9)
    x, u, p, ψ = bezier.eval(['x_i', 'sqrt(u_i u_i)', 'p', 'ψ'] @ ns, **arguments)

    with export.mplfigure('flow.png', dpi=150) as fig: # plot velocity as field, pressure as contours, streamlines as dashed
        ax = fig.add_subplot(111, aspect='equal')
        im = export.triplot(ax, x, u, tri=bezier.tri, hull=bezier.hull, cmap='hot_r', clim=(0,1))
        fig.colorbar(im, label='velocity')
        ax.tricontour(x[:, 0], x[:, 1], bezier.tri, ψ, levels=numpy.percentile(ψ, numpy.arange(2,100,3)), colors='k', linestyles='solid', linewidths=.5, zorder=9)

    x = numpy.linspace(0, 1, 1001)
    v = domain.locate(ns.x, numpy.stack([x, numpy.repeat(.5, len(x))], 1), tol=1e-10).eval(ns.u[1], **arguments)
    imin = numpy.argmin(v)
    imax = numpy.argmax(v)
    with export.mplfigure('cross-hor.png', dpi=150) as fig:
        ax = fig.add_subplot(111, xlim=(0,1), title='horizontal cross section at y=0.5', xlabel='x-coordinate', ylabel='vertical velocity')
        ax.plot(x, v)
        ax.annotate(f'x={x[imax]:.3f}\nv={v[imax]:.3f}', xy=(x[imax], v[imax]), xytext=(x[imax], 0), arrowprops=dict(arrowstyle="->"), ha='center', va='center')
        ax.annotate(f'x={x[imin]:.3f}\nv={v[imin]:.3f}', xy=(x[imin], v[imin]), xytext=(x[imin], 0), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    y = numpy.linspace(0, 1, 1001)
    u = domain.locate(ns.x, numpy.stack([numpy.repeat(.5, len(y)), y], 1), tol=1e-10).eval(ns.u[0], **arguments)
    imin = numpy.argmin(u)
    with export.mplfigure('cross-ver.png', dpi=150) as fig:
        ax = fig.add_subplot(111, ylim=(0,1), title='vertical cross section at x=0.5', ylabel='y-coordinate', xlabel='horizontal velocity')
        ax.plot(u, y)
        ax.annotate(f'y={y[imin]:.3f}\nu={u[imin]:.3f}', xy=(u[imin], y[imin]), xytext=(0, y[imin]), arrowprops=dict(arrowstyle="->"), ha='left', va='center')


class test(testing.TestCase):

    def test_baseline(self):
        args0, args1 = main(nelems=3, degree=2, reynolds=100.)
        with self.subTest('stokes-velocity'):
            self.assertAlmostEqual64(args0['u'], '''
                eNo9jSsLwlAcxQ82gwOjoFZ1ONA9NK6vC2Ky2cU2QeziFxDBJIKfZLv/O92DYRqo6waDGBR3GXjKgfPg
                Bwh9A6AYA80TMOwCU0OkcwVonGMl8Uaa5C90i9+MipP2tn1gEgFXvgpDGqgRK+smPTWbaqqcdWMfeBPw
                YcDGBdLMJR3Y/X+zdkhHHrEHM6lENt25+OU8OUi8PUn+klm87lacqiN4uQrZ4tUCLh3g4AFrV6Q/uctG
                gQ==''')
        with self.subTest('stokes-pressure'):
            self.assertAlmostEqual64(args0['p'], '''
                eNoBHgDh/+vVsNEXy6jTbiq1z7Av9C0mLJkw1NDTLEEtEC/xNAwED0s=''')
        with self.subTest('navier-stokes-velocity'):
            self.assertAlmostEqual64(args1['u'], '''
                eNpjYAABz4sMDP81GRh69BgYTgBxkQlIdCVQ5Nb5R/oSV+KNW8/+NXlidMOE4WS+mZM5A8M1JQaGuLMt
                ejPOpBgtOs1vonRe0IT59ErjfWZAuasMDB1n9M9vOG2kL3B6l1H0uUXGRQZXjVYB9a2/xMCQfW7FueVn
                nM/5nf5xevqZxDOq5w4bCwLlOoD6XDV/n1t//s5ZvjPzTjmdDjx55+Slky/MGaDgHFB3vz4DgynQfS9O
                A3WcBIkCAB7aSkk=''')
        with self.subTest('navier-stokes-pressure'):
            self.assertAlmostEqual64(args1['p'], '''
                eNpz01W9oHVmuU7SJYtzgherdcr0n59dfiZT11yP97yCGQDN0Azu''')

    def test_mixed(self):
        args0, args1 = main(nelems=3, etype='mixed', degree=2, reynolds=100.)
        with self.subTest('stokes-velocity'):
            self.assertAlmostEqual64(args0['u'], '''
                eNpjYAABHx0Ghrbz+lcYGJZpR2hrnDtm/EObgWHvWSGD1WeuGTIwmJ9jYLAwnH32p8nKMzFmFqfVTJTP
                aBszMOw0AenebmJh9tuMgWHGWX9jQ3MGhr9nLA35zxRcWHOm4mzOBQaG55cZGCTPGV6fdUrhwtEzvhe2
                n+Y8k3RG+Mwio99nuoHqg4G48WzCmTignYUXDfXNzoedATuL4bMeA0Op9qczWqfXnTl2ioHhINAdHufv
                ntx18qoZSH7FSRAJAB13Sc0=''')
        with self.subTest('stokes-pressure'):
            self.assertAlmostEqual64(args0['p'], '''
                eNp7pKl+nf1KznmxS62ns/W+az/TNTL4ondU1/46t6GKKQDiJg1H''')
        with self.subTest('navier-stokes-velocity'):
            self.assertAlmostEqual64(args1['u'], '''
                eNpjYACBVVcYGH5fZAKSChcLLv05p2jsp8XA4Hhe4YrV2QmGDAxHzzMwuBpknHcz4T3nYGp7JslY+ewy
                YwaGamOQ7jyjzaZJZgwMYaeFTR4D6TfnvhgUnanXTzuz4WzmBQaG6MsMDEcueOpxnF5iwHZ+kfHaUypn
                n5xefpbrzEYjZ3MGhiogDr/YYbxbjYHhrH6lYcY55zNgZzGcNWBgUL0Uctr3zLzTt08xMOScZmCYdnbl
                qQMnpcxB8konQSQACVZG3A==''')
        with self.subTest('navier-stokes-pressure'):
            self.assertAlmostEqual64(args1['p'], '''
                eNrbqjVZs1/ry/n48z1nSrW9L83RkTmneNZMO/TCOUNbMwDktQ3z''')

    def test_compatible(self):
        args0, args1 = main(nelems=3, degree=2, reynolds=100., compatible=True)
        with self.subTest('stokes-velocity'):
            self.assertAlmostEqual64(args0['u'], '''
                eNpjYIAAvwvdBr9O2Zk90E8+rXQ6yxzGZ4CDTfr3z0H45hc2mjSagFgn9f1P15+G6Fc0PHSSgQEAx7kX
                6A==''')
        with self.subTest('stokes-pressure'):
            self.assertAlmostEqual64(args0['p'], '''
                eNoL1u+7NOfUR929ugvORxlU6W7V1TcUuyiif/PKCf1yUwDfRw2t''')
        with self.subTest('navier-stokes-velocity'):
            self.assertAlmostEqual64(args1['u'], '''
                eNpjYICA1HNRRkGnZ5r26m86bX3awvyBftS5C6dOmDHAwWxDmbMzTUEsrfMrTA6YgFjKV53OOJ0FsR7o
                F561OMnAAAC5tRfX''')
        with self.subTest('navier-stokes-pressure'):
            self.assertAlmostEqual64(args1['p'], '''
                eNoz1VYx3HT6t16w/uKz73Uv6R7RNzx35swh7XdXrQ0TzADbMQ6l''')

    def test_strong(self):
        args0, args1 = main(nelems=3, degree=2, reynolds=100., strongbc=True)
        with self.subTest('stokes-velocity'):
            self.assertAlmostEqual64(args0['u'], '''
                eNpjYMAPDl2wNEg9p2D8QeOQcafBJ9OTJ6abB5lD5E6eVb348oyVkfSZf8YFZ6RNZp6+ZwiTO3KGgUEP
                iNedYmDoPc3AsMGEgQGh77beyzPHzkqfYTpTcObp6Zmnlc7B5A5dOH4+9dyDMx807M50GvCdOnki8wRM
                DhcAAEYiNtQ=''')
        with self.subTest('stokes-pressure'):
            self.assertAlmostEqual64(args0['p'], '''
                eNoBHgDh/3fRlNYxy7PR0NVKz1ktVi1E1HowsdGJ07Qt/9PINA6QEUk=''')
        with self.subTest('navier-stokes-velocity'):
            self.assertAlmostEqual64(args1['u'], '''
                eNpjYMAPAs4H6M8zaDJOO91vKma628TihKx5kDlEbv2Z5fqFZ24aKZ/mNplmmGJy5eRbY5jcpdOGF+tP
                /9BPPW1gXH6W2eSpkds5mBz72fvnUs/EnHt+etXpCWdZzgSd3W8Ck1M9L3zGGyi/4Pz80+ZnNp44c9L8
                BEwOFwAA4RM3QA==''')
        with self.subTest('navier-stokes-pressure'):
            self.assertAlmostEqual64(args1['p'], '''
                eNoBHgDh//ot489SzMEsntHezWTPAC+jL+XN/8wF02UxTc1JNhf0ELc=''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Stokes,Navier-Stokes,Taylor-Hood,Raviard-Thomas:thumbnail=0
