from nutils import mesh, function, export, testing
from nutils.solver import System, LinesearchNewton
from nutils.expression_v2 import Namespace
from matplotlib.collections import LineCollection
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
    domain.center_hor = domain.trim(geom[1] - .5, maxrefine=0).boundary['trimmed'].sample('bezier', 9)
    domain.center_ver = domain.trim(geom[0] - .5, maxrefine=0).boundary['trimmed'].sample('bezier', 9)

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.ε = function.levicivita(2)
    ns.Σ = function.ones([domain.ndims])
    ns.Re = reynolds
    ns.uwall = numpy.stack([domain.boundary.indicator('top'), 0])
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    if not compatible:
        ns.u = domain.field('u', btype='std', degree=degree, shape=[2])
        ns.p = domain.field('p', btype='std', degree=degree-1)
        ns.ψ = domain.field('ψ', btype='std', degree=2)
    else:
        ns.u = function.field('u', function.vectorize([domain.basis('spline', degree=p) for p in degree - 1 + numpy.eye(2, dtype=int)]))
        ns.p = domain.field('p', btype='spline', degree=degree-1)
        ns.ψ = domain.field('ψ', btype='spline', degree=degree)
    ns.v = function.replace_arguments(ns.u, 'u:v')
    ns.q = function.replace_arguments(ns.p, 'p:q')
    ns.σ_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'
    ns.ω = 'ε_ij ∇_i(u_j)' # vorticity

    # weak formulation for Stokes flow, over-integrating for improved
    # efficiency when we turn this to Navier-Stokes later on
    res = domain.integral('∇_j(v_i) σ_ij dV' @ ns, degree=degree*3)
    res += domain.integral('q ∇_k(u_k) dV' @ ns, degree=degree*3)

    # strong enforcement of non-penetrating boundary conditions
    sqr = domain.boundary.integral('(u_k n_k)^2 dS' @ ns, degree=degree*2)
    cons = System(sqr, trial='u').solve_constraints(droptol=1e-15)
    cons['p'] = numpy.zeros(function.arguments_for(res)['p'].shape, dtype=bool)
    cons['p'].flat[0] = True # point constraint

    if strongbc:
        # strong enforcement of tangential boundary conditions
        sqr = domain.boundary.integral('(ε_ij n_i (u_j - uwall_j))^2 dS' @ ns, degree=degree*2)
        tcons = System(sqr, trial='u').solve_constraints(droptol=1e-15)
        cons['u'] = numpy.choose(numpy.isnan(cons['u']), [cons['u'], tcons['u']])
    else:
        # weak enforcement of tangential boundary conditions via Nitsche's method
        ns.N = 5 * degree * nelems # Nitsche constant based on element size = 1/nelems
        ns.nitsche_i = '(N v_i - (∇_j(v_i) + ∇_i(v_j)) n_j) / Re'
        res += domain.boundary.integral('(nitsche_i (u_i - uwall_i) - v_i σ_ij n_j) dS' @ ns, degree=2*degree)

    with log.context('stokes'):
        args = System(res, trial='u,p', test='v,q').solve(constrain=cons)
        postprocess(domain, ns, **args)

    # change to Navier-Stokes by adding convection
    res += domain.integral('v_i ∇_j(u_i) u_j dV' @ ns, degree=degree*3)
    if not compatible:
        # add consistent term for skew-symmetry
        res += domain.integral('.5 u_i v_i ∇_j(u_j) dV' @ ns, degree=degree*3)

    with log.context('navier-stokes'):
        args = System(res, trial='u,p', test='v,q').solve(arguments=args, constrain=cons, tol=1e-10, method=LinesearchNewton())
        postprocess(domain, ns, **args)

    u, ω = domain.locate(ns.x, [[.5, .5], [0, .95]], tol=1e-14).eval(['u_i', 'ω'] @ ns, **args)
    log.info(f'center velocity: {u[0,0]}, {u[0,1]}')
    log.info(f'center vorticity: {ω[0]}')
    log.info(f'upper-left (0,.95) vorticity: {ω[1]}')

    return u, ω


# Postprocessing in this script is separated so that it can be reused for the
# results of Stokes and Navier-Stokes, and because of the extra steps required
# for establishing streamlines.

def postprocess(domain, ns, **arguments):

    # reconstruct velocity streamlines
    sqr = domain.integral('Σ_i (u_i - ε_ij ∇_j(ψ))^2 dV' @ ns, degree=4)
    consψ = numpy.zeros(function.arguments_for(sqr)['ψ'].shape, dtype=bool)
    consψ.flat[0] = True # point constraint
    arguments = System(sqr, trial='ψ').solve(arguments=arguments, constrain={'ψ': consψ})

    bezier = domain.sample('bezier', 4)
    x, u, ψ, ω = bezier.eval(['x_i', 'sqrt(u_i u_i)', 'ψ', 'ω'] @ ns, **arguments)
    with export.mplfigure('velocity.png', dpi=150) as fig: # plot velocity as field, streamlines as contours
        ax = fig.add_subplot(111)
        im = export.triplot(ax, x, u, tri=bezier.tri, hull=bezier.hull, cmap='hot_r', clim=(0,1))
        fig.colorbar(im, label='velocity')
        ax.tricontour(*x.T, bezier.tri, ψ, levels=numpy.unique(numpy.percentile(ψ, numpy.arange(2,100,3))), colors='k', linestyles='solid', linewidths=.5, zorder=9)
    with export.mplfigure('vorticity.png', dpi=150) as fig: # plot vorticity as field with contours
        ax = fig.add_subplot(111)
        im = export.triplot(ax, x, ω, tri=bezier.tri, hull=bezier.hull, cmap='bwr', clim=(-5,5))
        fig.colorbar(im, label='vorticity')
        ax.tricontour(*x.T, bezier.tri, ω, levels=numpy.arange(-5, 6), colors='k', linestyles='solid', linewidths=.5, zorder=9)

    xv = numpy.stack(domain.center_hor.eval(['x_0', 'u_1'] @ ns, **arguments), axis=1)
    xmin, vmin = xv[numpy.argmin(xv[:,1])]
    xmax, vmax = xv[numpy.argmax(xv[:,1])]
    with export.mplfigure('cross-hor.png', dpi=150) as fig:
        ax = fig.add_subplot(111, xlim=(0,1), title='horizontal cross section at y=0.5', xlabel='x-coordinate', ylabel='vertical velocity')
        ax.add_collection(LineCollection(xv[domain.center_hor.tri]))
        ax.autoscale(enable=True, axis='y')
        ax.annotate(f'x={xmax:.5f}\nv={vmax:.5f}', xy=(xmax, vmax), xytext=(xmax, 0), arrowprops=dict(arrowstyle="->"), ha='center', va='center')
        ax.annotate(f'x={xmin:.5f}\nv={vmin:.5f}', xy=(xmin, vmin), xytext=(xmin, 0), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    uy = numpy.stack(domain.center_ver.eval(['u_0', 'x_1'] @ ns, **arguments), axis=1)
    umin, ymin = uy[numpy.argmin(uy[:,0])]
    with export.mplfigure('cross-ver.png', dpi=150) as fig:
        ax = fig.add_subplot(111, ylim=(0,1), title='vertical cross section at x=0.5', ylabel='y-coordinate', xlabel='horizontal velocity')
        ax.add_collection(LineCollection(uy[domain.center_hor.tri]))
        ax.autoscale(enable=True, axis='x')
        ax.annotate(f'y={ymin:.5f}\nu={umin:.5f}', xy=(umin, ymin), xytext=(0, ymin), arrowprops=dict(arrowstyle="->"), ha='left', va='center')


class test(testing.TestCase):

    def test_baseline(self):
        (ucc, uul), (ωcc, ωul) = main(nelems=3, degree=2, reynolds=100.)
        self.assertAlmostEqual(ucc[0], -0.19499, places=5)
        self.assertAlmostEqual(ucc[1], 0.04884, places=5)
        self.assertAlmostEqual(ωcc, -1.22863, places=5)
        self.assertEqual(uul[0], 0)
        self.assertAlmostEqual(uul[1], 0.05882, places=5)
        self.assertAlmostEqual(ωul, 0.65710, places=5)

    def test_mixed(self):
        (ucc, uul), (ωcc, ωul) = main(nelems=3, etype='mixed', degree=2, reynolds=100.)
        self.assertAlmostEqual(ucc[0], -0.19341, places=5)
        self.assertAlmostEqual(ucc[1], 0.03757, places=5)
        self.assertAlmostEqual(ωcc, -0.71609, places=5)
        self.assertEqual(uul[0], 0)
        self.assertAlmostEqual(uul[1], 0.03627, places=5)
        self.assertAlmostEqual(ωul, 1.79983, places=5)

    def test_compatible(self):
        (ucc, uul), (ωcc, ωul) = main(nelems=3, degree=2, reynolds=100., compatible=True)
        self.assertAlmostEqual(ucc[0], -0.21725, places=5)
        self.assertAlmostEqual(ucc[1], 0.04419, places=5)
        self.assertAlmostEqual(ωcc, -0.69778, places=5)
        self.assertEqual(uul[0], 0)
        self.assertAlmostEqual(uul[1], 0.10897, places=5)
        self.assertAlmostEqual(ωul, -0.10411, places=5)

    def test_strong(self):
        (ucc, uul), (ωcc, ωul) = main(nelems=3, degree=2, reynolds=100., strongbc=True)
        self.assertAlmostEqual(ucc[0], -0.18231, places=5)
        self.assertAlmostEqual(ucc[1], 0.05775, places=5)
        self.assertAlmostEqual(ωcc, -1.44979, places=5)
        self.assertEqual(uul[0], 0)
        self.assertEqual(uul[1], 0)
        self.assertAlmostEqual(ωul, 1.41277, places=5)


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=benchmark,Navier-Stokes,Taylor-Hood,Raviard-Thomas:thumbnail=0
