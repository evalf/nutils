from nutils import cli, export, function, testing
from nutils.mesh import gmsh
from nutils.solver import System
from nutils.SI import Length, Density, Viscosity, Velocity, Time, Pressure, Acceleration
from nutils.expression_v2 import Namespace
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import treelog as log
import numpy


@dataclass
class Domain:
    '''Parameters for the domain geometry.

    The default values match Table 1 of Turek and Hron [1].'''

    channel_length: Length = Length('2.5m')
    channel_height: Length = Length('.41m')
    x_center: Length = Length('.2m')
    y_center: Length = Length('.2m')
    cylinder_radius: Length = Length('5cm')
    structure_length: Length = Length('35cm')
    structure_thickness: Length = Length('2cm')
    elemsize: Length = Length('4mm')
    coarsening: float = 10.

    def generate_mesh(self):
        'Call gmsh to generate mesh and return topo, geom tuple.'''

        u = Length('m') # reference length for mesh generation

        topo, geom = gmsh(
            Path(__file__).parent/'turek.geo',
            dimension=2,
            order=2,
            numbers={
                'channel_length': self.channel_length/u,
                'channel_height': self.channel_height/u,
                'x_center': self.x_center/u,
                'y_center': self.y_center/u,
                'cylinder_radius': self.cylinder_radius/u,
                'structure_length': self.structure_length/u,
                'structure_thickness': self.structure_thickness/u,
                'elemsize': self.elemsize/u,
                'coarsening': self.coarsening})

        bezier = topo.sample('bezier', 2)
        bezier_structure = topo['fluid'].boundary['structure'].sample('bezier', 3)
        bezier_cylinder = topo['fluid'].boundary['cylinder'].sample('bezier', 3)
        A = topo.points['A'].sample('gauss', 1).eval(geom)
        with export.mplfigure('mesh.jpg', dpi=150) as fig:
            ax = fig.add_subplot(111)
            export.triplot(ax, bezier.eval(geom), hull=bezier.hull)
            export.triplot(ax, bezier_structure.eval(geom), hull=bezier_structure.tri, linewidth=1, linecolor='r')
            export.triplot(ax, bezier_cylinder.eval(geom), hull=bezier_cylinder.tri, linewidth=1, linecolor='b')
            ax.set_xlim(0, 2*self.channel_height/u)

        return topo, geom * u


@dataclass
class Solid:
    '''Parameters for the solid problem.'''

    density: Density = Density('10kg/L')
    poisson_ratio: float = .4
    shear_modulus: Pressure = Pressure('.5MPa')
    gravity: Acceleration = Acceleration('0m/s2')

    def lame_parameters(self):
        'Return tuple of first and second lame parameter.'

        return 2 * self.shear_modulus * self.poisson_ratio / (1 - 2 * self.poisson_ratio), self.shear_modulus

    def young(self):
        "Return Young's elasticity modulus."

        return 2 * self.shear_modulus * (1 + self.poisson_ratio)


@dataclass
class Fluid:
    '''Parameters for the fluid problem.'''

    density: Density = Density('1kg/L')
    viscosity: Viscosity = Viscosity('1Pa*s')
    velocity: Velocity = Velocity('1m/s')

    def reynolds(self, reference_length):
        'Return Reynolds number for given reference length'

        return self.density * self.velocity * reference_length / self.viscosity


@dataclass
class Dynamic:
    '''Parameters relating to time dependence.'''

    timestep: Time = Time('5ms') # simulation time step size
    endtime: Time = Time('10s') # total duration of the simulation
    init: Time = Time('2s') # duration of the ramp-up phase
    window: Time = Time('1s') # sliding window length for time series plots
    gamma: float = .5
    beta: float = .25

    def __post_init__(self):
        self.timeseries = defaultdict(deque(maxlen=round(self.window / self.timestep)).copy)

    def ramp_up(self, t):
        'Return inflow ramp-up scale factor at given time.'

        return .5 - .5 * numpy.cos(numpy.pi * min(t / self.init, 1))

    @property
    def times(self):
        'Return all configured time steps for the simulation.'

        return numpy.arange(1, self.endtime / self.timestep + .5) * self.timestep

    def add_and_plot(self, name, t, v, ax):
        'Add data point and plot time series for past window.'

        d = self.timeseries[name]
        d.append((t, v))
        times, values = numpy.stack(d, axis=1)
        ax.plot(times, values)
        ax.set_ylabel(name)
        ax.grid()
        ax.autoscale(enable=True, axis='x', tight=True)
        vmin, vmax = numpy.quantile(values, [0,1])
        vmean = (vmax + vmin) / 2
        values -= vmean
        icross, = numpy.nonzero(values[1:] * values[:-1] < 0)
        if len(icross) >= 4: # minimum of two up, two down
            tcross = (times[icross] * values[icross+1] - times[icross+1] * values[icross]) / (values[icross+1] - values[icross])
            ax.plot(tcross, [vmean] * len(icross), '+')
            ax.text(tcross[numpy.diff(tcross).argmax():][:2].mean(), vmean,
                s=f'{vmean:+.4f}\n±{(vmax-vmin)/2:.4f}\n↻{(tcross[2:]-tcross[:-2]).mean():.4f}',
                va='center', ha='center', multialignment='right')

    # The Newmark-beta scheme is used for time integration. For our formulation
    # to support both stationary and dynamic simulations, we take displacement
    # (solid) and velocity (fluid) as primary variables, with time derivatives
    # introduced via helper arguments that are updated after every solve.
    #
    # d = d0 + δt u0 + .5 δt^2 aβ, where aβ = (1-2β) a0 + 2β a
    # => δd = δt u0 + δt^2 [ .5 a0 + β δa ]
    # => δa = [ δd / δt^2 - u0 / δt - .5 a0 ] / β
    #
    # u = u0 + δt aγ, where aγ = (1-γ) a0 + γ a
    # => δu = δt [ a0 + γ δa ]
    # => δa = [ δu / δt - a0 ] / γ

    def newmark_defo_args(self, d, d0=0., u0δt=0., a0δt2=0., **args):
        δaδt2 = (d - d0 - u0δt - .5 * a0δt2) / self.beta
        uδt = u0δt + a0δt2 + self.gamma * δaδt2
        aδt2 = a0δt2 + δaδt2
        return dict(args, d=d+uδt+.5*aδt2, d0=d, u0δt=uδt, a0δt2=aδt2)

    def newmark_defo(self, d):
        D = self.newmark_defo_args(d, *[function.replace_arguments(d, [('d', t)]) for t in ('d0', 'u0δt', 'a0δt2')])
        return D['u0δt'] / self.timestep, D['a0δt2'] / self.timestep**2

    def newmark_velo_args(self, u, u0=0., a0δt=0., **args):
        aδt = a0δt + (u - u0 - a0δt) / self.gamma
        return dict(args, u=u+aδt, u0=u, a0δt=aδt)

    def newmark_velo(self, u):
        D = self.newmark_velo_args(u, *[function.replace_arguments(u, [('u', t)]) for t in ('u0', 'a0δt')])
        return D['a0δt'] / self.timestep


def main(domain: Domain = Domain(), solid: Optional[Solid] = Solid(), fluid: Optional[Fluid] = Fluid(), dynamic: Optional[Dynamic] = Dynamic()):
    '''Turek Hron benchmark problem

    This is a monolythic ALE (Arbitrary Lagrangian Eulerian) implementation of
    the fluid-structure interaction benchmark defined in 2006 by [Turek and
    Hron](https://doi.org/10.1007/3-540-34596-5_15). The implementation covers
    all fluid dynamics tests CFD1, CFD2 and CFD3, all structural tests CSM1,
    CSM2 and CSM3, and all interaction tests FSI1, FSI2 and FSI3, as well as
    freely defined variants thereof.'''

    assert solid or fluid, 'nothing to compute'

    if fluid:
        log.info('Re:', fluid.reynolds(2 * domain.cylinder_radius))
        if solid:
            log.info('Ae:', solid.young() / fluid.density / fluid.velocity**2)
            log.info('β:',  solid.density / fluid.density)

    topo, geom = domain.generate_mesh()

    bezier = topo['fluid'].sample('bezier', 3)
    bezier = bezier.subset(bezier.eval(geom[0]) < 2.2*domain.channel_height)
    bbezier = topo['fluid'].boundary['cylinder,structure'].sample('bezier', 3)

    res = 0.
    cons = {}
    args = {}

    ns = Namespace()
    ns.δ = function.eye(2)
    ns.xref = geom
    ns.define_for('xref', gradient='∇ref', jacobians=('dVref', 'dSref'))

    if solid:

        ns.ρs = solid.density
        ns.λs, ns.μs = solid.lame_parameters()
        ns.g = -solid.gravity * ns.δ[1]

        # Deformation, velocity and acceleration are defined on the entire
        # domain, solving for conservation of momentum on the solid domain and
        # a mesh continuation problem on the fluid domain. While the latter is
        # used only when fluid is enabled, we include it regardless both for
        # simplicity and for testing purposes.
        ns.d = topo.field('d', btype='std', degree=2, shape=(2,)) * domain.cylinder_radius # deformation at the end of the timestep
        if dynamic:
            ns.v, ns.a = dynamic.newmark_defo(ns.d)
        else:
            ns.a = Acceleration.wrap(function.zeros((2,)))

        # Deformed geometry
        ns.x_i = 'xref_i + d_i'
        ns.F_ij = '∇ref_j(x_i)' # deformation gradient tensor
        ns.C_ij = 'F_ki F_kj' # right Cauchy-Green deformation tensor
        ns.E_ij = '.5 (C_ij - δ_ij)' # Green-Lagrangian strain tensor
        ns.S_ij = 'λs E_kk δ_ij + 2 μs E_ij' # 2nd Piola–Kirchhoff stress
        ns.P_ij = 'F_ik S_kj' # 1st Piola–Kirchhoff stress
        ns.J = numpy.linalg.det(ns.F)

        # Momentum balance: ρs (a - g) = div P
        ns.dtest = function.replace_arguments(ns.d, 'd:dtest') / (solid.shear_modulus * domain.cylinder_radius**2)
        res += topo['solid'].integral('(∇ref_j(dtest_i) P_ij + dtest_i ρs (a_i - g_i)) dVref' @ ns, degree=4)

        # In the momentum balance above, the only test and trial dofs involved
        # are those that have support on the solid domain. The remaining trial
        # dofs will follow from minimizing a mesh energy functional, using the
        # solid deformation as a boundary constraint so that the continuation
        # problem does not feed back into the physics. To achieve this within a
        # monolythic setting, we establish a boolean array 'dfluid' to select
        # all dofs that are exclusively supported by the fluid domain, and use
        # it to restrict the minimization problem to the remaining dofs via the
        # linearize operation.
        mesh_energy = topo['fluid'].integral('C_kk - 2 log(J)' @ ns, degree=4) # Neo-Hookean with jacobian based stiffness
        sqr = topo['solid'].integral('d_k d_k dVref' @ ns, degree=4) / domain.cylinder_radius**4
        dfluid = numpy.isnan(System(sqr, trial='d').solve_constraints(droptol=1e-9)['d']) # true if dof is not supported by solid domain
        res += function.linearize(mesh_energy, {'d': function.arguments_for(res)['dtest'] * dfluid})

        # Deformation constraints: fixed at exterior boundary and cylinder
        sqr = topo.boundary.integral('d_k d_k dSref' @ ns, degree=4) / domain.cylinder_radius**3
        cons = System(sqr, trial='d').solve_constraints(droptol=1e-9, constrain=cons)

        # Zero initial deformation
        args['d'] = numpy.zeros(function.arguments_for(res)['d'].shape)

    else: # fully rigid solid

        ns.x = ns.xref
        ns.v = Velocity.wrap(function.zeros((2,)))
        ns.a = Acceleration.wrap(function.zeros((2,)))

    if fluid:

        ns.ρf = fluid.density
        ns.μf = fluid.viscosity

        ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))

        # The fluid velocity is defined relative to the mesh velocity, which
        # helps to impose the no slip condition on the elastic structure.
        # Furthermore, in ALE the material derivative convects only with this
        # relative component.
        ns.urel = topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
        if dynamic:
            ns.arel = dynamic.newmark_velo(ns.urel)
            ns.u_i = 'v_i + urel_i'
            ns.DuDt_i = 'a_i + arel_i + ∇_j(u_i) urel_j' # material derivative
        else:
            ns.u = ns.urel
            ns.DuDt_i = '∇_j(u_i) u_j'

        ns.p = topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius
        ns.σ_ij = 'μf (∇_j(u_i) + ∇_i(u_j)) - p δ_ij' # fluid stress tensor

        # Project Posseuille inflow at inlet (exact because of quadratic
        # velocity basis), parallel outflow at outlet, and no-slip conditions
        # for the relative velocity at the remaining boundaries.
        y = ns.xref[1] / domain.channel_height
        ns.uin = 6 * fluid.velocity * y * (1 - y) # Posseuille profile
        sqr = topo['fluid'].boundary['wall,cylinder,structure'].integral('urel_k urel_k dSref' @ ns, degree=4) / (domain.cylinder_radius * fluid.velocity**2)
        sqr += topo['fluid'].boundary['inlet'].integral('(urel_0 - uin)^2 dSref' @ ns, degree=4) / (domain.cylinder_radius * fluid.velocity**2)
        sqr += topo['fluid'].boundary['inlet,outlet'].integral('urel_1^2 dSref' @ ns, degree=4) / (domain.cylinder_radius * fluid.velocity**2)
        cons = System(sqr, trial='u').solve_constraints(droptol=1e-9, constrain=cons) # exact projection
        ucons = cons['u'] # save for ramp-up phase

        # Momentum balance: ρf Du/Dt = ∇σ => ∀v ∫ (v ρf Du/Dt + ∇v:σ) = ∮ v·σ·n = 0
        ns.utest = function.replace_arguments(ns.urel, 'u:utest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('(utest_i ρf DuDt_i + ∇_j(utest_i) σ_ij) dV' @ ns, degree=4)

        # Incompressibility: div u = 0
        ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('ptest ∇_k(u_k) dV' @ ns, degree=4)

        if solid:
            # The action of fluid stress on the solid is imposed weakly by
            # lifting the test functions into the fluid domain, using the
            # identity ∮ d·σ·n = ∫ ∇(d·σ) = ∫ (∇d:σ + d·∇σ) = ∫ (∇d:σ + ρf
            # d·Du/Dt). We need the inverse of the dfluid mask to exclude dofs
            # without support on the boundary.
            dsolid = ~dfluid # true if dof is (partially) supported by solid domain
            res += function.replace_arguments(
                topo['fluid'].integral('(dtest_i ρf DuDt_i + ∇_j(dtest_i) σ_ij) dV' @ ns, degree=4),
                {'dtest': function.arguments_for(res)['dtest'] * dsolid})

        # The net force will likewise be evaluated weakly, via the identity F =
        # ∮ σ·n = ∮ λ σ·n = ∫ ∇(λ σ) = ∫ (∇λ·σ + λ ∇σ) = ∫ (∇λ·σ + λ ρf Du/Dt),
        # with λ = 1 on the boundary and lifted into the fluid domain.
        lift = topo['fluid'].field('lift', btype='std', degree=2) # same basis as urel, but scalar
        sqr = topo['fluid'].boundary['cylinder,structure'].integral((lift - 1)**2, degree=4)
        lcons = System(sqr, trial='lift').solve_constraints(droptol=1e-9)
        ns.λ = function.replace_arguments(lift, {'lift': numpy.nan_to_num(lcons['lift'])})
        F = topo['fluid'].integral('-(∇_j(λ) σ_ij + λ ρf DuDt_i) dV' @ ns, degree=4)

        # Zero initial velocity
        args['u'] = numpy.zeros(function.arguments_for(res)['u'].shape)

        u_bz = function.factor(bezier(ns.u))
        p_bz = function.factor(bezier(ns.p)) - topo.points['B'].sample('gauss', 1)(ns.p)[0]

    x_bz = function.factor(bezier(ns.x))
    x_bbz = function.factor(bbezier(ns.x))

    trial = 'upd'[0 if fluid else 2:2 if not solid else 3]
    system = System(res, trial=list(trial), test=[t+'test' for t in trial])

    DL = uxy = None # for unit tests only

    for t in log.iter.fraction('timestep', dynamic.times) if dynamic else [Time.wrap(float('inf'))]:

        if dynamic:
            if solid:
                args = dynamic.newmark_defo_args(**args)
            if fluid:
                args = dynamic.newmark_velo_args(**args)
                cons['u'] = ucons * dynamic.ramp_up(t) # constrain inflow at end of time step

        args = system.solve(constrain=cons, arguments=args, tol=1e-9)

        x, xb = function.eval([x_bz, x_bbz], **args)
        if fluid:
            u, p = function.eval([u_bz, p_bz], **args)
            with export.mplfigure('solution.jpg', dpi=150) as fig:
                pstep = 25 * fluid.viscosity * fluid.velocity / domain.channel_height
                ax = fig.add_subplot(111, title=f'flow at t={t:.3s}, pressure contours every {pstep:.0Pa}', ylabel='[m]')
                vmax = 2 * fluid.velocity * (dynamic.ramp_up(t) if dynamic else 1)
                im = export.triplot(ax, x/'m', numpy.linalg.norm(u/'m/s', axis=1), tri=bezier.tri, cmap='inferno', clim=(0, vmax/'m/s'))
                ax.tricontour(*(x/'m').T, bezier.tri, p/pstep, numpy.arange(*numpy.quantile(numpy.ceil(p / pstep), [0,1])),
                    colors='white', linestyles='solid', linewidths=1, alpha=.33)
                fig.colorbar(im, orientation='horizontal', label=f'velocity [m/s]')
                export.triplot(ax, xb/'m', hull=bbezier.tri, linewidth=1)
                ax.set_xlim(0, 2*domain.channel_height/'m')
                ax.set_ylim(0, domain.channel_height/'m')

            D, L = DL = function.eval(F, arguments=args)
            log.info(f'drag: {D:N/m}')
            log.info(f'lift: {L:N/m}')
            if dynamic:
                with export.mplfigure('force.jpg', dpi=150) as fig:
                    dynamic.add_and_plot('drag [N/m]', t/'s', D/'N/m', ax=fig.add_subplot(211))
                    dynamic.add_and_plot('lift [N/m]', t/'s', L/'N/m', ax=fig.add_subplot(212, xlabel='time [s]'))

        if solid:
            if not fluid:
                with export.mplfigure('deformation.jpg', dpi=150) as fig:
                    ax = fig.add_subplot(111, title=f'deformation at t={t:.3s}', ylabel='[m]')
                    export.triplot(ax, x/'m', hull=bezier.hull)
                    export.triplot(ax, xb/'m', hull=bbezier.tri, linewidth=1)
                    ax.set_xlim(0, 2*domain.channel_height/'m')
                    ax.set_ylim(0, domain.channel_height/'m')

            ux, uy = uxy = topo.points['A'].sample('gauss', 1).eval(ns.d, arguments=args)[0]
            log.info(f'ux: {ux:mm}')
            log.info(f'uy: {uy:mm}')
            if dynamic:
                with export.mplfigure('tip-displacement.jpg', dpi=150) as fig:
                    dynamic.add_and_plot('ux [mm]', t/'s', ux/'mm', ax=fig.add_subplot(211))
                    dynamic.add_and_plot('uy [mm]', t/'s', uy/'mm', ax=fig.add_subplot(212, xlabel='time [s]'))

    return DL, uxy


def CFD1(elemsize=Length('4mm'), coarsening=10.):
    log.info('benchmark CFD1')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        None,
        Fluid(velocity=Velocity('0.2m/s')),
        None)
    log.info('reference drag: 14.29N/m')
    log.info('reference lift: 1.119N/m')


def CFD2(elemsize=Length('4mm'), coarsening=10.):
    log.info('benchmark CFD2')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        None,
        Fluid(),
        None)
    log.info('reference drag: 136.7N/m')
    log.info('reference lift: 10.53N/m')


def CFD3(elemsize=Length('4mm'), coarsening=10., timestep=Time('8ms'), gamma=.5):
    log.info('benchmark CFD3')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        None,
        Fluid(velocity=Velocity('2m/s')),
        Dynamic(timestep=timestep, window=Time('0.6s'), gamma=gamma))
    log.info('reference drag: 439.45N/m ± 5.6183N/m ~ 0.2275s')
    log.info('reference lift: -11.893N/m ± 437.81N/m ~.2275s')


def CSM1(elemsize=Length('4mm'), coarsening=10.):
    log.info('benchmark CSM1')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(density=Density('1kg/L'), gravity=Acceleration('2m/s2')),
        None,
        None)
    log.info('reference ux: -7.187mm')
    log.info('reference uy: -66.10mm')


def CSM2(elemsize=Length('4mm'), coarsening=10.):
    log.info('benchmark CSM2')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(density=Density('1kg/L'), gravity=Acceleration('2m/s2'), shear_modulus=Pressure('2MPa')),
        None,
        None)
    log.info('reference ux: -0.4690mm')
    log.info('reference uy: -16.97mm')


def CSM3(elemsize=Length('4mm'), coarsening=10., timestep=Time('8ms'), gamma=.5, beta=.25):
    log.info('benchmark CSM3')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(density=Density('1kg/L'), gravity=Acceleration('2m/s2')),
        None,
        Dynamic(timestep=timestep, window=Time('2s'), gamma=gamma, beta=beta))
    log.info('reference ux: -14.305mm ± 14.305mm ~ 0.9095s')
    log.info('reference uy: -63.607mm ± 65.160mm ~ 0.9095s')


def FSI1(elemsize=Length('4mm'), coarsening=10.):
    log.info('benchmark FSI1')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(density=Density('1kg/L')),
        Fluid(velocity=Velocity('0.2m/s')),
        None)
    log.info('reference ux: 0.0227mm')
    log.info('reference uy: 0.8209mm')
    log.info('reference drag: 14.295N/m')
    log.info('reference lift: 0.7638N/m')


def FSI2(elemsize=Length('4mm'), coarsening=10., timestep=Time('2ms'), gamma=.5, beta=.25):
    log.info('benchmark FSI2')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(),
        Fluid(),
        Dynamic(timestep=timestep, endtime=Time('35s'), gamma=gamma, beta=beta))
    log.info('reference ux: -14.58mm ± 12.44mm ~ 0.26s')
    log.info('reference uy: 1.23mm ± 80.6mm ~ 0.5s')
    log.info('reference drag: 208.83N/m ± 73.75N/m ~ 0.26s')
    log.info('reference lift: 0.88N/m ± 234.2N/m ~ 0.5s')

def FSI3(elemsize=Length('4mm'), coarsening=10., timestep=Time('2ms'), gamma=.5, beta=.25):
    log.info('benchmark FSI3')
    main(
        Domain(elemsize=elemsize, coarsening=coarsening),
        Solid(density=Density('1kg/L'), shear_modulus=Pressure('2MPa')),
        Fluid(velocity=Velocity('2m/s')),
        Dynamic(timestep=timestep, window=Time('.5s'), endtime=Time('20s'), gamma=gamma, beta=beta))
    log.info('reference ux: -2.69mm ± 2.53mm ~ 0.092s')
    log.info('reference uy: 1.48mm ± 34.38mm ~ 0.19s')
    log.info('reference drag: 457.3N/m ± 22.66N/m ~ 0.092s')
    log.info('reference lift: 2.22N/m ± 149.78N/m ~ 0.19s')


class test(testing.TestCase):

    domain = Domain(elemsize=Length('2cm'), coarsening=4., channel_length=Length('1m'))

    def test_csm(self):
        DL, uxy = main(
            self.domain,
            Solid(gravity=Acceleration('0.1m/s2')),
            None,
            None)
        self.assertEqual(DL, None)
        self.assertAllAlmostEqual(uxy/'mm', [-1.82506, -33.41303], delta=1e-4)

    def test_cfd(self):
        DL, uxy = main(
            self.domain,
            None,
            Fluid(viscosity=Viscosity('100Pa*s')),
            None)
        self.assertAllAlmostEqual(DL/'N/m', [5463.0, 110.3], delta=2)
        self.assertEqual(uxy, None)

    def test_fsi(self):
        DL, uxy = main(
            self.domain,
            Solid(shear_modulus=Pressure('1GPa')),
            Fluid(viscosity=Viscosity('100Pa*s')),
            None)
        self.assertAllAlmostEqual(DL/'N/m', [5463.0, 110.6], delta=2)
        self.assertAllAlmostEqual(uxy/'mm', [0.008, -0.006], delta=1e-2)

    def test_dyncsm(self):
        DL, uxy = main(
            self.domain,
            Solid(density=Density('1kg/L'), gravity=Acceleration('10m/s2')),
            None,
            Dynamic(timestep=Time('10ms'), endtime=Time('10ms')))
        self.assertEqual(DL, None)
        self.assertAllAlmostEqual(uxy/'mm', [-0.00011313, -0.24959199], delta=1e-7)

    def test_dyncfd(self):
        DL, uxy = main(
            self.domain,
            None,
            Fluid(viscosity=Viscosity('100Pa*s')),
            Dynamic(timestep=Time('10ms'), endtime=Time('10ms')))
        self.assertAllAlmostEqual(DL/'N/m', [0.8824, -0.0070], delta=1e-3)
        self.assertEqual(uxy, None)

    def test_dynfsi(self):
        DL, uxy = main(
            self.domain,
            Solid(density=Density('1kg/L')),
            Fluid(),
            Dynamic(timestep=Time('10ms'), endtime=Time('10ms')))
        self.assertAllAlmostEqual(DL/'N/m', [0.3430, -0.0004], delta=1e-2)
        self.assertAllAlmostEqual(uxy/'mm', [0.000025, -0.000000], delta=1e-5)


if __name__ == '__main__':
    cli.choose(main, CFD1, CFD2, CFD3, CSM1, CSM2, CSM3, FSI1, FSI2, FSI3)


# example:tags=FSI,benchmark problem
