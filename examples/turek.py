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


# Monolythic implementation of the fluid-structure interaction benchmark
# defined in the 2006 paper by Turek and Hron [1]. The implementation covers
# all fluid dynamics tests CFD1, CFD2 and CFD3, all structural tests CSM1, CSM2
# and CSM3, and all interaction tests FSI1, FSI2 and FSI3, as well as freely
# defined variants thereof.
#
# [1] https://doi.org/10.1007/3-540-34596-5_15
#
# Some results of this code are tabulated below, showing the output of function
# defaults and refinements in space and time along with the reference values
# obtained by Turek and Hron for comparison. Note that our values may not be
# exactly reproducable due to platform and version dependence of the gmsh mesh
# generator, but any differences should not be significant.
#
# CFD1
#
# | x      ndof | drag  N/m | lift N/m |
# | ----------- | --------- | -------- |
# | 1    83,423 | 14.292127 | 1.118669 |
# | 2   330,297 | 14.292488 | 1.118578 |
# | 4 1,311,820 | 14.292672 | 1.118978 |
# | <reference> | 14.29     | 1.119    |
#
# CFD2
#
# | x      ndof |  drag  N/m |  lift N/m |
# | ----------- | ---------- | --------- |
# | 1    83,423 | 136.691171 | 10.511853 |
# | 2   330,297 | 136.698533 | 10.525791 |
# | 4 1,311,820 | 136.699862 | 10.532045 |
# | <reference> | 136.7      | 10.53     |
#
# CFD3
#
# | x      ndof | drag          N/m @ Hz     | lift            N/m @ Hz     |
# | ----------- | -------------------------- | ---------------------------- |
# | 1    83,423 | 439.5852 ± 5.6856 @ 4.4170 | -11.2158 ± 440.8414 @ 4.4170 |
# | <reference> | 439.45   ± 5.6183 @ 4.3956 | -11.893  ± 437.81   @ 4.3956 |
#
# CSM1
#
# | x     ndof* | ux of A mm | uy of A mm |
# | ----------- | ---------- | ---------- |
# | 1    86,414 |  -7.183355 | -66.079644 |
# | 2   341,706 |  -7.186132 | -66.094239 |
# | 4 1,358,614 |  -7.187190 | -66.099802 |
# | <reference> |  -7.187    | -66.10     |
#
# CSM2
#
# | x     ndof* | ux of A mm | uy of A mm |
# | ----------- | ---------- | ---------- |
# | 1    86,414 |  -0.468709 | -16.967829 |
# | 2   341,706 |  -0.468896 | -16.971715 |
# | 4 1,358,614 |  -0.468967 | -16.973193 |
# | <reference> |  -0.4690   | -16.97     |
#
# CSM3
#
# | x     ndof* | ux of A         mm @ Hz     | uy of A         mm @ Hz     |
# | ----------- | --------------------------- | --------------------------- |
# | 1   174,648 | -14.6315 ± 14.6355 @ 1.0936 | -64.4979 ± 65.1812 @ 1.0941 |
# | <reference> | -14.305  ± 14.305  @ 1.0995 | -63.607  ± 65.160  @ 1.0995 |
#
# FSI1
#
# | x      ndof | ux of A mm | uy of A mm |  drag N/m | lift N/m |
# | ----------- | ---------- | ---------- | --------- | -------- |
# | 1   169,837 |   0.022673 |   0.817920 | 14.293220 | 0.764822 |
# | 2   672,003 |   0.022678 |   0.817709 | 14.293580 | 0.764745 |
# | 3 1,506,762 |   0.022676 |   0.819155 | 14.293711 | 0.764806 |
# | <reference> |   0.0227   |   0.8209   | 14.295    | 0.7638   |
#
# FSI2
#
# | x      ndof | ux of A     mm @ Hz  | uy of A  mm @ Hz  | drag       N/m @ Hz  | lift     N/m @ Hz  |
# | ----------- | -------------------- | ----------------- | -------------------- | ------------------ |
# | <reference> | -14.58 ± 12.44 @ 3.8 | 1.23 ± 80.6 @ 2.0 | 208.83 ± 73.75 @ 3.8 | 0.88 ± 234.2 @ 2.0 |
#
# FSI3
#
# | x      ndof | ux of A    mm @ Hz   | uy of A   mm @ Hz  | drag      N/m @ Hz   | lift      N/m @ Hz  |
# | ----------- | -------------------- | ------------------ | -------------------- | ------------------- |
# | <reference> |  -2.69 ± 2.53 @ 10.9 | 1.48 ± 34.38 @ 5.3 | 457.3 ± 22.66 @ 10.9 | 2.22 ± 149.78 @ 5.3 |
#
# (*) The dof counts of the CSM simulations include the degrees of freedom
# associated with the deformation of the fluid domain, which do not play a part
# in the physics of the beam. They were made an integral part of the solid
# problem for sake of overall simplicity.


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
    min_elemsize: Length = Length('2mm')
    max_elemsize: Length = Length('2cm')

    def generate_mesh(self):
        'Call gmsh to generate mesh and return topo, geom tuple.'''

        u = Length('m') # temporary reference length

        topo, geom = gmsh(Path(__file__).parent/'turek.geo', dimension=2, order=2, numbers={
            'channel_length': self.channel_length/u,
            'channel_height': self.channel_height/u,
            'x_center': self.x_center/u,
            'y_center': self.y_center/u,
            'cylinder_radius': self.cylinder_radius/u,
            'structure_length': self.structure_length/u,
            'structure_thickness': self.structure_thickness/u,
            'min_elemsize': self.min_elemsize/u,
            'max_elemsize': self.max_elemsize/u})

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

    density: Density = Density('1kg/L')
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

    timestep: Time = Time('10ms') # simulation time step size
    endtime: Time = Time('10s') # total duration of the simulation
    init: Time = Time('2s') # duration of the ramp-up phase
    window: Time = Time('1s') # sliding window length for time series plots
    theta: float = 0.5 # theta method parameter: 1 for fully implicit, 0.5 for Crank-Nicolson

    def __post_init__(self):
        self.timeseries = defaultdict(deque(maxlen=round(self.window / self.timestep)).copy)

    def ramp_up(self, t):
        'Return inflow ramp-up scale factor at given time.'

        return .5 - .5 * numpy.cos(numpy.pi * min(t / self.init, 1))

    @property
    def times(self):
        'Return all configured time steps for the simulation.'

        return (self.theta + numpy.arange(self.endtime / self.timestep)) * self.timestep

    def value_and_rate(self, v1, subs):
        'Given value at end of time step, return theta value and time derivative.'

        v0 = function.replace_arguments(v1, subs)
        return self.theta * v1 + (1 - self.theta) * v0, (v1 - v0) / self.timestep

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


def main(domain: Domain = Domain(), solid: Optional[Solid] = None, fluid: Optional[Fluid] = None, dynamic: Optional[Dynamic] = None):
    'Simulate solid and/or fluid dynamics/statics problem and plot results.'

    assert solid or fluid, 'nothing to compute'

    if fluid:
        log.info('Re:', fluid.reynolds(2 * domain.cylinder_radius))
        if solid:
            log.info('Ae:', solid.young() / fluid.density / fluid.velocity**2)
            log.info('β:',  solid.density / fluid.density)

    topo, geom = domain.generate_mesh()

    bezier = topo['fluid'].sample('bezier', 3)
    bezier = bezier.subset(bezier.eval(geom[0]) < 2*domain.channel_height)
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
        ns.d1 = topo.field('d', btype='std', degree=2, shape=(2,)) * domain.cylinder_radius # deformation at the end of the timestep
        if dynamic:
            ns.v1 = topo.field('v', btype='std', degree=2, shape=(2,)) * numpy.sqrt(solid.shear_modulus / solid.density) # velocity at the end of the timestep
            ns.dθ, ns.v = dynamic.value_and_rate(ns.d1, 'd:d0')
            ns.vθ, ns.a = dynamic.value_and_rate(ns.v1, 'v:v0')
        else:
            ns.dθ = ns.d1
            ns.a = Acceleration.wrap(function.zeros((2,)))

        # Geometry at time θ
        ns.xθ_i = 'xref_i + dθ_i'
        ns.Fθ_ij = '∇ref_j(xθ_i)' # deformation gradient tensor
        ns.Cθ_ij = 'Fθ_ki Fθ_kj' # right Cauchy-Green deformation tensor
        ns.Eθ_ij = '.5 (Cθ_ij - δ_ij)' # Green-Lagrangian strain tensor
        ns.Sθ_ij = 'λs Eθ_kk δ_ij + 2 μs Eθ_ij' # 2nd Piola–Kirchhoff stress
        ns.Pθ_ij = 'Fθ_ik Sθ_kj' # 1st Piola–Kirchhoff stress

        # Geometry at end of time step
        ns.x1 = ns.xref + ns.d1
        ns.F1_ij = '∇ref_j(x1_i)' # deformation gradient tensor
        ns.C1_ij = 'F1_ki F1_kj' # right Cauchy-Green deformation tensor
        ns.J1 = numpy.linalg.det(ns.F1)

        # Momentum balance: ρs (a - g) = div P
        ns.dtest = function.replace_arguments(ns.d1, 'd:dtest') / (solid.shear_modulus * domain.cylinder_radius**2)
        res += topo['solid'].integral('(∇ref_j(dtest_i) Pθ_ij + dtest_i ρs (a_i - g_i)) dVref' @ ns, degree=4)

        # In the momentum balance above, the only test and trial dofs involved
        # are those that have support on the solid domain. The remaining trial
        # dofs will follow from minimizing a mesh energy functional, using the
        # solid deformation as a boundary constraint so that the continuation
        # problem does not feed back into the physics. To achieve this within a
        # monolythic setting, we establish a boolean array 'dfluid' to select
        # all dofs that are exclusively supported by the fluid domain, and use
        # it to restrict the minimization problem to the remaining dofs via the
        # replace_arguments operation.
        mesh_energy = topo['fluid'].integral('C1_kk - 2 log(J1)' @ ns, degree=4) # Neo-Hookean with jacobian based stiffness
        sqr = topo['solid'].integral('d1_k d1_k' @ ns, degree=4) / 'm2'
        dfluid = numpy.isnan(System(sqr, trial='d').solve_constraints(droptol=1e-10)['d']) # true if dof is not supported by solid domain
        res += function.linearize(mesh_energy, {'d': function.arguments_for(res)['dtest'] * dfluid})

        if dynamic:
            # Compatibility: v = vθ
            ns.vtest = function.replace_arguments(ns.v1, 'v:vtest') * (solid.density / solid.shear_modulus / domain.cylinder_radius**2)
            res += topo.integral('vtest_i (v_i - vθ_i) dVref' @ ns, degree=4)

            # Zero initial velocity
            args['v'] = numpy.zeros(function.arguments_for(res)['v'].shape)

        # Deformation constraints: fixed at exterior boundary and cylinder
        sqr = topo.boundary.integral(ns.d1 @ ns.d1, degree=4) / 'm2'
        cons = System(sqr, trial='d').solve_constraints(droptol=1e-10, constrain=cons)

        # Zero initial deformation
        args['d'] = numpy.zeros(function.arguments_for(res)['d'].shape)

    else: # fully rigid solid

        ns.xθ = ns.x1 = ns.xref
        ns.vθ = ns.v1 = Velocity.wrap(function.zeros((2,)))
        ns.a = Acceleration.wrap(function.zeros((2,)))

    if fluid:

        ns.ρf = fluid.density
        ns.μf = fluid.viscosity

        ns.define_for('xθ', gradient='∇θ', normal='nθ', jacobians=('dVθ', 'dSθ'))
        ns.define_for('x1', gradient='∇1', normal='n1', jacobians=('dV1', 'dS1'))

        # The fluid velocity is defined relative to the mesh velocity, which
        # helps to impose the no slip condition on the elastic structure.
        # Furthermore, the material derivative convects only with this relative
        # component.
        ns.urel1 = topo['fluid'].field('u', btype='std', degree=2, shape=(2,)) * fluid.velocity
        if dynamic:
            ns.urelθ, ns.arel = dynamic.value_and_rate(ns.urel1, 'u:u0')
            ns.uθ_i = 'vθ_i + urelθ_i'
            ns.u1_i = 'v1_i + urel1_i'
            ns.DuDt_i = 'a_i + arel_i + ∇θ_j(uθ_i) urelθ_j' # material derivative
        else:
            ns.uθ = ns.u1 = ns.urel1
            ns.DuDt_i = '∇θ_j(uθ_i) uθ_j'

        ns.p = topo['fluid'].field('p', btype='std', degree=1) * fluid.viscosity * fluid.velocity / domain.cylinder_radius
        ns.σθ_ij = 'μf (∇θ_j(uθ_i) + ∇θ_i(uθ_j)) - p δ_ij' # fluid stress tensor

        # Project Posseuille inflow at inlet (exact because of quadratic
        # velocity basis), parallel outflow at outlet, and no-slip conditions
        # for the relative velocity at the remaining boundaries.
        y = ns.xref[1] / domain.channel_height
        uin = 6 * fluid.velocity * y * (1 - y) # Posseuille profile
        sqr = topo['fluid'].boundary['wall,cylinder,structure'].integral('urel1_k urel1_k' @ ns, degree=4) / 'm2/s2'
        sqr += topo['fluid'].boundary['inlet'].integral((ns.urel1[0] - uin)**2 * ns.dSref, degree=4) / 'm3/s2'
        sqr += topo['fluid'].boundary['inlet,outlet'].integral(ns.urel1[1]**2, degree=4) / 'm2/s2'
        cons = System(sqr, trial='u').solve_constraints(droptol=1e-10, constrain=cons) # exact projection
        ucons = cons['u'] # save for ramp-up phase

        # Momentum balance: ρf Du/Dt = ∇σ => ∀v ∫ (v ρf Du/Dt + ∇v:σ) = ∮ v·σ·n = 0
        ns.utest = function.replace_arguments(ns.urel1, 'u:utest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('(utest_i ρf DuDt_i + ∇θ_j(utest_i) σθ_ij) dVθ' @ ns, degree=4)

        # Incompressibility: div u = 0
        ns.ptest = function.replace_arguments(ns.p, 'p:ptest') / fluid.viscosity / fluid.velocity**2
        res += topo['fluid'].integral('ptest ∇1_k(u1_k) dV1' @ ns, degree=4)

        if solid:
            # The action of fluid stress on the solid is imposed weakly by
            # lifting the test functions into the fluid domain, using the
            # identity ∮ d·σ·n = ∫ ∇(d·σ) = ∫ (∇d:σ + d·∇σ) = ∫ (∇d:σ + ρf
            # d·Du/Dt). We need the inverse of the dfluid mask to exclude dofs
            # without support on the boundary.
            dsolid = ~dfluid # true if dof is (partially) supported by solid domain
            res += function.replace_arguments(topo['fluid'].integral('(dtest_i ρf DuDt_i + ∇θ_j(dtest_i) σθ_ij) dVθ' @ ns, degree=4),
                {'dtest': function.arguments_for(res)['dtest'] * dsolid})

        # The net force will likewise be evaluated weakly, via the identity F =
        # ∮ σ·n = ∮ λ σ·n = ∫ ∇(λ σ) = ∫ (∇λ·σ + λ ∇σ) = ∫ (∇λ·σ + λ ρf Du/Dt),
        # with λ = 1 on the boundary and lifted into the fluid domain.
        lift = topo['fluid'].field('lift', btype='std', degree=2) # same basis as urel, but scalar
        sqr = topo['fluid'].boundary['cylinder,structure'].integral((lift - 1)**2, degree=4)
        lcons = System(sqr, trial='lift').solve_constraints(droptol=1e-10)
        ns.λ = function.replace_arguments(lift, {'lift': numpy.nan_to_num(lcons['lift'])})
        Fθ = topo['fluid'].integral('-(∇θ_j(λ) σθ_ij + λ ρf DuDt_i) dVθ' @ ns, degree=4)

        # Zero initial velocity
        args['u'] = numpy.zeros(function.arguments_for(res)['u'].shape)

        u_bz = function.factor(bezier(ns.uθ))
        p_bz = function.factor(bezier(ns.p)) - topo.points['B'].sample('gauss', 1)(ns.p)[0]

    x_bz = function.factor(bezier(ns.xθ))
    x_bbz = function.factor(bbezier(ns.xθ))

    trial = 'updv'[0 if fluid else 2:2 if not solid else 3 if not dynamic else 4]
    system = System(res, trial=list(trial), test=[t+'test' for t in trial])

    ret = [None, None] # for unit tests only

    for tθ in log.iter.fraction('timestep', dynamic.times) if dynamic else [Time.wrap(float('inf'))]:

        if dynamic:
            if solid:
                args['d0'] = args['d']
                args['v0'] = args['v']
            if fluid:
                args['u0'] = args['u']
                cons['u'] = ucons * dynamic.ramp_up(tθ + dynamic.timestep * (1 - dynamic.theta)) # constrain inflow at end of time step

        args = system.solve(constrain=cons, arguments=args, tol=1e-9)

        x, xb = function.eval([x_bz, x_bbz], **args)
        if fluid:
            u, p = function.eval([u_bz, p_bz], **args)
            with export.mplfigure('solution.jpg', dpi=150) as fig:
                pstep = 12 * fluid.viscosity * fluid.velocity / domain.channel_height
                ax = fig.add_subplot(111, title=f'flow at t={tθ:.3s}, pressure contours every {pstep:.0Pa}', ylabel='[m]')
                vmax = 2 * fluid.velocity * (dynamic.ramp_up(tθ) if dynamic else 1)
                im = export.triplot(ax, x/'m', numpy.linalg.norm(u/'m/s', axis=1), tri=bezier.tri, cmap='inferno', clim=(0, vmax/'m/s'))
                ax.tricontour(*(x/'m').T, bezier.tri, p/pstep, numpy.arange(*numpy.quantile(numpy.ceil(p / pstep), [0,1])),
                    colors='white', linestyles='solid', linewidths=1, alpha=.33)
                fig.colorbar(im, orientation='horizontal', label=f'velocity [m/s]')
                export.triplot(ax, xb/'m', hull=bbezier.tri, linewidth=1)
                ax.set_xlim(0, 2*domain.channel_height/'m')
                ax.set_ylim(0, domain.channel_height/'m')

            D, L = ret[0] = function.eval(Fθ, arguments=args)
            log.info(f'lift: {L:N/m}')
            log.info(f'drag: {D:N/m}')
            if dynamic:
                with export.mplfigure('force.jpg', dpi=150) as fig:
                    dynamic.add_and_plot('lift [N/m]', tθ/'s', L/'N/m', ax=fig.add_subplot(211))
                    dynamic.add_and_plot('drag [N/m]', tθ/'s', D/'N/m', ax=fig.add_subplot(212, xlabel='time [s]'))

        if solid:
            if not fluid:
                with export.mplfigure('deformation.jpg', dpi=150) as fig:
                    ax = fig.add_subplot(111, title=f'deformation at t={tθ:.3s}', ylabel='[m]')
                    export.triplot(ax, x/'m', hull=bezier.hull)
                    export.triplot(ax, xb/'m', hull=bbezier.tri, linewidth=1)
                    ax.set_xlim(0, 2*domain.channel_height/'m')
                    ax.set_ylim(0, domain.channel_height/'m')

            ux, uy = ret[1] = topo.points['A'].sample('gauss', 1).eval(ns.dθ, arguments=args)[0]
            log.info(f'uy: {uy:mm}')
            log.info(f'ux: {ux:mm}')
            if dynamic:
                with export.mplfigure('tip-displacement.jpg', dpi=150) as fig:
                    dynamic.add_and_plot('uy [mm]', tθ/'s', uy/'mm', ax=fig.add_subplot(211))
                    dynamic.add_and_plot('ux [mm]', tθ/'s', ux/'mm', ax=fig.add_subplot(212, xlabel='time [s]'))

    return ret


def CFD1(elemsize=Length('2mm'), coarsening=10.):
    log.info('benchmark CFD1')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        fluid=Fluid(velocity=Velocity('0.2m/s')))


def CFD2(elemsize=Length('2mm'), coarsening=10.):
    log.info('benchmark CFD2')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        fluid=Fluid())


def CFD3(elemsize=Length('2mm'), coarsening=10., timestep=Time('10ms')):
    log.info('benchmark CFD3')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        fluid=Fluid(velocity=Velocity('2m/s')),
        dynamic=Dynamic(timestep=timestep, window=Time('0.6s')))


def CSM1(elemsize=Length('2mm'), coarsening=10.):
    log.info('benchmark CSM1')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(gravity=Acceleration('2m/s2')))


def CSM2(elemsize=Length('2mm'), coarsening=10.):
    log.info('benchmark CSM2')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(gravity=Acceleration('2m/s2'), shear_modulus=Pressure('2MPa')))


def CSM3(elemsize=Length('2mm'), coarsening=10., timestep=Time('10ms')):
    log.info('benchmark CSM3')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(gravity=Acceleration('2m/s2')),
        dynamic=Dynamic(timestep=timestep, window=Time('2s')))


def FSI1(elemsize=Length('2mm'), coarsening=10.):
    log.info('benchmark FSI1')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(),
        fluid=Fluid(velocity=Velocity('0.2m/s')))


def FSI2(elemsize=Length('2mm'), coarsening=10., timestep=Time('5ms'), theta=.5):
    log.info('benchmark FSI2')
    # To compute the FSI2 benchmark at a larger timestep you need to introduce
    # some numerical dissipation for stability via the theta parameter. For
    # example, a timestep of 10ms requires a theta of at least 0.51; without it
    # onset of instability will show around the 15 second mark.
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(density=Density('10kg/L')),
        fluid=Fluid(),
        dynamic=Dynamic(timestep=timestep, endtime=Time('35s'), theta=theta))


def FSI3(elemsize=Length('2mm'), coarsening=10., timestep=Time('5ms'), theta=.5):
    log.info('benchmark FSI3')
    main(
        domain=Domain(min_elemsize=elemsize, max_elemsize=coarsening*elemsize),
        solid=Solid(shear_modulus=Pressure('2MPa')),
        fluid=Fluid(velocity=Velocity('2m/s')),
        dynamic=Dynamic(timestep=timestep, window=Time('.5s'), endtime=Time('20s'), theta=theta))


class test(testing.TestCase):

    domain = Domain(min_elemsize=Length('2cm'), max_elemsize=Length('8cm'), channel_length=Length('1m'))

    def test_csm(self):
        DL, uxy = main(self.domain, solid=Solid(gravity=Acceleration('1m/s2')))
        self.assertEqual(DL, None)
        self.assertAllAlmostEqual(uxy/'mm', [-1.82506, -33.41303], delta=1e-4)

    def test_cfd(self):
        DL, uxy = main(self.domain, fluid=Fluid(viscosity=Viscosity('100Pa*s')))
        self.assertAllAlmostEqual(DL/'N/m', [5463.0, 110.3], delta=2)
        self.assertEqual(uxy, None)

    def test_fsi(self):
        DL, uxy = main(self.domain, solid=Solid(shear_modulus=Pressure('1GPa')), fluid=Fluid(viscosity=Viscosity('100Pa*s')))
        self.assertAllAlmostEqual(DL/'N/m', [5463.0, 110.6], delta=2)
        self.assertAllAlmostEqual(uxy/'mm', [0.008, -0.006], delta=1e-2)

    def test_dyncsm(self):
        DL, uxy = main(self.domain, solid=Solid(gravity=Acceleration('10m/s2')), dynamic=Dynamic(endtime=Time('10ms')))
        self.assertEqual(DL, None)
        self.assertAllAlmostEqual(uxy/'mm', [-0.00011313, -0.24959199], delta=1e-7)

    def test_dyncfd(self):
        DL, uxy = main(self.domain, fluid=Fluid(viscosity=Viscosity('100Pa*s')), dynamic=Dynamic(endtime=Time('10ms')))
        self.assertAllAlmostEqual(DL/'N/m', [0.4411, -0.0032], delta=1e-3)
        self.assertEqual(uxy, None)

    def test_dynfsi(self):
        DL, uxy = main(self.domain, solid=Solid(), fluid=Fluid(), dynamic=Dynamic(endtime=Time('10ms')))
        self.assertAllAlmostEqual(DL/'N/m', [0.1718, -0.0004], delta=1e-3)
        self.assertAllAlmostEqual(uxy/'mm', [0.0000125, -0.0000001], delta=1e-6)


if __name__ == '__main__':
    cli.choose(main, CFD1, CFD2, CFD3, CSM1, CSM2, CSM3, FSI1, FSI2, FSI3)
