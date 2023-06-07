#! /usr/bin/env python3
#
# In this script we solve the Cahn-Hilliard equation, which models the unmixing
# of two phases (φ=+1 and φ=-1) under the influence of surface tension. It is a
# mixed equation of two scalar equations for phase φ and chemical potential η::
#
#     dφ/dt = -div(J(η))
#     η = ψ'(φ) σ / ε - σ ε Δ(φ)
#
# along with constitutive relations for the flux vector J = -M ∇η and the
# double well potential ψ = .25 (φ² - 1)², and subject to boundary conditions
# ∂ₙφ = -σd / σ ε and ∂ₙη = 0. Parameters are the interface thickness ε, fluid
# surface tension σ, differential wall surface tension σd, and mobility M.
#
# Cahn-Hilliard is a diffuse interface model, which means that phases do not
# separate sharply, but instead form a transition zone between the phases. The
# transition zone has a thickness proportional to ε, as is readily confirmed in
# one dimension, where a steady state solution on an infinite domain is formed
# by η(x) = 0, φ(x) = tanh(x / √2 ε).
#
# The main driver of the unmixing process is the double well potential ψ that
# is proportional to the mixing energy, taking its minima at the pure phases
# φ=+1 and φ=-1. The interface between the phases adds an energy contribution
# proportional to its length. At the wall we have a phase-dependent fluid-solid
# energy. Over time, the system minimizes the total energy::
#
#     E(φ) = ∫_Ω ψ(φ) σ / ε + ∫_Ω .5 σ ε ‖∇φ‖² + ∫_Γ (σm + φ σd)
#            \                \                  \
#             mixing energy    interface energy   wall energy
#
# Proof: the time derivative of E followed by substitution of the strong form
# and boundary conditions yields dE/dt = ∫_Ω η dφ/dt = -∫_Ω M ‖∇η‖² ≤ 0. □
#
# Switching to discrete time we set dφ/dt = (φ - φ0) / dt and add a stabilizing
# perturbation term δψ(φ, φ0) to the doube well potential for reasons outlined
# below. This yields the following time discrete system::
#
#     φ = φ0 - dt div(J(η))
#     η = (ψ'(φ) + δψ'(φ, ψ0)) σ / ε - σ ε Δ(φ)
#
# with the equivalent weak formulation::
#
#     ∂/∂η ∫_Ω [ .5 dt J(η)·∇η - η (φ - φ0) ] = 0
#     ∂/∂φ ∫_Ω [ E(φ) + δψ(φ, φ0) σ / ε - η φ ] = 0
#
# For stability we wish for the perturbation δψ to be such that the time
# discrete system preserves the energy dissipation property E(φ) ≤ E(φ0) for
# any timestep dt. To derive suitable perturbation terms to this effect, we
# define without loss of generality δψ'(φ, φ0) = .5 (φ - φ0) f(φ, φ0) and
# derive the following condition for unconditional stability::
#
#     E(φ) - E(φ0) = ∫_Ω .5 (1 - φ² - .5 (φ + φ0)² - f(φ, φ0)) (φ - φ0)² σ / ε
#                  - ∫_Ω (.5 σ ε ‖∇φ - ∇φ0‖² + dt M ‖∇η‖²) ≤ 0
#
# The inequality holds true if the perturbation f is bounded from below such
# that f(φ, φ0) ≥ 1 - φ² - .5 (φ + φ0)². To keep the energy minima at the pure
# phases we additionally impose that f(±1, φ0) = 0, and select 1 - φ² as a
# suitable upper bound which we will call "nonlinear".
#
# We next observe that η is linear in φ if f(φ, φ0) = g(φ0) - φ² - (φ + φ0)²
# for any function g, which dominates if g(φ0) ≥ 1 + .5 (φ + φ0)². While this
# cannot be made to hold true for all φ, we make it hold for -√2 ≤ φ, φ0 ≤ √2
# by defining g(φ0) = 2 + 2 ‖φ0‖ + φ0², which we will call "linear". This
# scheme further satisfies a weak minima preservation f(±1, ±‖φ0‖) = 0.
#
# We have thus arrived at the three stabilization schemes implemented here:
#
# - nonlinear: f(φ, φ0) = 1 - φ²
# - linear: f(φ, φ0) = 2 + 2 ‖φ0‖ - 2 φ (φ + φ0)
# - none: f(φ, φ0) = 0 (not unconditionally stable)
#
# The stab enum in this script defines the schemes in terms of δψ to allow
# Nutils to construct the residuals through automatic differentiation.
#
# NOTE: This script uses dimensional quantities and requires the nutils.SI
# module, which is installed separate from the the core nutils.

from nutils import mesh, function, solver, numeric, export, cli, testing
from nutils.expression_v2 import Namespace
from nutils.SI import Length, Time, Density, Tension, Energy, Pressure, Velocity, parse
import numpy
import itertools
import enum
import treelog as log


class stab(enum.Enum):
    nonlinear = '.25 dφ^2 (1 - φ^2 + φ dφ (2 / 3) - dφ^2 / 6)'
    linear = '.25 dφ^2 (2 + 2 abs(φ0) - (φ + φ0)^2)'
    none = '0'  # not unconditionally stable


def main(size: Length, epsilon: Length, mobility: Time/Density, stens: Tension,
         wtensn: Tension, wtensp: Tension, nelems: int, etype: str, degree: int,
         timestep: Time, tol: Energy/Length, endtime: Time, seed: int, circle: bool,
         stab: stab, showflux: bool):
    '''
    Cahn-Hilliard equation on a unit square/circle.

    .. arguments::

       size [10cm]
         Domain size.
       epsilon [2mm]
         Interface thickness; defaults to an automatic value based on the
         configured mesh density if left unspecified.
       mobility [1mL*s/kg]
         Mobility.
       stens [50mN/m]
         Surface tension.
       wtensn [30mN/m]
         Wall surface tension for phase -1.
       wtensp [20mN/m]
         Wall surface tension for phase +1.
       nelems [0]
         Number of elements along domain edge. When set to zero a value is set
         automatically based on the configured domain size and epsilon.
       etype [rectilinear]
         Type of elements (square/triangle/mixed).
       degree [2]
         Polynomial degree.
       timestep [.5s]
         Time step.
       tol [1nJ/m]
         Newton tolerance.
       endtime [1min]
         End of the simulation.
       seed [0]
         Random seed for the initial condition.
       circle [no]
         Select circular domain as opposed to a unit square.
       stab [linear]
         Stabilization method (linear/nonlinear/none).
       showflux [no]
         Overlay flux vectors on phase plot
    '''

    nmin = round(size / epsilon)
    if nelems <= 0:
        nelems = nmin
        log.info('setting nelems to {}'.format(nelems))
    elif nelems < nmin:
        log.warning('mesh is too coarse, consider increasing nelems to {:.0f}'.format(nmin))

    log.info('contact angle: {:.0f}°'.format(numpy.arccos((wtensn - wtensp) / stens) * 180 / numpy.pi))

    if circle:
        domain, geom = mesh.unitcircle(nelems, etype)
        geom = (geom + 1) / 2
    else:
        domain, geom = mesh.unitsquare(nelems, etype)

    bezier = domain.sample('bezier', 5)  # sample for surface plots
    grid = domain.locate(geom, numeric.simplex_grid([1, 1], 1/40), maxdist=1/nelems, skip_missing=True, tol=1e-5)  # sample for quivers
    basis = domain.basis('std', degree=degree)

    ns = Namespace()
    ns.x = size * geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.add_field(('φ', 'φ0'), basis)
    ns.add_field('η', basis * stens / epsilon) # basis scaling to give η the required unit
    ns.ε = epsilon
    ns.σ = stens
    ns.σmean = (wtensp + wtensn) / 2
    ns.σdiff = (wtensp - wtensn) / 2
    ns.σwall = 'σmean + φ σdiff'
    ns.dφ = 'φ - φ0'
    ns.ψ = '.25 (φ^2 - 1)^2'
    ns.δψ = stab.value
    ns.M = mobility
    ns.J_i = '-M ∇_i(η)'
    ns.dt = timestep

    nrg_mix = domain.integral('(ψ σ / ε) dV' @ ns, degree=7)
    nrg_iface = domain.integral('.5 σ ε ∇_k(φ) ∇_k(φ) dV' @ ns, degree=7)
    nrg_wall = domain.boundary.integral('σwall dS' @ ns, degree=7)
    nrg = nrg_mix + nrg_iface + nrg_wall + domain.integral('(δψ σ / ε - η dφ + .5 dt J_k ∇_k(η)) dV' @ ns, degree=7)

    numpy.random.seed(seed)
    args = dict(φ=numpy.random.normal(0, .5, basis.shape)) # initial condition

    with log.iter.fraction('timestep', range(round(endtime / timestep))) as steps:
        for istep in steps:

            E = numpy.stack(function.eval([nrg_mix, nrg_iface, nrg_wall], **args))
            log.user('energy: {0:,.0μJ/m} ({1[0]:.0f}% mixture, {1[1]:.0f}% interface, {1[2]:.0f}% wall)'.format(numpy.sum(E), 100*E/numpy.sum(E)))

            args['φ0'] = args['φ']
            args = solver.optimize(['φ', 'η'], nrg / tol, arguments=args, tol=1)

            with export.mplfigure('phase.png') as fig:
                ax = fig.add_subplot(aspect='equal', xlabel='[mm]', ylabel='[mm]')
                x, φ = bezier.eval(['x_i', 'φ'] @ ns, **args)
                im = ax.tripcolor(*(x/'mm').T, bezier.tri, φ, shading='gouraud', cmap='bwr')
                im.set_clim(-1, 1)
                fig.colorbar(im)
                if showflux:
                    x, J = grid.eval(['x_i', 'J_i'] @ ns, **args)
                    log.info('largest flux: {:.1mm/h}'.format(numpy.max(numpy.hypot(J[:, 0], J[:, 1]))))
                    ax.quiver(*(x/'mm').T, *(J/'m/s').T, color='r')
                    ax.quiver(*(x/'mm').T, *-(J/'m/s').T, color='b')
                ax.autoscale(enable=True, axis='both', tight=True)

    return args

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.


if __name__ == '__main__':
    cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.


class test(testing.TestCase):

    def test_initial(self):
        args = main(size=parse('10cm'), epsilon=parse('5cm'), mobility=parse('1μL*s/kg'),
                     stens=parse('50mN/m'), wtensn=parse('30mN/m'), wtensp=parse('20mN/m'), nelems=3,
                     etype='square', degree=2, timestep=parse('1h'), tol=parse('1nJ/m'),
                     endtime=parse('1h'), seed=0, circle=False, stab=stab.linear, showflux=True)
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ0'], '''
                eNoBYgCd/xM3LjTtNYs3MDcUyt41uc14zjo0LzKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3Tcnx
                L07NzTa4NlMyETREyrPIGMxYMl82VDbjy1/M8clZyf3IRjday6XLmMl6NRnJMF4tqQ==''')

    def test_square(self):
        args = main(size=parse('10cm'), epsilon=parse('5cm'), mobility=parse('1μL*s/kg'),
                     stens=parse('50mN/m'), wtensn=parse('30mN/m'), wtensp=parse('20mN/m'), nelems=3,
                     etype='square', degree=2, timestep=parse('1h'), tol=parse('1nJ/m'),
                     endtime=parse('2h'), seed=0, circle=False, stab=stab.linear, showflux=True)
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ'], '''
                eNoBYgCd/zE1EjX1NaA2+TXiMxkz0TS9NL01ajaRNZoxYNElNRM1LDUlNZQw0cqgysI1nTWcNN4xLsuk
                ybDJvDWaNTQ07s7nysnJ6ckPNQY1CzNozKjK58kOysQ0zTQKM73M3coVyjfKR9cuPg==''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoBYgCd/3TIkccBNkQ6IDqIN3/MF8cSx9Y02TmdOVHLMcecxxLIEjQUOAHOa8a1xWw3izb2M9UzPMc0
                xmnGpzibODY34tETyJHHp8hbyWU2xzZTydfIOsrNyo3Gi8jCyyXIm8hkzD3K1IAxtQ==''')

    def test_multipatchcircle(self):
        args = main(size=parse('10cm'), epsilon=parse('5cm'), mobility=parse('1μL*s/kg'),
                     stens=parse('50mN/m'), wtensn=parse('30mN/m'), wtensp=parse('20mN/m'), nelems=3,
                     etype='multipatch', degree=2, timestep=parse('1h'), tol=parse('1nJ/m'),
                     endtime=parse('2h'), seed=0, circle=True, stab=stab.linear, showflux=True)
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ'], '''
                eNoNyD9IlHEYB3BIUyI0HaSkyEkl4nzf3/M8EDk4Zw5x4BK5RAbRICWIUMshoaKBEaZ3mprhUi29w3ku
                ZUtH0PN93p9/eBvuriWEMghsuCk0x8+nIolsyD85kIKMyl05LXNyVR5KhXO8SZ+4SHU87brdyOWvvMq1
                rDROOddC1eBNUPR9Pm8TtoeRcDW4mfy41LN7favZT1mbL1sBt9BgGZyId23Y+hGiBmwJbuOcvUKikR5p
                UbtRjxea1Xmd1SE/EHfYGdpxmXAQVX2Z7Nva8S+gFRG/ZuK//JMjnqWZ8AEvUiku43fY6HPyTO6IcYEn
                2Ume0/yeTnHkInfQ9Z2WqIdaacMthptuKHyXGvbNpZX4i7XZUlBOffb7dtJG0Qvyz+OSrQXbqaq/YR9x
                Nm60NJowiBb8cY/cZHCN2/kxHfkJu+fqOzO4gBXLokkOOcsNPONou9fSXbn4fLyHX7iCDuxoXp/qE9Ti
                Ir7pso7pgr7V+/pBt3T92BXtxH8H7Mmc''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoBggF9/lU4VTgzOIo4jzhIOCc3Lzd5Ngw4xTf1N1o3YzV70II3ijZoNuIyZcpFybM1jzX0zgU28DVG
                zyLKzMjGyXfICsjfx73Hl8dxMzgzgctEylLKFsqEyU3I2snuyJvHZMc7yLzHKctJy6rKqci4yGDIrcqK
                yvLID8royADInMcIyJ7HI8jnx13HcMdax2fHctPiz7DJHzZENjI1V8jHx7AxHMpqx2PHpMgAyDk4SDgR
                OH84gThhONA3ejY/OJw3Y84NyYY2C8x1NqU2BjfJyQXKYy9rN303ATZmNnY3TTaMNmI0e8mGyf7KKjYz
                Nj80DDD+Mn3MhsqCM/bLsslNyRo3BzcuNUbLEMk7yDfI08nXyWjJu818zdTK+8hRyLPJt8jYx4/HEMiy
                x5wzXTNbzP02CTdQNmPKFslCNc7MRcjSx5DJZ8hiOG04JDjJN4020jBZySAwQsvxyJDInMlnyCzIxMes
                x3DHmMeAx1fHVMdfx0vHS8dsxzzHl8csxz3HSceixxfInGzEOA==''')
