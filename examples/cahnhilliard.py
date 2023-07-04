from nutils import mesh, function, solver, numeric, export, testing
from nutils.expression_v2 import Namespace
from nutils.SI import Length, Time, Density, Tension, Energy, Pressure, Velocity, parse
import numpy
import itertools
import enum
import treelog as log

# NOTE: This script uses dimensional quantities and requires the nutils.SI
# module, which is installed separate from the the core nutils.


class stab(enum.Enum):
    nonlinear = '.25 dφ^2 (1 - φ^2 + φ dφ (2 / 3) - dφ^2 / 6)'
    linear = '.25 dφ^2 (2 + 2 abs(φ0) - (φ + φ0)^2)'
    none = '0' # not unconditionally stable


def main(size: Length = parse('10cm'),
         epsilon: Length = parse('1mm'),
         mobility: Time/Density = parse('1mL*s/kg'),
         stens: Tension = parse('50mN/m'),
         wtensn: Tension = parse('30mN/m'),
         wtensp: Tension = parse('20mN/m'),
         nelems: int = 0,
         etype: str = 'rectilinear',
         degree: int = 1,
         timestep: Time = parse('.5s'),
         tol: Energy/Length = parse('1nJ/m'),
         endtime: Time = parse('1min'),
         seed: int = 0,
         circle: bool = True,
         stab: stab = stab.linear,
         showflux: bool = True):

    '''Unmixing of immiscible fluids

    Solves the Cahn-Hilliard equation, which models the unmixing of two phases
    (`φ=+1` and `φ=-1`) under the influence of surface tension. It is a mixed
    equation of two scalar equations for phase `φ` and chemical potential `η`:

        dφ/dt = -div(J(η))
        η = ψ'(φ) σ / ε - σ ε Δ(φ)

    along with constitutive relations for the flux vector `J = -M ∇η` and the
    double well potential `ψ = .25 (φ² - 1)²`, and subject to boundary conditions
    `∂ₙφ = -σd / σ ε` and `∂ₙη = 0`. Parameters are the interface thickness `ε`,
    fluid surface tension `σ`, differential wall surface tension `σd`, and
    mobility `M`.

    Cahn-Hilliard is a diffuse interface model, which means that phases do not
    separate sharply, but instead form a transition zone between the phases. The
    transition zone has a thickness proportional to `ε`, as is readily confirmed
    in one dimension, where a steady state solution on an infinite domain is
    formed by `η(x) = 0`, `φ(x) = tanh(x / √2 ε)`.

    The main driver of the unmixing process is the double well potential `ψ` that
    is proportional to the mixing energy, taking its minima at the pure phases
    `φ=+1` and `φ=-1`. The interface between the phases adds an energy
    contribution proportional to its length. At the wall we have a
    phase-dependent fluid-solid energy. Over time, the system minimizes the total
    energy:

        E(φ) := ∫_Ω ψ(φ) σ / ε + ∫_Ω .5 σ ε ‖∇φ‖² + ∫_Γ (σm + φ σd)
                \                \                  \ 
                 mixing energy    interface energy   wall energy

    Proof: the time derivative of `E` followed by substitution of the strong form
    and boundary conditions yields `dE/dt = ∫_Ω η dφ/dt = -∫_Ω M ‖∇η‖² ≤ 0`. □

    Switching to discrete time we set `dφ/dt = (φ - φ0) / dt` and add a
    stabilizing perturbation term `δψ(φ, φ0)` to the double well potential for
    reasons outlined below. This yields the following time discrete system:

        φ = φ0 - dt div(J(η))
        η = (ψ'(φ) + δψ'(φ, φ0)) σ / ε - σ ε Δ(φ)

    For stability we wish for the perturbation `δψ` to be such that the time
    discrete system preserves the energy dissipation property `E(φ) ≤ E(φ0)` for
    any timestep `dt`. To derive suitable perturbation terms to this effect, we
    define without loss of generality `δψ'(φ, φ0) = .5 (φ - φ0) f(φ, φ0)` and
    derive the following condition for unconditional stability:

        E(φ) - E(φ0) = ∫_Ω .5 (1 - φ² - .5 (φ + φ0)² - f(φ, φ0)) (φ - φ0)² σ / ε
                     - ∫_Ω (.5 σ ε ‖∇φ - ∇φ0‖² + dt M ‖∇η‖²) ≤ 0

    The inequality holds true if the perturbation `f` is bounded from below such
    that `f(φ, φ0) ≥ 1 - φ² - .5 (φ + φ0)²`. To keep the energy minima at the
    pure phases we additionally impose that `f(±1, φ0) = 0`, and select `1 - φ²`
    as a suitable upper bound which we will call "nonlinear".

    We next observe that `η` is linear in `φ` if `f(φ, φ0) = g(φ0) - φ² - (φ +
    φ0)²` for any function `g`, which dominates if `g(φ0) ≥ 1 + .5 (φ + φ0)²`.
    While this cannot be made to hold true for all `φ`, we make it hold for `-√2
    ≤ φ, φ0 ≤ √2` by defining `g(φ0) = 2 + 2 |φ0| + φ0²`, which we will call
    "linear". This scheme further satisfies a weak minima preservation `f(±1,
    ±|φ0|) = 0`.

    We have thus arrived at the three stabilization schemes implemented here:

    - nonlinear: `f(φ, φ0) = 1 - φ²`
    - linear: `f(φ, φ0) = 2 + 2 |φ0| - 2 φ (φ + φ0)`
    - none: `f(φ, φ0) = 0` (not unconditionally stable)

    Finally, we observe that the weak formulation:

        ∀ δη: ∫_Ω [ dt J(η)·∇δη - δη (φ - φ0) ] = 0
        ∀ δφ: ∫_Ω [ δφ (ψ'(φ) + δψ'(φ, φ0)) σ / ε + σ ε ∇(δφ)·∇(φ) ] = ∫_Γ -δφ σd

    is equivalent to the optimization problem `∂F/∂φ = ∂F/∂η = 0`, where

        F(φ, φ0, η) := E(φ) + ∫_Ω [ .5 dt J(η)·∇η + δψ(φ, φ0) σ / ε - η (φ - φ0) ]

    For this reason, the `stab` enum in this script defines the stabilizing term
    `δψ`, rather than `f`, allowing Nutils to construct the residuals through
    automatic differentiation using the `optimize` method.

    Parameters
    ----------
    size
        Domain size.
    epsilon
        Interface thickness; defaults to an automatic value based on the
        configured mesh density if left unspecified.
    mobility
        Mobility.
    stens
        Surface tension.
    wtensn
        Wall surface tension for phase -1.
    wtensp
        Wall surface tension for phase +1.
    nelems
        Number of elements along domain edge. When set to zero a value is set
        automatically based on the configured domain size and epsilon.
    etype
        Type of elements (square/triangle/mixed).
    degree
        Polynomial degree.
    timestep
        Time step.
    tol
        Newton tolerance.
    endtime
        End of the simulation.
    seed
        Random seed for the initial condition.
    circle
        Select circular domain as opposed to a unit square.
    stab
        Stabilization method (linear/nonlinear/none).
    showflux
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

    basis = domain.basis('std', degree=degree)
    bezier = domain.sample('bezier', 5)  # sample for surface plots
    if showflux:
        grid = domain.locate(geom, numeric.simplex_grid([1, 1], 1/40), maxdist=1/nelems, skip_missing=True, tol=1e-5)  # sample for quivers

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

    nrg_mix = domain.integral('(ψ σ / ε) dV' @ ns, degree=degree*4)
    nrg_iface = domain.integral('.5 σ ε ∇_k(φ) ∇_k(φ) dV' @ ns, degree=degree*4)
    nrg_wall = domain.boundary.integral('σwall dS' @ ns, degree=degree*2)
    nrg = nrg_mix + nrg_iface + nrg_wall + domain.integral('(δψ σ / ε - η dφ + .5 dt J_k ∇_k(η)) dV' @ ns, degree=degree*4)

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
                im = ax.tripcolor(*(x/'mm').T, bezier.tri, φ, shading='gouraud', cmap='coolwarm')
                im.set_clim(-1, 1)
                fig.colorbar(im)
                if showflux:
                    x, v = grid.eval(['x_i', 'φ J_i'] @ ns, **args)
                    log.info('largest flux: {:.2mm/s}'.format(numpy.max(numpy.hypot(v[:, 0], v[:, 1]))))
                    ax.quiver(*(x/'mm').T, *(v/'m/s').T)
                ax.autoscale(enable=True, axis='both', tight=True)

    return args


class test(testing.TestCase):

    def test_initial(self):
        args = main(epsilon=parse('5cm'), mobility=parse('1μL*s/kg'), nelems=3, degree=2, timestep=parse('1h'), endtime=parse('1h'), circle=False)
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ0'], '''
                eNoBYgCd/xM3LjTtNYs3MDcUyt41uc14zjo0LzKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3Tcnx
                L07NzTa4NlMyETREyrPIGMxYMl82VDbjy1/M8clZyf3IRjday6XLmMl6NRnJMF4tqQ==''')

    def test_square(self):
        args = main(epsilon=parse('5cm'), mobility=parse('1μL*s/kg'), nelems=3, degree=2, timestep=parse('1h'), endtime=parse('2h'), circle=False)
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ'], '''
                eNoBYgCd/zE1EjX1NaA2+TXiMxkz0TS9NL01ajaRNZoxYNElNRM1LDUlNZQw0cqgysI1nTWcNN4xLsuk
                ybDJvDWaNTQ07s7nysnJ6ckPNQY1CzNozKjK58kOysQ0zTQKM73M3coVyjfKR9cuPg==''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoBYgCd/3TIkccBNkQ6IDqIN4HMF8cSx9Y02DmdOVHLMcecxxLIEjQUOAHOa8a1xWw3izb1M9kzPMc0
                xmnGpzibODY359ETyJHHp8hbyWU2xzZSydfIOsrNyo3GjMjAyyXIm8hkzD3K1ggxvA==''')

    def test_multipatchcircle(self):
        args = main(epsilon=parse('5cm'), mobility=parse('1μL*s/kg'), nelems=3, etype='multipatch', degree=2, timestep=parse('1h'), endtime=parse('2h'))
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ'], '''
                eNoNyE9Ik3EYB3BwZohodpCSIk9NQub7/n7PA5IHu7Y6xKBbgUQK0mGUMIS6jIgmJRZitc3MEi/VxQVz
                XsouDcHn+7w///B22HYU/wTBOuwUmcfPp8Yhr/JfrnOJJ3iU2/g1X+YHXKMcrdnvVLYt9NwMmlTfBn2g
                ZhL71OZMl214H72yu+6KmtFdpPxFbzjcu3Rl59rmafdMe1xVS7iFdk2jKdjRcb0JHxGQhriDs/oeoRTk
                SMoyiJN4JVl5I7OSdLeDqJ6y2ybtj6Ahb8NDXTr+PLpRoEWy9If2qUCz9oV/n+ZsJajil9/hcvyS77JS
                iSbJcJEStGxbqWC+mHp/zc7bIdttV82cv2aS/ufYuOurLATr2qPzXjX2wx3qCZ1AHMbNBBVd8rZiDXdD
                v+FM0KEJdGIEXfhtHppJ7ypdpEf2yGV0zER60ziPBc2ik/9Rltpp2tituCb6c8G5YBcHGEAU21KUKXmC
                ZlzAT3knjyUvn+SefJVNWTl2TXrxHyMaybw=''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoBggF9/lU4VTgzOIo4kDhIOCY3Lzd5Ngw4xjf1N1o3YzV70II3ijZoNuIyZcpFybM1jjXxzgY28TVH
                zyLKzMjGyXfICsjfx73Hl8dyMzozgstEylLKFsqEyU3I2snuyJvHZMc7yLzHKctJy6rKqci4yGDIrcqK
                yvLID8royADInMcIyJ7HI8jnx13HcMdax2fHWNPez6/JHzZENjI1V8jHx7AxHMpqx2PHpMgAyDk4SDgR
                OH84gThhONA3ejY/OJw3Y84NyYY2C8x1NqU2BjfJyQTKYC9rN303ATZmNnY3TTaMNmI0e8mFyf3KKjYz
                Nj80DDD+Mn3MhsqCM/bLsslNyRo3BzcuNUbLEMk7yDfI0snXyWjJu817zdTK+8hRyLPJt8jYx4/HEMiy
                x5wzXTNbzP02CTdQNmLKFslCNc7MRcjSx5DJZ8hiOG04JDjJN4020jBZySIwQsvxyJDInMlnyCzIxMes
                x3DHmMeAx1fHVMdfx0vHS8dsxzzHl8csxz3HSceixxfIfsLEFg==''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Cahn-Hilliard
