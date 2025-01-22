from nutils import mesh, function, numeric, export, testing
from nutils.solver import System
from nutils.expression_v2 import Namespace
from nutils.SI import Length, Time, Density, Tension, Energy, Pressure, Velocity, parse
import numpy
import treelog as log


def main(size: Length = parse('10cm'),
         epsilon: Length = parse('1mm'),
         mobility: Time/Density = parse('1mL*s/kg'),
         stens: Tension = parse('50mN/m'),
         wtensn: Tension = parse('30mN/m'),
         wtensp: Tension = parse('20mN/m'),
         nelems: int = 0,
         etype: str = 'rectilinear',
         degree: int = 1,
         timestep: Time = parse('.1s'),
         tol: Energy/Length = parse('1nJ/m'),
         endtime: Time = parse('1min'),
         seed: int = 0,
         circle: bool = True,
         stable: bool = False,
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
                ╲                ╲                  ╲
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
    any timestep `dt`. To derive a suitable perturbation term to this effect, we
    define without loss of generality `δψ'(φ, φ0) = .5 (φ - φ0) f(φ, φ0)` and
    derive the following condition for unconditional stability:

        E(φ) - E(φ0) = ∫_Ω .5 (1 - φ² - .5 (φ + φ0)² - f(φ, φ0)) (φ - φ0)² σ / ε
                     - ∫_Ω (.5 σ ε ‖∇φ - ∇φ0‖² + dt M ‖∇η‖²) ≤ 0

    The inequality holds true if the perturbation `f` is bounded from below such
    that `f(φ, φ0) ≥ 1 - φ² - .5 (φ + φ0)²`. To keep the energy minima at the
    pure phases we additionally impose that `f(±1, φ0) = 0`, and select `1 - φ²`
    as a suitable upper bound. For unconditional stability we thus obtained the
    perturbation gradient `δψ'(φ, φ0) = .5 (φ - φ0) (1 - φ²)`.

    Finally, we observe that the weak formulation:

        ∀ δη: ∫_Ω [ dt J(η)·∇δη - δη (φ - φ0) ] = 0
        ∀ δφ: ∫_Ω [ δφ (ψ'(φ) + δψ'(φ, φ0)) σ / ε + σ ε ∇(δφ)·∇(φ) ] = ∫_Γ -δφ σd

    is equivalent to the optimization problem `∂F/∂φ = ∂F/∂η = 0`, where

        F(φ, φ0, η) := E(φ) + ∫_Ω [ .5 dt J(η)·∇η + δψ(φ, φ0) σ / ε - η (φ - φ0) ]

    For this reason, this script defines the stabilizing term `δψ`, rather than
    its derivative `δψ'`, allowing Nutils to construct the residuals through
    symbolic differentiation.

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
    stable
        Enable unconditional stability at the expense of dissipation.
    showflux
        Overlay flux vectors on phase plot.
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

    ns = Namespace()
    ns.x = geom * size
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.φ = domain.field('φ', btype='std', degree=degree)
    ns.dφ = ns.φ - function.replace_arguments(ns.φ, 'φ:φ0')
    ns.η = domain.field('η', btype='std', degree=degree) * (stens / epsilon)
    ns.dt = function.field('dt') * timestep
    ns.ε = epsilon
    ns.σ = stens
    ns.σmean = (wtensp + wtensn) / 2
    ns.σdiff = (wtensp - wtensn) / 2
    ns.σwall = 'σmean + φ σdiff'
    ns.ψ = '.25 (φ^2 - 1)^2'
    ns.δψ = '.25 dφ^2 (1 - φ^2 + 2 φ dφ / 3 - dφ^2 / 6)' if stable else '0'
    ns.M = mobility
    ns.J_i = '-M ∇_i(η)'
    ns.v_i = 'φ J_i'

    nrg_mix = function.factor(domain.integral('(ψ σ / ε) dV' @ ns, degree=degree*4))
    nrg_iface = function.factor(domain.integral('.5 σ ε ∇_k(φ) ∇_k(φ) dV' @ ns, degree=degree*4))
    nrg_wall = function.factor(domain.boundary.integral('σwall dS' @ ns, degree=degree*2))
    nrg = nrg_mix + nrg_iface + nrg_wall + function.factor(domain.integral('(δψ σ / ε - η dφ + .5 dt J_k ∇_k(η)) dV' @ ns, degree=degree*4))

    bezier = domain.sample('bezier', 5) # sample for surface plots
    bezier_x = bezier.eval(ns.x)
    bezier_φ = function.factor(bezier(ns.φ))
    if showflux:
        grid = domain.locate(geom, numeric.simplex_grid([1, 1], 1/40), maxdist=1/nelems, skip_missing=True, tol=1e-5) # sample for quivers
        grid_x = grid.eval(ns.x)
        grid_v = function.factor(grid(ns.v))

    system = System(nrg / tol, trial='φ,η')

    numpy.random.seed(seed)
    args = dict(φ=numpy.random.normal(0, .5, system.argshapes['φ'])) # initial condition

    with log.iter.fraction('timestep', range(round(endtime / timestep))) as steps:
        for istep in steps:

            E = numpy.stack(function.eval([nrg_mix, nrg_iface, nrg_wall], **args))
            log.user('energy: {0:,.0μJ/m} ({1[0]:.0f}% mixture, {1[1]:.0f}% interface, {1[2]:.0f}% wall)'.format(numpy.sum(E), 100*E/numpy.sum(E)))

            args = system.step(timestep=1., timesteparg='dt', suffix='0', arguments=args, tol=1, maxiter=5)

            with export.mplfigure('phase.png') as fig:
                ax = fig.add_subplot(aspect='equal', xlabel='[mm]', ylabel='[mm]')
                im = ax.tripcolor(*(bezier_x/'mm').T, bezier.tri, function.eval(bezier_φ, **args), shading='gouraud', cmap='coolwarm')
                im.set_clim(-1, 1)
                fig.colorbar(im)
                if showflux:
                    v = function.eval(grid_v, **args)
                    log.info('largest flux: {:.2mm/s}'.format(numpy.max(numpy.hypot(v[:, 0], v[:, 1]))))
                    ax.quiver(*(grid_x/'mm').T, *(v/'m/s').T)
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
                eNoBYgCd/y41EjX2NZ829DXcMxUz0jTANL41ajaNNZox/9EoNRY1LDUkNZAw1cqnysI1njWdNNkxMMuk
                ybDJuTWXNTE0587oysjJ58kSNQM1ATNqzKjK58kNytA00DQJM8bM38oTyjfKbwku0w==''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoBYgCd/1PIicccNko6IzqRNwzMEccMx/M05TmfOTTLMceexwzI1TMiOMLNZsa3xXU3rjZdNE4zO8cr
                xlrGoziaOEA3os8VyJLHk8hlyTw2sDZXydPISMoPy5zGe8i7yzfIncgAzGLKwgYwXw==''')

    def test_multipatchcircle(self):
        args = main(epsilon=parse('5cm'), mobility=parse('1μL*s/kg'), nelems=3, etype='multipatch', degree=2, timestep=parse('1h'), endtime=parse('2h'))
        with self.subTest('concentration'):
            self.assertAlmostEqual64(args['φ'], '''
                eNoNz01IlFEUBmByEcVsWkiBoKHYoh9nvnvPOa5GcCE1gqNjDZOBBUM1iSYYEf2JEGZE0SoIokWMCCYk
                0qZaJKE5Qvae+33XkUhwFi4iWhSKGEELc/vsnhG5IxekVSp8lk/SkPQLyQZ3cJbumwNSJUU+zLW019T5
                Io/xV3pvvyc+uln9RSX6a++aCb+lKZ2yA3bQ/IgH4UP9g2rzMzgUPIhT1O4yOqlP9Lqe1169qFll1KMZ
                GYziHUqYx1M8x0tM4y1m0Ojm9baKbuPDrp2zCVtjjsZPh7Nar60svEANlDc90brmuItreJX20zVzJRqV
                YUlJTIb5DXXZmOzwHN9kT5FdD/o4zXu4SF9si8mv1FLFFuxnkwselZPh+ImSPxKlQ+8KblMXltv863DR
                eW3SRXQmkk0t/lIYc0n1aDdTQUe8HOVdVis4Q0LP7Atz9diIK2iG23iN0lQ2Y8vH3Y1vneG29us/HHQD
                +htLeIVPuIdTyOEyHqMPKdza/ebRg25MYJ/+BxBNvrM=''')
        with self.subTest('chemical-potential'):
            self.assertAlmostEqual64(args['η'], '''
                eNoNzzFLW1EUAOAtDiLFB0ohggUVTG2i9HnvPS9DpkjAIg7WBG0GWxwEoRUqtkJQEETR4CKCSDsIYjEY                                                                                          
                nIriYojlnHPvTSqablVBOgSpCCFQF8HK9wu+AziErzAHt+pOLhTH4TP0wIOKyr8myU+gEWpVxXX0KrVS                                                                                          
                FMJQL76xoBR+xJcQgDHh06O0jtPoqoDa764zv+gCV3DgbDdULgzZRv2PumiZ07zE87zOyDlup04apHkq                                                                                          
                UQc38xfaoB9UpRp2+JzO+fLRKY/wPWWoRcVUWoKI23v2c7+X8K6hD7blsUUWHng+D6GsgjJvVh8HkxCD                                                                                          
                BdUhP9iAiqgZlVdCPZPb9rd75zbJoCrJG7FhX7iO+9Pd7a641a7XZrG4UwjZpAmbkJnVZ7qs1/SATnKG                                                                                          
                muiP9pmsGbVh7ed39F2XdIPdKp7oT7xJw3JROvKNeF6I6aecgxOIw47aFwGb4xqO8Ht+y6+4im0Upzna                                                                                          
                o15MYRYrGKEEpvEIHZqiCczgFUYpT/8Bk47KLA==''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=Cahn-Hilliard
