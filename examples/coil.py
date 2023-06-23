from nutils import export, function, mesh, solver, testing
from nutils.expression_v2 import Namespace
import functools
import numpy


def main(nelems: int = 50,
         degree: int = 3,
         freq: float = 0.,
         rwire: float = .0025,
         rcoil: float = 0.025,
         nturns: int = 1):

    '''Current-induced magnetic field

    Computes the magnetic field induced by a DC or AC current in one or several
    toroidal conductors. This problem is modeled with the quasi-static
    [magnetic vector potential][1] with Lorenz gauge:

        ∇_j(∇_j(A_i)) = -μ0 J_i

    where `A` is the magnetic vector potential, `J` the current density and `μ0`
    the magnetic permeability. The magnetic field `B` is then given by the curl
    of the magnetic vector potential. The current density is the sum of an
    external current `Jext` and the current induced by the magnetic field,
    `Jind`. The external current is given by

        Jext_i = (I / π rwire²) cos(ω t) eθ_i

    inside the conductor and zero everywhere else, where `ω = 2 π f`. The induced
    current follows from [Faraday's law of induction][2] and [Ohm's law][3]:

        Jind_i = -σ ∂_t(A_i)

    where `σ` is the conductivity, which is non-zero only inside the conductor.

    We can solve the temporal component of `A` by letting `A_i = Re(Â_i exp(j ω
    t))`. This problem in terms of `Â` is:

        ∇_j(∇_j(Â_i) = -μ0 Ĵ_i

    with

        Ĵext_i = (I / π rwire²) eθ_i

    and

        Ĵind_i = -j ω σ Â_i

    [1]: https://en.wikipedia.org/wiki/Magnetic_vector_potential
    [2]: https://en.wikipedia.org/wiki/Faraday%27s_law_of_induction
    [3]: https://en.wikipedia.org/wiki/Ohm%27s_law

    Parameters
    ----------
    nelems
        Number of elements per spatial dimension.
    degree
        Polynomial degree of discretized magnetic vector potential.
    freq
        Alternating current frequency; a value of 0 corresponds to direct
        current.
    rwire
        Radius of the wire.
    rcoil
        Radius of the coil, must be larger than `rwire`.
    nturns
        Number of windings in the coil, spaced by a distance of `4 rwire`.
    '''

    ns = Namespace()
    ns.j = 1j
    ns.π = numpy.pi
    ns.f = freq
    ns.ω = '2 π f'
    ns.μ0 = '4e-7 π'  # magnetic permeability in vacuum
    ns.σ = 5.988e7  # conductivity of copper
    ns.rcoil = rcoil  # radius of coil (to the center of the wire)
    ns.rwire = rwire  # radius of the wire

    # The problem is axisymmetric in the z-axis and symmetric in z=0. We create a
    # 2D domain `RZ` covering the quarter plane [0,inf)^2, multiply this with a
    # 1D domain `REV` with one element spanning [-π,π] and transform the geometry
    # and vector bases from cylindrical to cartesian coordinates. The `RZ`
    # domain is mapped from [0,1] to [0,inf) using an arctanh transform. Finally,
    # a Neumann boundary condition is used at z=0 to obtain symmetry in z=0.

    RZ, ns.rz0 = mesh.rectilinear([numpy.linspace(0, 1, nelems)]*2, space='RZ')
    REV, ns.θ = mesh.line([-numpy.pi, numpy.pi], bnames=['start', 'end'], space='Θ')
    REV0 = REV.refined[:1].boundary['end'].sample('bezier', 2)
    ns.rz = numpy.arctanh(ns.rz0) * 2 * rcoil
    ns.r, ns.z = ns.rz

    # Trimming of the wires. The centers of the wires are located at
    # `rcoil,zwires`.

    ns.zwires = (numpy.arange(nturns) - (nturns - 1) / 2) * 4 * rwire
    ns.dwires = ns.rwire - numpy.sqrt((ns.r - ns.rcoil)**2 + functools.reduce(numpy.minimum, (ns.z - ns.zwires)**2))
    RZ = RZ.withsubdomain(coil=RZ[:-1, :-1].trim(ns.dwires/ns.rwire, maxrefine=4))

    ns.rot = numpy.stack([function.scatter(function.trignormal(ns.θ), 3, [0, 1]), function.kronecker(1., 0, 3, 2)])
    ns.eθ = numpy.stack(['-sin(θ)', 'cos(θ)', '0'] @ ns)

    X = RZ * REV
    ns.x = ns.rz @ ns.rot
    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'), curl='curl')
    ns.add_field(('A', 'Atest'), RZ.basis('spline', degree=degree, removedofs=[[0, -1], [-1]])[:,numpy.newaxis] * ns.eθ, dtype=complex)
    ns.B_i = 'curl_ij(A_j)'
    ns.E_i = '-j ω A_i'
    ns.Jind_i = 'σ E_i'
    ns.I = 1
    ns.Jext_i = 'eθ_i I / π rwire^2'
    ns.J_i = 'Jext_i + Jind_i'

    res = REV.integral(RZ.integral('-∇_j(Atest_i) ∇_j(A_i) dV' @ ns, degree=2*degree), degree=0)
    res += REV.integral(RZ['coil'].integral('μ0 Atest_i J_i dV' @ ns, degree=2*degree), degree=0)

    args = solver.solve_linear('A:Atest,', res)

    # Since the coordinate transformation is singular at r=0 we can't evaluate
    # `B` (the curl of `A`) at r=0. We circumvent this problem by projecting `B`
    # on a basis.

    ns.Borig = ns.B
    ns.add_field(('B', 'Btest'), RZ.basis('spline', degree=degree), ns.rot, dtype=complex)
    res = REV.integral(RZ.integral('Btest_i (B_i - Borig_i) dV' @ ns, degree=2*degree), degree=0)
    args = solver.solve_linear('B:Btest,', res, arguments=args)

    with export.mplfigure('magnetic-potential-1.png', dpi=300) as fig:
        ax = fig.add_subplot(111, aspect='equal', xlabel='$x_0$', ylabel='$x_2$', adjustable='datalim')
        # Magnetic vector potential. `r < 0` is the imaginary part, `r > 0` the
        # real part.
        smpl = REV0 * RZ[:-1, :-1].sample('bezier', 5)
        r, z, A, Bmag = smpl.eval(['r', 'z', 'A_1', 'sqrt(real(B_i) real(B_i)) + sqrt(imag(B_i) imag(B_i)) j'] @ ns, **args)
        Amax = abs(A).max()
        Bmax = abs(Bmag).max()
        levels = numpy.linspace(-Amax, Amax, 32)[1:-1]
        r = numpy.concatenate([r, r], axis=0)
        z = numpy.concatenate([z, -z], axis=0)
        A = numpy.concatenate([A, A], axis=0)
        Bmag = numpy.concatenate([Bmag, Bmag], axis=0)
        tri = numpy.concatenate([smpl.tri+i*smpl.npoints for i in range(2)])
        hull = numpy.concatenate([smpl.hull+i*smpl.npoints for i in range(2)])
        imBi = ax.tripcolor(-r, z, tri, Bmag.imag, shading='gouraud', cmap='Greens')
        imBi.set_clim(0, Bmax)
        ax.tricontour(-r, z, tri, -A.imag, colors='k', linewidths=.5, levels=levels)
        imBr = ax.tripcolor(r, z, tri, Bmag.real, shading='gouraud', cmap='Greens')
        imBr.set_clim(0, Bmax)
        ax.tricontour(r, z, tri, A.real, colors='k', linewidths=.5, levels=levels)
        # Current density (wires only). `r < 0` is the imaginary part, `r > 0` the
        # real part.
        smpl = REV0 * RZ['coil'].sample('bezier', 5)
        r, z, J = smpl.eval(['r', 'z', 'J_1'] @ ns, **args)
        Jmax = abs(J).max()
        r = numpy.concatenate([r, r], axis=0)
        z = numpy.concatenate([z, -z], axis=0)
        J = numpy.concatenate([J, J], axis=0)
        tri = numpy.concatenate([smpl.tri+i*smpl.npoints for i in range(2)])
        imJi = ax.tripcolor(-r, z, tri, -J.imag, shading='gouraud', cmap='bwr')
        imJi.set_clim(-Jmax, Jmax)
        imJr = ax.tripcolor(r, z, tri, J.real, shading='gouraud', cmap='bwr')
        imJr.set_clim(-Jmax, Jmax)
        # Minor ticks at element edges.
        rticks = RZ.boundary['bottom'].interfaces.sample('gauss', 0).eval(ns.r)
        ax.set_xticks(numpy.concatenate([-rticks[::-1], [0], rticks]), minor=True)
        zticks = RZ.boundary['left'].interfaces.sample('gauss', 0).eval(ns.z)
        ax.set_yticks(numpy.concatenate([-zticks[::-1], [0], zticks]), minor=True)
        ax.tick_params(direction='in', which='minor', bottom=True, top=True, left=True, right=True)
        # Real and imag indicator.
        spine = next(iter(ax.spines.values()))
        ax.axvline(0, color='k')
        ax.text(0, .95, '← imag ', horizontalalignment='right', verticalalignment='bottom', transform=ax.get_xaxis_transform())
        ax.text(0, .95, ' real →', horizontalalignment='left', verticalalignment='bottom', transform=ax.get_xaxis_transform())
        ax.set_xlim(-2*rcoil, 2*rcoil)
        ax.set_ylim(-2*rcoil, 2*rcoil)
        fig.colorbar(imJr, label='$J_1$')
        fig.colorbar(imBr, label='$|B|$')

    if freq == 0:
        ns.δ = function.eye(3)
        # Reference: https://physics.stackexchange.com/a/355183
        ns.Bexact = ns.δ[2] * ns.μ0 * ns.I * ns.rcoil**2 / 2 * ((ns.rcoil**2 + (ns.z - ns.zwires)**2)**(-3/2)).sum()
        smpl = REV0 * RZ[:-1, :-1].boundary['left'].sample('bezier', 5)
        B, Bexact, z = smpl.eval(['real(B_2)', 'Bexact_2', 'z'] @ ns, **args)
        z = numpy.concatenate([-z[::-1], z])
        B = numpy.concatenate([B[::-1], B])
        Bexact = numpy.concatenate([Bexact[::-1], Bexact])
        with export.mplfigure('magnetic-field-x2-axis.png', dpi=300) as fig:
            ax = fig.add_subplot(111, xlabel='$x_2$', ylabel='$B_2$', title='$B_2$ at $x_0 = x_1 = 0$')
            ax.plot(z, B, label='FEM')
            ax.plot(z, Bexact, label='exact', linestyle='dotted')
            ax.legend()
            # Minor ticks at element edges.
            zticks = RZ.boundary['left'].interfaces.sample('gauss', 0).eval(ns.z)
            ax.set_xticks(numpy.concatenate([-zticks[::-1], [0], zticks]), minor=True)
            ax.tick_params(axis='x', direction='in', which='minor', bottom=True, top=True)
            ax.set_xlim(-2*rcoil, 2*rcoil)

    return args


class test(testing.TestCase):

    def test_dc(self):
        args = main(nelems=16, degree=2)
        with self.subTest('A.real'):
            self.assertAlmostEqual64(args['A'].real, '''
                eNoNke9rzWEYh5NzVmtnvud5nvv+3PdzTn7lIIRlL3Rq/wArinFGaytFo6xjTedISMwsJsNksbJYtlIS
                U9pqLcqJKL9ytL3xYm92kpkQ2vL9B67P9el6TS/oHuVpPb13zW7WZu2U2WaG4t8CF8xWVsS+YgZF3MYu
                /OYLTHyFyijrXllrNxvEqxaVa1S/yJBk5CfaEUMnz1MzPXcxV23JVAWjOq4D2qAL9YakZBAp9HKE99F9
                99E+NcWgw5/yaT+tJzWm3WLlEiI4wu9oKdW6TTYTL/m//oPf4T9rvU7IXvmE7RjjFB+lAXfZjsRrk2uT
                qxM3fcSfDTfaJSqn8YubeJhKbtIG5kdiImESHX5ez2iFXpWk9MHjPE/Rckq4jDnhO/0xv8SPhfwZOScq
                d7EG/VzGW0ODHvNdS+GDa7pTy/WJNMgcrmMlBln4ALW6l2aZrtCk/pO3cksaRaSALCrRx8pt1OX+mLzk
                5JDUS01ILmEYOWzEJB/nKGep1y22j/AYD3AH3chjD6oRxRu+yDVcpN3U49K2wAV+xqP8kPu5i9u4jjfw
                HI1Ta9ihya2zLdRCh+kg7adGqqMtlKZVFKNpN+JyboFL2f8Z6oV2''')

    def test_ac_5(self):
        args = main(nelems=16, degree=2, freq=1000., nturns=5)
        with self.subTest('A.imag'):
            self.assertAlmostEqual64(args['A'].imag, '''
                eNoNkEtIlGEYhRcWBqVWNgsxujBImxJmIIwEc2FJCYVaQxEErRIvkyFOhqTB4CLSpE3TxQiLCjXQEdJN
                QUkZKMxCyxYhmpvvPe/7ft//K11AvPSvzuJZnPOcOZ3TeV1SX3PtERu3n2zEfXRNXqkfXU6s9P5O8wiP
                8nue5kXeLhXyRHL0mVbZApfn1fj1K4MYwgjeYQLzWEcZpzhPXkqtHrAhF/Pqlzdpg9YpG/kIowb3wGjg
                rTImSU3YUTfhN9MNaqcEdVIfDdIaNeAPUnxGQpplS12Fn0+7KItCVBhkLSVpC17jNG/wFxnWRbvgtZmk
                +WxumQ4zY6rNX/ODmrGfp7hH4vrIdnuPzQuTMU9Nv/lpeswe+h407Aic6qRcr9iT3jE6SoepjE5RJXXR
                MOXiPkI8xBfkoEbtNu859dNAsGycvhFRGHFM4Xjwx3nZqbvtrCtCEQ4hivLALoE+zGIvt/IvvhbwTX3j
                0khjDB8wjQX8QyFXcidP8j7plbCuaZeLcYwv8VVu5HZ+wG85w6sckZsyI+c0xza6YimWiJTICamSy3Jd
                7spAwLL1rKa12t5xLdqirdqmtzWp3ZrSVzquGXVaYC/ar/ah+w/zsU82''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=electro-magnetism:thumbnail=0
