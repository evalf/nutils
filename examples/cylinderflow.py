#! /usr/bin/env python3
#
# In this script we solve the Navier-Stokes equations around a cylinder, using
# the same Raviart-Thomas discretization as in
# :ref:`examples/drivencavity-compatible.py` but in curvilinear coordinates.
# The mesh is constructed such that all elements are shape similar, growing
# exponentially with radius such that the artificial exterior boundary is
# placed at large (configurable) distance.

from nutils import mesh, function, solver, util, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy
import treelog


def main(nelems: int, degree: int, reynolds: float, rotation: float, timestep: float, maxradius: float, seed: int, endtime: float):
    '''
    Flow around a cylinder.

    .. arguments::

       nelems [24]
         Element size expressed in number of elements along the cylinder wall.
         All elements have similar shape with approximately unit aspect ratio,
         with elements away from the cylinder wall growing exponentially.
       degree [3]
         Polynomial degree for velocity space; the pressure space is one degree
         less.
       reynolds [1000]
         Reynolds number, taking the cylinder radius as characteristic length.
       rotation [0]
         Cylinder rotation speed.
       timestep [.04]
         Time step
       maxradius [25]
         Target exterior radius; the actual domain size is subject to integer
         multiples of the configured element size.
       seed [0]
         Random seed for small velocity noise in the intial condition.
       endtime [inf]
         Stopping time.
    '''

    elemangle = 2 * numpy.pi / nelems
    melems = int(numpy.log(2*maxradius) / elemangle + .5)
    treelog.info('creating {}x{} mesh, outer radius {:.2f}'.format(melems, nelems, .5*numpy.exp(elemangle*melems)))
    domain, geom = mesh.rectilinear([melems, nelems], periodic=(1,))
    domain = domain.withboundary(inner='left', outer='right')

    ns = Namespace()
    ns.δ = function.eye(domain.ndims)
    ns.Σ = function.ones([domain.ndims])
    ns.uinf = function.Array.cast([1, 0])
    ns.r = .5 * function.exp(elemangle * geom[0])
    ns.Re = reynolds
    ns.phi = geom[1] * elemangle  # add small angle to break element symmetry
    ns.x_i = 'r (cos(phi) δ_i0 + sin(phi) δ_i1)'
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    J = ns.x.grad(geom)
    detJ = function.determinant(J)
    ubasis = function.matmat(function.vectorize([
        domain.basis('spline', degree=(degree, degree-1), removedofs=((0,), None)),
        domain.basis('spline', degree=(degree-1, degree))]), J.T) / detJ
    pbasis = domain.basis('spline', degree=degree-1) / detJ
    ns.u = function.dotarg('u', ubasis)
    ns.v = function.dotarg('v', ubasis)
    ns.p = function.dotarg('p', pbasis)
    ns.q = function.dotarg('q', pbasis)
    ns.sigma_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'
    ns.N = 10 * degree / elemangle  # Nitsche constant based on element size = elemangle/2
    ns.nitsche_i = '(N v_i - (∇_j(v_i) + ∇_i(v_j)) n_j) / Re'
    ns.rotation = rotation
    ns.uwall_i = '0.5 rotation (-sin(phi) δ_i0 + cos(phi) δ_i1)'

    inflow = domain.boundary['outer'].select(-ns.uinf.dotnorm(ns.x), ischeme='gauss1')  # upstream half of the exterior boundary
    sqr = inflow.integral('Σ_i (u_i - uinf_i)^2' @ ns, degree=degree*2)
    ucons = solver.optimize('u', sqr, droptol=1e-15)  # constrain inflow semicircle to uinf
    cons = dict(u=ucons)

    numpy.random.seed(seed)
    sqr = domain.integral('Σ_i (u_i - uinf_i)^2' @ ns, degree=degree*2)
    udofs0 = solver.optimize('u', sqr) * numpy.random.normal(1, .1, len(ubasis))  # set initial condition to u=uinf with small random noise
    state0 = dict(u=udofs0)

    res = domain.integral('(v_i ∇_j(u_i) u_j + ∇_j(v_i) sigma_ij) dV' @ ns, degree=9)
    res += domain.boundary['inner'].integral('(nitsche_i (u_i - uwall_i) - v_i sigma_ij n_j) dS' @ ns, degree=9)
    res += domain.integral('q ∇_k(u_k) dV' @ ns, degree=9)
    uinertia = domain.integral('v_i u_i dV' @ ns, degree=9)

    bbox = numpy.array([[-2, 46/9], [-2, 2]])  # bounding box for figure based on 16x9 aspect ratio
    bezier0 = domain.sample('bezier', 5)
    bezier = bezier0.subset((bezier0.eval((ns.x-bbox[:, 0]) * (bbox[:, 1]-ns.x)) > 0).all(axis=1))
    interpolate = util.tri_interpolator(bezier.tri, bezier.eval(ns.x), mergetol=1e-5)  # interpolator for quivers
    spacing = .05  # initial quiver spacing
    xgrd = util.regularize(bbox, spacing)

    with treelog.iter.plain('timestep', solver.impliciteuler('u:v,p:q', residual=res, inertia=uinertia, arguments=state0, timestep=timestep, constrain=cons, newtontol=1e-10)) as steps:
        for istep, state in enumerate(steps):

            t = istep * timestep
            x, u, normu, p = bezier.eval(['x_i', 'u_i', 'sqrt(u_i u_i)', 'p'] @ ns, **state)
            ugrd = interpolate[xgrd](u)

            with export.mplfigure('flow.png', figsize=(12.8, 7.2)) as fig:
                ax = fig.add_axes([0, 0, 1, 1], yticks=[], xticks=[], frame_on=False, xlim=bbox[0], ylim=bbox[1])
                im = ax.tripcolor(x[:, 0], x[:, 1], bezier.tri, p, shading='gouraud', cmap='jet')
                import matplotlib.collections
                ax.add_collection(matplotlib.collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1, alpha=.5))
                ax.quiver(xgrd[:, 0], xgrd[:, 1], ugrd[:, 0], ugrd[:, 1], angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9, alpha=.5)
                ax.plot(0, 0, 'k', marker=(3, 2, t*rotation*180/numpy.pi-90), markersize=20)
                cax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
                cax.tick_params(labelsize='large')
                fig.colorbar(im, cax=cax)

            if t >= endtime:
                break

            xgrd = util.regularize(bbox, spacing, xgrd + ugrd * timestep)

    return state0, state

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

    def test_rot0(self):
        state0, state = main(nelems=6, degree=3, reynolds=100, rotation=0, timestep=.1, maxradius=25, seed=0, endtime=.05)
        with self.subTest('initial condition'):
            self.assertAlmostEqual64(state0['u'], '''
                eNoNzLEJwkAYQOHXuYZgJyZHLjmMCIJY2WcEwQFcwcYFbBxCEaytzSVnuM7CQkEDqdVCG//uNe/rqEcI
                p+pSwTzQAez90cM06kZQuLWDrV5oGJf9EupEJ7CyqYXUTAyM7C2HmZyNf8p57S2lB95It8KNgmH1PcNB
                /UTEvUVpojqGdpEV8IlfIt7znSiZ+QPSaDIR''', atol=2e-13)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(state['u'], '''
                eNoBkABv/+o0szWg04bKlsogMVI4JjcmMXXI+cfizb05/Dk4MBHGEcaPNDo8ljuTNibE4sNpznI9VD02
                M5zCnsJazE0+Hj76NsPByMH/yl43nDNlyGnIsy+YNz44MspbyD/IWcwHOMM4AzXAxsPGGjAJOUM7GcgU
                xePEqckvO+g8+DcOwyfD4zfjPFY+sMfJwavBhDNPPpLTRUE=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(state['p'], '''
                eNoBSAC3/4zF18aozR866DpHNSk8JDonOdw4k8VzNaHBk8PFOyI+Gj9vPPRA/T/LQDtBIECaP0i5yLsA
                wL9FwkabQsJJTbc2ubJHw7ZvRq/qITA=''')

    def test_rot1(self):
        state0, state = main(nelems=6, degree=3, reynolds=100, rotation=1, timestep=.1, maxradius=25, seed=0, endtime=.05)
        with self.subTest('initial condition'):
            self.assertAlmostEqual64(state0['u'], '''
                eNoNzLEJwkAYQOHXuYZgJyZHLjmMCIJY2WcEwQFcwcYFbBxCEaytzSVnuM7CQkEDqdVCG//uNe/rqEcI
                p+pSwTzQAez90cM06kZQuLWDrV5oGJf9EupEJ7CyqYXUTAyM7C2HmZyNf8p57S2lB95It8KNgmH1PcNB
                /UTEvUVpojqGdpEV8IlfIt7znSiZ+QPSaDIR''', atol=2e-13)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(state['u'], '''
                eNoBkABv/+M0tzVcKYfKlMr1MFE4JzdBMXbI+cfMzb05/Dk/MBHGEcaPNDo8ljuUNibE4sNjznI9VD02
                M5zCnsJazE0+Hj76NsPByMH/ynk3WTR/yH7I5TGsNzk4HcpUyDnILMwBOMQ4BzXBxsTGpDAKOUM7GMgU
                xePEp8kvO+g8+DcOwyfD5DfkPFY+sMfJwavBhDNPPpo5RbI=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(state['p'], '''
                eNoBSAC3/4rF3Ma4ziE65zoUNSg8JToqOd04k8VbNaDBlMPJOyI+Gj9sPPRA/T/MQDtBH0CYP0e5yLsD
                wL9FwkaaQsJJTbc3ubJHw7ZvRqV7IP8=''')
