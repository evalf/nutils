#! /usr/bin/env python3
#
# In this script we solve the Navier-Stokes equations around a cylinder, using
# the same Raviart-Thomas discretization as in
# :ref:`examples/drivencavity-compatible.py` but in curvilinear coordinates.
# The mesh is constructed such that all elements are shape similar, growing
# exponentially with radius such that the artificial exterior boundary is
# placed at large (configurable) distance.

from nutils import mesh, function, solver, util, export, cli, testing, numeric
from nutils.expression_v2 import Namespace
import numpy
import treelog


class PostProcessor:

    def __init__(self, topo, ns, timestep, region=4., aspect=16/9, figscale=7.2, spacing=.05):
        self.ns = ns
        self.figsize = aspect * figscale, figscale
        self.bbox = numpy.array([[-.5, aspect-.5], [-.5, .5]]) * region
        self.bezier = topo.select(function.min(*(ns.x-self.bbox[:,0])*(self.bbox[:,1]-ns.x))).sample('bezier', 5)
        self.spacing = spacing
        self.timestep = timestep
        self.t = 0.
        self.initialize_xgrd()
        self.topo = topo

    def initialize_xgrd(self):
        self.orig = numeric.floor(self.bbox[:,0] / (2*self.spacing)) * 2 - 1
        nx, ny = numeric.ceil(self.bbox[:,1] / (2*self.spacing)) * 2 + 2 - self.orig
        self.vacant = numpy.hypot(
            self.orig[0] + numpy.arange(nx)[:,numpy.newaxis],
            self.orig[1] + numpy.arange(ny)) > self.ns.R.eval() / self.spacing
        self.xgrd = (numpy.stack(self.vacant[1::2,1::2].nonzero(), axis=1) * 2 + self.orig + 1) * self.spacing

    def regularize_xgrd(self):
        # use grid rounding to detect and remove oldest points that have close
        # neighbours and introduce new points into vacant spots
        keep = numpy.zeros(len(self.xgrd), dtype=bool)
        vacant = self.vacant.copy()
        for i, ind in enumerate(numeric.round(self.xgrd / self.spacing) - self.orig): # points are ordered young to old
            if all(ind >= 0) and all(ind < vacant.shape) and vacant[tuple(ind)]:
                vacant[tuple(ind)] = False
                keep[i] = True
        roll = numpy.arange(vacant.ndim)-1
        for _ in roll: # coarsen all dimensions using 3-point window
            vacant = numeric.overlapping(vacant.transpose(roll), axis=0, n=3)[::2].all(1)
        newpoints = numpy.stack(vacant.nonzero(), axis=1) * 2 + self.orig + 1
        self.xgrd = numpy.concatenate([newpoints * self.spacing, self.xgrd[keep]], axis=0)

    def __call__(self, args):
        x, p = self.bezier.eval(['x_i', 'p'] @ self.ns, **args)
        logr = numpy.log(numpy.hypot(*self.xgrd.T) / self.ns.R.eval())
        φ = numpy.arctan2(self.xgrd[:,1], self.xgrd[:,0]) + numpy.pi
        ugrd = self.topo.locate(self.ns.grid, numpy.stack([logr, φ], axis=1), eps=1, tol=1e-5).eval(self.ns.u, **args)
        with export.mplfigure('flow.png', figsize=self.figsize) as fig:
            ax = fig.add_axes([0, 0, 1, 1], yticks=[], xticks=[], frame_on=False, xlim=self.bbox[0], ylim=self.bbox[1])
            im = ax.tripcolor(*x.T, self.bezier.tri, p, shading='gouraud', cmap='jet')
            export.plotlines_(ax, x.T, self.bezier.hull, colors='k', linewidths=.1, alpha=.5)
            ax.quiver(*self.xgrd.T, *ugrd.T, angles='xy', width=1e-3, headwidth=3e3, headlength=5e3, headaxislength=2e3, zorder=9, alpha=.5, pivot='tip')
            ax.plot(0, 0, 'k', marker=(3, 2, self.t*self.ns.rotation.eval()*180/numpy.pi-90), markersize=20)
            cax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
            cax.tick_params(labelsize='large')
            fig.colorbar(im, cax=cax)
        self.t += self.timestep
        self.xgrd += ugrd * self.timestep
        self.regularize_xgrd()


def main(nelems: int, degree: int, reynolds: float, rotation: float, radius: float, timestep: float, maxradius: float, seed: int, endtime: float):
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
       radius [.5]
         Cylinder radius.
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
    ns.Re = reynolds
    ns.grid = geom * elemangle
    ns.R = radius
    ns.r = 'R exp(grid_0)'
    ns.φ = 'grid_1'
    ns.x_i = 'r (cos(φ) δ_i0 + sin(φ) δ_i1)'
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    J = ns.x.grad(geom)
    detJ = function.determinant(J)
    ns.add_field(('u', 'v'), function.matmat(function.vectorize([
        domain.basis('spline', degree=(degree, degree-1), removedofs=((0,), None)),
        domain.basis('spline', degree=(degree-1, degree))]), J.T) / detJ)
    ns.add_field(('p', 'q'), domain.basis('spline', degree=degree-1) / detJ)
    ns.sigma_ij = '(∇_j(u_i) + ∇_i(u_j)) / Re - p δ_ij'
    ns.N = 10 * degree / elemangle  # Nitsche constant based on element size = elemangle/2
    ns.nitsche_i = '(N v_i - (∇_j(v_i) + ∇_i(v_j)) n_j) / Re'
    ns.rotation = rotation
    ns.uwall_i = '0.5 rotation (-sin(φ) δ_i0 + cos(φ) δ_i1)'

    inflow = domain.boundary['outer'].select(-ns.uinf.dotnorm(ns.x), ischeme='gauss1')  # upstream half of the exterior boundary
    sqr = inflow.integral('Σ_i (u_i - uinf_i)^2' @ ns, degree=degree*2)
    cons = solver.optimize('u,', sqr, droptol=1e-15)  # constrain inflow semicircle to uinf

    numpy.random.seed(seed)
    sqr = domain.integral('Σ_i (u_i - uinf_i)^2' @ ns, degree=degree*2)
    args0 = solver.optimize('u,', sqr) # set initial condition to u=uinf
    args0['u'] = args0['u'] * numpy.random.normal(1, .1, len(args0['u'])) # add small random noise

    res = domain.integral('(v_i ∇_j(u_i) u_j + ∇_j(v_i) sigma_ij) dV' @ ns, degree=9)
    res += domain.boundary['inner'].integral('(nitsche_i (u_i - uwall_i) - v_i sigma_ij n_j) dS' @ ns, degree=9)
    res += domain.integral('q ∇_k(u_k) dV' @ ns, degree=9)
    uinertia = domain.integral('v_i u_i dV' @ ns, degree=9)

    postprocess = PostProcessor(domain, ns, timestep)

    with treelog.iter.plain('timestep', solver.impliciteuler('u:v,p:q', residual=res, inertia=uinertia, arguments=args0, timestep=timestep, constrain=cons, newtontol=1e-10)) as steps:
        for istep, args in enumerate(steps):

            postprocess(args)

            if istep * timestep >= endtime:
                break

    return args0, args

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
        args0, args = main(nelems=6, degree=3, reynolds=100, radius=.5, rotation=0, timestep=.1, maxradius=25, seed=0, endtime=.05)
        with self.subTest('initial condition'):
            self.assertAlmostEqual64(args0['u'], '''
                eNoNzLEJwkAYQOHXuYZgJyZHLjmMCIJY2WcEwQFcwcYFbBxCEaytzSVnuM7CQkEDqdVCG//uNe/rqEcI
                p+pSwTzQAez90cM06kZQuLWDrV5oGJf9EupEJ7CyqYXUTAyM7C2HmZyNf8p57S2lB95It8KNgmH1PcNB
                /UTEvUVpojqGdpEV8IlfIt7znSiZ+QPSaDIR''', atol=2e-13)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(args['u'], '''
                eNoBkABv/+o0szWg04bKlsogMVI4JjcmMXXI+cfizb05/Dk4MBHGEcaPNDo8ljuTNibE4sNpznI9VD02
                M5zCnsJazE0+Hj76NsPByMH/yl43nDNlyGnIsy+YNz44MspbyD/IWcwHOMM4AzXAxsPGGjAJOUM7GcgU
                xePEqckvO+g8+DcOwyfD4zfjPFY+sMfJwavBhDNPPpLTRUE=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(args['p'], '''
                eNoBSAC3/4zF18aozR866DpHNSk8JDonOdw4k8VzNaHBk8PFOyI+Gj9vPPRA/T/LQDtBIECaP0i5yLsA
                wL9FwkabQsJJTbc2ubJHw7ZvRq/qITA=''')

    def test_rot1(self):
        args0, args = main(nelems=6, degree=3, reynolds=100, radius=.5, rotation=1, timestep=.1, maxradius=25, seed=0, endtime=.05)
        with self.subTest('initial condition'):
            self.assertAlmostEqual64(args0['u'], '''
                eNoNzLEJwkAYQOHXuYZgJyZHLjmMCIJY2WcEwQFcwcYFbBxCEaytzSVnuM7CQkEDqdVCG//uNe/rqEcI
                p+pSwTzQAez90cM06kZQuLWDrV5oGJf9EupEJ7CyqYXUTAyM7C2HmZyNf8p57S2lB95It8KNgmH1PcNB
                /UTEvUVpojqGdpEV8IlfIt7znSiZ+QPSaDIR''', atol=2e-13)
        with self.subTest('velocity'):
            self.assertAlmostEqual64(args['u'], '''
                eNoBkABv/+M0tzVcKYfKlMr1MFE4JzdBMXbI+cfMzb05/Dk/MBHGEcaPNDo8ljuUNibE4sNjznI9VD02
                M5zCnsJazE0+Hj76NsPByMH/ynk3WTR/yH7I5TGsNzk4HcpUyDnILMwBOMQ4BzXBxsTGpDAKOUM7GMgU
                xePEp8kvO+g8+DcOwyfD5DfkPFY+sMfJwavBhDNPPpo5RbI=''')
        with self.subTest('pressure'):
            self.assertAlmostEqual64(args['p'], '''
                eNoBSAC3/4rF3Ma4ziE65zoUNSg8JToqOd04k8VbNaDBlMPJOyI+Gj9sPPRA/T/MQDtBH0CYP0e5yLsD
                wL9FwkaaQsJJTbc3ubJHw7ZvRqV7IP8=''')
