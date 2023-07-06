from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
from dataclasses import dataclass
from typing import Union
import treelog as log
import numpy


@dataclass
class FCM:
    '''Finite cell method

    Generate a topology and geometry by meshing the unit square, followed by
    removing the circular cut-out using the Finite Cell Method. Then restrict a
    basis from the base topology to the trimmed topology for analysis.

    Parameters
    ----------
    nelems
        Number of elements along one dimension in the base mesh.
    etype
        Element type to use in the base mesh (square/triangle/mixed/multipatch).
    btype
        Basis type, options depending on the select element type.
    degree
        Polynomial degree of the basis functions.
    maxrefine
        Quad-tree refinement depth of the finite cell procedure.
    '''

    nelems: int = 9
    etype: str = 'square'
    btype: str = 'std'
    degree: int = 2
    maxrefine: int = 2

    def generate(self, radius):
        topo0, geom = mesh.unitsquare(self.nelems, self.etype)
        topo = topo0.trim(numpy.linalg.norm(geom) - radius, maxrefine=self.maxrefine, name='hole')
        basis = topo.basis(self.btype, degree=self.degree)
        return topo.withboundary(sym='left,bottom', far='top,right'), geom, basis, self.degree


@dataclass
class NURBS:
    '''Non-Uniform Radional B-Splines

    Generate a 1x2 structured topology, map it using quadratic NURBS to a
    square domain with circular cut-out, and refine several times before
    constructing a NURBS basis for analysis.

    Parameters
    ----------
    nrefine
        Number of refinement levels for analysis.
    '''

    nrefine: int = 2

    def generate(self, radius):
        topo, geom0 = mesh.rectilinear([1, 2])
        bsplinebasis = topo.basis('spline', degree=2)
        controlweights = numpy.ones(12)
        controlweights[1:3] = .5 + .25 * numpy.sqrt(2)
        weightfunc = bsplinebasis @ controlweights
        nurbsbasis = bsplinebasis * controlweights / weightfunc
        # create geometry function
        A = 0, 0, 0
        B = (2**.5-1) * radius, .3 * (radius+1) / 2, 1
        C = radius, (radius+1) / 2, 1
        controlpoints = numpy.array([[A, B, C, C], [C, C, B, A]]).T.reshape(-1, 2)
        geom = nurbsbasis @ controlpoints
        # refine topology
        if self.nrefine:
            topo = topo.refine(self.nrefine)
            bsplinebasis = topo.basis('spline', degree=2)
            sqr = topo.integral((function.dotarg('w', bsplinebasis) - weightfunc)**2, degree=9)
            controlweights = solver.optimize('w', sqr)
            nurbsbasis = bsplinebasis * controlweights / weightfunc
        return topo.withboundary(hole='left', sym='top,bottom', far='right'), geom, nurbsbasis, 5


def main(mode: Union[FCM, NURBS] = NURBS,
         radius: float = .5,
         traction: float = .1,
         poisson: float = .3):

    '''Deformation of an infinite plate with a circular hole

    Solves the linear plane strain elasticity problem for an infinite plate
    with a circular hole under tension. We do this by placing the circle in the
    origin of a unit square, imposing symmetry conditions on the left and
    bottom ("sym"), and Dirichlet conditions constraining the displacements to
    the analytical solution to the right and top ("far").

    The script can be run in two modes: by specifying `mode=FCM`, the circular
    hole is cut out of a regular finite element mesh by means of the Finite Cell
    Method; by specifying `mode=NURBS` a Non-Uniform Rational BSpline geometry is
    created to map a regular domain onto the desired shape. Either mode supports
    sub-parameters which can be specified from the command-line by attaching them
    in curly braces (e.g. `FCM{nelems=20,degree=1}`).

    Parameters
    ----------
    mode
        Discretization strategy: FCM (Finite Cell Method) or NURBS.
    radius
        Cut-out radius.
    traction
        Far field traction (relative to Young's modulus).
    poisson
        Poisson's ratio, nonnegative and strictly smaller than 1/2.
    '''

    topo, geom, basis, degree = mode.generate(radius)

    ns = Namespace()
    ns.δ = function.eye(topo.ndims)
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.λ = 2 * poisson
    ns.μ = 1 - poisson
    ns.add_field(('u', 'v'), basis, shape=(2,))
    ns.X_i = 'x_i + u_i'
    ns.ε_ij = '(∇_j(u_i) + ∇_i(u_j)) / 2'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'
    ns.r2 = 'x_k x_k'
    ns.R2 = radius**2 / ns.r2
    ns.k = (3-poisson) / (1+poisson) # plane stress parameter
    ns.scale = traction * (1+poisson) / 2
    ns.uexact_i = 'scale (x_i ((k + 1) (.5 + R2) + (1 - R2) R2 (x_0^2 - 3 x_1^2) / r2) - 2 δ_i1 x_1 (1 + (k - 1 + R2) R2))'
    ns.du_i = 'u_i - uexact_i'
    ns.dr = numpy.sqrt(ns.r2) - radius

    radiuserr = topo.boundary['hole'].integrate('dr^2 dS' @ ns, degree=9)**.5
    log.info('hole radius exact up to L2 error {:.2e}'.format(radiuserr))

    sqr = topo.boundary['sym'].integral('(u_i n_i)^2 dS' @ ns, degree=degree*2)
    cons = solver.optimize(('u',), sqr, droptol=1e-15)

    sqr = topo.boundary['far'].integral('du_k du_k dS' @ ns, degree=20)
    cons = solver.optimize(('u',), sqr, droptol=1e-15, constrain=cons)

    res = topo.integral('∇_j(v_i) σ_ij dV' @ ns, degree=degree*2)
    args = solver.solve_linear('u:v', res, constrain=cons)

    bezier = topo.sample('bezier', 5)
    X, σxx = bezier.eval(['X_i', 'σ_00'] @ ns, **args)
    export.triplot('stressxx.png', X, σxx, tri=bezier.tri, hull=bezier.hull, clim=(numpy.nanmin(σxx), numpy.nanmax(σxx)), cmap='jet')

    err = numpy.sqrt(topo.integrate(['du_k du_k dV', '∇_j(du_i) ∇_j(du_i) dV'] @ ns, degree=max(degree, 3)*2, arguments=args))
    log.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

    return err, cons, args


class test(testing.TestCase):

    def test_spline(self):
        err, cons, args = main(mode=FCM(nelems=4, btype='spline'))
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00033, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00674, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGBoYGBAxvrnGBow4X89g3NQFSjQwLAGq7i10Wus4k+NfM8fNWZgOGL89upc47WX0ozvXjAzPn1e
                1TjnPACrACoJ''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpbb3bMjIHhxzkGBhMgtgdi/XMqp8RPvjLxOPPCcNq5Fn3Pcxr6luf+6xmcm2LMwLDQePf5c0bTzx8x
                5DnvaTjnnIFhzbmlQPH5xgvu6hvfvbjaSPRCsJHeeWuj1+c2AMU3Gi++Wmu885Kp8d4LP41WnH9q5Hv+
                KFD8iPHbq3ON115KM757wcz49HlV45zzAL8gQC8=''')

    def test_mixed(self):
        err, cons, args = main(mode=FCM(nelems=4, etype='mixed'))
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00024, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00740, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjaGDADhlwiOEU1z8HZusbgukkg5BzRJqKFRoa1oD1HzfceA5NH9FmgKC10SuwOdONpM7DxDYa77gM
                MueoMQPDEePzV2Hic42XXmoynnQRxvc3dryQbnz3Aoj91Mj3vJnx6fOqxjnnAQzkV94=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNoNzEEoQ3EcB/AXVymHtdqBkyLx3v/3LTQHtHJQKKHZ0YXMQS6sSM2BcrKMqbHTotUOw4GrthzWfr//
                e6+nHJYUyUopKSnlHT717Vvfr0cpSWCWr/FVs1GuZdHKmb6QGMYRN9Qev1irXFUVTtAfG8agb5gtc5LX
                zQj/WDm+s3b8bsBncosZZsNUvGEt8YI1w2lcSQRrMg9Pp/FmZ2EYOfTYmnIyR+PShLi+oA8pq5DsqxoH
                qEvGaFdG1AErtclP6pnvVYnz/u4MVj2OZrfg53OceElE3Q482p9U0d0I2FGEnRK16SQdyjfFtEOuTFOv
                DFGDi7QsxxSSIoIPGby7Jdy4l/5PxVeGeFu4dWLYtk+Rd5JI2SaKOoh2PYVRvYi6/qWCvNKE9KMqnViR
                fyhZkYI=''')

    def test_nurbs0(self):
        err, cons, args = main(mode=NURBS(nrefine=0))
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00200, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .02271, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYGBoQIIggMZXOKdmnHRe3vjh+cvGDAwA6w0LgQ==''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYJh07qLhhnOTjb0vTDdmAAKVcy/1u85lGYforQDzFc6pGSedlzd+eP4ykA8AvkQRaA==''')

    def test_nurbs2(self):
        err, cons, args = main(mode=NURBS(nrefine=2))
        with self.subTest('l2-error'):
            self.assertAlmostEqual(err[0], .00009, places=5)
        with self.subTest('h1-error'):
            self.assertAlmostEqual(err[1], .00286, places=5)
        with self.subTest('constraints'):
            self.assertAlmostEqual64(cons['u'], '''
                eNpjYGBoIAKCwCBXp3kuysDjnLXR+3NPjTzPqxrnAnHeeQvjk+dTjZ9d2GG85soJYwYGAPkhPtE=''')
        with self.subTest('left-hand side'):
            self.assertAlmostEqual64(args['u'], '''
                eNpjYOg890mv85yM4axz0kYHz+00Yj6vZJxzPtWY+0KPMffFucaml+caMwBB5LlCvYhzCw0qzu0wPHyu
                0sjlPIsx14VoY/6LvcaxlxYZz7myCKzO+dwWPZdzBwzqz20z/Hguxmj2+TtGHRdsjHdfbDB2v7zUeMXV
                pWB1VucC9B3OORmuOCdhZHR+ktGu87eNbC6oGstfLDA+eWm1seG19WB1Buf+6ruce2p469wco9Dzb4wm
                n2c23nZe3djqQqpx88XNxrOv7gOr0zwXZeBxztro/bmnRp7nVY1zgTjvvIXxSaBfnl3YYbzmygmgOgDU
                Imlr''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=elasticity,FCM,NURBS
