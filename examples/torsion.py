from nutils import mesh, function, solver, export, testing
from nutils.expression_v2 import Namespace
import treelog as log
import numpy as np


def main(length: float = 2*np.pi,
         thickness: float = .5,
         rotation: float = 90.,
         increment: float = 5.,
         elemsize: float = .5,
         poisson: float = .4,
         restol: float = 1e-8,
         trim: float = np.pi/2,
         stretch: float = 1.,
         degree: int = 2):

    '''Hyperelastic cylinder under torque

    This script simulates the deformation under torque of a hollow cylinder
    with a circular cutout in its wall. The cylinder is modeled as a
    Neo-Hookean solid that is clamped on both ends while being twisted at fixed
    angular increments.

    Dimensions are normalized to the radius of the cylinder, measured at half
    the thickness of the material. Since the problem is entirely kinematic,
    stiffness does not affect the resulting shape (it merely scales the energy)
    which leaves compressibility as the only relevant material parameter. We
    therefore define the strain energy density function simply as

        W = F : F - 3 - 2 log(|F|) + D (|F| - 1)^2

    Here F is the deformation gradient with respect to the reference geometry,
    and D is defined as ν / (.5-ν), using Poisson's ratio ν consistent with
    linear theory. The equilibrium state follows from minimizing the integrated
    energy subject to constraints.

    More information on Neo-Hookean and other hyperelastic materials can be found
    at <https://en.wikipedia.org/wiki/Hyperelastic_material>.

    Parameters
    ----------
    length
        Tube length.
    thickness
        Tube thickness.
    rotation
        Final rotation angle (degrees)
    increment
        Approximate angle increment (degrees)
    elemsize
        Approximate element size.
    poisson
        Poisson's ratio; value in the range [0,0.5).
    trim
        Radius or hole.
    stretch
        Length stretch factor.
    restol
        Newton tolerance.
    degree
        Polynomial degree.
    '''

    zgrid = length * np.linspace(-.5, .5, round(length / elemsize)+1)
    θgrid = np.linspace(-np.pi, np.pi, round(2 * np.pi / elemsize)+1)
    cylinder, (z, θ) = mesh.rectilinear([zgrid, θgrid], periodic=(1,))
    φ = θ - (z / length * np.pi / 180) * function.Argument('φ', shape=())
    if trim:
        cylinder = cylinder.trim(θ**2 + z**2 - trim**2, maxrefine=2)
    extrusion, r = mesh.line([1 - thickness/2, 1 + thickness/2], space='T')
    topo = cylinder * extrusion
    bezier = topo.boundary.sample('bezier', 5)

    ns = Namespace()
    ns.X = np.stack([z, r * np.sin(θ), r * np.cos(θ)]) # reference geometry
    ns.Xφ = np.stack([z * stretch, r * np.sin(φ), r * np.cos(φ)])
    ns.define_for('X', gradient='∇', jacobians=('dV',))
    ns.add_field('u', topo.basis('spline', degree=degree,
        removedofs=((0,-1),None,None)), shape=(3,)) # deformation, clamped
    ns.x_i = 'Xφ_i + u_i' # deformed geometry
    ns.F_ij = '∇_j(x_i)'
    ns.J = np.linalg.det(ns.F)
    ns.D = poisson / (.5 - poisson)
    ns.W = 'F_ij F_ij - 3 - 2 log(J) + D (J - 1)^2' # Neo-Hookean energy density

    energy = topo.integral('W dV' @ ns, degree=degree*2)

    args = {}
    clim = (0, 1) if stretch == 1 else None # fix scale for undeformed configuration
    for args['φ'] in np.linspace(0, rotation, round(rotation / increment) + 1):
        with log.context('{φ:.1f} deg', **args):
            args = solver.minimize('u,', energy, arguments=args).solve(restol)
            x, W = bezier.eval(['x_i', 'W'] @ ns, **args)
            export.triplot('energy.jpg', x, W, tri=bezier.tri, hull=bezier.hull,
                clim=clim, cmap='inferno_r', vlabel='strain energy density')
            clim = None # proceed with autmatic scaling

    return args


class test(testing.TestCase):

    def test_torque(self):
        args = main(rotation=1., increment=1., elemsize=1., poisson=.25)
        self.assertAlmostEqual64(args['u'], '''
            eNoN0stLE3AcAHAC58Ieq9SDh8Z87ffcQrCQJBcVJFMKKUzaQSExeihKFzvEwsBU6GJlJCYpLCQKLXQY
            VjazmHWI7ff4/rYx3Tx4CSotw2lQff6Gz3jJZKzAVMe4aYFWYDCszyY+xNZQ2FlhAvgxroBWsru0FjlI
            CD9E/WQXfYPXSRbqJBuoEf8gfuwlXtqGM+YaycAQfCUdMK5P0pvwJ34ApbUHOlG7XlVjOEc3mE2cVt+h
            evmcEvBNW9RTZCGc2rG1aIqewNvAyoppgnmYpBmWYufZBvPx+3iOVxJBfvMBMkALXXFyCUJ8Qjv1Oq/X
            F5TDlauTyrBZydUvZpcBucpGhcV4cB4cK2womde30JAZ1JvIi/NJ49K9kiZy2wRMNznIz7AuGmRlrJeF
            6TKt4VE27irHRfywe4ECf+e2cSZfuFIQF0fcZapYzLlXxCdxivWpGkmZX1apGJ0R2SYI+WTEcTXRRK4j
            H+omWygEefB2cSwxr2+YZjSoD/GXsk9NMS39co4iNSMUOx1NgZO/j5SpJK+KrAi3rIuW40XxMbJAiTga
            sfHP4rnsol75RfayCmVXNbzebGlOlyGTnKKv4S+2sjGUDWm1F0/vP6dceB+xqGKK5ayUdKe0Sx+zyVHx
            AHvEhI6THaJeP6KpaK6+AlWikrh0jhggl9VSNE6SqlR6GFdWmWJP5Hbp489KKp1rqCIWRgHcBGHcSuoS
            tc4CM+2UqAXuYoWHdVZpxKR1CPeYdm2jDZCjreiOzkAzztcdUEdeqf8TTI/eQCOwR/txUAdVG16LTxoH
            8cBF009+quOwTv4BeBNCLA==''')

    def test_stretch(self):
        args = main(rotation=0., elemsize=2., poisson=.25, restol=1e-8, trim=0., stretch=1.1, degree=3)
        self.assertAlmostEqual64(args['u'], '''
            eNodjy2OwlAURu8GipmgJ6gRff25jCGT1LOFBtd0AxVTMyEhIRgEVbgG2y3gCQmGdx+9ONSIUeOmG5j3
            YT7x5eQkp3MqP+71RvLimtuX3clJR3YinSP6TP1Pc278XvmkRB/TzkXpk2fwGwb/xxNpo7G8ucFc7EO2
            Zm33crjP7cL/RO88GKIVb/2e+XAnomkb/abgZwx+yeCPvJCyhydQeGqFpwjhKXt4AoWnVniKEJ6yhydQ
            eGqFpwjhyRN0fcfoqmJ0ZQZdeYIu//uuKkZXZtCVJ+jyP4NHV2bQ9Q9mTH+1''')


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)


# example:tags=finite strain,energy minimization
