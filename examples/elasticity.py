#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, numeric, solver
import numpy, unittest


def main(
    nelems: 'number of elements' = 12,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    degree: 'polynomial degree' = 2,
    figures: 'create figures' = True,
    solvetol: 'solver tolerance' = 1e-10,
 ):

  # construct mesh
  verts = numpy.linspace(0, 1, nelems+1)
  domain, geom = mesh.rectilinear([verts,verts])

  # create namespace
  ns = function.Namespace(default_geometry_name='x0')
  ns.x0 = geom
  ns.basis = domain.basis('spline', degree=degree).vector(2)
  ns.u_i = 'basis_ni ?lhs_n'
  ns.x_i = 'x0_i + u_i'
  ns.lmbda = lmbda
  ns.mu = mu
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'

  # construct dirichlet boundary constraints
  sqr = domain.boundary['left'].integral('u_k u_k' @ ns, geometry=ns.x0, degree=2)
  sqr += domain.boundary['right'].integral('(u_0 - .5)^2' @ ns, geometry=ns.x0, degree=2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # construct residual
  res = domain.integral('basis_ni,j stress_ij' @ ns, geometry=ns.x0, degree=2)

  # solve system and substitute the solution in the namespace
  lhs = solver.solve_linear('lhs', res, constrain=cons)
  ns = ns(lhs=lhs)

  # plot solution
  if figures:
    points, colors = domain.elem_eval([ns.x, ns.stress[0,1]], ischeme='bezier3', separate=True)
    with plot.PyPlot('stress', ndigits=0) as plt:
      plt.mesh(points, colors, tight=False)
      plt.colorbar()

  return lhs, cons


class test(unittest.TestCase):

  def test_p1(self):
    lhs, cons = main(nelems=4, degree=1, figures=False, solvetol=0)
    numeric.assert_allclose64(cons,
      'eNoz0TE01Yk20IHBvMQ8srGpgYFBqq4JQZo6tmHBsQB8/T8F')
    numeric.assert_allclose64(lhs,
      'eNqNkEsOQyEIRTeECZ8L6lqaDrv/LVRB07xZ4+AA90iMIHF6Md0jZvFpIBGgaPboT65umoR50iY/+pNb'
      'dynKTMacj/7kzsz/8PfSGLY2OGnvY5OprTJHLQZ4F8M4CVUcZ5WezsoknahLgM7rwLicGLWns5azf+M6'
      'KKfznr2/xYZHGg==')

  def test_p2(self):
    lhs, cons = main(nelems=4, degree=2, figures=False, solvetol=0)
    numeric.assert_allclose64(lhs,
      'eNqVkEsSAyEIRC+kVbb89CypLOf+VwiCVuIyNYuH0EAzXCDl1cr3G7DxVCnSmgVp9os7f3SYnZ/KBWO9'
      'nSZ88eS3jtBnEi3YVC6e/NYxGW0iKXLz5FPn/to//L1dZpvrpm4atw6BU0v1gCLwCq1KlblsSrGJ+A1K'
      'HAXYsuMCD2JG9QpH4NKYPhTIFslthBR4ILtFY3h16TaSCVWK1g7SbAFZtijlOpcit1Aa4xGK3iWde5Dr'
      'vDL2FlpG3h+923IP')
    numeric.assert_allclose64(cons,
      'eNoz0TE01Yk20EHAvMQ8mmJTAwODVF0Toml6ug0HjgUAg5ZcPw==')


if __name__ == '__main__':
  cli.run(main)
