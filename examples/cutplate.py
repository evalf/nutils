#! /usr/bin/env python3

from nutils import *
import unittest


def main(
    nelems: 'number of elements, 0 for triangulation' = 0,
    maxrefine: 'maxrefine level for trimming' = 2,
    radius: 'cut-out radius' = .5,
    degree: 'polynomial degree' = 1,
    poisson: 'poisson ratio' = .25,
    figures: 'create figures' = True,
  ):

  ns = function.Namespace(default_geometry_name='x0')
  ns.lmbda = poisson / (1+poisson) / (1-2*poisson)
  ns.mu = .5 / (1+poisson)

  # construct domain and basis
  if nelems > 0:
    verts = numpy.linspace(0, 1, nelems+1)
    domain0, ns.x0 = mesh.rectilinear([verts, verts])
  else:
    assert degree == 1, 'degree must be 1 for triangular mesh'
    domain0, ns.x0 = mesh.demo()
  domain = domain0.trim(function.norm2(ns.x0) - radius, maxrefine=maxrefine)
  ns.ubasis = domain.basis('spline', degree=degree).vector(2)

  # populate namespace
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.x_i = 'x0_i + u_i'
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
  ns.R2 = radius**2
  ns.r2 = 'x0_k x0_k'
  ns.k = 3 - 4 * poisson # plane strain parameter
  ns.uexact_i = '''.1 <x0_0 ((k + 1) / 2 + (1 + k) R2 / r2 + (1 - R2 / r2) (x0_0^2 - 3 x0_1^2) R2 / r2^2),
                       x0_1 ((k - 3) / 2 + (1 - k) R2 / r2 + (1 - R2 / r2) (3 x0_0^2 - x0_1^2) R2 / r2^2)>_i'''
  ns.uerr_i = 'u_i - uexact_i'

  # construct dirichlet boundary constraints
  sqr = domain.boundary['left'].integral('u_0^2' @ ns, geometry=ns.x0, degree=degree*2)
  sqr += domain.boundary['bottom'].integral('u_1^2' @ ns, geometry=ns.x0, degree=degree*2)
  sqr += domain.boundary['top,right'].integral('uerr_k uerr_k' @ ns, geometry=ns.x0, degree=max(degree,3)*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # construct residual
  res = domain.integral('ubasis_ni,j stress_ij' @ ns, geometry=ns.x0, degree=degree*2)

  # solve system
  lhs = solver.solve_linear('lhs', res, constrain=cons)

  # vizualize result
  ns = ns(lhs=lhs)
  if figures:
    vonmises = 'sqrt(stress_ij stress_ij - stress_ii stress_jj / 2)' @ ns
    x, colors = domain.simplex.elem_eval([ns.x, vonmises], ischeme='bezier5', separate=True)
    with plot.PyPlot('solution') as plt:
      plt.mesh(x, colors, cmap='jet')
      plt.colorbar()

  # evaluate error
  err = numpy.sqrt(domain.integrate(['uerr_k uerr_k' @ ns, 'uerr_i,j uerr_i,j' @ ns], geometry=ns.x0, degree=max(degree,3)*2))
  log.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

  return err, cons, lhs


def conv(degree=1, nrefine=4):

  l2err, h1err = numpy.array([main(nelems=2**(1+irefine), degree=degree)[0] for irefine in log.range('refine', nrefine)]).T
  h = .5**numpy.arange(nrefine)

  with plot.PyPlot('convergence') as plt:
    plt.subplot(211)
    plt.loglog(h, l2err, 'k*--')
    plt.slope_triangle(h, l2err)
    plt.ylabel('L2 error')
    plt.grid(True)
    plt.subplot(212)
    plt.loglog(h, h1err, 'k*--')
    plt.slope_triangle(h, h1err)
    plt.ylabel('H1 error')
    plt.grid(True)


class test(unittest.TestCase):

  def test_tri_p1_refine2(self):
    err, cons, lhs = main(degree=1, maxrefine=2, figures=False)
    numeric.assert_allclose64(err, 'eNp7rF1rCAAFPAG9')
    numeric.assert_allclose64(cons, 'eNpjaHhnLGTM0PDQkKHhw01TFYaGwlsMDTA43STXhKFBxwTErj'
      'nHeY6hoe8sQ4PjWZh8rjpDw9ubDA3v1K9eZGg4c56hAQByvx3s')
    numeric.assert_allclose64(lhs, 'eNoBUACv/5sz7jMSM8oy4TGAMfDZNSQ5MnHazjNHNDI0aTRONJc'
      '0bTQCNCw0bTPrznzOCc4wzo7No81BzYTN481yzm/ObSdQ0O3Zi9LuJ9XRKtDMzyvPk+wnvw==')

  def test_quad_p2_refine2(self):
    err, cons, lhs = main(nelems=4, degree=2, maxrefine=2, figures=False)
    numeric.assert_allclose64(err, 'eNpr0kjRAwADegE9')
    numeric.assert_allclose64(cons, 'eNpjaIi7l6Iw995sJYYGGDyqj2D/MESwWYwR7PXGE0wmmGSYGJ'
      'owmLyHi/ueRajwOyutDGNPP6t8Dy5z7g7crphzb27fvbLm4rPz189VnAMAG9QxcQ==')
    numeric.assert_allclose64(lhs, 'eNoBjABz/4g3Xt5kIJ3emyJCybk1szJJMO4vxS9eNGs0mDOMMhE'
      'y+DFxNGs0+jN0MxwzBDN8NH00STT/M8MzrzOQNJA0aDQxNAA07zOjNz7OX81hzU3NMsiCy9TNdc1bzU7'
      'NGyPxzszOM866zZfNI94PJALRMc9FzgDO3CKm1L/Ry8+3zlzO7Nvd1KzR5s/XznjO0+ZFww==')

  def test_quad_p2_refine3(self):
    err, cons, lhs = main(nelems=4, degree=2, maxrefine=3, figures=False)
    numeric.assert_allclose64(err, 'eNqr1UjUAwADYAE1')
    numeric.assert_allclose64(cons, 'eNpjaIi7l6Iw995sJYYGGDyqj2D/MESwWYwR7PXGE0wmmGSYGJ'
      'owmLyHi/ueRajwOyutDGNPP6t8Dy5z7g7crphzb27fvbLm4rPz189VnAMAG9QxcQ==')
    numeric.assert_allclose64(lhs, 'eNoBjABz/583Xt5kIJ3emyIwybs1tDJKMO4vxS9eNGw0mDOMMhE'
      'y+DFxNGs0+jN0MxwzBDN8NH00STQANMMzrzOQNJA0aDQxNAA07zOtNz/OYM1hzU3NIMh/y9TNdc1bzU7'
      'NGyPvzsvOM866zZfNI94yIwLRMs9FzgDO3CKn1MDRy8+3zlzO7Nvd1KzR5s/XznjOhRRE6A==')


if __name__ == '__main__':
  cli.choose(main, conv)
