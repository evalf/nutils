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
    numeric.assert_allclose64(err,
      'eNoz0TE01Yk2sjAzT9U10zE1MDFN1TWNBQA3pQUL')
    numeric.assert_allclose64(cons,
      'eNpdT0EOwyAM+xCRapME8pZphx249v/HEjoqbQcU20G2owVWXufnLGjOIVqAqjlTc3gdYgsL1OaiFXPE'
      'kP4VwSUm+X/UdqQT2bgdEeEbLwMDLCPEVe3J6tTjIeGHbvLjH6w7e7bL+jeprSOxBDmr+v2Djn3M+wKf'
      'VEHS')
    numeric.assert_allclose64(lhs,
      'eNo1UMmNBEEIS6iQMGcRy2ienX8KO0Dvy5Y5bLADPx9Y4SE7yJBBqDXWVX7ITyC00TmtkWD+I3k8UA/d'
      'k+p3C5ApIHw2CjgGGTko/7rqoCWvnut81xlVMwfdRGRgGQcHfEiYLfHyGnLFJi5lvc0VvIGvce5U2Ta7'
      'SLykI/mREu3oJMz1Xtnm2Vb4tcTRvBihRKoFEsRukcC8iDT6Yv/+AR6yS2Q=')

  def test_quad_p2_refine2(self):
    err, cons, lhs = main(nelems=4, degree=2, maxrefine=2, figures=False)
    numeric.assert_allclose64(err,
      'eNoz0TE01Yk2NbAwSNU11zE0MDZK1TWNBQA28wT6')
    numeric.assert_allclose64(cons,
      'eNp1kLEOwzAIRH8oSIA5DN9SdeiQNf8/1pBOlTNYh+6dT9h2CI7X9bkOEjBOimOm60m5jKFRhtqcpZX6'
      'P8q5wtgyN9gTE+F1z/YMaKY25k/RquBWFiuV0OYyHbsuyjHjaYEFkQWHZOxeR8E5KkDiaduEA92voboN'
      'wPyugHpWgiSxHF8Dc38cqYX0YDpvB9K17y/9CXCk')
    numeric.assert_allclose64(lhs,
      'eNotUcmRBDEIS6hdxSUwsWzts/NPYWzUL2GQLI54FM+fqvq7/FkKwbv2U532rj4Jt30TFlUXV0Xou+KJ'
      'kH2xvc8bj5X1oJUPShvf4pdnlnFRY897Z+1bPyCDgSB/98eX4W+hzvf4qiKJYsPzkA8nb2cW6oy6VAwC'
      '5IfXh6APqBeNTzd1raTuBLMaVOM2uFoivsCUgXOWpbZtyOaYKVdpUrUTRbLaJ3fMxlx7VrxCMKtb4ckA'
      'bVQX7nT3G2mWNDuuCqZzsaUhNHCkUV3KIIHp73RnwzW9R8jTQ97znsDShN4AhzqOXxOWTYcGVSrCjy02'
      'yXHOT7KO1f8Pzy6Cyw==')

  def test_quad_p2_refine3(self):
    err, cons, lhs = main(nelems=4, degree=2, maxrefine=3, figures=False)
    numeric.assert_allclose64(err,
      'eNoz0TE01Yk2NTA2TtU11zE0MDJL1TWNBQA24gT7')
    numeric.assert_allclose64(cons,
      'eNp1kLEOwzAIRH8oSIA5DN9SdeiQNf8/1pBOlTNYh+6dT9h2CI7X9bkOEjBOimOm60m5jKFRhtqcpZX6'
      'P8q5wtgyN9gTE+F1z/YMaKY25k/RquBWFiuV0OYyHbsuyjHjaYEFkQWHZOxeR8E5KkDiaduEA92voboN'
      'wPyugHpWgiSxHF8Dc38cqYX0YDpvB9K17y/9CXCk')
    numeric.assert_allclose64(lhs,
      'eNotUsuRxTAIa8jM8BOYWnb2mP5beDb4JAISAhxfgvUnEvmRLRIwPtorK/SjOgnTfRPqmRcpk+sjX+6a'
      'F8s5PsLSNGs8QSOXzjfr5akGLopvv7hjdz1yS6PDh1/8+I2yuflil3dQEIOvrzk/7LyKDG/r6ELGFxi+'
      'Wz7E+GD0LP50XZd880pVnwaZdQekYrcXqExguTs41xqy2qgpj3+XdiCGLIqnQjc0qT4xnUtOmzPiiFA6'
      'ooRPsLnGXKL8qqz3PmpxHrXhWSKlT0wBdEm3anNV7sBxZoi7wgk0ZLZz4Kk9xgkaNQ4Fa7IwT2P1eTw6'
      'P8NkIG31/wPhOYLe')


if __name__ == '__main__':
  cli.choose(main, conv)
