#! /usr/bin/env python3

from nutils import *
import unittest


def main(
    nelems: 'number of elements, 0 for triangulation' = 0,
    maxrefine: 'maxrefine level for trimming' = 2,
    radius: 'cut-out radius' = .5,
    degree: 'polynomial degree' = 1,
    poisson: 'poisson ratio' = .25,
    withplots: 'create plots' = True,
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
  ns = ns | dict(lhs=lhs)
  if withplots:
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
    retvals = main(degree=1, maxrefine=2, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNplUkuuwzAIvE4rGYnBgM1x3iLb3n9Z40+kly4iwwQzHxnlpQX2Lq+XdO8XeTFWv8jKwD5/n4LmctEY
      QtU8E3O45kjWBLXRtGKOuKhvEDLBbJ6faENuEmlyNiLCTz0XGGBJQa5qN1eXGncTzreKf/tD6uEe6lL+
      amrryJpCJNLqnBCHnDXDMzSW34fv6BU5drwbtyXw6b9V6+vHzgBuc5OAp0dhtOX/4IvhmQv6wk82qF7z
      JAVPxb8hkcUmP0lRiz184qKu3Nat0DVsQ8oujLM4EZIwbzM7x0EFzux+wiSBz4dzJ0rVeT+l9xdsEpFC'''))

  def test_quad_p2_refine2(self):
    retvals = main(nelems=4, degree=2, maxrefine=2, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqlUkFuxDAI/M6uFCQzBmye00Ou+/9jMTg9bFOpag8RBMYwMPDxkIP1eTwe2iafNA5uvZ+kR8ReH6+D
      WJucNI/hFnGPQMdYAchIu1DvH5pjFbnLmaj+lOPofpLc51QzB+lzW00L5bSN858nMs/D7K4WeV/E7wlE
      Un0lO7vfTUdzj0Zs3m8Rppr1MYFbgIpVCYXNhSB2jYiF0xpnCjLLEYxylKtsCMPM0br/QhwaIrkdkebL
      ei/6GMg5gZFyX4oBpQBgtU2Z+T9tzJUPk3QuFYHpG99q+4tR2D6zLydr+VIWXdq2qQ6YsVWrfsYZ/6va
      zGvwWI0Ol9ybN7kctHL2ARBjkQwwuiYLGmw5Fk1TKzBjP3+7DJKmJaP0S0/f2KFSr79di2LxXpJLK0hX
      2wIPLuz7ARFiyjwPtyVvOLA9S8hQzn+P6vkJim73Tw=='''))

  def test_quad_p2_refine3(self):
    retvals = main(nelems=4, degree=2, maxrefine=3, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqlUklqBDEM/M4M2KB9eU4Oc53/H2NLnkCGDoTk0EhtlbYq4bjJQL2P202B5TF9IJA/po719vx4jokK
      6z2Gp/Fj5nrgDYhB4mU36v0jSNpFrmImqj/FEGE1keuYasVIOI7VsqRYFrD+Maji6GZXtWay+08DrKDm
      DjJmXm0346w20ZIvEaZa9SmILgEq1iWULDZiYup6seUAYIVIoh0hb0dRX8Ig2nL5F+JMdyg2RJqVFOjh
      nLnscsqetYhaAaJmDyWK1bAonHnP9VKRKOHgsdmHVoGj/2tq+VKWWODYqk+IdFTrPMPu+0e1sXRb1Kh7
      FG8JwsehZvJ1AJutKDCxVbXpaI0J6/1mIsnJ+n4ZU8C6g7BDS5TUSa57O724FmZpYVCgs1lPS3Vs7PsB
      TcI98DqPNIJyFtvZvVVP738e1f0TqD33ag=='''))


if __name__ == '__main__':
  cli.choose(main, conv)
