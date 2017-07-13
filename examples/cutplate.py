#! /usr/bin/env python3

from nutils import *


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
  cons = model.optimize('lhs', sqr, droptol=1e-15)

  # construct residual
  res = domain.integral('ubasis_ni,j stress_ij' @ ns, geometry=ns.x0, degree=degree*2)

  # solve system
  lhs = model.solve_linear('lhs', res, constrain=cons)

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


def unittest():

  retvals = main(degree=1, maxrefine=2, withplots=False)
  assert debug.checkdata( retvals, '''
    eNplUkuuxDAIu86MFCRMgITjvEW3c//lhHwqvc6iinEJ2FZQXlpg7/J6Sfd2kRdj9YusDO7z9yloLheN
    JlTNMzmHa7YkJqiNohVzxEV9k5BJZvH8RBtykkiTMxERfvAcYIDlCnJVu3d1qXEX4Xyr+Dc/pJ7dQ13K
    X0VtHYkpRCKtzg5xyBkzPENj+X34jl6Rbce7cVsCn/5btVg/dgZwm5MEPD0Koy3/h18bnrmgL/5kg+o1
    T1LwVPwbEllYn+AkRS1284mLunJbt0JXjw0pGxgnOBGSMPftcuU4VoEzu58wSeDz4dyJUnXeT+n9BWsz
    kUM=''')

  retvals = main(nelems=4, degree=2, maxrefine=2, withplots=False)
  assert debug.checkdata( retvals, '''
    eNqlUktqBUEIvM4LtKC23+NkMdt3/2W6tRPIYwIhWQzaVumoJY2HDNK38XgoUV7gg3DaBTpW7Pn+HECK
    ckEMT5sX5ApM9h1g8bKb9foxJu8id5iJ6k8Yrb9fIPeYamEsM47VsqxUFqneFFw4udldLci5G79vYIGa
    G5yUeTcdxBkNyHLeMky16nMw3xJUrEsoW2wGUOqK2HIQqSCWaEfY21HqsksYIln0+QtxwOf+lwyZWdtJ
    2RmrOedukn2WPWMxtwLMJrVFiXqHee3FPLDsUZE58vCxt4+dN6NUoepavpTlKXhsqcNEfFRr9Ywq/le1
    iUxrNeoZtbdE+XQY2zkHAMTR5NVNVQMnb06YnixiPVnfLwMEtblyENDkFtZVrMu8XssCjuSC3cRUOwI7
    Nff1gIAppM5jyYzlsJ1Zlgwn+59H9fYBrtX3Zw==''')

  retvals = main(nelems=4, degree=2, maxrefine=3, withplots=False)
  assert debug.checkdata( retvals, '''
    eNqlU8tuAzEI/J1EWiTzhs/pYa/5/2NtcCo13UpVeohgmQEDQ/C4yYF6P243HYIn+IGD4gQ9Zuzx8TgA
    J3BCHJ7GJ+QMMPkKkHjZxXr90UhaRa4wE9XfMMQxH5FrTLUwEo5ttSwplh1Y3xhUOLrZVS1IXo1fNzBB
    zQUyZl5NB7FHA7TkS4apVn0KokuCinUJJYvFAEydEZvOGFgQSbQj5O0odtkpDOKK8R/EAbfItQURqu2k
    4KjmnLnsdMrusYhaAaLeHkpUXlgUz7z7eqpIlGPzsbc/WgWO/q6u5UtZYhnbVn1CpK1a5xn2u2+qjZhe
    q1H3EhJyiGyHepPPA1jbyiITm6xscLQaE8LUmoykO+v7ZYAM6zLC1k9pbq6rdPaPa2GOzkYZ9T8DVtsC
    Ozb39YCA0KPOI22NMJ257X5yyjA6+59Hdf8EpD73dg==''')


if __name__ == '__main__':
  cli.choose(main, conv, unittest)
