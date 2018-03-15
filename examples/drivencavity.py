#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, numeric, solver, _
import numpy, unittest


@log.title
def postprocess(domain, ns):

  # confirm that velocity is pointwise divergence-free
  div = domain.integrate('(u_k,k)^2' @ ns, geometry=ns.x, degree=9)**.5
  log.info('velocity divergence: {:.2e}'.format(div))

  # plot velocity field as streamlines, pressure field as contours
  x, u, p = domain.elem_eval([ ns.x, ns.u, ns.p ], ischeme='bezier9', separate=True)
  with plot.PyPlot('flow') as plt:
    tri = plt.mesh(x, mergetol=1e-5)
    plt.tricontour(tri, p, every=.01, linestyles='solid', alpha=.333)
    plt.colorbar()
    plt.streamplot(tri, u, spacing=.01, linewidth=-10, color='k', zorder=9)


def main(
    nelems: 'number of elements' = 12,
    viscosity: 'fluid viscosity' = 1e-3,
    density: 'fluid density' = 1,
    degree: 'polynomial degree' = 2,
    warp: 'warp domain (downward bend)' = False,
    figures: 'create figures' = True,
  ):

  log.user( 'reynolds number: {:.1f}'.format(density / viscosity) ) # based on unit length and velocity

  # create namespace
  ns = function.Namespace()
  ns.viscosity = viscosity
  ns.density = density

  # construct mesh
  verts = numpy.linspace( 0, 1, nelems+1 )
  domain, ns.x0 = mesh.rectilinear( [verts,verts] )

  # construct bases
  ns.uxbasis, ns.uybasis, ns.pbasis, ns.lbasis = function.chain([
    domain.basis( 'spline', degree=(degree+1,degree), removedofs=((0,-1),None) ),
    domain.basis( 'spline', degree=(degree,degree+1), removedofs=(None,(0,-1)) ),
    domain.basis( 'spline', degree=degree ),
    [1], # lagrange multiplier
  ])
  ns.ubasis_ni = '<uxbasis_n, uybasis_n>_i'

  # construct geometry
  if not warp:
    ns.x = ns.x0
  else:
    xi, eta = ns.x0
    ns.x = (eta+2) * function.rotmat(xi*.4)[:,1] - (0,2) # slight downward bend
    ns.J_ij = 'x_i,x0_j'
    ns.detJ = function.determinant(ns.J)
    ns.ubasis_ni = 'ubasis_nj J_ij / detJ' # piola transform
    ns.pbasis_n = 'pbasis_n / detJ'

  # populate namespace
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.p = 'pbasis_n ?lhs_n'
  ns.l = 'lbasis_n ?lhs_n'
  ns.sigma_ij = 'viscosity (u_i,j + u_j,i) - p δ_ij'
  ns.c = 5 * (degree+1) / domain.boundary.elem_eval(1, geometry=ns.x, ischeme='gauss2', asfunction=True)
  ns.nietzsche_ni = 'viscosity (c ubasis_ni - (ubasis_ni,j + ubasis_nj,i) n_j)'
  ns.top = domain.boundary.indicator('top')
  ns.utop_i = 'top <n_1, -n_0>_i'

  # solve stokes flow
  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n (u_k,k + l) + lbasis_n p' @ ns, geometry=ns.x, degree=2*(degree+1))
  res += domain.boundary.integral('nietzsche_ni (u_i - utop_i)' @ ns, geometry=ns.x, degree=2*(degree+1))
  lhs0 = solver.solve_linear('lhs', res)
  if figures:
    postprocess(domain, ns(lhs=lhs0))

  # solve navier-stokes flow
  res += domain.integral('density ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*(degree+1))
  lhs1 = solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
  if figures:
    postprocess(domain, ns(lhs=lhs1))

  return lhs0, lhs1


class test(unittest.TestCase):

  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=False, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqFj0EOAzEIAz8UJHAIWd5S9dj/f2GBDVVvPWHhiU10iI4XAbw/tAYgnpOwOYSOJa45MS0AC2MychFC'
      'JYXDzuJvxgGCnJyiDRIA9VS5MsGzjN5TG9TOT9uTRd+w7fA81a5Vk0T2KmQaW8VefD6jFo93OGr+HOCz'
      '0EaaaICa6Bbqmm7pEn7fWJVLrA==')
    numeric.assert_allclose64(lhs1,
      'eNpVj8ltBTEMQxsaAdqXWoIcp/8WvkUnAXKiQT6Qsj/izxd5Vr0UT+TkKqlJvOSPTQw0xF7KE7DzGofg'
      '3EdngBy1BmGWdTume5MDQH9HKGvL46+cNMTXULukmvcNOhWlUiZwhH3+zUWL3NafGZKe2ki5oKaq9yAZ'
      'X0Kcew3x/dQJygf7o4lfFivKJQ0gOfslp4dxejdj/iRAnRlEhGGDvz+PoEvw')

  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=2, warp=False, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqVj8sNxDAIRBsykvnjWlZ7TP8tbEwy0V5jXxC8YQYbbOOjqX6QD1pRaxcsagfZIOOauwgR7YaUZKPq'
      'c12Fzh4Vi+zGklgv0Rf+QNnjIoQ1/wmV4t4ltXpXOPsesF/AvD+BBEggCeiTC24wI7iJzb6Rz3dQnK6h'
      'tRu6Iu+lqffStO5ksm1Wc992apitj6zyJubzoYGEoIGEoCEkIURBEkIUJJnfH/O8bL4=')
    numeric.assert_allclose64(lhs1,
      'eNpVkVtuBSAIRDekCcPbtTT97P63UAFv0n5pYBwOoy7o+tKD/Nm2NkTpZ+uCO+rcSpl1MclubHY+I/Xo'
      'znYTmQpLV5LMnla8OhDXVhiSq8PHTp3H0S+Y4gk82itMpqBK7RUaPkAPlTFgm+P4X0KJ8QrUCrb0CSFD'
      'x1wjsnAxDwQNY0wTAkvx35detLeQWjR3Fijwj/dw8GxYGVzFh1P9TSfY2yzxsD5ZUGehkbfgC6S9OhAx'
      'glAdb2tvUbES3lTHEifeRe39VnvecJxoznxeKL64IXn/yA4ZnC1q3K5pM81IGidnsyPohej7F3rzeSo=')

  def test_p1_warped(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=True, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqFjsENBDEIAxsKUkgIkFpO99z+WzgI69vnvow8lo00lvahwcwXrRayU2nYDEPaEtmp08Uu0gBagHib'
      '57HZy3jtQIA4SJYBhCGeZCgfg1316O0TAIE8a3cX/cvi6OnonFIvS6/duUfVm/IhslYcFkRtnAdc63dE'
      'kECAkMAKYQYrGOnfH4IBS84=')
    numeric.assert_allclose64(lhs1,
      'eNo9j0sOAzEIQy80SOEPZ6m6nPtfocGZaTZYwfiBXWzXhyykbvLLKnkqiXTfZJetyqnhqjcFGjEfxC0Q'
      'pVlTWVgxauEOoayYzVWGrAdCzgzHG04SBqw4gypsfho5tI3lclAoLHqsf1ytTogXs9Nit3JvwitGiMhz'
      'lRqAzBkIWavPrDpCRAIraqsNl8XOLSbrUHg/HFXZgGRXj9XZkRl7+TGs7w+8L0wG')


if __name__ == '__main__':
  cli.run(main)
