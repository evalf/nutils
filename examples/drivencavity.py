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
    postprocess(domain, ns | dict(lhs=lhs0))

  # solve navier-stokes flow
  res += domain.integral('density ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*(degree+1))
  lhs1 = solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
  if figures:
    postprocess(domain, ns | dict(lhs=lhs1))

  return lhs0, lhs1


class test(unittest.TestCase):

  def test_p1(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=False, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqFTzkSAzEI+5A9A+KweUsm5f7/Cwt4t0mTCg0SktDBNj4ToH1NGwD3nFgU19RhHFoT4n5NT0IItUig'
      'DQLOZ/HXA7QOAeECLzEZQJ8qtRdIpGbu+Vzom0qiP2lCLUkza2ZFbXz4tt2VmZe3RLwKpO2m5xn1lK5k'
      '1OMUiC4yU7JKArUoRQqepzSkwApEe2RMg0yxIjLEK4S+N1c4S6w=')
    numeric.assert_allclose64(lhs1,
      'eNpVT7ltBEEMa2gH0P/UYjjc/lvwiPIFF1EQKZKyh/35ORZZ7/HHo33wiHK8xx5tb6CzvCcuQUazuAqC'
      'osJ9sEUTCtWo9ejC7RXk4A1JEJGMlI/5Ed+F6CpFLZeoUJhyqmDDZP0V5zXNxtW9VlJdQwllDKrIKoQb'
      'VdkIDdn+T9P27ZaNSxI4cOg2NjLDcF8iVK8ixF8GUiOCl7vaZNDvH5IHS/I=')

  def test_p2(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=2, warp=False, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqVkcsNAyEMRBtaJGzjD7VEOab/FjIeVlFyjLggeJ43y65L/HpY2noNv8aO2r0RNXuNdY0lNXsTqsoD'
      'LS2i5nOfjU2yJSp9sDXqT/QPP1An4bGORSy/CdOe8NZvZoULSfEDTC74pRtpsQng0xW0Mxe4/XSDkTcQ'
      'BgkYSeiawUoiMAfMYTTbjrxD0+7QZPuR2ZpAem7OiCx+aJVbE/OzMMMCGFlELYsekdYgFTOnIpocIapI'
      'I2hy/4IddR4jzyM837kcbbo=')
    numeric.assert_allclose64(lhs1,
      'eNpVkUtuBDEIRC/klijzP0uU5dz/CmPALSUru6FcvKJlQdePJOLz6HrAQp9HFsxQ5yMUURfl6MazjWmk'
      '5tkVU5apbG5tkOqrteqAbRSK2NXZqa1Mg/c3OY/A5uKvpwi1l4vbACWyOhvhM8TT/hKyXwtYg8oVgnlm'
      'bT8MUbiYB4yJuGmWsLn4z0sr2lMI4QkEcvzjzTarhMytOJw9RXqBx4OgN1ngYr27oH4iHqdgCyTtCbiN'
      'wEXGW1vITWHrbHUskVcJ0fu3QI2zjWjOyt7Dis/PkmzeOt9sLLrbNXTGa2EdnPDmTUZnp98vZe15Gw==')

  def test_p1_warped(self):
    lhs0, lhs1 = main(nelems=3, viscosity=1e-2, degree=1, warp=True, figures=False)
    numeric.assert_allclose64(lhs0,
      'eNqFjrutQzEMQxeyAFN/z/Lwyrv/CpHkpEiVSgIPQVIXbP0RA/yQLQCnL3EIHtJlqqevpPpDXsCFWyCc'
      'GHKQe4SfGWXAABTpsA8oQbMJO7IFpNtc1rhRjrcz3b/bOHiGmdrdUaloxUX0TtZ9e+XIWBA+DlKzeqKI'
      'x6Rx+t1eloktx25HGW4GZ+x+qmVPRtVYP9UyQA7P1P3/AnrIS7Y=')
    numeric.assert_allclose64(lhs1,
      'eNo9T8utBDEIayiRMH9qWe1x+m9hA5n3TljY2FgXbH22OuezbWkGem4WomfrUsro6SbybD8El/dio3hA'
      'SowCDJ5TdbMBAslmglLGyzmGMMAn7TXf7DqxbJgLhtolomJikVazcfVq6X9cUl3XE6Ovm5+8OJ+AogEz'
      '/7XS8QdiXgRR3luxAcw+ZlK3Llj1luJXCXTuKZWRExJV1FLDre3n+RbQ9welMUvq')


if __name__ == '__main__':
  cli.run(main)
