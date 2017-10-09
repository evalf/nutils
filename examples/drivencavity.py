#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, debug, solver, _
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
    withplots: 'create plots' = True,
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
  ns.top = domain.boundary['top'].indicator()
  ns.utop_i = 'top <n_1, -n_0>_i'

  # solve stokes flow
  res = domain.integral('ubasis_ni,j sigma_ij + pbasis_n (u_k,k + l) + lbasis_n p' @ ns, geometry=ns.x, degree=2*(degree+1))
  res += domain.boundary.integral('nietzsche_ni (u_i - utop_i)' @ ns, geometry=ns.x, degree=2*(degree+1))
  lhs0 = solver.solve_linear('lhs', res)
  if withplots:
    postprocess(domain, ns | dict(lhs=lhs0))

  # solve navier-stokes flow
  res += domain.integral('density ubasis_ni u_i,j u_j' @ ns, geometry=ns.x, degree=3*(degree+1))
  lhs1 = solver.newton('lhs', res, lhs0=lhs0).solve(tol=1e-10)
  if withplots:
    postprocess(domain, ns | dict(lhs=lhs1))

  return lhs0, lhs1


class test(unittest.TestCase):

  def test_p1(self):
    retvals = main(nelems=3, viscosity=1e-2, degree=1, warp=False, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqFUltuBDEIu86uNFS8Ew7U+1+hBMKq/eoXCHtsYoaelz6k7+f1AmZc32APM8WpwAuz0cco9FQWT4In
      IMhnkI3SaYL9Dv7VuIRkCp5mACBmrk8VS5NRCpg5DACD/HJrLfiIreA4q/q2qkC0rCji6CW78T5GPT9e
      iahHLxBS1KEMYwgwjHGBsRmXMcGvJ6MF9VV7moe3gJAdXMIqHTGSXgYVb7KtsN2sE5ZdDKkjHI3Y+yBJ
      qDom4It6iSsObKT1MGlmXm03sM/ZTjpLOlpCjT92tola9dpkKtGp4Koqn/vc8ElxnwEpdY5Lo/zzL6lX
      LuQSJ5cigqI2M3bUIWVvLPtEiqqIxTDr01ew7x/oQZfn'''))

  def test_p2(self):
    retvals = main(nelems=3, viscosity=1e-2, degree=2, warp=False, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqVU22O7DAIu86s1DzFfOdAc/8rvATC7uzPrSo1IsYYQ/G85IF8Pa8XO+t76DOWxToHEMt7yDMEMc/B
      iDgDFOQJZZ2rDjzzKkB0Aots/RH6h/oNhVohCOyfCKZAclGs5DKFngtoAea/n3c0usGj0aPh39q6Yhcc
      XZFkZp/Yz3vYrmwcJ8DL/JI6X1KXjLhDDpb99LdzAMlGIzQRnyr324mdNzqx80YnjpYzWk/LGa2n5Wze
      PX1ZqGtwUcEMZf6MOAflGsJujMolmEd5oMwVIc5ITNWLZUtX2EqXIijtWppDX4bMoHk9UfPkcuUKiMzk
      cnErQVcqoYQN8mWfCtmvTbAcq1wguNQRnRJx5KISGClGaZYJxJrjFotc0RFyVxHT8UvvIqfqkGu6rVPs
      Vp/Q21lt2ZbVXsz0QjxynJjCtUJes9vjluLW2kI5W5izvNu4/B5E77SSc5tjc9Y3LheOPt8mmdXy8f1F
      WJSS9e4O9PyfW05UZ4th34vy9R8GDefR'''))

  def test_p1_warped(self):
    retvals = main(nelems=3, viscosity=1e-2, degree=1, warp=True, withplots=False)
    self.assertTrue(debug.checkdata(retvals, '''
      eNqFUstxBTEIa2ffjMkYzM8Fvf5bCAazOeYEg1gJyYvj4YH8Gc8DhIhfkBFlnwpkKwY8hHmfupztCxqA
      FgC4zU+z0WvwL0cvAAZyyBqIAftBSDEH6KpZ7xwagEb+1C4XvGTRzDPRtbhO5lm6a1PRm2IiLBKNBaJG
      eYBr3d4rvdEL0ButAi3TKi0yf0ZEC6yUFtitjiDaifN0O1VllYEA9CZL2fiqhJFw5aesImUEV35r0zm5
      rggIYm40eQTHKUuC9w1YCrCjpidRSRVQLmevnM9t2bRMsOnKvBinnoaIrqvFKYgRXJLMWUHakiQhUq94
      Kq542vLCNEsF80XDlNtOEdu+8zdBSU6N499kP7+4m5gf'''))


if __name__ == '__main__':
  cli.run(main)
