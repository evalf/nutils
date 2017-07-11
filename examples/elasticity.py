#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, debug, model
import numpy


def main(
    nelems: 'number of elements' = 12,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    degree: 'polynomial degree' = 2,
    withplots: 'create plots' = True,
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
  cons = model.optimize('lhs', sqr, droptol=1e-15)

  # construct residual
  res = domain.integral('basis_ni,j stress_ij' @ ns, geometry=ns.x0, degree=2)

  # solve system and substitute the solution in the namespace
  lhs = model.solve_linear('lhs', res, constrain=cons)
  ns |= dict(lhs=lhs)

  # plot solution
  if withplots:
    points, colors = domain.elem_eval([ns.x, ns.stress[0,1]], ischeme='bezier3', separate=True)
    with plot.PyPlot('stress', ndigits=0) as plt:
      plt.mesh(points, colors, tight=False)
      plt.colorbar()

  return lhs, cons


def unittest():

  retvals = main(nelems=4, degree=1, withplots=False, solvetol=0)
  assert debug.checkdata(retvals, '''
    eNqlkEsKwzAMRK8Tg1ysnz/H6aLb3H9Zx7LaOhQKKVjMSG/AQgibAEqAbUs3+HzInB+xQxQxZVn6yUmZ
    hgrrUG649JNz0anYTFNae+OabP5LT+vmKn2sQKXUQ/souo8OKyc8VIjUQ+6jw5pLGyGh9gpNHx3WkthC
    zO+Q+eiwX/W05X7fL9fFw/zz5bcKEJ7x0YpY''')

  retvals = main(nelems=4, degree=2, withplots=False, solvetol=0)
  assert debug.checkdata(retvals, '''
    eNq1ksEOwyAIhl+nTXQRENDH2aHXvv9xFqRJ1+ywLEtqvr/wi0gLaakJ6pqWpTzS29NA+5Y5cSlqpE4X
    znj4oGPd8qjXjvdBZb4w4tNHQMUJziJyYcSnr5LSJDpZroy4+0Z/5RveJ8C92M1QBe2mDKOypHyKyOSw
    aod2UKhWG4oqmOEUkclhbQLoW6TYaQRuOEVkclgbixUTISMCiW8JEZkc1ibkjdVmROR5SojI5LCO3+I+
    k/25/3X9/tX+3eGntab1Bd2X0og=''')


if __name__ == '__main__':
  cli.choose(main, unittest)
