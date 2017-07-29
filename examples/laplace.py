#! /usr/bin/env python3

from nutils import mesh, plot, cli, function, log, debug, solver
import numpy


def main(
    nelems: 'number of elements' = 10,
    degree: 'polynomial degree' = 1,
    basistype: 'basis function' = 'spline',
    solvetol: 'solver tolerance' = 1e-10,
    withplots: 'create plots' = True,
  ):

  # construct mesh
  verts = numpy.linspace(0, numpy.pi/2, nelems+1)
  domain, geom = mesh.rectilinear([verts, verts])

  # create namespace
  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(basistype, degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.fx = 'sin(x_0)'
  ns.fy = 'exp(x_1)'

  # construct residual
  res = domain.integral('-basis_n,i u_,i' @ ns, geometry=ns.x, degree=degree*2)
  res += domain.boundary['top'].integral('basis_n fx fy' @ ns, geometry=ns.x, degree=degree*2)

  # construct dirichlet constraints
  sqr = domain.boundary['left'].integral('u^2' @ ns, geometry=ns.x, degree=degree*2)
  sqr += domain.boundary['bottom'].integral('(u - fx)^2' @ ns, geometry=ns.x, degree=degree*2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # find lhs such that res == 0 and substitute this lhs in the namespace
  lhs = solver.solve_linear('lhs', res, constrain=cons)
  ns |= dict(lhs=lhs)

  # plot solution
  if withplots:
    points, colors = domain.elem_eval([ns.x, ns.u], ischeme='bezier9', separate=True)
    with plot.PyPlot('solution', index=nelems) as plt:
      plt.mesh(points, colors, cmap='jet')
      plt.colorbar()

  # evaluate error against exact solution fx fy
  err = domain.integrate('(u - fx fy)^2' @ ns, geometry=ns.x, degree=degree*2)**.5
  log.user('L2 error: {:.2e}'.format(err))

  return cons, lhs, err


def unittest():

  retvals = main(nelems=4, degree=1, withplots=False, solvetol=0)
  assert debug.checkdata(retvals, '''
    eNqVjjsKAzEMRK+zCzJ49LHk46Rwu/cvYylsICRNCjHijTUe0KEEO+k4dJqv5tQgPHMR8w1ig86ciwFY
    bZKE22pK1+P6GMeQX3yKxS+Ojv1evvhu838ZG47UENdUsHKGI3Sk3uXQNYqbeirztFTRzJN3WciY5dd/
    W+esO1WrvLs8NF4+onxhrbyNyz9phPXVBp1PMQRX2g==''')

  retvals = main(nelems=4, degree=2, withplots=False, solvetol=0)
  assert debug.checkdata(retvals, '''
    eNqlkDtuxDAMRK9jAxSgISV+jpNi271/uSQdp0jgKgaEJ2lmbI5BxyLsk45jcAi/hhFEojg2Zm6ceF0c
    0CW1sW15EYSI+RqL3l/vP2urypPmy+xJiwh/0jDzeRKzwz8qsFiPKxta3Dx7RDOgGFv7fNdS077PkZJC
    WC5FRmjTeBfvqpjgyxfRehqLwt7+Nc2Ld30g+n2w+m76Fa2L2+X32fmfX5KB9R3wXwG7ApjFk8DlUDo/
    ykV8bg==''')


if __name__ == '__main__':
  cli.choose(main, unittest)
