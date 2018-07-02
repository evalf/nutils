#! /usr/bin/env python3

from nutils import *
import numpy, unittest
from matplotlib import collections


def main(
    nelems: 'number of elements' = 12,
    lmbda: 'first lamé constant' = 1.,
    mu: 'second lamé constant' = 1.,
    degree: 'polynomial degree' = 2,
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
  bezier = domain.sample('bezier', 3)
  x, stress = bezier.eval([ns.x, ns.stress[0,1]])
  with export.mplfigure('stress.png') as fig:
    ax = fig.add_subplot(111, aspect='equal')
    ax.autoscale(enable=True, axis='both', tight=True)
    im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, stress, shading='gouraud', cmap='jet')
    ax.add_collection(collections.LineCollection(x[bezier.hull], colors='k', linewidths=.5, alpha=.1))
    fig.colorbar(im)

  return lhs, cons


class test(unittest.TestCase):

  def test_p1(self):
    lhs, cons = main(nelems=4, degree=1, solvetol=0)
    numeric.assert_allclose64(lhs, 'eNpjYICBFGMxYyEgTjFebDLBpB2IF5tkmKaYJgJxhukPOIRrYBA'
      '1CjJgYFh3/vXZMiMVQwaGO+e6zvYY2QBZR86VnO2FsorPAgAXLB7S')
    numeric.assert_allclose64(cons, 'eNpjYICDBnzwhykMMhCpAwEBQ08XYg==')

  def test_p2(self):
    lhs, cons = main(nelems=4, degree=2, solvetol=0)
    numeric.assert_allclose64(lhs, 'eNpjYECAPKNKw3lAWGmYZyRlwmfyxviNMZ+JlAmvKZcpMxBymfK'
      'abjXdYroZCLcAWT+QIJIxDGcMAwyY9f5e3HDe5Fyc0XvDQv3+C4LnFp3tMPpg+F2f84LAuYqzrUasRuf'
      '1DS/8Plt9tsvol+ErfbELbOfKzgIAw+QzeA==')
    numeric.assert_allclose64(cons, 'eNpjYEACDaTBH6YIyECBOcgQAP8cIg8=')


if __name__ == '__main__':
  cli.run(main)
