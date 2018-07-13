#! /usr/bin/env python3

import nutils, numpy

def main(etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (h/th-std/spline)' = 'h-std',
         degree: 'polynomial degree' = 2,
         nrefine: 'number of refinement steps (-1 for unlimited)' = -1):

  domain, geom = nutils.mesh.unitsquare(2, etype)

  x, y = geom * 2 - 1
  exact = (x**2 + y**2)**(1/3) * nutils.function.cos(nutils.function.arctan2(y+x, y-x) * (2/3))
  domain = domain.trim(exact-1e-15, maxrefine=0)
  linreg = nutils.util.linear_regressor()

  for irefine in nutils.log.count('level'):

    ns = nutils.function.Namespace()
    ns.x = geom
    ns.basis = domain.basis(btype, degree=degree)
    ns.u = 'basis_n ?lhs_n'
    ns.du = ns.u - exact

    sqr = domain.boundary['trimmed'].integral('u^2' @ ns, geometry=ns.x, degree=degree*2)
    cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

    sqr = domain.boundary.integral('du^2' @ ns, geometry=ns.x, degree=7)
    cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15, constrain=cons)

    res = domain.integral('basis_n,k u_,k' @ ns, geometry=ns.x, degree=degree*2)
    lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

    ndofs = len(ns.basis)
    error = numpy.sqrt(domain.integrate(['du^2' @ ns, 'du_,k du_,k' @ ns], geometry=ns.x, degree=7, arguments=dict(lhs=lhs)))
    rate, offset = linreg.add(numpy.log(len(ns.basis)), numpy.log(error))
    nutils.log.user('ndofs: {ndofs}, L2 error: {error[0]:.2e} ({rate[0]:.2f}), H1 error: {error[1]:.2e} ({rate[1]:.2f})'.format(ndofs=len(ns.basis), error=error, rate=rate))

    bezier = domain.sample('bezier', 9)
    x, u, du = bezier.eval([ns.x, ns.u, ns.du], arguments=dict(lhs=lhs))
    nutils.export.triplot('sol.jpg', x, u, tri=bezier.tri, hull=bezier.hull)
    nutils.export.triplot('err.jpg', x, du, tri=bezier.tri, hull=bezier.hull)

    if irefine == nrefine:
      break

    refdom = domain.refined
    ns.refbasis = refdom.basis(btype, degree=degree)
    indicator = refdom.integrate('refbasis_n,k u_,k' @ ns, geometry=ns.x, degree=degree*2, arguments=dict(lhs=lhs))
    indicator -= refdom.boundary.integrate('refbasis_n u_,k n_k' @ ns, geometry=ns.x, degree=degree*2, arguments=dict(lhs=lhs))
    mask = indicator**2 > numpy.mean(indicator**2)

    domain = domain.refined_by(elem.transform[:-1] for elem in domain.refined.supp(ns.refbasis, mask))

  return ndofs, error, rate, lhs

if __name__ == '__main__':
  nutils.cli.run(main)

import unittest

class test(unittest.TestCase):

  def test_square_quadratic(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='square', degree=2)
    self.assertEqual(ndofs, 149)
    numpy.testing.assert_almost_equal(error, [0.00104, 0.05495], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.066, -0.478], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNo1jyFrQnEUxd/qTH4FeciEFf3fc/7NNBwsOI'
      'wmMSz5wGA2zaDFtPT8Bi8Iq6bBZJuCRdC4Z1OEgdgeGHbCdtM9F845v9vhkK+MfOw//BE5Vnnr'
      'n6RC1DDAt1Tkty5zdQv+pm4DVOUJgszV5Bhq27oQR3TYR5NnXvuCj5niASW+M2XGHmfI4QcjJl'
      'zwjmPMLcE9u3xhni207REnFNjgGjc4uJ09Y4kr7vEl3cIYM6QIXdE2ZpjgDXuskVdSjzFX5Usl'
      'clP7tIk8DWUu1Fnw/8RTeZZK76o9FV0QRG6jhpMIRiI864NLpWg7USSiLLGpy6ocuoNr29xy+q'
      'SPX7TTY6M=')

  def test_triangle_quadratic(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='triangle', degree=2)
    self.assertEqual(ndofs, 98)
    numpy.testing.assert_almost_equal(error, [0.00219, 0.08451], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.111, -0.548], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNoljS0Og1AQhB9JazhGg0S9/QsGWw8hyAaBaM'
      'BxAEQNZ+AWNRhMRdMDIBpMTWUtBtl9sJMVs5mdr+KO7/yjIwc0sy8x13yjM71hAmMGOnGKRueK'
      'kdTy5VA+3MtLLtKrD/USiS8tPzXZsjF7diDXkuIEM8fa7/oCZXRKqjghDzPdA+a0QGEFC+v+St'
      'tsrBWcRnhAg4IjLJvPKaMVEtoZpS1UB/RUf+Q6Og8=')

  def test_mixed_linear(self):
    ndofs, error, rate, lhs = main(nrefine=2, etype='mixed', degree=1)
    self.assertEqual(ndofs, 34)
    numpy.testing.assert_almost_equal(error, [0.00714, 0.18546], decimal=5)
    numpy.testing.assert_almost_equal(rate, [-1.143, -0.545], decimal=3)
    nutils.numeric.assert_allclose64(lhs, 'eNrLNltnpmlabcphLm7GAATZ5vfMjppnm8P41a'
      'Yg+Wyzlbo7DUH8yCs+F3wuRF6JuXTx3E7DmEsMDCt1AXoBFaA=')
