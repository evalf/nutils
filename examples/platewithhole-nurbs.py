#!/usr/bin/env python3

"""
This example demonstrates how to perform NURBS-based isogeometric analysis
for the elastic analysis of an infinite plate with a circular hole under
tension. A detailed description of the testcase can be found in:
Hughes et al. 'Isogeometric analysis: CAD, finite elements, NURBS, exact 
geometry and mesh refinement', Computer Methods in Applied Mechanics and 
Engineering, Elsevier, 2005, 194, 4135-4195.
"""

import nutils, numpy, matplotlib.collections


def main(
    nrefine = 2,
    radius: 'hole radius' = 0.5,
    poisson: "poisson's ratio" = 0.3,
    traction: "far field traction (relative to Young's modulus)" = .1,
  ):

  # create the coarsest level parameter domain
  domain, geom0 = nutils.mesh.rectilinear([1, 2])
  bsplinebasis = domain.basis('spline', degree=2)
  controlweights = numpy.ones(12)
  controlweights[1:3] = .5 + .25 * numpy.sqrt(2)
  weightfunc = bsplinebasis.dot(controlweights)
  nurbsbasis = bsplinebasis * controlweights / weightfunc

  # create geometry function
  indices = [0,2], [1,2], [2,1], [2,0]
  controlpoints = numpy.concatenate([
    numpy.take([0, 2**.5-1, 1], indices) * radius,
    numpy.take([0, .3, 1], indices) * (radius+1) / 2,
    numpy.take([0, 1, 1], indices)])
  geom = (nurbsbasis[:,numpy.newaxis] * controlpoints).sum(0)

  radiuserr = numpy.sqrt(domain.boundary['left'].integrate((nutils.function.norm2(geom) - radius)**2, degree=9))
  nutils.log.info('hole radius exact up to L2 error {:.2e}'.format(radiuserr))

  # refine domain
  if nrefine:
    domain = domain.refine(nrefine)
    bsplinebasis = domain.basis('spline', degree=2)
    controlweights = domain.project(weightfunc, onto=bsplinebasis, geometry=geom0, ischeme='gauss9')
    nurbsbasis = bsplinebasis * controlweights / weightfunc

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.lmbda = 2 * poisson
  ns.mu = 1 - poisson
  ns.ubasis = nurbsbasis.vector(2)
  ns.u_i = 'ubasis_ni ?lhs_n'
  ns.X_i = 'x_i + u_i'
  ns.strain_ij = '(u_i,j + u_j,i) / 2'
  ns.stress_ij = 'lmbda strain_kk δ_ij + 2 mu strain_ij'
  ns.r2 = 'x_k x_k'
  ns.R2 = radius**2 / ns.r2
  ns.k = (3-poisson) / (1+poisson) # plane stress parameter
  ns.scale = traction * (1+poisson) / 2
  ns.uexact_i = 'scale (x_i ((k + 1) (0.5 + R2) + (1 - R2) R2 (x_0^2 - 3 x_1^2) / r2) - 2 δ_i1 x_1 (1 + (k - 1 + R2) R2))'
  ns.du_i = 'u_i - uexact_i'

  sqr = domain.boundary['top,bottom'].integral('(u_i n_i)^2' @ ns, degree=9)
  sqr += domain.boundary['right'].integral('du_k du_k' @ ns, geometry=ns.x, degree=20)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  # construct residual
  res = domain.integral('ubasis_ni,j stress_ij' @ ns, geometry=ns.x, degree=9)

  # solve system
  lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

  # vizualize result
  bezier = domain.sample('bezier', 9)
  X, stressxx = bezier.eval([ns.X, ns.stress[0,0]], arguments=dict(lhs=lhs))
  nutils.export.triplot('stressxx.jpg', X, stressxx, tri=bezier.tri, hull=bezier.hull, clim=(numpy.nanmin(stressxx), numpy.nanmax(stressxx)))

  # evaluate error
  err = numpy.sqrt(domain.integrate(['du_k du_k' @ ns, 'du_i,j du_i,j' @ ns], geometry=ns.x, degree=9, arguments=dict(lhs=lhs)))
  nutils.log.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

  return err, cons, lhs

if __name__ == '__main__':
  nutils.cli.run(main)

import unittest

class test(unittest.TestCase):

  def test0(self):
    err, cons, lhs = main(nrefine=0)
    nutils.numeric.assert_allclose64(err, 'eNoT1r6hDwACsAFG')
    nutils.numeric.assert_allclose64(cons, 'eNqLuMnQAIJnNSC08XUNYznjy8YQXoYmhC6+I'
      'nsu7/ze873aAH5yESc=')
    nutils.numeric.assert_allclose64(lhs, 'eNoBMADP/1jZ0DGVM5YzzSjfL2kzqDMz1ygzHj'
      'PTM5LOr85F0GgpJc6GzrIuc9Qdzm7Pvc+NKyFrF1c=')

  def test2(self):
    err, cons, lhs = main(nrefine=2)
    nutils.numeric.assert_allclose64(err, 'eNqzUn2kDQADSgFt')
    nutils.numeric.assert_allclose64(cons, 'eNrzec7QgA7NpTDFuJ5iiuXJYIpteIgpdkM+y'
      'sDa6KmRqrGqsYVxqvEO4xPGmKoSsZgm8whTbLEcptjT+5hifxU1z3mce3/O83zu+bzzJ88/u7D'
      'xSvUdAExTStA=')
    nutils.numeric.assert_allclose64(lhs, 'eNoB8AAP/0zn8i4cMRsyuTIiM2UzjDOdM50zNx'
      'pxLqEwuDF5MgQzWzONM6IzojMK5bQuwDC2MVwy3DI8M4AzpTOlM24cUC9CMRgykjLbMiUzcDOr'
      'M68zsOH9L+UxnDLsMgMzJzNlM7MzvjPYH1owOzLlMiUzJTM4M2UzuDPIM4nOic6azsHOA89szw'
      'vQC9E102EcWc5YznjOw85EzwrQD9Fd0pzUHOJDzkTOf87xzpvPiNC70UfTqNWjHjrOQM6ozjLP'
      'us880B/RyNIz1uXfMM5EztrOVc+Tz7bPOdCE0ZfV/SEpzkjO785Jz23Pbs/Jz+bQsdR73GyTdW'
      'c=')
