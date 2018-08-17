#! /usr/bin/env python3
#
# In this script we solve the linear plane strain elasticity problem for an
# infinite plate with a circular hole under tension. We do this by placing the
# circle in the origin of a unit square, imposing symmetry conditions on the
# left and bottom, and Dirichlet conditions constraining the displacements to
# the analytical solution to the right and top. The traction-free circle is
# removed by means of the Finite Cell Method (FCM).

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), type of basis function (std or
# spline, with availability depending on element type), polynomial degree, far
# field traction, number of refinement levels for FCM, the cutout radius and
# Poisson's ratio.

def main(nelems: 'number of elementsa long edge' = 9,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (std/spline)' = 'std',
         degree: 'polynomial degree' = 2,
         traction: "far field traction (relative to Young's modulus)" = .1,
         maxrefine: 'maxrefine level for trimming' = 2,
         radius: 'cut-out radius' = .5,
         poisson: 'poisson ratio' = .3):

  domain0, geom = nutils.mesh.unitsquare(nelems, etype)
  domain = domain0.trim(nutils.function.norm2(geom) - radius, maxrefine=maxrefine)

  ns = nutils.function.Namespace()
  ns.x = geom
  ns.lmbda = 2 * poisson
  ns.mu = 1 - poisson
  ns.ubasis = domain.basis(btype, degree=degree).vector(2)
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

  sqr = domain.boundary['left,bottom'].integral('(u_i n_i)^2 d:x' @ ns, degree=degree*2)
  sqr += domain.boundary['top,right'].integral('du_k du_k d:x' @ ns, degree=20)
  cons = nutils.solver.optimize('lhs', sqr, droptol=1e-15)

  res = domain.integral('ubasis_ni,j stress_ij d:x' @ ns, degree=degree*2)
  lhs = nutils.solver.solve_linear('lhs', res, constrain=cons)

  bezier = domain.sample('bezier', 5)
  X, stressxx = bezier.eval([ns.X, ns.stress[0,0]], arguments=dict(lhs=lhs))
  nutils.export.triplot('stressxx.jpg', X, stressxx, tri=bezier.tri, hull=bezier.hull)

  err = numpy.sqrt(domain.integrate(['du_k du_k d:x', 'du_i,j du_i,j d:x'] @ ns, degree=max(degree,3)*2, arguments=dict(lhs=lhs)))
  nutils.log.user('errors: L2={:.2e}, H1={:.2e}'.format(*err))

  return err, cons, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line. For
# example, to keep with the default arguments simply run :sh:`python3
# platewithhole.py`. To select mixed elements and quadratic basis functions add
# :sh:`python3 platewithhole.py etype=mixed degree=2`.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

import unittest

class test(unittest.TestCase):

  def test_spline(self):
    err, cons, lhs = main(nelems=4, etype='square', degree=2, btype='spline')
    nutils.numeric.assert_allclose64(err, 'eNpbpd6jCwAEZgGL')
    nutils.numeric.assert_allclose64(cons, 'eNpjaPC5XybfdX+dIkMDDP7TQ7ANDBFsayME+'
      '6nRUeMjxnON04zNjFWNYaL655B0nrNUgrFrzrHeh7Ffn/sNt8v3/Nk7X66uuXT3wunzOecBJ0s'
      'yCg==')
    nutils.numeric.assert_allclose64(lhs, 'eNoBjABz/6I2TN92H4rfriEeyuw05zGFLykv/i'
      '6UM6EzzjLEMUkxMDGlM58zLzOrMlMyOzKwM7EzfTM1M/ky5TLFM8QznTNmMzYzJTPLNvjONM4/'
      'zi/OGclHzJfOSs45zjDOOSK7z5fPC8+cznzOBd/D3d3RFdAuz+vO+yGg1bnSvdCoz03Pzdz01a'
      'zS3dDLz2zPaQdIRw==')

  def test_mixed(self):
    err, cons, lhs = main(nelems=4, etype='mixed', degree=2, btype='std')
    nutils.numeric.assert_allclose64(err, 'eNpjU9+jCwACNgEX')
    nutils.numeric.assert_allclose64(cons, 'eNpjaGCAwx4pGMv/8UYZGFvrgagCkNZnaEgyY'
      'GjABw0NGRqOG+JXY23E0DDdCMTaaMzQcNT4iDGIPde4CUz7G6cD6adGZsaqxvjNgUD9c0BbgTj'
      'kHEwk+jE2dTVA+Y3nTsmB2GYPsZv1CqhG6jyItePye8XLd69dBbGXXZp0EUQ7Xrh7gaHB9/zp8'
      'znnAW7uYcc=')
    nutils.numeric.assert_allclose64(lhs, 'eNoNzcsrRGEYB2CxlbKY1CSXhUJxzvf+Co0Fml'
      'IWTCExdjaEBSuTSI0FiymRaxgrl9QsBgu2mqFc3vc75zCliGmQUaKkZCH+gKcnQaM4gI11rFaG'
      '3Gn1aJ6rAPlS0XzTGDG+zWOz/MFVlG1kGAGzx1yAF11YwBo2oKmDMrFDcRVSLmqkeqXUvTpVmw'
      'hjALvYRhCF+KAydCJKQfoim1qpliK0RBEsI4o9xBHDOPz/exAG8uBDL37oiapQghlp48/L2GUO'
      'u2WRp3mIT/iXa7iOW9jLGzzJ1WywhxX3cTvvy7Bc6RerO1VuhaVJ+vWbuOWCS2VKZnmMkxzls4'
      'Ln2yynKrly3encWHHtsjx2rp4Xv3akQl65/1+4E2nn0Hkvdu4S10f2hLVlz1kRXaAb9J3elWY5'
      'l0H5AxDbnCE=')
