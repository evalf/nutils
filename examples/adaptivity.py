#! /usr/bin/env python3

from nutils import *
import unittest
from matplotlib import collections, patches


class MakePlots:

  def __init__(self, geom, optimalrate, aoicenter, aoiarea, aoicircle):
    self.geom = geom
    self.ndofs = []
    self.error_exact = []
    self.error_estimate = []
    self.optimalrate = optimalrate
    if aoicircle:
      self.aoipatch = patches.Circle
      self.aoiargs = aoicenter, numpy.sqrt(aoiarea/numpy.pi)
    else:
      self.aoipatch = patches.Rectangle
      self.aoiargs = numpy.array(aoicenter) - aoiarea**.5/2, aoiarea**.5, aoiarea**.5

  def __call__(self, domain, sol, ndofs, error_estimate, error_exact):
    bezier = domain.sample('bezier', 9)
    x, colors = bezier.eval([self.geom, sol])
    with export.mplfigure('sol.png') as fig:
      ax = fig.add_subplot(111, aspect='equal')
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, colors, shading='gouraud', cmap='jet')
      ax.add_collection(collections.LineCollection(x[bezier.hull], colors='k', linewidths=.1))
      ax.add_patch(self.aoipatch(*self.aoiargs, facecolor='none', edgecolor='k', linewidth=1, linestyle='dashed'))
      ax.autoscale(enable=True, axis='both', tight=True)
      fig.colorbar(im)
    self.ndofs.append(ndofs)
    self.error_exact.append(error_exact)
    self.error_estimate.append(error_estimate)
    optimal = numpy.power(self.ndofs, -self.optimalrate, dtype=float)
    with export.mplfigure('conv.png') as fig:
      ax = fig.add_subplot(111, xlabel='degrees of freedom')
      ax.loglog(self.ndofs, self.error_exact, '-', label='error')
      ax.loglog(self.ndofs, self.error_estimate, '--', label='error estimate')
      ax.loglog(self.ndofs, optimal / optimal[-1] * self.error_exact[-1], ':', label='optimal rate: {}'.format(round(self.optimalrate, 3)))
      ax.legend(loc=3, frameon=False)


def main(
    degree: 'number of elements' = 1,
    circle: 'use circular area of interest (default square)' = False,
    uniform: 'use uniform refinement (default adaptive)' = False,
    basistype: 'basis function' = 'h-std',
    nrefine: 'maximum allowed number of refinements' = 7,
  ):

  # construct domain
  verts = numpy.linspace(-1, 1, 7)
  basetopo, geom = mesh.rectilinear([verts, verts])
  aoi = basetopo.trim(1/9/numpy.pi - ((geom+.5)**2).sum(-1), maxrefine=5) if circle else basetopo[1:2,1:2]
  domain = (basetopo.withboundary(outside=...) - basetopo[3:,:3].withboundary(inside=...)).withsubdomain(aoi=aoi)

  # construct exact sulution (used for boundary conditions and error evaluation)
  exact = (geom**2).sum(-1)**(1./3) * function.sin((2/3) * function.arctan2(-geom[1], -geom[0]))
  flux = exact.ngrad(geom)

  # sanity check
  harmonicity = numpy.sqrt(domain.integrate(exact.laplace(geom)**2, geometry=geom, degree=9))
  log.info('exact solution lsqr harmonicity:', harmonicity)

  # prepare plotting
  makeplots = MakePlots(geom, 2/3 if uniform else degree, [-.5, -.5], 1/9, circle)

  # start adaptive refinement
  for irefine in log.count('level', start=1):

    # construct, solve course domain primal/dual problem
    basis = domain.basis(basistype, degree=degree)
    laplace = function.outer(basis.grad(geom)).sum(-1)
    matrix = domain.integrate(laplace, geometry=geom, degree=5)
    rhsprimal = domain.boundary['inside'].integrate(basis * flux, geometry=geom, degree=99)
    rhsdual = domain['aoi'].integrate(basis, geometry=geom, degree=5)
    cons = domain.boundary['outside'].project(exact, degree=9, geometry=geom, onto=basis)
    lhsprimal = matrix.solve(rhsprimal, constrain=cons)
    lhsdual = matrix.solve(rhsdual, constrain=cons&0)
    primal = basis.dot(lhsprimal)
    dual = basis.dot(lhsdual)

    # construct, solve refined domain primal/dual problem
    finedomain = domain.refined
    finebasis = finedomain.basis(basistype, degree=degree)
    finelaplace = function.outer(finebasis.grad(geom)).sum(-1)
    finematrix = finedomain.integrate(finelaplace, geometry=geom, degree=5)
    finerhsdual = finedomain['aoi'].integrate(finebasis, geometry=geom, degree=5)
    finecons = finedomain.boundary['outside'].project(0, degree=5, geometry=geom, onto=finebasis)
    finelhsdual = finematrix.solve(finerhsdual, constrain=finecons)

    # evaluate error estimate
    dlhsdual = finelhsdual - finedomain.project(dual, onto=finebasis, geometry=geom, degree=5)
    ddualw = finebasis * dlhsdual
    error_est_w = finedomain.boundary['inside'].integrate(ddualw * flux, geometry=geom, degree=99)
    error_est_w -= finedomain.integrate((ddualw.grad(geom) * primal.grad(geom) ).sum(-1), geometry=geom, degree=5)
    error_estimate = abs(error_est_w).sum()
    error_exact = domain['aoi'].integrate(exact - primal, geometry=geom, degree=99)
    log.user('error estimate: {:.2e} ({:.1f}% accurate)'.format(error_estimate, 100.*error_estimate/error_exact))

    # plot solution and error convergence
    makeplots(domain, primal, len(lhsprimal), error_estimate, error_exact)

    if irefine >= nrefine:
      break

    # refine mesh
    if uniform:
      domain = domain.refined
    else:
      mask = error_est_w**2 > numpy.mean(error_est_w**2)
      domain = domain.refined_by(elem.transform[:-1] for elem in domain.refined.supp(finebasis, mask))

  return lhsprimal, error_est_w


class test(unittest.TestCase):

  def test_p1_h_std(self):
    lhsprimal, error_est_w = main(degree=1, circle=False, uniform=False, basistype='h-std', nrefine=2)
    numeric.assert_allclose64(lhsprimal, 'eNoBhAB7/2s2sDVfNKMjoMtQypXJlzbjNZk0l9Vkyx3Ka'
      'snA0gXL18k0yWjKfsn3yGrKrskdybXIfskeycPIdcj3yLXIdcg6yF4ocyhDLHA1JjXGNEs0izNAMs02h'
      'DYoNq019DSrM+o2pzZTNuM1PDX6M0bUrtEKN802gTYgNpY1rjR90DfOZs630OhqQfw=')
    numeric.assert_allclose64(error_est_w, 'eNp1kE0oBGEYx/83lNq1a2c/5uu1yUFc7IWkZFkHRVx'
      'WboqiXERJSpxw29PiwEFauVBq243k6qKQcvKxY2bM7K58RknDM+O8U//mfX/P/3metz9Q+mtiaQnY8AO'
      'VIvDC/9Obu2uzN+jKRc1y6U1PC3X5mSBQIQMyAwYl2zOkXTJLj8ht3AB1rROvcfitcCXvGhli7QrQrdv'
      'sK3AQ+uVdgsXNhYAF8pXRtjGntmTcBKa5CcPiXn3EqH5COuNKv/mTah2hBNdIvjGaw+SU/9nZnVaB1D0'
      'QIX2QmlWbbunx/HlOMxpY0oiwVnM/9129o9mVlQegSL7VOyBG/3rFpm79uCr24PGCLVZvsqQZU5K+YWf'
      'SEaXQQpu+KIlZUpToYwAYcHKbKBx64kLsqVMuGpVMeeyTR7R3pYsvSDw/J9qOH/L12yeaIVD/KaU2RPc'
      'pp//Hn31L8LXuZdFbnJTmfS+iKmyHvcpFOKNkxZow0JMDLDWvjktr/J4AjJL+AIyZgFs=')

  def test_p1_th_std(self):
    lhsprimal, error_est_w = main(degree=1, circle=False, uniform=False, basistype='th-std', nrefine=2)
    numeric.assert_allclose64(lhsprimal, 'eNoBhAB7/2s2sDVfNKMjoMtQypXJlzbjNZk0l9Vkyx3Ka'
      'snA0gXL18k0yWjKfsn3yGrKrskdybXIfskeycPIdcj3yLXIdcg6yCQ1PzVdNbE2ZDYDNoE1wDRoM802h'
      'DYoNq019DShM+o2pzZTNuM1PDX2M9bS7MsKN802gTYgNpY1rjR90EbLUcucyv1wQqw=')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAN3BVuyDEwzBVnYOCRZWD4IA0R3XXf4U2S5JE'
      'Hzi855D492yqj9qpckoFhM1ClvAIDQ4QcSM3EZwKK/54Zy9uKBQF1PQGKK4LFzz+/Ir/yxXag2PmHQPO'
      'fgcQSnnJI/5fml/knVi3FwNAAVMcOtC0DLNf+4q5EiVjOi39iH0WBYkD5/UB8Vgy3m78B5epFJ4i1Al2'
      'dATTHRH6ZuLo8SGb1C6C7HjAwGAPxPiBuew4SXfQs/NWFh09f6ChMe2GsYP1y/cNfIvskQTKdjxkY3gD'
      'VTb/PwOAGpLUegUQFnu0VdHssJMyg0CgyT2HaS7dH00SFnoBkdgN1WQJd/x1oWyUQOwNF9wNdEQqWzXm'
      '9Syhcxu2ti/ybFzwKj577y6c8/fzIVfq1nLS0lxRIxR9gqAQCXcwANEMGqP8EMNSigPxicLj/Ed/xaYK'
      '0ikCHrPCbQrk60Q+yT2SWKAk/uqi0/dFhOUUlBgZvYIj+e/LqSabcJplKoD4bGQYGAEjUgPU=')

  def test_p2_h_spline(self):
    lhsprimal, error_est_w = main(degree=2, circle=False, uniform=False, basistype='h-spline', nrefine=1)
    numeric.assert_allclose64(lhsprimal, 'eNoBbgCR/2s2GTYnNQoz9szZyufJlcl/Ni82OTUuM9jMv'
      '8rSyYHJrzZgNoQ1LDN1zH3Km8lRyek2rDbKNSo0AMwZylrJF8kpN9w2kjbNM4TKrMkKydfIssm2ySHJw'
      'ciVyIfJBsnByHfIV8gXydfIlchXyDrIutg53Q==')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAH5kiUCAXItEvd5rspEywx7dXat0fefeXX+wC'
      'S2yfi/urv83cSi5+1C4i/bn1pKOjFz8SrIQiSq5VpfPL/UepTkYc/5Lc8O/bCSDT75USh02B9qs9mSpx'
      '9qinG8aRDjOu50qvQ147isgKBn0Bya6XPP3V+NOPx3Xu7FB9L1j5eJmb16rxIxBuQHN+TQMmMRwqPWBX'
      'cFOwe+8pESjx9v0WQXwgkxym7SkpYsfnpO9X5D1QVWBUPiMXImLw6J87A4Ke2Q1Hs0UXJTzIr3rCIMjA'
      'ceJL16JbU6ZeKLza9YRJmYPBWeCaTLlcvMu8lv6i4GAMDq+R1SZ7Xpq92iaW/O/mZgaFYVl+yWTLn7RG'
      'xYq4SQcwQAgDpQ2yr')

  def test_p1_h_std_circle(self):
    lhsprimal, error_est_w = main(degree=1, circle=True, uniform=False, basistype='h-std', nrefine=1)
    numeric.assert_allclose64(lhsprimal, 'eNoBUACv/2s2sDVgNHPpoMtQypXJljbgNZM0RtRkyx3Ka'
      'snMNic25DSm0gnL1sk0yQk3gDaZNcHPYsp+yffIbMq1yR7JtciAyR/JxMh1yPfItch1yDrIoQ4s1w==')
    numeric.assert_allclose64(error_est_w, 'eNpjYMAHIhUZGFwUGBgqJBgYJGUZGD5IQ8QD7s15ZKL'
      'S8/jW/SkilrL7nunJeL0MloTIqd9mYNgExGlAtb/lGRhS5SDiU+7lylmrCdw/edf76S2FoKcF8g3iIVD'
      'zqpUZGHJUGRhCgHYZAO1cLgsRT1BIlLyu9P7+JNXUexaq/x/MlJ/4MkSGgcFPhYFh730GBoknIFWrHok'
      '/eiC/RO7kcyOgeZVA+54D8X2w3OrnyU9KXxo/Y5E8Ic7AkAWUzwViFglMvwIAeXg9bw==')


if __name__ == '__main__':
  cli.run(main)
