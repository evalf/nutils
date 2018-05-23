#! /usr/bin/python3

from nutils import mesh, cli, log, function, plot, numeric, solver, util, export
import numpy, unittest
from matplotlib import collections


class MakePlots:

  def __init__(self, domain, ns):
    self.sample = domain.sample('bezier', 7)
    self.ns = ns
    self.index = 0

  def __call__(self, **arguments):
    self.index += 1
    x, u = self.sample.eval([self.ns.x, self.ns.u], arguments=arguments)
    with export.mplfigure('solution{}'.format(self.index)) as fig:
      ax = fig.add_subplot(111)
      if self.sample.ndims == 1:
        ax.add_collection(collections.LineCollection(numpy.array([x[:,0], u]).T[self.sample.tri], colors='k'))
      elif self.sample.ndims == 2:
        ax.set_aspect('equal')
        im = ax.tripcolor(x[:,0], x[:,1], self.sample.tri, u, shading='gouraud', cmap='jet')
        ax.add_collection(collections.LineCollection(x[self.sample.hull], colors='k', linewidths=.5, alpha=.1))
        fig.colorbar(im)
      ax.autoscale(axis='x', tight=True)
      ax.autoscale(axis='y', tight=self.sample.ndims>1)


def main(
    nelems: 'number of elements' = 20,
    degree: 'polynomial degree' = 1,
    timescale: 'time scale (timestep=timescale/nelems)' = .5,
    tol: 'solver tolerance' = 1e-5,
    ndims: 'spatial dimension' = 1,
    endtime: 'end time, 0 for no end time' = 0,
    figures: 'create figures' = True,
 ):

  # construct mesh
  domain, geom = mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims))

  # prepare residual
  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis('discont', degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.f = '.5 u^2'
  ns.C = 1

  # construct residual and inertia vector
  res = domain.integral('-basis_n,0 f' @ ns, geometry=ns.x, degree=5)
  res += domain.interfaces.integral('-[basis_n] n_0 ({f} - .5 C [u] n_0)' @ ns, geometry=ns.x, degree=5)
  inertia = domain.integral('basis_n u' @ ns, geometry=ns.x, degree=5)

  # construct initial condition (centered gaussian)
  lhs0 = domain.project('exp(-?y_i ?y_i)(y_i = 5 (x_i - 0.5_i))' @ ns, onto=ns.basis, geometry=ns.x, degree=5)

  # prepare plotting
  makeplots = MakePlots(domain, ns) if figures else lambda **args: None

  # start time stepping
  timestep = timescale/nelems
  for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, newtontol=tol)):
    makeplots(lhs=lhs)
    if endtime and itime * timestep >= endtime:
      break

  return res.eval(arguments=dict(lhs=lhs)), lhs


class test(unittest.TestCase):

  def test_1d_p1(self):
    res, lhs = main(ndims=1, nelems=10, timescale=.1, degree=1, endtime=.01, figures=False)
    numeric.assert_allclose64(res, 'eNoBKADX/0EjOSuY1xUtjC+JMNcyNDNLMwgzw81EzUHM8cy0zsD'
      'P89i20xgrvNxPhBPC')
    numeric.assert_allclose64(lhs, 'eNrbocann6u3yqjTyMLUwfSw2TWzKPNM8+9mH8wyTMNNZxptMir'
      'W49ffpwYAI6cOVA==')

  def test_1d_p2(self):
    res, lhs = main(ndims=1, nelems=10, timescale=.1, degree=2, endtime=.01, figures=False)
    numeric.assert_allclose64(res, 'eNoBPADD/6kgyiOI2Pspriq7K4cuhC9sMLsxNjKXMpkyPDJ/MfD'
      'PPM5yzTjNd83azV3PMdBa0a7TNdXK2FXXL9xP31+mHuU=')
    numeric.assert_allclose64(lhs, 'eNrr0c7SrtWfrD/d4JHRE6Ofxj6mnqaKZofNDpjZmQeYB5pHmL8'
      'we23mb5ZvWmjKY/LV6KPRFIMZ+o368dp92gCxZxZG')

  def test_2d_p1(self):
    res, lhs = main(ndims=2, nelems=4, timescale=.1, degree=1, endtime=.01, figures=False)
    numeric.assert_allclose64(res, 'eNoBgAB//yYduB+R2yHZjyYvKJvWV9YvKI8mV9ab1rgfJh0h2ZH'
      'b7CVvKBklqycdLpUvmy4/MJUvHS4/MJsubyjsJasnGSX12mPYVh5+4LzRMNBI0oPQMNC80YPQSNJj2PX'
      'afuBWHivbtdjS4kHgQ9Tu0kDZk9fu0kPUk9dA2bXYK9tB4NLih2hBgw==')
    numeric.assert_allclose64(lhs, 'eNoNyKENhEAQRuGEQsCv2SEzyQZHDbRACdsDJNsBjqBxSBxBHIg'
      'J9xsqQJ1Drro1L1/eYBZceGz8njrRyacm8UQLBvPYCw1airpyUVYSJLhKijK4IC01WDnqqxvX8OTl427'
      'aU73sctPGr3qqceBnRzOjo0xy9JpJR73m6R6YMZo/Q+FCLQ==')


if __name__ == '__main__':
  cli.run(main)
