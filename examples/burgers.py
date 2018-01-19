#! /usr/bin/python3

from nutils import mesh, cli, log, function, plot, numeric, solver, _
import numpy, unittest


class MakePlots:

  def __init__(self, domain):
    self.domain = domain
    self.index = 0

  def __call__(self, ns):
    self.index += 1
    xp, up = self.domain.elem_eval([ns.x, ns.u], ischeme='bezier7', separate=True)
    with plot.PyPlot('solution', index=self.index) as plt:
      plt.mesh(xp, up)
      plt.clim(0, 1)
      plt.colorbar()


def main(
    nelems: 'number of elements' = 20,
    degree: 'polynomial degree' = 1,
    timescale: 'time scale (timestep=timescale/nelems)' = .5,
    tol: 'solver tolerance' = 1e-5,
    ndims: 'spatial dimension' = 1,
    endtime: 'end time, 0 for no end time' = 0,
    figures: 'create figures' = True,
 ):

  # construct mesh, basis
  ns = function.Namespace()
  domain, ns.x = mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims))
  ns.basis = domain.basis('discont', degree=degree)
  ns.u = 'basis_n ?lhs_n'

  # construct initial condition (centered gaussian)
  lhs0 = domain.project('exp(-?y_i ?y_i)(y_i = 5 (x_i - 0.5_i))' @ ns, onto=ns.basis, geometry=ns.x, degree=5)

  # prepare residual
  ns.f = '.5 u^2'
  ns.C = 1
  res = domain.integral('-basis_n,0 f' @ ns, geometry=ns.x, degree=5)
  res += domain.interfaces.integral('-[basis_n] n_0 ({f} - .5 C [u] n_0)' @ ns, geometry=ns.x, degree=5)
  inertia = domain.integral('basis_n u' @ ns, geometry=ns.x, degree=5)

  # prepare plotting
  makeplots = MakePlots(domain) if figures else lambda ns: None

  # start time stepping
  timestep = timescale/nelems
  for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, newtontol=tol)):
    makeplots(ns(lhs=lhs))
    if endtime and itime * timestep >= endtime:
      break

  return res.eval(arguments=dict(lhs=lhs)), lhs


class test(unittest.TestCase):

  def test_1d_p1(self):
    res, lhs = main(ndims=1, nelems=10, timescale=.1, degree=1, endtime=.01, figures=False)
    numeric.assert_allclose64(res,
      'eNoljskNRDEIQxvCUiBsqWU0R/pv4QdyepaxhZXY6LfVpJAkS1fBCZrihSCTtDY4PQpGm9dqXmRBiYX1'
      'MWTIPEQoWwdxzKcB9jVR3MgZoZbvJB7xhHY9CNvk/T1t3EFbg+/C/wc4pyZV')
    numeric.assert_allclose64(lhs,
      'eNolzsENwDAIA8CFEgkDJnSWqs/uv0KBvk4yFuALXLeK8d1nwanv5oLQ2ksl2wwdzVFzL8PaQ/iYEi0E'
      'pZU+punkqZi+HU7fZl/lF2TuWOR/N3x0slXhqb+eD+HPJKs=')

  def test_1d_p2(self):
    res, lhs = main(ndims=1, nelems=10, timescale=.1, degree=2, endtime=.01, figures=False)
    numeric.assert_allclose64(res,
      'eNotj8mNBDAIBBMyEs1NLKN5Ov8UZsH7qqKNLbcd+Pk0LC71MVe7VIekmy/lARcuxd+S+VDCemdwXvKD'
      '0h5Kiw89gWEq17AqZdkcL+/dd46dSUx1xUtspRL7NHVmvyTkJcn5RCX8XYc+AYrfkZfNDwmhtSKBmi7k'
      '/i8W7Fuzpab39weWJTmO')
    numeric.assert_allclose64(lhs,
      'eNotz8kNxDAMA8CGHED3Uctin+m/hZiyXwMLAkXbYl8/Meb3iSXiCrnw9m0rVK6RSel9DPqY3VDd6SgJ'
      '3TRgOnK2FgU7SMcOO2aePOzpKqrJK9bR22dfi648OVx5e0Rf544y5rtns53+ElCEzv9M4P8DFyQ2QQ==')

  def test_2d_p1(self):
    res, lhs = main(ndims=2, nelems=4, timescale=.1, degree=1, endtime=.01, figures=False)
    numeric.assert_allclose64(res,
      'eNpNkUsSwzAIQy9kZsCY31k6Xeb+V2iMA+lKBGn0SLIGyfiQkFwQQ1RwKygTXeADJkVcYIPceOtilK3g'
      'SJhDkNm/U8k2OlrtRevyxhHLoXjQVndde8+MK59Db18HBc4LZBDJ2jrVUmtfud4/ueotTvc+HHBlzIEn'
      '5fHTLPJWwf1a9xAasrvv6xftUjDmlRtySsxrdbitDhfhZRaimWas5y8Yn49Nwn7OuXvON1Sx7JOpeO5C'
      '9QwvtzO8VoXb6nAjGtqIB/r9AX6LebM=')
    numeric.assert_allclose64(lhs,
      'eNpVkcsRwyAMRBuCGf0RtWRyTP8tJBJISU6Wdtf7wJaBOh4ojq9pY+pGP4N6DkQMr6ljuhqkI7w8FCZL'
      'B40/u3yNjpZTyW4vXJcXbS7c+Sr5onR4OydFw5AhuHbsBLJiJwGLJ2JUc+uVa/3murhIVdykKSAHaUjp'
      'LOGdh9qmyaad90Iyi242OvoC/tUrV3rlurhIVdykqab386ud64OkYQAUA8O9n7OcQ267fw7gz6lkGx2t'
      '8qZ1+8U936ubdqU=')


if __name__ == '__main__':
  cli.run(main)
