#! /usr/bin/python3

from nutils import mesh, cli, log, function, plot, debug, solver, _
import numpy


class MakePlots:

  def __init__(self, domain, video):
    self.domain = domain
    self.plt = video and plot.PyPlotVideo('solution')
    self.index = 0

  def __call__(self, ns):
    self.index += 1
    xp, up = self.domain.elem_eval([ns.x, ns.u], ischeme='bezier7', separate=True)
    with self.plt if self.plt else plot.PyPlot('solution', index=self.index) as plt:
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
    withplots: 'create plots' = True,
 ):

  # construct mesh, basis
  ns = function.Namespace()
  domain, ns.x = mesh.rectilinear([numpy.linspace(0,1,nelems+1)]*ndims, periodic=range(ndims))
  ns.basis = domain.basis('discont', degree=degree)
  ns.u = 'basis_n ?lhs_n'

  # construct initial condition (centered gaussian)
  lhs0 = domain.project('exp(-?y_i ?y_i) | ?y_i = 5 (x_i - 0.5_i)' @ ns, onto=ns.basis, geometry=ns.x, degree=5)

  # prepare residual
  ns.f = '.5 u^2'
  ns.C = 1
  res = domain.integral('-basis_n,0 f' @ ns, geometry=ns.x, degree=5)
  res += domain.interfaces.integral('-[basis_n] n_0 ({f} - .5 C [u] n_0)' @ ns, geometry=ns.x, degree=5)
  inertia = domain.integral('basis_n u' @ ns, geometry=ns.x, degree=5)

  # prepare plotting
  makeplots = MakePlots(domain, video=withplots=='video') if withplots else lambda ns: None

  # start time stepping
  timestep = timescale/nelems
  for itime, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', res, inertia, timestep, lhs0, newtontol=tol)):
    makeplots(ns | dict(lhs=lhs))
    if endtime and itime * timestep >= endtime:
      break

  return res.eval(arguments=dict(lhs=lhs)), lhs


def unittest():

  retvals = main(ndims=1, nelems=10, timescale=.1, degree=1, endtime=.01, withplots=False)
  assert debug.checkdata(retvals, '''
    eNotkNsNRCEIBdvxJpDwBguy/xZWcL/GCHgGGZYB+wdrqbkcLBAyOhiAVhIHE1wq+oIr6qCDMlHzYh+8
    48L2mDpkliamdcEBt8dMIAdNC96WmUXzeiWJzHcw5o5FdfHJ3awjpJbchtdWSL2beKTbxbW5hcaxot+/
    rvZc1GKC059rFsW4Et9UvbRolkr2ff130nQfxvOtzaO79f0FU8yK1yOaQl7t9cH3A58wSvg=''')

  retvals = main(ndims=1, nelems=10, timescale=.1, degree=2, endtime=.01, withplots=False)
  assert debug.checkdata(retvals, '''
    eNotkdkNxDAIRNtJJCxxHwWl/xbWwH49AnhgCMGjQPbC8xSpf6dATew7CYcr6zsBhEnf8dum1mQ3nO9b
    +I4BpVSTi61pQdQMwclnhgwLffM1eUOP5mEVnsCSp/Nk0JYqIjfjvJnA2EDYdZ+TbECUuCVL7Q0PucQE
    7C0Yd4T9A3XUsVmc1r7vDViJxyGbjMMkXoc1ewnlTCKUO0ibPox2pCBmuOxlFUy3HtY6l+rZLEcZVlto
    Rq5e9wkk5ugl6ehZ2fRLoi5p+++d9l389/GZIxRzCCqy3X+Px4zzJ1l5fL7w/gDuxG+1''')

  retvals = main(ndims=2, nelems=4, timescale=.1, degree=1, endtime=.01, withplots=False)
  assert debug.checkdata(retvals, '''
    eNpNktkNAzEIRNtJJFviMIcLSv8txOCF5AsvMzsPHzhea6C8x+uFgvqZe4gKRJ3KiJ/pYxJu/0wb6MZR
    F4NEnQ6wc7HR9F8pZwttrfSidXjjkGVllm+M6q4r+syQfd96snXghlNlIIZfBqllrX75uv/4Krc4nftw
    pitDLpjQQiHzeyICsa2z2Lolss/0CyJ0GnPSJjom/ie1uaU2F+HHLEQzzVjuLRjdw0Zhv+MY33FIxTJP
    SOHOBXqDl9s93p9U5pba3IiGNqKg56Xgcrq0jQ9WPBdEjLlBF9mXwcGQwUk/l6B8hPUT2lpKOTu9cB1e
    tGno+Su5USq8PRcoGyJroe18GrAsvmlBVkQ8fe5++br/+Dq4SBXcpLlg3VeocbdHsRUHdYbaKsmmeN+H
    SaqRzUq3b8D//fJVv3wdXKQKbtKU50aJosb240UfQQEwFgzP/pzXHXLrc3MA/q+Us4W2VnjTOr1w7/H+
    Aj2F8Gw=''')


if __name__ == '__main__':
  cli.choose(main, unittest)
