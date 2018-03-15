#! /usr/bin/env python3

from nutils import mesh, plot, cli, log, function, numeric, numeric, solver
import numpy, unittest


class MakePlots( object ):

  def __init__(self, domain, nsteps, namespace):
    self.domain = domain
    self.namespace = namespace
    self.energies = numpy.empty((nsteps, 4))
    self.index = 0

  def __call__(self, lhs):
    ns = self.namespace(lhs=lhs)
    self.energies[self.index,:2] = self.domain.integrate(['(c^2 - 1)^2 / 2 epsilon^2' @ ns, '.5 c_,k c_,k' @ ns], geometry=ns.x, degree=4)
    self.energies[self.index,2] = self.domain.boundary.integrate('abs(ewall) + ewall c' @ ns, geometry=ns.x, degree=4)
    self.energies[self.index,3] = self.energies[self.index,:3].sum()
    x, c = self.domain.elem_eval([ns.x, ns.c], ischeme='bezier4', separate=True)
    with plot.PyPlot('flow', index=self.index) as plt:
      plt.axes(yticks=[], xticks=[])
      plt.mesh(x, c)
      plt.colorbar()
      plt.clim(-1, 1)
      plt.axes([.07,.05,.35,.25], yticks=[], xticks=[], axisbg='w').patch.set_alpha(.8)
      for energy, name in zip(self.energies[:self.index+1].T, ['mixture','interface','wall','total']):
        plt.plot(numpy.arange(len(energy))[::-1], energy[::-1], '-o', markevery=self.index+1, label=name)
      plt.legend( numpoints=1, frameon=False, fontsize=8 )
      plt.xlim(0, len(self.energies))
      plt.ylim(0, self.energies[0,3])
      plt.xlabel('time')
      plt.ylabel('energy')
    self.index += 1


def main(
    nelems: 'number of elements' = 20,
    epsilon: 'epsilon, 0 for automatic (based on nelems)' = 0,
    timestep: 'time step' = .01,
    maxtime: 'end time' = 1.,
    theta: 'contact angle (degrees)' = 90,
    init: 'initial condition (random/bubbles)' = 'random',
    figures: 'create figures' = True,
  ):

  mineps = 1./nelems
  if not epsilon:
    log.info('setting epsilon={}'.format(mineps))
    epsilon = mineps
  elif epsilon < mineps:
    log.warning('epsilon under crititical threshold: {} < {}'.format(epsilon, mineps))

  # create namespace
  ns = function.Namespace()
  ns.epsilon = epsilon
  ns.ewall = .5 * numpy.cos( theta * numpy.pi / 180 )

  # construct mesh
  xnodes = ynodes = numpy.linspace(0,1,nelems+1)
  domain, ns.x = mesh.rectilinear( [ xnodes, ynodes ] )

  # prepare bases
  ns.cbasis, ns.mubasis = function.chain([
    domain.basis('spline', degree=2),
    domain.basis('spline', degree=2)
  ])

  # polulate namespace
  ns.c = 'cbasis_n ?lhs_n'
  ns.c0 = 'cbasis_n ?lhs0_n'
  ns.mu = 'mubasis_n ?lhs_n'
  ns.f = '(6 c0 - 2 c0^3 - 4 c) / epsilon^2'

  # construct initial condition
  if init == 'random':
    numpy.random.seed(0)
    lhs0 = numpy.random.normal(0, .5, ns.cbasis.shape)
  elif init == 'bubbles':
    R1 = .25
    R2 = numpy.sqrt(.5) * R1 # area2 = .5 * area1
    ns.cbubble1 = function.tanh((R1-function.norm2(ns.x-(.5+R2/numpy.sqrt(2)+.8*ns.epsilon)))/ns.epsilon)
    ns.cbubble2 = function.tanh((R2-function.norm2(ns.x-(.5-R1/numpy.sqrt(2)-.8*ns.epsilon)))/ns.epsilon)
    sqr = domain.integral('(c - cbubble1 - cbubble2 - 1)^2 + mu^2' @ ns, geometry=ns.x, degree=4)
    lhs0 = solver.optimize('lhs', sqr)
  else:
    raise Exception( 'unknown init %r' % init )

  # construct residual
  res = domain.integral('epsilon^2 cbasis_n,k mu_,k + mubasis_n (mu + f) - mubasis_n,k c_,k' @ ns, geometry=ns.x, degree=4)
  res -= domain.boundary.integral('mubasis_n ewall' @ ns, geometry=ns.x, degree=4)
  inertia = domain.integral('cbasis_n c' @ ns, geometry=ns.x, degree=4)

  # solve time dependent problem
  nsteps = numeric.round(maxtime/timestep)
  makeplots = MakePlots(domain, nsteps, ns) if figures else lambda *args: None
  for istep, lhs in log.enumerate('timestep', solver.impliciteuler('lhs', target0='lhs0', residual=res, inertia=inertia, timestep=timestep, lhs0=lhs0)):
    makeplots(lhs)
    if istep == nsteps:
      break

  return lhs0, lhs


class test(unittest.TestCase):

  def test(self):
    lhs0, lhs = main(nelems=8, init='bubbles', timestep=.01, maxtime=.05, figures=False)
    numeric.assert_allclose64(lhs0, 'eNrdjjEKwlAMhgVv4FgnF3dpfY+ugquTuDjY3cEDCJ08SWfRUs'
      'RFBDf/P48Kgh28ghS6t1CM1zCBQELyffG4YS6+9HjDCSkOmhvuJAv8UUSn3R5HrblO5vZjuq4AcUamu7'
      '74o4+ZmYVUeOOBOy7oMWLXLWTKhLWsZMgGNzgUqJDwFTxNbJZ5yZMyqVe1PM3EhvY6bl2qrrNyVhKb0A'
      '5sE3juoPZMyUMu8+u4Cbay5u/HVF0NSrbOc2v22fmT+AJFl32r')
    numeric.assert_allclose64(lhs, 'eNotzT1IQlEUwHGIMBycGhuFdAgV877o3eOSBdGioCAhLg4uToF'
      'L9UBorBcRNgbSLDgJpk09X9x7/Mgi4YU0hIL0FkOaGoKOGXf73XP+ZzNYDu6wlL+AH7InhXygVw4urWd'
      'YJXCCtnz8tx2WYbtKP/SCU2nJDqmQKX8l0A/tBarowol8l6/0U8ATfMEqHmAJHa0qbmGEyracShcJ+C0'
      'WZ3Xv4KlHZUtOpKNlsR/FVnJKWpk1O9SpYpzZCiqLyieb3RZU3sK6N0dyH0j45mbJCA6e0jST8J0+58W'
      'V0NRV0OEYtsEJQz7kV+JW6OoaXID2ZyM+4pqqqyvwBmMoQRQWoMu7fBXWSDbCy+EG7IPNdX7Gddoaw3L'
      '4G+4gCTe8odbUY+qUoEFyDl9Uuuxk29tUjtJWkiSvxsS1WTSddGuBOjc0ExMe09N0N4d0u0vlBm1dk2j'
      'GkTG3M15Ts+2i6W4eGYfGL86izcg=')


if __name__ == '__main__':
  cli.run(main)
