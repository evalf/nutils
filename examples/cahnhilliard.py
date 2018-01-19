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
    numeric.assert_allclose64(lhs0,
      'eNrdktuthDAMRBsCyYnftazu5+2/hQ3xA0EJCx9YxDmZ8YSOwcfnNGD4P+k4xWDsAlFtF8TOu7BhseSC'
      'seRmGIVzFANgcfD6k7tuoINfxRTYuwl5dfDqWL1BIYLE5Z+Fo8RpneT00Fe41TqvVhszIAiYtJnenHSm'
      'bizaqEKebktd4dgp/NC0gJiqxN5hyUeLHmcIkUu1PmfXZlteAxU9gDJoZo9R9iRQTfJ0FnmF0dMrv62w'
      'gFN9f1Xn/gpTJjqugawseJq/o+10anrlt+QVzsX3gnPMjNYTMd7zvu9JBttLNbzyWuKK5hDnI2TAS+p4'
      'Xzp958rvwZXVEtc4dKW8sTzLOj8vXV+TO9jKoSfXZltfAw1oN8NvvH9fWJHkhw==')
    numeric.assert_allclose64(lhs,
      'eNptlFuy4zAIRDdkV/EQINYyNZ93/1u4kng4zswXJJaPuhuScaFcf4yFf265nNh25eH6c48L1c/3t8qk'
      '/cU9HeA0zpqNwMhmQz6aoiHiOcoWJ5lZkzo8qNOTwabJwKbii1raiibTbFeyGTARlaDaSBhBPPJBVHip'
      '5uW0tBWNjCMBcbKgkmZjngnAc09qJYx7Ore22uoaqJZAxXByC1Ae9lB322DKhuA9iM6u7bbCAqJspXIN'
      'nUffGt2porbM2XWjULh8ptqDqOzabqkr2hzxeWKcVPIzKEXS7w2pofYcKrpyWtIKZgrzSEU6dTj6P9uG'
      'XzOtMVRq5bOUFYzHnsVaUAAI+98r3PvxjLQnUKGVz5LWNASjaKate3g1JrobgaiIcz2gC8e+bleIz7Sf'
      '0/7l2KnA47M26EblgxZCCaJ7ErOCJdG+iPJJLEUPJ98/v7JdOd6T4qVy3FnT5R7vrQovZ6UnOQQjnHq8'
      't66hUxnjM4y561Q/vKkzeJ1Q6Uo9xZmpUzzuZTgJOpgfX7Y3btWztPwkXfmUr9JTnJGTAIPMKUfJenbo'
      'pjFP0zOrpCuhcpaKirT+xo6ye23rOM082ewGN5KfYXWFV0htrkQ1ah3UWBT12hiYX6skn4OrwCuox2DL'
      'ahZxWFuN8//2s7agplepd1htsXU1a505Lv/+AgpBcDw=')


if __name__ == '__main__':
  cli.run(main)
