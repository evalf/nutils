#! /usr/bin/env python3
#
# In this script we solve the Cahn-Hiilliard equation, which models the
# unmixing of two phases under the effect of surface tension.

import nutils, numpy

# The main function defines the parameter space for the script. Configurable
# parameters are the mesh density (in number of elements along an edge),
# element type (square, triangle, or mixed), type of basis function (std or
# spline, with availability depending on element type), polynomial degree, the
# epsilon parameter, contactangle, timestep, stop criterion, random seed, and a
# boolean flag for making the domain circular as opposed to a unit square.

def main(nelems: 'number of elements' = 20,
         etype: 'type of elements (square/triangle/mixed)' = 'square',
         btype: 'type of basis function (std/spline)' = 'std',
         degree: 'polynomial degree' = 2,
         epsilon: 'epsilon, 0 for automatic (based on nelems)' = 0,
         contactangle: 'wall contact angle (degrees)' = 90,
         timestep: 'time step' = .01,
         mtol: 'stop when chemical potential is peak to peak below threshold' = .01,
         seed: 'random seed' = 0,
         circle: 'select circular domain' = False):

  mineps = 1./nelems
  if not epsilon:
    nutils.log.info('setting epsilon={}'.format(mineps))
    epsilon = mineps
  elif epsilon < mineps:
    nutils.log.warning('epsilon under crititical threshold: {} < {}'.format(epsilon, mineps))

  domain, geom = nutils.mesh.unitsquare(nelems, etype)
  bezier = domain.sample('bezier', 5) # sample for plotting

  ns = nutils.function.Namespace()
  if not circle:
    ns.x = geom
  else:
    ns.xi = (geom-.5) * (.5*numpy.pi)
    ns.x_i = '<sin(xi_0) cos(xi_1), cos(xi_0) sin(xi_1)>_i / sqrt(2)'
  ns.epsilon = epsilon
  ns.ewall = .5 * numpy.cos(contactangle * numpy.pi / 180)
  ns.cbasis, ns.mbasis = nutils.function.chain([domain.basis('std', degree=degree)] * 2)
  ns.c = 'cbasis_n ?lhs_n'
  ns.c0 = 'cbasis_n ?lhs0_n'
  ns.m = 'mbasis_n ?lhs_n'
  ns.f = '(6 c0 - 2 c0^3 - 4 c) / epsilon^2' # convex/concave splitting of double well potential derivative

  res = domain.integral('(epsilon^2 mbasis_n,k m_,k + cbasis_n,k c_,k) d:x' @ ns, degree=7)
  res -= domain.integral('cbasis_n (m + f) d:x' @ ns, degree=7)
  res += domain.boundary.integral('cbasis_n ewall d:x' @ ns, degree=7)
  inertia = domain.integral('mbasis_n c d:x' @ ns, degree=7)

  energy = dict( # energy breakdown
    mixture = domain.integral('(c^2 - 1)^2 d:x / 2 epsilon^2' @ ns, degree=4),
    interfaces = domain.integral('.5 c_,k c_,k d:x' @ ns, degree=4),
    wall = domain.boundary.integral('(abs(ewall) + ewall c) d:x' @ ns, degree=4))

  numpy.random.seed(seed)
  lhs0 = numpy.random.normal(0, .5, ns.cbasis.shape) # initial condition

  for lhs in nutils.log.iter('timestep', nutils.solver.impliciteuler('lhs', target0='lhs0', residual=res, inertia=inertia, timestep=timestep, lhs0=lhs0)):

    E = nutils.sample.eval_integrals(*energy.values(), lhs=lhs)
    nutils.log.user('energy: {:.3f} ({})'.format(sum(E), ', '.join('{:.0f}% {}'.format(100*e/sum(E), n) for e, n in sorted(zip(E, energy), reverse=True))))

    x, c, m = bezier.eval(['x_i', 'c', 'm'] @ ns, lhs=lhs)
    nutils.export.triplot('phase.png', x, c, tri=bezier.tri, hull=bezier.hull, clim=(-1,1))
    nutils.export.triplot('chempot.png', x, m, tri=bezier.tri, hull=bezier.hull)

    if numpy.ptp(m) < mtol:
      break

  return lhs0, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.

if __name__ == '__main__':
  nutils.cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategicly chosen return values for routine regression testing. Here we use
# the standard :mod:`unittest` framework, with
# :func:`nutils.numeric.assert_allclose64` facilitating the embedding of
# desired results as compressed base64 data.

class test(nutils.testing.TestCase):

  def _checkrand(self, lhs0):
    nutils.numeric.assert_allclose64(lhs0, 'eNoBxAA7/xM3LjTtNYs3MDcUyt41uc14zjo0L'
      'zKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3TcnxL07NzTa4NlMyETREyrPIGMxYMl82VDb'
      'jy1/M8clZyf3IRjday6XLmMl6NRnJDs1Ayh00WMu1yQHRUDSsMKIz7MoEzM/KCMxwyvjIlzLQy'
      'xTJdjQ5yjEwWjX3MTk2n8kwNMbKTsoay1DMWDC8ycM1eTQyyb42NzdKNmLN5skSNs/LXDbnMuw'
      '19DNKNREtGTfui1ut')

  def test_square(self):
    lhs0, lhs = main(nelems=3, timestep=1, mtol=.1)
    self._checkrand(lhs0)
    nutils.numeric.assert_allclose64(lhs, 'eNqbZTbHzMHsiGmpCd9V1gszzWaZ2ZjtMQ01eX'
      'V+xbk0szSgzAaTDxdNTkue1jbTMpM15TJqP/335PeT100vmyqYaJ3tPNV1svNknmmKqYJR+On3'
      'J01Pmp9MMY0y/WIYCOSZn7Q82XCi8UTXiSkn5pxYBISovJYTrSd6T0wD8xae6ATCCSemn5gLlu'
      'sFwiknZp9YcGIpEE4Ewhkn5p1YfGIFEKLyAN6wcSE=')

  def test_contactangle(self):
    lhs0, lhs = main(nelems=3, timestep=1, mtol=.1, contactangle=45)
    self._checkrand(lhs0)
    nutils.numeric.assert_allclose64(lhs, 'eNqzNsszkzZbbfrdOOus6Jlss5lmPmbPTQtNtp'
      '6be8bZrNTss6mW6SMDv9OnTokDZRpMbxl7nNE89fTkItNHpl0mT8+fOzX3ZP7J3yb+ph1G206z'
      'n7I+KXWyyOSeibK+1ulzJyVP/joRZhJp0m6yyeSyyXsgDAfy2kw2mlw0eWvyxiTLJNtkgslmk3'
      'Mmz4CwzqTeZLbJNpOzJo+AcIrJVJO1JkdMbpi8BsLlJitM9gHNeGLy2eQLkLfSZL/JFZOnJl+B'
      'EAAJrlyi')

  def test_mixedcircle(self):
    lhs0, lhs = main(nelems=3, timestep=1, mtol=.1, circle=True, etype='mixed')
    self._checkrand(lhs0)
    nutils.numeric.assert_allclose64(lhs, 'eNrTM31uImDqY1puGmwia1prssNY37TERNM01e'
      'SOkYuJlck6Q1ED9TP9px+fOmq82FjtfKFJiM6CK70mBsZixmUXgk9XnMo7VX6661zL+cZz58+l'
      'n0s6e/PM7DOvjDTOvTz97tS8c6xn9pzYemLHiQMn9p9YDyS3nth4YteJbUCRHUByO5DcfGLDie'
      'UnlpyYA2RtP7HpxJ4T64Aih8Bwz4k1QPF5QJ3rgap3ntgCVAHRe+bEbiBr5YmDQBMBKJ13Eg==')
