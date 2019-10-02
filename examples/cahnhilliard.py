#! /usr/bin/env python3
#
# In this script we solve the Cahn-Hiilliard equation, which models the
# unmixing of two phases under the effect of surface tension.

from nutils import mesh, function, solver, sample, export, cli, testing
import numpy, treelog

def main(nelems:int, etype:str, btype:str, degree:int, epsilon:float, contactangle:float, timestep:float, mtol:float, seed:int, circle:bool):
  '''
  Cahn-Hilliard equation on a unit square/circle.

  .. arguments::

     nelems [20]
       Number of elements along domain edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), with availability depending on the
       configured element type.
     degree [2]
       Polynomial degree.
     epsilon [0]
       Interface thickness, 0 for automatic based on the configured mesh
       density.
     contactangle [90]
       Wall contact angle in degrees.
     timestep [.01]
       Time step.
     mtol [.01]
       Threshold value for chemical potential peak to peak difference, used as
       a stop criterion.
     seed [0]
       Random seed for the initial condition.
     circle [no]
       Select circular domain as opposed to a unit square.
  '''

  mineps = 1./nelems
  if not epsilon:
    treelog.info('setting epsilon={}'.format(mineps))
    epsilon = mineps
  elif epsilon < mineps:
    treelog.warning('epsilon under crititical threshold: {} < {}'.format(epsilon, mineps))

  domain, geom = mesh.unitsquare(nelems, etype)
  bezier = domain.sample('bezier', 5) # sample for plotting

  ns = function.Namespace()
  if not circle:
    ns.x = geom
  else:
    ns.xi = (geom-.5) * (.5*numpy.pi)
    ns.x_i = '<sin(xi_0) cos(xi_1), cos(xi_0) sin(xi_1)>_i / sqrt(2)'
  ns.epsilon = epsilon
  ns.ewall = .5 * numpy.cos(contactangle * numpy.pi / 180)
  ns.cbasis, ns.mbasis = function.chain([domain.basis('std', degree=degree)] * 2)
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

  with treelog.iter.plain('timestep', solver.impliciteuler('lhs', target0='lhs0', residual=res, inertia=inertia, timestep=timestep, lhs0=lhs0)) as steps:
    for lhs in steps:

      E = sample.eval_integrals(*energy.values(), lhs=lhs)
      treelog.user('energy: {:.3f} ({})'.format(sum(E), ', '.join('{:.0f}% {}'.format(100*e/sum(E), n) for e, n in sorted(zip(E, energy), reverse=True))))

      x, c, m = bezier.eval(['x_i', 'c', 'm'] @ ns, lhs=lhs)
      export.triplot('phase.png', x, c, tri=bezier.tri, hull=bezier.hull, clim=(-1,1))
      export.triplot('chempot.png', x, m, tri=bezier.tri, hull=bezier.hull)

      if numpy.ptp(m) < mtol:
        break

  return lhs0, lhs

# If the script is executed (as opposed to imported), :func:`nutils.cli.run`
# calls the main function with arguments provided from the command line.

if __name__ == '__main__':
  cli.run(main)

# Once a simulation is developed and tested, it is good practice to save a few
# strategic return values for regression testing. The :mod:`nutils.testing`
# module, which builds on the standard :mod:`unittest` framework, facilitates
# this by providing :func:`nutils.testing.TestCase.assertAlmostEqual64` for the
# embedding of desired results as compressed base64 data.

class test(testing.TestCase):

  def _checkrand(self, lhs0):
    with self.subTest('initial condition'): self.assertAlmostEqual64(lhs0, '''
      eNoBxAA7/xM3LjTtNYs3MDcUyt41uc14zjo0LzKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3Tcnx
      L07NzTa4NlMyETREyrPIGMxYMl82VDbjy1/M8clZyf3IRjday6XLmMl6NRnJDs1Ayh00WMu1yQHRUDSs
      MKIz7MoEzM/KCMxwyvjIlzLQyxTJdjQ5yjEwWjX3MTk2n8kwNMbKTsoay1DMWDC8ycM1eTQyyb42NzdK
      NmLN5skSNs/LXDbnMuw19DNKNREtGTfui1ut''')

  @testing.requires('matplotlib')
  def test_square(self):
    lhs0, lhs = main(nelems=3, etype='square', btype='std', degree=2, epsilon=0, contactangle=90, timestep=1, mtol=.1, seed=0, circle=False)
    self._checkrand(lhs0)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNqbZTbHzMHsiGmpCd9V1gszzWaZ2ZjtMQ01eXV+xbk0szSgzAaTDxdNTkue1jbTMpM15TJqP/335PeT
      100vmyqYaJ3tPNV1svNknmmKqYJR+On3J01Pmp9MMY0y/WIYCOSZn7Q82XCi8UTXiSkn5pxYBISovJYT
      rSd6T0wD8xae6ATCCSemn5gLlusFwiknZp9YcGIpEE4Ewhkn5p1YfGIFEKLyAN6wcSE=''')

  @testing.requires('matplotlib')
  def test_contactangle(self):
    lhs0, lhs = main(nelems=3, etype='square', btype='std', degree=2, epsilon=0, contactangle=45, timestep=1, mtol=.1, seed=0, circle=False)
    self._checkrand(lhs0)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNqzNsszkzZbbfrdOOus6Jlss5lmPmbPTQtNtp6be8bZrNTss6mW6SMDv9OnTokDZRpMbxl7nNE89fTk
      ItNHpl0mT8+fOzX3ZP7J3yb+ph1G206zn7I+KXWyyOSeibK+1ulzJyVP/joRZhJp0m6yyeSyyXsgDAfy
      2kw2mlw0eWvyxiTLJNtkgslmk3Mmz4CwzqTeZLbJNpOzJo+AcIrJVJO1JkdMbpi8BsLlJitM9gHNeGLy
      2eQLkLfSZL/JFZOnJl+BEAAJrlyi''')

  @testing.requires('matplotlib')
  def test_mixedcircle(self):
    lhs0, lhs = main(nelems=3, etype='mixed', btype='std', degree=2, epsilon=0, contactangle=90, timestep=1, mtol=.1, seed=0, circle=True)
    self._checkrand(lhs0)
    with self.subTest('left-hand side'): self.assertAlmostEqual64(lhs, '''
      eNrTM31uImDqY1puGmwia1prssNY37TERNM01eSOkYuJlck6Q1ED9TP9px+fOmq82FjtfKFJiM6CK70m
      BsZixmUXgk9XnMo7VX6661zL+cZz58+ln0s6e/PM7DOvjDTOvTz97tS8c6xn9pzYemLHiQMn9p9YDyS3
      nth4YteJbUCRHUByO5DcfGLDieUnlpyYA2RtP7HpxJ4T64Aih8Bwz4k1QPF5QJ3rgap3ntgCVAHRe+bE
      biBr5YmDQBMBKJ13Eg==''')
