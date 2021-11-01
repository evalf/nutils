#! /usr/bin/env python3
#
# In this script we solve the Cahn-Hiilliard equation, which models the
# unmixing of two phases under the effect of surface tension.

from nutils import mesh, function, solver, export, cli, testing
from nutils.expression_v2 import Namespace
import numpy, treelog, itertools, enum, typing

class stab(enum.Enum):
  none = '0' # for educational purposes only
  linear = '.5 dc^2 (6 - 6 c^2 + 8 c dc - 3 dc^2) / epsilon^2'
  optimal = '.5 dc^2 (1 - dc^2 / 12) / epsilon^2'

def main(nelems:int, etype:str, btype:str, degree:int, epsilon:typing.Optional[float],
         contactangle:float, timestep:float, mtol:float, seed:int, circle:bool, stab:stab):
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
     epsilon []
       Interface thickness; defaults to an automatic value based on the
       configured mesh density if left unspecified.
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
     stab [linear]
       Stabilization method (linear/optimal/none).
  '''

  mineps = 1./nelems
  if epsilon is None:
    treelog.info('setting epsilon={}'.format(mineps))
    epsilon = mineps
  elif epsilon < mineps:
    treelog.warning('epsilon under crititical threshold: {} < {}'.format(epsilon, mineps))

  domain, geom = mesh.unitsquare(nelems, etype)
  bezier = domain.sample('bezier', 5) # sample for plotting

  ns = Namespace()
  if not circle:
    ns.x = geom
  else:
    angle = (geom-.5) * (numpy.pi/2)
    ns.x = function.sin(angle) * function.cos(angle)[[1,0]] / numpy.sqrt(2)
  ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
  ns.epsilon = epsilon
  ns.ewall = .5 * numpy.cos(contactangle * numpy.pi / 180)
  ns.cbasis = ns.mbasis = domain.basis('std', degree=degree)
  ns.c = function.dotarg('c', ns.cbasis)
  ns.dc = ns.c - function.dotarg('c0', ns.cbasis)
  ns.m = function.dotarg('m', ns.mbasis)
  ns.F = '.5 (c^2 - 1)^2 / epsilon^2'
  ns.dF = stab.value
  ns.dt = timestep

  nrg_mix = domain.integral('F dV' @ ns, degree=7)
  nrg_iface = domain.integral('.5 ∇_k(c) ∇_k(c) dV' @ ns, degree=7)
  nrg_wall = domain.boundary.integral('(abs(ewall) + c ewall) dS' @ ns, degree=7)
  nrg = nrg_mix + nrg_iface + nrg_wall + domain.integral('(dF - m dc - .5 dt epsilon^2 ∇_k(m) ∇_k(m)) dV' @ ns, degree=7)

  numpy.random.seed(seed)
  state = dict(c=numpy.random.normal(0,.5,ns.cbasis.shape), m=numpy.random.normal(0,.5,ns.mbasis.shape)) # initial condition

  with treelog.iter.plain('timestep', itertools.count()) as steps:
   for istep in steps:

    E = function.eval([nrg_mix, nrg_iface, nrg_wall], **state)
    treelog.user('energy: {0:.3f} ({1[0]:.0f}% mixture, {1[1]:.0f}% interface, {1[2]:.0f}% wall)'.format(sum(E), 100*numpy.array(E)/sum(E)))

    x, c, m = bezier.eval(['x_i', 'c', 'm'] @ ns, **state)
    export.triplot('phase.png', x, c, tri=bezier.tri, clim=(-1,1))
    export.triplot('chempot.png', x, m, tri=bezier.tri)

    if numpy.ptp(m) < mtol:
      break

    state['c0'] = state['c']
    state = solver.optimize(['c', 'm'], nrg, arguments=state, tol=1e-10)

  return state

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

  @testing.requires('matplotlib')
  def test_initial(self):
    state = main(nelems=3, etype='square', btype='std', degree=2, epsilon=None, contactangle=90, timestep=1, mtol=float('inf'), seed=0, circle=False, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['c'], '''
      eNoBYgCd/xM3LjTtNYs3MDcUyt41uc14zjo0LzKzNm812jFhNNMzwDYgzbMzV8o0yCM1rzWeypE3Tcnx
      L07NzTa4NlMyETREyrPIGMxYMl82VDbjy1/M8clZyf3IRjday6XLmMl6NRnJMF4tqQ==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['m'], '''
      eNoBYgCd/w7NQModNFjLtckB0VA0rDCiM+zKBMzPygjMcMr4yJcy0MsUyXY0OcoxMFo19zE5Np/JMDTG
      yk7KGstQzFgwvMnDNXk0Msm+Njc3SjZizebJEjbPy1w25zLsNfQzSjURLRk3Qt4uBQ==''')

  @testing.requires('matplotlib')
  def test_square(self):
    state = main(nelems=3, etype='square', btype='std', degree=2, epsilon=None, contactangle=90, timestep=1, mtol=.1, seed=0, circle=False, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['c'], '''
      eNoBYgCd/5o2nDZANsQ1dTQO1QXQmTaaNjw2vDVVNOrPqM5mNmY2xDWwNPDRNMsZyys2KjYdNQoyh8v9
      yffJ1zXTNSA0Ks2JyorJicluNWQ1IDJXy+/JNck3yWQ1WjX0MVHL78k3yTnJbYgt7Q==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['m'], '''
      eNpVyU0KgCAABeGrK9SiQCiQCsrwB8GjzLF8uJNvNYzBsuP5yGKmWlhxXKMSmxzchPGceF4iVU55+Ck0
      masD28JDNQ==''')

  @testing.requires('matplotlib')
  def test_contactangle(self):
    state = main(nelems=3, etype='square', btype='std', degree=2, epsilon=None, contactangle=45, timestep=1, mtol=.1, seed=0, circle=False, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['c'], '''
      eNoBYgCd/zs2bjYbNqs19zNqzRXMazaZNkw25zVxNLXOncxDNnU28zUqNeIwTsvKyhc2TDaANdozSMwp
      yuXJojXiNYo05c/Oyp3Jb8n7NE81iDK2ywfKO8kayXI03jQjLyrLzskZyfrIkb8vTg==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['m'], '''
      eNoNzCEOggAYgNFLvbPguILeQRpWgyQPoAYCuBGhWHVzGshOLegMSvffS1/5EqmF2sUnTKJylbO3l6mZ
      pb2TZ5jLrDWObmGlUDroDWFjq43H3dcvaqdz9TCGP1tYLVU=''')

  @testing.requires('matplotlib')
  def test_mixedcircle(self):
    state = main(nelems=3, etype='mixed', btype='std', degree=2, epsilon=None, contactangle=90, timestep=1, mtol=.1, seed=0, circle=True, stab=stab.linear)
    with self.subTest('concentration'): self.assertAlmostEqual64(state['c'], '''
      eNoBYgCd/y415zQQNUw1dzVTNB01fTS4My81dDQpNWU03DJENDo0rjEVMCfMj8vjysUzozMmz3E0VCyg
      1I00MDMWM3bQU8t4ym7Kd8uKzoTPgc7PzmfOYs3ZzJvM6jIozunL7sqezgXMYsUuJg==''')
    with self.subTest('chemical-potential'): self.assertAlmostEqual64(state['m'], '''
      eNoljEkKgEAQxP7/B/EkiCCO27gOuJxy8FEGpSEUoaoTkYWTgyAjHRuTZpGzHGgpKchMMz2JRnN/l6j1
      uctge2W08W8fdlPF5ccXGqBI7Q==''')
