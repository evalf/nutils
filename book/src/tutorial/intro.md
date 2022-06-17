# Tutorial

In this tutorial we will explore Nutils' main building blocks by solving a
simple 1D Laplace problem. The tutorial assumes knowledge of the
[Python](https://www.python.org/) programming language, as well as familiarity
with the third party modules [Numpy](https://numpy.org/) and
[Matplotlib](https://matplotlib.org/). It also assumes knowledge of advanced
calculus, weak formulations, and the Finite Element Method, and makes heavy use
of [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation).

## Whetting your Appetite

The computation that we will work towards amounts to about 20 lines of Nutils
code, including visualization. The entire script is presented below, in
copy-pasteable form suitable for interactive exploration using for example
ipython. In the sections that follow we will go over these lines ones by one
and explain the relevant concepts involved.

```python
from nutils import function, mesh, solver
from nutils.expression_v2 import Namespace
import numpy
from matplotlib import pyplot as plt

topo, geom = mesh.rectilinear([numpy.linspace(0, 1, 5)])

ns = Namespace()
ns.x = geom
ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
ns.basis = topo.basis('spline', degree=1)
ns.u = function.dotarg('lhs', ns.basis)

sqr = topo.boundary['left'].integral('u^2 dS' @ ns, degree=2)
cons = solver.optimize('lhs', sqr, droptol=1e-15)
# optimize > constrained 1/5 dofs
# optimize > optimum value 0.00e+00

res = topo.integral('∇_i(basis_n) ∇_i(u) dV' @ ns, degree=0)
res -= topo.boundary['right'].integral('basis_n dS' @ ns, degree=0)
lhs = solver.solve_linear('lhs', residual=res, constrain=cons)
# solve > solving 4 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 9e-16±1e-15

bezier = topo.sample('bezier', 32)
nanjoin = lambda array, tri: numpy.insert(array.take(tri.flat, 0).astype(float),
    slice(tri.shape[1], tri.size, tri.shape[1]), numpy.nan, axis=0)
sampled_x = nanjoin(bezier.eval('x_0' @ ns), bezier.tri)
def plot_line(func, **arguments):
    plt.plot(sampled_x, nanjoin(bezier.eval(func, **arguments), bezier.tri))
    plt.xlabel('x_0')
    plt.xticks(numpy.linspace(0, 1, 5))

plot_line(ns.u, lhs=lhs)
```
![output](intro-44113917.svg)

You are encouraged to execute this code at least once before reading on, as the
code snippets that follow may assume certain products to be present in the
namespace. In particular the `plot_line` function is used heavily in the
ensuing sections.
