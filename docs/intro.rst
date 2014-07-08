Introduction
============

To get one thing out of the way first, note that Nutils is not your classical
Finite Element program. It does not have menus, no buttons to click, nothing to
make a screenshot of. To get it to do *anything* some programming is going to
be required.

That said, let's see what Nutils can be instead.


Design
------

Nutils is a programming library, providing components that are rich enough to
handle a wide range of problems by simply linking them together. This blurs the
line between classical graphical user interfaces and a programming environment,
both of which serve to offer some degree of mixing and matching of available
components. The former has a lower entry bar, whereas the latter offers more
flexibility, the possibility to extend the toolkit with custom algorithms, and
the possibility to pull in third party modules. It is our strong belief that on
the edge of science where Nutils strives to be a great degree of extensibility
is adamant.

(One of the lesser interesting possibilities this gives is to write a dedicated,
Nutils powered GUI application using any toolkit of preference.)

What Nutils specifically does not offer are problem specific components, like
one could imagine a "crack growth" module or "solve navier stokes" function. As
a primary design principle we aim for a Nutils application to be closely
readable as a high level mathematical problem description; `i.e.` the weak
form, domain, boundary conditions, time stepping of Newton iterations, etc. It
is the supporting operations like integrating over a domain or taking gradients
of compound functions that are being kept out of sight as much as possible.


Quick demo
----------

As a small but representative demonstration of what is involved in setting up a
problem in Nutils we solve the `Laplace problem
<https://en.wikipedia.org/wiki/Laplace%27s_equation>`_ on a unit square, with
zero Dirichlet conditions on the left and bottom boundaries, unit flux at the
top and a natural boundary condition at the right. We begin by creating a
structured ``nelems`` ⅹ ``nelems`` Finite Element mesh using the built-in
generator::

    verts = numpy.linspace( 0, 1, nelems+1 )
    domain, geom = mesh.rectilinear( [verts,verts] )

Here ``domain`` is topology representing an interconnected set of elements, and
``geometry`` is a mapping from the topology onto ℝ², representing it placement
in physical space. This strict separation of topological and geometric
information is key design choice in Nutils.

Proceeding to specifying the problem, we create a second order spline basis
``funcsp`` which doubles as trial and test space (`u` resp. `v`). We build a
``matrix`` by integrating ``laplace`` = `∇v · ∇u` over the domain, and a ``rhs``
vector by integrating `v` over the top boundary. The Dirichlet constraints are
projected over the left and bottom boundaries to find constrained coefficients
``cons``. Remaining coefficients are found by solving the system in ``lhs``.
Finally these are contracted with the basis to form our ``solution`` function::

    funcsp = domain.splinefunc( degree=2 )
    laplace = function.outer( funcsp.grad(geom) ).sum()
    matrix = domain.integrate( laplace, geometry=geom, ischeme='gauss2' )
    rhs = domain.boundary['top'].integrate( funcsp, geometry=geom, ischeme='gauss1' )
    cons = domain.boundary['left,bottom'].project( 0, ischeme='gauss1', geometry=geom, onto=funcsp )
    lhs = matrix.solve( rhs, constrain=cons, tol=1e-8, symmetric=True )
    solution = funcsp.dot(lhs)
    
The ``solution`` function is a mapping from the topology onto ℝ. Sampling this
together with the ``geometry`` generates arrays that we can use for plotting::

    points, colors = domain.elem_eval( [ geom, solution ], ischeme='bezier4', separate=True )
    with plot.PyPlot( 'solution', index=index ) as plt:
      plt.mesh( points, colors, triangulate='bezier' )
      plt.colorbar()
