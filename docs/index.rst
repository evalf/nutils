.. Nutils documentation master file, created by
   sphinx-quickstart on Tue Jun 10 16:51:22 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Nutils's documentation!
==================================

Nutils: open source numerical utilities for Python, is a collaborative
programming effort aimed at the creation of a modern, general purpose
programming library for `Finite Element
<https://en.wikipedia.org/wiki/Finite_element_method>`_ applications and
related computational methods. Identifying features are a heavily object
oriented design, strict separation of topology and geometry, and CAS-like
function arithmetic such as found in Maple and Mathematica. Primary design
goals are:

  * **Readability**. Finite element scripts built on top of Nutils should focus
    on work flow and maths, unobscured by Finite Element infrastructure.
  * **Flexibility**. The Nutils are tools; they do not enforce a strict work
    flow. Missing components can be added locally without loosing
    interoperability.
  * **Compatibility**. Exposed objects are of native python type or allow for
    easy conversion to leverage third party tools.
  * **Speed**. Nutils are self-optimizing and support parallel computation.
    Typical scripting inefficiencies are discouraged by design.


Filosophy
---------

Nutils is not your classical Finite Element program. It does not have menus, no
buttons to click, nothing to make a screenshot of. To get it to do *anything*
some programming is going to be required.

That being said, the components that Nutils offers are rich enough to handle a
wide range of problems without adding any further algorithms. This blurs the
line between classical graphical user interfaces and a programming environment,
both of which serve to offer flexible configuration of available components.
The former has a lower entry bar, whereas the latter offers more flexibility in
mixing and matching, the possibility to extend the toolkit with custom
algorithms, and the possibility to pull in third party modules. It is our
strong belief that on the edge of science where Nutils strives to be a great
degree of extensibility is adamant. Naturally, one of the lesser interesting
possibilities this gives is to write a dedicated, Nutils powered GUI
application using any toolkit of preference.

One thing that Nutils does not offer is problem specific components, like a
"crack growth" module or "solve navier stokes" function. As a primary design
principle we aim for a Nutils application to be closely readable as a high level
mathematical problem description; `i.e.` the weak form, domain, boundary
conditions, time stepping of Newton iterations, etc. It is the supporting
operations like integrating over a domain or taking gradients of compound
functions that are being kept out of sight as much as possible.


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


Library
-------

The Nutils are separated in modules focussing on topics such as mesh generation,
function manipulation, debugging, plotting, etc. They are designed to form
relatively independent units, though some components such as output logging run
through all. Others, such as topology and element, operate in tight connection,
but are divided for reasons of scope and scale. A typical Nutils application
uses methods from all modules, although, as seen above, very few modules require
direct access for standard computations.

See the full overview for API information:

.. toctree::
   :maxdepth: 1

   library


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

