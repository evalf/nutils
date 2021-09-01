Getting Started
===============

The following is a quick guide to running your first Nutils simulation in three
simple steps.


Step 1: Install Nutils
----------------------

With Python version 3.5 or newer installed, Nutils and its dependencies can be
installed via the `Python Package Index <https://pypi.org/project/nutils/>`_
using the pip package installer. In a terminal window::

    python -m pip install --user nutils

Most applications will require `Matplotlib <https://matplotlib.org/>`_ for
visualization, but since this is not a direct dependency it needs to be
installed separately::

    python -m pip install --user matplotlib

Another useful utility is `BottomBar <https://github.com/evalf/bottombar>`_,
which Nutils will use to provide runtime information in the bottom line of the
terminal window::

    python -m pip install --user bottombar


Step 2: Create a simulation script
----------------------------------

Open a text editor and create a file ``poisson.py`` with the following
contents:

.. code:: python

    from nutils import mesh, function, solver, export, cli

    def main(nelems: int = 10):
      domain, x = mesh.unitsquare(nelems, etype='square')
      u = function.dotarg('udofs', domain.basis('std', degree=1))
      g = u.grad(x)
      J = function.J(x)
      cons = solver.optimize('udofs',
        domain.boundary.integral(u**2 * J, degree=2), droptol=1e-12)
      udofs = solver.optimize('udofs',
        domain.integral((g @ g / 2 - u) * J, degree=1), constrain=cons)
      bezier = domain.sample('bezier', 3)
      x, u = bezier.eval([x, u], udofs=udofs)
      export.triplot('u.png', x, u, tri=bezier.tri, hull=bezier.hull)

    cli.run(main)

Note that while we could make the script even shorter by avoiding the main
function and ``cli.run``, the above structure is preferred as it automatically
sets up a logging environment, activates a matrix backend and handles command
line parsing.


Step 3: Run the simulation
--------------------------

Back in the terminal, the simulation can now be started by running::

    python poisson.py

This should produce the following output::

    nutils v7.0
    optimize > constrained 40/121 dofs
    optimize > optimum value 0.00e+00
    optimize > solve > solving 81 dof system to machine precision using arnoldi solver
    optimize > solve > solver returned with residual 6e-17
    optimize > optimum value -1.75e-02
    u.png
    log written to file:///home/myusername/public_html/poisson.py/log.html

If the terminal is reasonably modern (Windows users may want to install the new
`Windows Terminal <https://aka.ms/windowsterminal>`_) then the messages are
coloured for extra clarity, and a BottomBar will be shown for the duration of
the simulation.

The last line of the log shows the location of the simultaneously generated
html file that holds the `same log <_logs/examples%2B2Fpoisson.py/index.html>`_
as well as a link to the generated image.


Next steps and support
----------------------

Be sure to read the :ref:`install` guide for alternative installation methods,
as well as tips for setting up the broader compute environment.

After that, the best way to get going is by reading the :ref:`tutorial`, to
familiarize yourself with Nutils' concepts and syntax, and by studying the
:ref:`examples` that demonstrate implementations of several solid and fluid
mechanics problems. Most simulations will have components in common with the
example scripts, so a mix-and-match approach is a good way to start building
your own script.

Documentation of individual functions can be found in the :ref:`api_reference`.
For questions that are not answered by the API reference there is the
nutils-users support channel at `#nutils-users:matrix.org
<https://matrix.to/#/#nutils-users:matrix.org>`_. Note that you will need to
create an account at any Matrix server in order to join this channel.

Finally, if you are using Nutils in academic research, please consider `citing
Nutils <https://doi.org/10.5281/zenodo.822369>`_.
