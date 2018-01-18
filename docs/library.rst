Library
=======

The Nutils are separated in modules focussing on topics such as mesh generation,
function manipulation, debugging, plotting, etc. They are designed to form
relatively independent units, though some components such as output logging run
through all. Others, such as topology and element, operate in tight connection,
but are divided for reasons of scope and scale. A typical Nutils application
uses methods from all modules, although, as seen above, very few modules require
direct access for standard computations.

What follows is an automatically generated API reference.

.. toctree::
   :maxdepth: 1

   library/topology
   library/function
   library/expression
   library/core
   library/config
   library/element
   library/log
   library/matrix
   library/mesh
   library/numeric
   library/parallel
   library/util
   library/plot
   library/cache
   library/cli
   library/solver
   library/transform
