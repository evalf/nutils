The Nutils Project
==================

The nutils project is a collaborative programming effort aimed at the
creation of a general purpose python programming library for setting up finite
element computations. Identifying features are a heavily object oriented
design, strict separation of topology and geometry, and CAS-like function
arithmetic such as found in maple and mathematica. Primary design goals
are:

  * __Readability__. Finite element scripts built on top of nutils should focus
    on work flow and maths, unobscured by finite element infrastructure.
  * __Flexibility__. The nutils are tools; they do not enforce a strict work
    flow. Missing components can be added locally without loosing
    interoperability.
  * __Compatibility__. Exposed objects are of native python type or allow for
    easy conversion to leverage third party tools.
  * __Speed__. Nutils are self-optimizing and support parallel computation.
    Typical scripting inefficiencies are discouraged by design.

The nutils are under active development, and are presently in use for
academic research by Phd and MSc students.
