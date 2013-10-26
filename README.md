The Nutils Project
==================

The nutils project is a collaborative programming effort aimed at the creation
of a general purpose python programming library for setting up finite element
computations. Identifying features are a heavily object oriented design, strict
separation of topology and geometry, and CAS-like function arithmatic such as
found in maple and mathematica. Primary design goals are:

  * __Readability__. Finite element scripts built on top of finity should focus
    on workflow and maths, unobscured by finite element infrastructure.
  * __Flexibility__. Finity is a toolbox; it does not enforce how its tools are
    to be used. Missing components can be added locally without loosing
    interoperability.
  * __Compatibility__. Exposed objects are of native python type or allow for
    easy conversion to leverage third party tools.
  * __Speed__. Finity components are self-optimizing and support parallel
    computation. Typical scripting inefficiencies are discouraged by design.

The nutils are under active development, and are presently in use by PhD and
MSc students for research on a variety of topics.
