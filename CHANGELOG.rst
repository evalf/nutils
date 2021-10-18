Changelog
=========

Nutils is being actively developed and the API is continuously evolving.
The following overview lists user facing changes as well as newly added
features in inverse chronological order.


New in v7.0 (in development)
----------------------------

- New: expression and namespace version 2

  The ``nutils.expression`` module has been renamed to
  :mod:`nutils.expression_v1`, the ``nutils.function.Namespace`` class to
  :class:`nutils.expression_v1.Namespace` and the :mod:`nutils.expression_v2`
  module has been added, featuring a new
  :class:`~nutils.expression_v2.Namespace`. The version 2 of the namespace v2
  has an expression language that differs slightly from version 1, most notably
  in the way derivatives are written. The old namespace remains available for
  the time being. All examples are updated to the new namespace. You are
  encouraged to use the new namespace for newly written code.

- Changed: bifurcate has been replaced by spaces

  In the past using functions on products of :class:`~nutils.topology.Topology`
  instances required using ``function.bifurcate``. This has been replaced by
  the concept of 'spaces'. Every topology is defined in a space, identified by
  a name (:class:`str`). Functions defined on some topology are considered
  constant on other topologies (defined on other spaces).

  If you want to multiply two topologies, you have to make sure that the
  topologies have different spaces, e.g. via the ``space`` parameter of
  :func:`nutils.mesh.rectilinear`. Example:

  >>> from nutils import mesh, function
  >>> Xtopo, x = mesh.rectilinear([4], space='X')
  >>> Ytopo, y = mesh.rectilinear([2], space='Y')
  >>> topo = Xtopo * Ytopo
  >>> geom = function.concatenate([x, y])

- Changed: function.Array shape must be constant

  Resulting from to the function/evaluable split introduced in #574, variable
  length axes such as relating to integration points or sparsity can stay
  confined to the evaluable layer. In order to benefit from this situation and
  improve compatibility with Numpy's arrays, :class:`nutils.function.Array`
  objects are henceforth limited to constant shapes. Additionally:

  * The sparsity construct ``nutils.function.inflate`` has been removed;
  * The :func:`nutils.function.Elemwise` function requires all element arrays
    to be of the same shape, and its remaining use has been deprecated in
    favor of :func:`nutils.function.get`;
  * Aligning with Numpy's API, :func:`nutils.function.concatenate` no longer
    automatically broadcasts its arguments, but instead demands that all
    dimensions except for the concatenation axis match exactly.

- Changed: locate arguments

  The :func:`nutils.topology.Topology.locate` method now allows ``tol`` to be
  left unspecified if ``eps`` is specified instead, which is repurposed as stop
  criterion for distances in element coordinates. Conversely, if only ``tol``
  is specified, a corresponding minimal ``eps`` value is set automatically to
  match points near element edges. The ``ischeme`` and ``scale`` arguments are
  deprecated and replaced by ``maxdist``, which can be left unspecified in
  general. The optional ``weights`` argument results in a sample that is
  suitable for integration.

- Moved: unit from types to separate module

  The ``unit`` type has been moved into its own :mod:`nutils.unit` module, with
  the old location ``types.unit`` now holding a forward method. The forward
  emits a deprecation warning prompting to change ``nutils.types.unit.create``
  (or its shorthand ``nutils.types.unit``) to :func:`nutils.unit.create`.

- Removed: loading libraries from .local

  Libraries that are installed in odd locations will no longer be automatically
  located by Nutils (see b8b7a6d5 for reasons). Instead the user will need to
  set the appropriate environment variable, prior to starting Python. In
  Windows this is the ``PATH`` variable, in Linux and OS X ``LD_LIBRARY_PATH``.

  Crucially, this affects the MKL libraries when they are user-installed via
  pip. By default Nutils selects the best available matrix backend that it
  finds available, which could result in it silently falling back on Scipy or
  Numpy. To confirm that the path variable is set correctly run your
  application with ``matrix=mkl`` to force an error if MKL cannot be loaded.

- Function module split into ``function`` and ``evaluable``

  The function module has been split into a high-level, numpy-like ``function``
  module and a lower-level ``evaluable`` module. The ``evaluable`` module is
  agnostic to the so-called points axis. Scripts that don't use custom
  implementations of ``function.Array`` should work without modification.

  Custom implementations of the old ``function.Array`` should now derive from
  ``evaluable.Array``. Furthermore, an accompanying implementation of
  ``function.Array`` should be added with a ``prepare_eval`` method that
  returns the former.

  The following example implementation of an addition

  >>> class Add(function.Array):
  ...   def __init__(self, a, b):
  ...     super().__init__(args=[a, b], shape=a.shape, dtype=a.dtype)
  ...   def evalf(self, a, b):
  ...     return a+b

  should be converted to

  >>> class Add(function.Array):
  ...   def __init__(self, a: function.Array, b: function.Array) -> None:
  ...     self.a = a
  ...     self.b = b
  ...     super().__init__(shape=a.shape, dtype=a.dtype)
  ...   def prepare_eval(self, **kwargs) -> evaluable.Array:
  ...     a = self.a.prepare_eval(**kwargs)
  ...     b = self.b.prepare_eval(**kwargs)
  ...     return Add_evaluable(a, b)
  ...
  >>> class Add_evaluable(evaluable.Array):
  ...   def __init__(self, a, b):
  ...     super().__init__(args=[a, b], shape=a.shape, dtype=a.dtype)
  ...   def evalf(self, a, b):
  ...     return a+b

- Solve multiple residuals to multiple targets

  In problems involving multiple fields, where formerly it was required to
  :func:`nutils.function.chain` the bases in order to construct and solve a
  block system, an alternative possibility is now to keep the residuals and
  targets separate and reference the several parts at the solving phase::

      # old, still valid approach
      >>> ns.ubasis, ns.pbasis = function.chain([ubasis, pbasis])
      >>> ns.u_i = 'ubasis_ni ?dofs_n'
      >>> ns.p = 'pbasis_n ?dofs_n'

      # new, alternative approach
      >>> ns.ubasis = ubasis
      >>> ns.pbasis = pbasis
      >>> ns.u_i = 'ubasis_ni ?u_n'
      >>> ns.p = 'pbasis_n ?p_n'

      # common: problem definition
      >>> ns.σ_ij = '(u_i,j + u_j,i) / Re - p δ_ij'
      >>> ures = topo.integral('ubasis_ni,j σ_ij d:x d:x' @ ns, degree=4)
      >>> pres = topo.integral('pbasis_n u_,kk d:x' @ ns, degree=4)

      # old approach: solving a single residual to a single target
      >>> dofs = solver.newton('dofs', ures + pres).solve(1e-10)

      # new approach: solving multiple residuals to multiple targets
      >>> state = solver.newton(['u', 'p'], [ures, pres]).solve(1e-10)

  In the new, multi-target approach, the return value is no longer an array but
  a dictionary that maps a target to its solution. If additional arguments were
  specified to newton (or any of the other solvers) then these are copied into
  the return dictionary so as to form a complete state, which can directly be
  used as an arguments to subsequent evaluations.

  If an argument is specified for a solve target then its value is used as an
  initial guess (newton, minimize) or initial condition (thetamethod). This
  replaces the ``lhs0`` argument which is not supported for multiple targets.

- New thetamethod argument ``historysuffix`` deprecates ``target0``

  To explicitly refer to the history state in :func:`nutils.solver.thetamethod`
  and its derivatives ``impliciteuler`` and ``cranknicolson``, instead of
  specifiying the target through the ``target0`` parameter, the new argument
  ``historysuffix`` specifies only the suffix to be added to the main target.
  Hence, the following three invocations are equivalent::

      # deprecated
      >>> solver.impliciteuler('target', residual, inertia, target0='target0')
      # new syntax
      >>> solver.impliciteuler('target', residual, inertia, historysuffix='0')
      # equal, since '0' is the default suffix
      >>> solver.impliciteuler('target', residual, inertia)

- In-place modification of newton, minimize, pseudotime iterates

  When :class:`nutils.solver.newton`, :class:`nutils.solver.minimize` or
  :class:`nutils.solver.pseudotime` are used as iterators, the generated
  vectors are now modified in place. Therefore, if iterates are stored for
  analysis, be sure to use the ``.copy`` method.

- Deprecated ``function.elemwise``

  The function ``function.elemwise`` has been deprecated. Use
  ``function.Elemwise`` instead::

      >>> function.elemwise(topo.transforms, values) # deprecated
      >>> function.Elemwise(values, topo.f_index) # new

- Removed ``transforms`` attribute of bases

  The ``transforms`` attribute of bases has been removed due to internal
  restructurings. The ``transforms`` attribute of the topology on which the
  basis was created can be used as a replacement::

      >>> reftopo = topo.refined
      >>> refbasis = reftopo.basis(...)
      >>> supp = refbasis.get_support(...)
      >>> #topo = topo.refined_by(refbasis.transforms[supp]) # no longer valid
      >>> topo = topo.refined_by(reftopo.transforms[supp]) # still valid


New in v6.0 "garak-guksu"
-------------------------

Release date: `2020-04-29 <https://github.com/evalf/nutils/releases/tag/v6.0>`_.

- Sparse module

  The new :mod:`nutils.sparse` module introduces a data type and a suite
  of manipulation methods for arbitrary dimensional sparse data. The
  existing integrate and integral methods now create data of this type
  under the hood, and then convert it to a scalar, Numpy array or
  :class:`nutils.matrix.Matrix` upon return. To prevent this conversion
  and receive the sparse objects instead use the new
  :func:`nutils.sample.Sample.integrate_sparse` or
  :func:`nutils.sample.eval_integrals_sparse`.

- External dependency for parsing gmsh files

  The :func:`nutils.mesh.gmsh` method now depends on the external
  `meshio <https://github.com/nschloe/meshio>`_ module to parse .msh
  files::

      $ python3 -m pip install --user --upgrade meshio

- Change dof order in basis.vector

  When creating a vector basis using ``topo.basis(..).vector(nd)``, the
  order of the degrees of freedom changed from grouping by vector
  components to grouping by scalar basis functions::

      [b0,  0]         [b0,  0]
      [b1,  0]         [ 0, b0]
      [.., ..] old     [b1,  0]
      [bn,  0] ------> [ 0, b1]
      [ 0, b0]     new [.., ..]
      [.., ..]         [bn,  0]
      [ 0, bn]         [ 0, bn]

  This should not affect applications unless the solution vector is
  manipulated directly, such as might happen in unit tests. If required
  for legacy purposes the old vector can be retrieved using ``old =
  new.reshape(-1,nd).T.ravel()``. Note that the change does not extend
  to :func:`nutils.function.vectorize`.

- Change from stickybar to bottombar

  For :func:`nutils.cli.run` to draw a status bar, it now requires the
  external `bottombar <https://github.com/evalf/bottombar>`_ module to
  be installed::

      $ python3 -m pip install --user bottombar

  This replaces stickybar, which is no longer used. In addition to the
  log uri and runtime the status bar will now show the current memory
  usage, if that information is available. On Windows this requires
  `psutil` to be installed; on Linux and OSX it should work by default.

- Support for gmsh 'msh4' file format

  The :func:`nutils.mesh.gmsh` method now supports input in the 'msh4'
  file format, in addition to the 'msh2' format which remains supported
  for backward compatibility. Internally, :func:`nutils.mesh.parsegmsh`
  now takes file contents instead of a file name.

- New command line option: gracefulexit

  The new boolean command line option ``gracefulexit`` determines what
  happens when an exception reaches :func:`nutils.cli.run`. If true
  (default) then the exception is handled as before and a system exit is
  initiated with an exit code of 2. If false then the exception is
  reraised as-is. This is useful in particular when combined with an
  external debugging tool.

- Log tracebacks at debug level

  The way exceptions are handled by :func:`nutils.cli.run` is changed
  from logging the entire exception and traceback as a single error
  message, to logging the exceptions as errors and tracebacks as debug
  messages. Additionally, the order of exceptions and traceback is fully
  reversed, such that the most relevant message is the first thing shown
  and context follows.

- Solve leniently to relative tolerance in Newton systems

  The :class:`nutils.solver.newton` method now sets the relative
  tolerance of the linear system to ``1e-3`` unless otherwise specified
  via ``linrtol``. This is mainly useful for iterative solvers which can
  save computational effort by having their stopping criterion follow
  the current Newton residual, but it may also help with direct solvers
  to warn of ill conditioning issues. Iterations furthermore use
  :func:`nutils.matrix.Matrix.solve_leniently`, thus proceeding after
  warning that tolerances have not been met in the hope that Newton
  convergence might be attained regardless.

- Linear solver arguments

  The methods :class:`nutils.solver.newton`,
  :class:`nutils.solver.minimize`, :class:`nutils.solver.pseudotime`,
  :func:`nutils.solver.solve_linear` and :func:`nutils.solver.optimize`
  now receive linear solver arguments as keyword arguments rather than
  via the ``solveargs`` dictionary, which is deprecated. To avoid name
  clashes with the remaining arguments, argument names must be prefixed
  by ``lin``::

      >>> solver.solve_linear('lhs', res,
      ...   solveargs=dict(solver='gmres')) # deprecated syntax

      >>> solver.solve_linear('lhs', res,
      ...   linsolver='gmres') # new syntax

- Iterative refinement

  Direct solvers enter an iterative refinement loop in case the first
  pass did not meet the configured tolerance. In machine precision mode
  (atol=0, rtol=0) this refinement continues until the residual
  stagnates.

- Matrix solver tolerances

  The absolute and/or relative tolerance for solutions of a linear
  system can now be specified in :func:`nutils.matrix.Matrix.solve` via
  the ``atol`` resp. ``rtol`` arguments, regardless of backend and
  solver. If the backend returns a solution that violates both
  tolerances then an exception is raised of type
  :class:`nutils.matrix.ToleranceNotReached`, from which the solution
  can still be obtained via the `.best` attribute. Alternatively the new
  method :func:`nutils.matrix.Matrix.solve_leniently` always returns a
  solution while logging a warning if tolerances are not met. In case
  both tolerances are left at their default value or zero then solvers
  are instructed to produce a solution to machine precision, with
  subsequent checks disabled.

- Use stringly for command line parsing

  Nutils now depends on stringly (version 1.0b1) for parsing of command
  line arguments. The new implementation of :func:`nutils.cli.run` is
  fully backwards compatible, but the preferred method of annotating
  function arguments is now as demonstrated in all of the examples.

  For new Nutils installations Stringly will be installed automatically
  as a dependency. For existing setups it can be installed manually as
  follows::

      $ python3 -m pip install --user --upgrade stringly

- Fixed and fallback lengths in (namespace) expressions

  The ``nutils.function.Namespace`` has two new arguments:
  ``length_<indices>`` and ``fallback_length``. The former can be used
  to assign fixed lengths to specific indices in expressions, say index
  ``i`` should have length 2, which is used for verification and
  resolving undefined lengths. The latter is used to resolve remaining
  undefined lengths::

      >>> ns = nutils.function.Namespace(length_i=2, fallback_length=3)
      >>> ns.eval_ij('δ_ij') # using length_i
      Array<2,2>
      >>> ns.eval_jk('δ_jk') # using fallback_length
      Array<3,3>

- Treelog update

  Nutils now depends on treelog version 1.0b5, which brings improved
  iterators along with other enhancements. For transitional convenience
  the backwards incompatible changes have been backported in the
  ``nutils.log`` wrapper, which now emits a warning in case the
  deprecated methods are used. This wrapper is scheduled for deletion
  prior to the release of version 6.0. To update treelog to the most
  recent version use::

      python -m pip install -U treelog

- Unit type

  The new ``nutils.types.unit`` allows for the creation of a unit system for
  easy specification of physical quantities. Used in conjunction with
  :func:`nutils.cli.run` this facilitates specifying units from the command
  line, as well as providing a warning mechanism against incompatible units::

      >>> U = types.unit.create(m=1, s=1, g=1e-3, N='kg*m/s2', Pa='N/m2')
      >>> def main(length=U('2m'), F=U('5kN')):
      ...   topo, geom = mesh.rectilinear([numpy.linspace(0,length,10)])

      # python myscript.py length=25cm # OK
      # python myscript.py F=10Pa # error!

- Sample basis

  Samples now provide a :func:`nutils.sample.Sample.basis`: an array
  that for any point in the sample evaluates to the unit vector
  corresponding to its index. This new underpinning of
  :func:`nutils.sample.Sample.asfunction` opens the way for sampled
  arguments, as demonstrated in the last example below::

      >>> H1 = mysample.asfunction(mydata) # mysample.eval(H1) == mydata
      >>> H2 = mysample.basis().dot(mydata) # mysample.eval(H2) == mydata
      >>> ns.Hbasis = mysample.basis()
      >>> H3 = 'Hbasis_n ?d_n' @ ns # mysample.eval(H3, d=mydata) == mydata

- Higher order gmsh geometries

  Gmsh element support has been extended to include cubic and quartic
  meshes in 2D and quadratic meshes in 3D, and parsing the msh file is
  now a cacheable operation. Additionally, tetrahedra now define bezier
  points at any order.

- Repository location

  The Nutils repository has moved to
  https://github.com/evalf/nutils.git. For the time being the old
  address is maintained by Github as an alias, but in the long term you
  are advised to update your remote as follows::

      git remote set-url origin https://github.com/evalf/nutils.git


New in v5.0 "farfalle"
----------------------

Release date: `2019-06-11 <https://github.com/evalf/nutils/releases/tag/v5.0>`_.

- Matrix matmul operator, solve with multiple right hand sides

  The ``Matrix.matvec`` method has been deprecated in favour of the new
  ``__matmul__`` (@) operator, which supports multiplication arrays of
  any dimension. The :func:`nutils.matrix.Matrix.solve` method has been
  extended to support multiple right hand sides::

      >>> matrix.matvec(lhs) # deprecated
      >>> matrix @ lhs # new syntax
      >>> matrix @ numpy.stack([lhs1, lhs2, lhs3], axis=1)
      >>> matrix.solve(rhs)
      >>> matrix.solve(numpy.stack([rhs1, rhs2, rhs3], axis=1)

- MKL's fgmres method

  Matrices produced by the ``MKL`` backend now support the
  :func:`nutils.matrix.Matrix.solve` argument solver='fmgres' to use Intel
  MKL's fgmres method.

- Thetamethod time target

  The :class:`nutils.solver.thetamethod` class, as well as its special
  cases ``impliciteuler`` and ``cranknicolson``, now have a
  ``timetarget`` argument to specify that the formulation contains a
  time variable::

      >>> res = topo.integral('...?t... d:x' @ ns, degree=2)
      >>> solver.impliciteuler('dofs', res, ..., timetarget='t')

- New leveltopo argument for trimming

  In :func:`nutils.topology.Topology.trim`, in case the levelset cannot
  be evaluated on the to-be-trimmed topology itself, the correct
  topology can now be specified via the new ``leveltopo`` argument.

- New unittest assertion assertAlmostEqual64

  :class:`nutils.testing.TestCase` now facilitates comparison against
  base64 encoded, compressed, and packed data via the new method
  :func:`nutils.testing.TestCase.assertAlmostEqual64`. This replaces
  ``numeric.assert_allclose64`` which is now deprecated and scheduled
  for removal in Nutils 6.

- Fast locate for structured topology, geometry

  A special case :func:`nutils.topology.Topology.locate` method for
  structured topologies checks of the geometry is an affine
  transformation of the natural configuration, in which case the trivial
  inversion is used instead of expensive Newton iterations::

      >>> topo, geom = mesh.rectilinear([2, 3])
      >>> smp = topo.locate(geom/2-1, [[-.1,.2]])
      # locate detected linear geometry: x = [-1. -1.] + [0.5 0.5] xi ~+2.2e-16

- Lazy references, transforms, bases

  The introduction of sequence abstractions :mod:`nutils.elementseq` and
  :mod:`nutils.transformseq`, together with and a lazy implementation of
  :class:`nutils.function.Basis` basis functions, help to prevent the
  unnecessary generation of data. In hierarchically refined topologies,
  in particular, this results in large speedups and a much reduced
  memory footprint.

- Switch to treelog

  The ``nutils.log`` module is deprecated and will be replaced by the
  externally maintained `treelog <https://github.com/evalf/treelog>`_,
  which is now an installation dependency.

- Replace pariter, parmap by fork, range.

  The :mod:`nutils.parallel` module is largely rewritten. The old
  methods ``pariter`` and ``parmap`` are replaced by the
  :func:`nutils.parallel.fork` context, combined with the shared
  :func:`nutils.parallel.range` iterator::

      >>> indices = parallel.range(10)
      >>> with parallel.fork(nprocs=2) as procid:
      >>>   for index in indices:
      >>>     print('procid={}, index={}'.format(procid, index))


New in v4.0 "eliche"
--------------------

Release date: `2018-08-22 <https://github.com/evalf/nutils/releases/tag/v4.0>`_.

- Spline basis continuity argument

  In addition to the ``knotmultiplicities`` argument to define the
  continuity of basis function on structured topologies, the
  :func:`nutils.topology.Topology.basis` method now supports the
  ``continuity`` argument to define the global continuity of basis
  functions. With negative numbers counting backwards from the
  ``degree``, the default value of ``-1`` corresponds to a knot
  multiplicity of 1.

- Eval arguments

  Functions of type ``nutils.function.Evaluable`` can receive
  arguments in addition to element and points by depending on instances
  of :func:`nutils.function.Argument` and having their values specified
  via `nutils.sample.Sample.eval`::

      >>> f = geom.dot(function.Argument('myarg', shape=geom.shape))
      >>> f = 'x_i ?myarg_i' @ ns # equivalent operation in namespace
      >>> topo.sample('uniform', 1).eval(f, myarg=numpy.ones(geom.shape))

- The d:-operator

  Namespace expression syntax now includes the ``d:`` Jacobian operator,
  allowing one to write ``'d:x' @ ns`` instead of ``function.J(ns.x)``.
  Since including the Jacobian in the integrand is preferred over
  specifying it separately, the ``geometry`` argument of
  :func:`nutils.topology.Topology.integrate` is deprecated::

      >>> topo.integrate(ns.f, geometry=ns.x) # deprecated
      >>> topo.integrate(ns.f * function.J(ns.x)) # was and remains valid
      >>> topo.integrate('f d:x' @ ns) # new namespace syntax

- Truncated hierarchical bsplines

  Hierarchically refined topologies now support basis truncation, which
  reduces the supports of individual basis functions while maintaining
  the spanned space. To select between truncated and non-truncated the
  basis type must be prefixed with 'th-' or 'h-', respectively. A
  non-prefixed basis type falls back on the default implementation that
  fails on all types but discont::

      >>> htopo.basis('spline', degree=2) # no longer valid
      >>> htopo.basis('h-spline', degree=2) # new syntax for original basis
      >>> htopo.basis('th-spline', degree=2) # new syntax for truncated basis
      >>> htopo.basis('discont', degree=2) # still valid

- Transparent function cache

  The :mod:`nutils.cache` module provides a memoizing function decorator
  :func:`nutils.cache.function` which reads return values from cache in
  case a set of function arguments has been seen before. It is similar
  in function to Python's `functools.lru_cache`, except that the cache
  is maintained on disk and :func:`nutils.types.nutils_hash` is used to
  compare arguments, which means that arguments need not be Python
  hashable. The mechanism is activated via :func:`nutils.cache.enable`::

      >>> @cache.function
      >>> def f(x):
      >>>   return x * 2
      >>>
      >>> with cache.enable():
      >>>   f(10)

  If :func:`nutils.cli.run` is used then the cache can also be enabled
  via the new ``--cache`` command line argument. With many internal
  Nutils functions already decorated, including all methods in the
  :func:`nutils.solver` module, transparent caching is available out of
  the box with no further action required.

- New module: types

  The new :mod:`nutils.types` module unifies and extends components
  relating to object types. The following preexisting objects have been
  moved to the new location::

      util.enforcetypes -> types.apply_annotations
      util.frozendict -> types.frozendict
      numeric.const -> types.frozenarray

- MKL matrix, Pardiso solver

  The new ``MKL`` backend generates matrices that are powered by Intel's Math
  Kernel Library, which notably includes the reputable Pardiso solver. This
  requires ``libmkl`` to be installed, which is conveniently available through
  pip::

      $ pip install mkl

  When :func:`nutils.cli.run` is used the new matrix type is selected
  automatically if it is available, or manually using ``--matrix=MKL``.

- Nonlinear minimization

  For problems that adhere to an energy structure, the new solver method
  :func:`nutils.solver.minimize` provides an alternative mechanism that
  exploits this structure to robustly find the energy minimum::

      >>> res = sqr.derivative('dofs')
      >>> solver.newton('dofs', res, ...)
      >>> solver.minimize('dofs', sqr, ...) # equivalent

- Data packing

  Two new methods, :func:`nutils.numeric.pack` and its inverse
  :func:`nutils.numeric.unpack`, provide lossy compression to floating
  point data. Primarily useful for regression tests, the convenience
  method ``numeric.assert_allclose64`` combines data packing with zlib
  compression and base64 encoding for inclusion in Python codes.


New in v3.0 "dragon beard"
--------------------------

Release date: `2018-02-05 <https://github.com/evalf/nutils/releases/tag/v3.0>`_.

- New: function.Namespace

  The ``nutils.function.Namespace`` object represents a container
  of :class:`nutils.function.Array` instances::

      >>> ns = function.Namespace()
      >>> ns.x = geom
      >>> ns.basis = domain.basis('std', degree=1).vector(2)

  In addition to bundling arrays, arrays can be manipulated using index
  notation via string expressions using the ``nutils.expression``
  syntax::

      >>> ns.sol_i = 'basis_ni ?dofs_n'
      >>> f = ns.eval_i('sol_i,j n_j')

- New: Topology.integral

  Analogous to :func:`nutils.topology.Topology.integrate`, which
  integrates a function and returns the result as a (sparse) array, the
  new method :func:`nutils.topology.Topology.integral` with identical
  arguments results in an ``nutils.sample.Integral`` object for
  postponed evaluation::

      >>> x = domain.integrate(f, geometry=geom, degree=2) # direct
      >>> integ = domain.integral(f, geometry=geom, degree=2) # indirect
      >>> x = integ.eval()

  Integral objects support linear transformations, derivatives and
  substitutions. Their main use is in combination with routines from the
  :mod:`nutils.solver` module.

- Removed: TransformChain, CanonicalTransformChain

  Transformation chains (sequences of transform items) are stored as
  standard tuples. Former class methods are replaced by module methods::

      >>> elem.transform.promote(ndims) # no longer valid
      >>> transform.promote(elem.transform, ndims) # new syntax

  In addition, every ``edge_transform`` and ``child_transform`` of
  Reference objects is changed from (typically unit-length)
  ``TransformChain`` to :class:`nutils.transform.TransformItem`.

- Changed: command line interface

  Command line parsers :func:`nutils.cli.run` or
  :func:`nutils.cli.choose` dropped support for space separated
  arguments (--arg value), requiring argument and value to be joined by
  an equals sign instead::

      $ python script.py --arg=value

  Boolean arguments are specified by omitting the value and prepending
  'no' to the argument name for negation::

      $ python script.py --pdb --norichoutput

  For convenience, leading dashes have been made optional::

      $ python script.py arg=value pdb norichoutput

- New: Topology intersections (deprecates common_refinement)

  Intersections between topologies can be made using the ``&`` operator.
  In case the operands have different refinement patterns, the resulting
  topology will consist of the common refinements of the intersection::

      >>> intersection = topoA & topoB
      >>> interface = topo['fluid'].boundary & ~topo['solid'].boundary

- Changed: Topology.indicator

  The :func:`nutils.topology.Topology.indicator` method is moved from
  subtopology to parent topology, i.e. the topology you want to evaluate
  the indicator on, and now takes the subtopology is an argument::

    >>> ind = domain.boundary['top'].indicator() # no longer valid
    >>> ind = domain.boundary.indicator(domain.boundary['top']) # new syntax
    >>> ind = domain.boundary.indicator('top') # equivalent shorthand

- Changed: Evaluable.eval

  The ``nutils.function.Evaluable.eval`` method accepts a flexible
  number of keyword arguments, which are accessible to ``evalf`` by
  depending on the ``EVALARGS`` token. Standard keywords are
  ``_transforms`` for transformation chains, ``_points`` for integration
  points, and ``_cache`` for the cache object::

    >>> f.eval(elem, 'gauss2') # no longer valid
    >>> ip, iw = elem.getischeme('gauss2')
    >>> tr = elem.transform, elem.opposite
    >>> f.eval(_transforms=tr, _points=ip) # new syntax

- New: numeric.const

  The ``numeric.const`` array represents an immutable, hashable array::

      >>> A = numeric.const([[1,2],[3,4]])
      >>> d = {A: 1}

  Existing arrays can be wrapped into a ``const`` object by adding
  ``copy=False``. The ``writeable`` flag of the original array is set to
  False to prevent subsequent modification::

      >> A = numpy.array([1,2,3])
      >> Aconst = numeric.const(A, copy=False)
      >> A[1] = 4
      ValueError: assignment destination is read-only

- New: function annotations

  The ``util.enforcetypes`` decorator applies conversion methods to
  annotated arguments::

      >>> @util.enforcetypes
      >>> def f(a:float, b:tuple)
      >>>   print(type(a), type(b))
      >>> f(1, [2])
      <class 'float'> <class 'tuple'>

  The decorator is by default active to constructors of cache.Immutable
  derived objects, such as function.Evaluable.

- Changed: Evaluable._edit

  Evaluable objects have a default edit implementation that
  re-instantiates the object with the operand applied to all constructor
  arguments. In situations where the default implementation is not
  sufficient it can be overridden by implementing the ``edit`` method
  (note: without the underscore)::

      >>> class B(function.Evaluable):
      >>>   def __init__(self, d):
      >>>     assert isinstance(d, dict)
      >>>     self.d = d
      >>>   def edit(self, op):
      >>>     return B({key: op(value) for key, value in self.d.items()})

- Changed: function derivatives

  The ``nutils.function.derivative`` ``axes`` argument has been
  removed; ``derivative(func, var)`` now takes the derivative of
  ``func`` to all the axes in ``var``::

      >>> der = function.derivative(func, var,
      ...         axes=numpy.arange(var.ndim)) # no longer valid
      >>> der = function.derivative(func, var) # new syntax

- New module: cli

  The ``nutils.util.run`` function is deprecated and replaced by two new
  functions, :func:`nutils.cli.choose` and :func:`nutils.cli.run`. The
  new functions are very similar to the original, but have a few notable
  differences:

    - ``cli.choose`` requires the name of the function to be executed
      (typically 'main'), followed by any optional arguments
    - ``cli.run`` does not require the name of the function to be executed,
      but only a single one can be specified
    - argument conversions follow the type of the argument's default
      value, instead of the result of ``eval``
    - the ``--tbexplore`` option for post-mortem debugging is replaced
      by ``--pdb``, replacing Nutils' own traceback explorer by Python's
      builtin debugger
    - on-line debugging is provided via the ctrl+c signal handler
    - function annotations can be used to describe arguments in both
      help messages and logging output (see examples)

- New module: solver

  The :mod:`nutils.solver` module provides infrastructure to facilitate
  formulating and solving complicated nonlinear problems in a structured
  and largely automated fashion.

- New: topology.with{subdomain,boundary,interfaces,points}

  Topologies have been made fully immutable, which means that the old
  setitem operation is no longer supported. Instead, to add a
  subtopology to the domain, its boundary, its interfaces, or points,
  any of the methods :func:``withsubdomain``, ``withboundary``,
  ``withinterfaces``, and ``withpoints``, respectively, will return a
  copy of the topology with the desired groups added::

      >> topo.boundary['wall'] = topo.boundary['left,top'] # no longer valid
      >> newtopo = topo.withboundary(wall=topo.boundary['left,top']) # new syntax
      >> newtopo = topo.withboundary(wall='left,top') # equivalent shorthand
      >> newtopo.boundary['wall'].integrate(...)

- New: circular symmetry

  Any topology can be revolved using the new
  ``nutils.topology.Topology.revolved`` method, which interprets the
  first geometry dimension as a radius and replaces it by two new
  dimensions, shifting the remaining axes backward. In addition to the
  modified topology and geometry, simplifying function is returned as
  the third return value which replaces all occurrences of the
  revolution angle by zero. This should only be used after all gradients
  have been computed::

      >> rdomain, rgeom, simplify = domain.revolved(geom)
      >> basis = rdomain.basis('spline', degree=2)
      >> M = function.outer(basis.grad(rgeom)).sum(-1)
      >> rdomain.integrate(M, geometry=rgeom, ischeme='gauss2', edit=simplify)

- Renamed mesh.gmesh to mesh.gmsh; added support for periodicity

  The gmsh importer was unintentionally misnamed as gmesh; this has been
  fixed. With that the old name is deprecated and will be removed in
  future. In addition, support for the non-physical mesh format and
  externally supplied boundary labels has been removed (see the unit
  test tests/mesh.py for examples of valid .geo format). Support is
  added for periodicity and interface groups.


New in v2.0 "chuka men"
-----------------------

Release date: `2016-02-18 <https://github.com/evalf/nutils/releases/tag/v2.0>`_.

- Changed: jump sign

  The jump operator has been changed according to the following
  definition: ``jump(f) = opposite(f) - f``. In words, it represents the
  value of the argument from the side that the normal is pointing
  toward, minus the value from the side that the normal is pointing away
  from. Compared to the old definition this means the sign is flipped.

- Changed: Topology objects

  The Topology base class no longer takes a list of elements in its
  constructor. Instead, the ``__iter__`` method should be implemented by
  the derived class, as well as ``__len__`` for the number of elements,
  and getelem(index) to access individual elements. The 'elements'
  attribute is deprecated.

  The :class:`nutils.topology.StructuredTopology` object no longer
  accepts an array with elements. Instead, an 'axes' argument is
  provided with information that allows it to generate elements in the
  fly. The 'structure' attribute is deprecated. A newly added ``shape``
  tuple is now a documented attribute.

- Changed: properties dumpdir, outdir, outrootdir

  Two global properties have been renamed as follows::

      dumpdir -> outdir
      outdir -> outrootdir

  The ``outrootdir`` defaults to ~/public_html and can be redefined from
  the command line or in the .nutilsrc configuration file. The outdir
  defaults to the current directory and is redefined by ``util.run``,
  nesting the name/date/time subdirectory sequence under ``outrootdir``.

- Changed: sum axis argument

  The behaviour of ``nutils.function.sum`` is inconsistent with that
  of the Numpy counterparts. In case no axes argument is specified,
  Numpy sums over all axes, whereas Nutils sums over the last axis. To
  undo this mistake and transition to Numpy's behaviour, calling sum
  without an axes argument is deprecated and will be forbidden in Nutils
  3.0. In Nutils 4.0 it will be reintroduced with the corrected meaning.

- Changed: strict dimension equality in function.outer

  The :func:`nutils.function.outer` method allows arguments of different
  dimension by left-padding the smallest prior to multiplication. There
  is no clear reason for this generality and it hinders error checking.
  Therefore in future in ``function.outer(a, b)``, ``a.ndim`` must equal
  ``b.ndim``. In a brief transition period non-equality emits a warning.

- Changed: Evaluable base class

  Relevant only for custom ``nutils.function.Evaluable`` objects,
  the ``evalf`` method changes from constructor argument to
  instance/class method::

      >> class MyEval( function.Evaluable):
      >>   def __init__(self, ...):
      >>     function.Evaluable(args=[...], shape=...)
      >>   def evalf( self, ...):
      >>     ...

  Moreover, the ``args`` argument may only contain Evaluable objects.
  Static information is to be passed through ``self``.

- Removed: _numeric C-extension

  At this point Nutils is pure Python. It is no longer necessary to run
  make to compile extension modules. The numeric.py module remains
  unchanged.

- Periodic boundary groups

  Touching elements of periodic domains are no longer part of the
  ``boundary`` topology. It is still available as boundary of an
  appropriate non-periodic subtopology::

      >> domain.boundary['left'] # no longer valid
      >> domain[:,:1].boundary['left'] # still valid

- New module: transform

  The new :mod:`nutils.transform` module provides objects and operations
  relating to affine coordinate transformations.

- Traceback explorer disabled by default

  The new command line switch ``--tbexplore`` activates the traceback
  explorer on program failure. To change the default behavior add
  ``tbexplore=True`` to your .nutilsrc file.

- Rich output

  The new command line switch ``--richoutput`` activates color and
  unicode output. To change the default behavior add ``richoutput=True``
  to your .nutilsrc file.


Older releases
--------------

- v1.0 "bakmi" was released `2014-08-04
  <https://github.com/evalf/nutils/releases/tag/v1.0>`_.

- v0.0 "anelli" was released `2013-10-28
  <https://github.com/evalf/nutils/releases/tag/v0.0>`_.
