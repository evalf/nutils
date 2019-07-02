Changelog
=========

Nutils is being actively developed and the API is continuously evolving. The
following overview lists user facing changes as well as newly added features in
inverse chronological order.

Changes since version 5.0
-------------------------

- New namespace expression syntax for jacobians and normals

  The syntax for the jacobian (determinant) of a geometry ``x`` in a namespace
  expression has changed from ``d:x`` to ``J:x``. If the jacobian is to be
  evaluated on a boundary or an interface, this has to be made explicit using
  ``J^:x``.

  The syntax for the normal of a geometry ``x`` in a namespace expression has
  changed from ``n_x_i`` to ``n:x_i``. There is no new syntax for the normal of
  the default geometry, formerly ``n_i``.

  Using the old syntax for the jacobian or the normal will generate a
  deprecation warning. The old syntax will be disabled as of version 6.0.

- Repository location

  The Nutils repository has moved to https://github.com/evalf/nutils.git. For
  the time being the old address is maintained by Github as an alias, but in
  the long term you are advised to update your remote as follows::

      git remote set-url origin https://github.com/evalf/nutils.git
