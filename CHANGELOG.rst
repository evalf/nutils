Changelog
=========

Nutils is being actively developed and the API is continuously evolving. The
following overview lists user facing changes as well as newly added features in
inverse chronological order.

Changes since version 5.0
-------------------------

- Higher order gmsh geometries

  Gmsh element support has been extended to include cubic and quartic meshes in
  2D and quadratic meshes in 3D, and parsing the msh file is now a cacheable
  operation. Additionally, tetrahedra now define bezier points at any order.

- Repository location

  The Nutils repository has moved to https://github.com/evalf/nutils.git. For
  the time being the old address is maintained by Github as an alias, but in
  the long term you are advised to update your remote as follows::

      git remote set-url origin https://github.com/evalf/nutils.git
