Changelog
=========

Nutils is being actively developed and the API is continuously evolving. The
following overview lists user facing changes as well as newly added features in
inverse chronological order.

Changes since version 5.0
-------------------------

- Sample basis

  Samples now provide a :func:`nutils.sample.Sample.basis`: an array that for
  any point in the sample evaluates to the unit vector corresponding to its
  index. This new underpinning of :func:`nutils.sample.Sample.asfunction` opens
  the way for sampled arguments, as demonstrated in the last example below:

      >>> H1 = mysample.asfunction(mydata) # mysample.eval(H1) == mydata
      >>> H2 = mysample.basis().dot(mydata) # mysample.eval(H2) == mydata
      >>> ns.Hbasis = mysample.basis()
      >>> H3 = 'Hbasis_n ?d_n' @ ns # mysample.eval(H3, d=mydata) == mydata

- Higher order gmsh geometries

  Gmsh element support has been extended to include cubic and quartic meshes in
  2D and quadratic meshes in 3D, and parsing the msh file is now a cacheable
  operation. Additionally, tetrahedra now define bezier points at any order.

- Repository location

  The Nutils repository has moved to https://github.com/evalf/nutils.git. For
  the time being the old address is maintained by Github as an alias, but in
  the long term you are advised to update your remote as follows::

      git remote set-url origin https://github.com/evalf/nutils.git
