Changelog
=========

Nutils is being actively developed and the API is continuously evolving. The
following overview lists user facing changes as well as newly added features in
inverse chronological order.

Changes since version 5.0
-------------------------

- Treelog update

  Nutils now depends on treelog version 1.0b5, which brings improved iterators
  along with other enhancements. For transitional convenience the backwards
  incompatible changes have been backported in the :mod:`nutils.log` wrapper,
  which now emits a warning to indicate that the fallback is active. This
  wrapper is scheduled for deletion prior to the release of version 6.0. To
  update treelog to the most recent version use::

      python -m pip install -U treelog

- Unit type

  The new :class:`nutils.types.unit` allows for the creation of a unit system
  for easy specification of physical quantities. Used in conjuction with
  :func:`nutils.cli.run` this facilitates specifying units from the command
  line, as well as providing a warning mechanism against incompatible units.

      >>> U = types.unit.create(m=1, s=1, g=1e-3, N='kg*m/s2', Pa='N/m2')
      >>> def main(length=U('2m'), F=U('5kN')):
      ...   topo, geom = mesh.rectilinear([numpy.linspace(0,length,10)])

    | $ python myscript.py length=25cm # OK
    | $ python myscript.py F=10Pa # error!

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
