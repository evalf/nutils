# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from . import util, warnings
import contextlib, numpy, os, treelog as log

@contextlib.contextmanager
@util.positional_only
def mplfigure(name, kwargs=...):
  '''Matplotlib figure context, convenience function.

  Returns a :class:`matplotlib.figure.Figure` object suitable for
  `object-oriented plotting`_. Upon exit the result is written to the currently
  active logger.

  .. _`object-oriented plotting`: https://matplotlib.org/gallery/api/agg_oo_sgskip.html

  .. requires:: matplotlib

  Args
  ----
  name : :class:`str`
      The filename of the resulting figure.
  **kwargs :
      Keyword arguments are passed on unchanged to the constructor of the
      :class:`matplotlib.figure.Figure` object.
  '''

  import matplotlib.figure, matplotlib.backends.backend_agg
  fig = matplotlib.figure.Figure(**kwargs)
  with log.userfile(name, 'wb') as f:
    yield fig
    if f:
      matplotlib.backends.backend_agg.FigureCanvas(fig) # sets reference via fig.set_canvas
      try:
        fig.savefig(f, format=os.path.splitext(name)[1][1:])
      finally:
        fig.set_canvas(None) # break circular reference

def triplot(name, points, values=None, *, tri=None, hull=None, cmap=None, clim=None, linewidth=.1, linecolor='k'):
  if (tri is None) != (values is None):
    raise Exception('tri and values can only be specified jointly')
  with mplfigure(name) as fig:
    ax = fig.add_subplot(111)
    if points.shape[1] == 1:
      if tri is not None:
        import matplotlib.collections
        ax.add_collection(matplotlib.collections.LineCollection(numpy.array([points[:,0], values]).T[tri]))
      if hull is not None:
        for x in points[hull[:,0],0]:
          ax.axvline(x, color=linecolor, linewidth=linewidth)
      ax.autoscale(enable=True, axis='x', tight=True)
      if clim is None:
        ax.autoscale(enable=True, axis='y', tight=False)
      else:
        ax.set_ylim(clim)
    elif points.shape[1] == 2:
      ax.set_aspect('equal')
      if tri is not None:
        im = ax.tripcolor(points[:,0], points[:,1], tri, values, shading='gouraud', cmap=cmap, rasterized=True)
        if clim is not None:
          im.set_clim(clim)
        fig.colorbar(im)
      if hull is not None:
        import matplotlib.collections
        ax.add_collection(matplotlib.collections.LineCollection(points[hull], colors=linecolor, linewidths=linewidth, alpha=1 if tri is None else .5))
      ax.autoscale(enable=True, axis='both', tight=True)
    else:
      raise Exception('invalid spatial dimension: {}'.format(points.shape[1]))

@util.positional_only
def vtk(name, cells, points, kwargs=...):
  '''Export data to a VTK file.

  This method provides a simple interface to the `VTK file format`_ with a
  number of important restrictions:

  *   Simplex-only. This makes it possible to define the mesh by a combination
      of vertex coordinates and a connectivity table.
  *   Legacy mode. The newer XML based format is more complex and does not
      provide benefits within the constraints set by this method.
  *   Binary mode. This allows for direct output of binary data, which aids
      speed, accuracy and file size.

  Beyond the mandatory file name, connectivity table, and vertex coordinates,
  any additional data sets can be provided as keyword arguments, where the keys
  are the names by which the arrays are stored. The data can be either vertex
  or point data, with the distinction made based on the length of the array.

  .. _`VTK file format`: https://www.vtk.org/VTK/img/file-formats.pdf

  Args
  ----
  name : :class:`str`
    Destination file name (without vtk extension).
  tri : :class:`int` array
    Triangulation.
  x : :class:`float` array
    Vertex coordinates.
  **kwargs :
    Cell and/or point data
  '''

  vtkcelltype = {
    3: numpy.array( 5, dtype='>u4'), # VTK_TRIANGLE
    4: numpy.array(10, dtype='>u4')} # VTK_TETRA
  vtkndim = {
    1: 'SCALARS {} {} 1\nLOOKUP_TABLE default\n',
    2: 'VECTORS {} {}\n',
    3: 'TENSORS {} {}\n'}
  vtkdtype = {
    numpy.dtype('>i1'): 'char',  numpy.dtype('>u1'): 'unsigned_char',
    numpy.dtype('>i2'): 'short', numpy.dtype('>u2'): 'unsigned_short',
    numpy.dtype('>i4'): 'int',   numpy.dtype('>u4'): 'unsigned_int',
    numpy.dtype('>f4'): 'float', numpy.dtype('>f8'): 'double'}
  def vtkarray(a): # convert to big endian data and zero-pad all axes to length 3
    a = numpy.asarray(a)
    if any(n > 3 for n in a.shape[1:]):
      raise Exception('invalid shape: {}'.format(a.shape))
    e = numpy.zeros([len(a)] + [3] * (a.ndim-1), dtype='>{0.kind}{0.itemsize}'.format(a.dtype))
    if e.dtype not in vtkdtype:
      raise Exception('invalid data type: {}'.format(a.dtype))
    e[tuple(map(slice, a.shape))] = a
    return e

  assert cells.ndim == points.ndim == 2
  npoints, ndims = points.shape
  ncells, nverts = cells.shape

  if nverts not in vtkcelltype:
    raise Exception('invalid point dimension: {}'.format(ndims))

  points = vtkarray(points)
  gathered = util.gather((len(array), (name, vtkarray(array))) for name, array in kwargs.items())

  for n, items in gathered:
    if n != npoints and n != ncells:
      raise Exception('data length matches neither points nor cells: {}'.format(', '.join(dname for dname, array in items)))

  t_cells = numpy.empty((ncells, nverts+1), dtype='>u4')
  t_cells[:,0] = nverts
  t_cells[:,1:] = cells

  name_vtk = name + '.vtk'
  with log.userfile(name_vtk, 'wb') as vtk:
    vtk.write(b'# vtk DataFile Version 3.0\nvtk output\nBINARY\nDATASET UNSTRUCTURED_GRID\n')
    vtk.write('POINTS {} {}\n'.format(npoints, vtkdtype[points.dtype]).encode('ascii'))
    points.tofile(vtk)
    vtk.write('CELLS {} {}\n'.format(ncells, t_cells.size).encode('ascii'))
    t_cells.tofile(vtk)
    vtk.write('CELL_TYPES {}\n'.format(ncells).encode('ascii'))
    vtkcelltype[nverts].repeat(ncells).tofile(vtk)
    for n, items in gathered:
      vtk.write('{}_DATA {}\n'.format('POINT' if n == npoints else 'CELL', n).encode('ascii'))
      for dname, array in items:
        vtk.write(vtkndim[array.ndim].format(dname, vtkdtype[array.dtype]).encode('ascii'))
        array.tofile(vtk)

# vim:sw=2:sts=2:et
