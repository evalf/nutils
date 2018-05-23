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

from . import config, log, core, util
import contextlib, numpy

@contextlib.contextmanager
@util.positional_only('name')
def mplfigure(*args, **kwargs):
  '''Matplotlib figure context, convenience function.

  Returns a :class:`matplotlib.figure.Figure` object suitable for
  `object-oriented plotting`_. Upon exit the result is saved using the
  agg-backend in all formats configured via :attr:`nutils.config.imagetype`,
  and the resulting filenames written to log.

  .. _`object-oriented plotting`: https://matplotlib.org/gallery/api/agg_oo_sgskip.html

  Args
  ----
  name : :class:`str`
      The filename (without extension) of the resulting figure(s)
  **kwargs :
      Keyword arguments are passed on unchanged to the constructor of the
      :class:`matplotlib.figure.Figure` object.
  '''

  name, = args
  formats = config.imagetype.split(',')
  with log.context(name):
    import matplotlib.backends.backend_agg
    fig = matplotlib.figure.Figure(**kwargs)
    matplotlib.backends.backend_agg.FigureCanvas(fig) # sets reference via fig.set_canvas
    yield fig
    for fmt in formats:
      with log.context(fmt), core.open_in_outdir(name+'.'+fmt, 'wb') as f:
        fig.savefig(f, format=fmt)
    fig.set_canvas(None) # break circular reference
  log.user(', '.join(name+'.'+fmt for fmt in formats))

@util.positional_only('name', 'tri', 'x')
def vtk(*args, **kwargs):
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

  name, cells, points = args
  assert cells.ndim == points.ndim == 2
  ndims = points.shape[1]
  assert cells.shape[1] == ndims + 1

  if ndims == 2:
    points = numpy.concatenate([points, numpy.zeros_like(points[:,:1])], axis=1)
    celltype = 5 # VTK_TRIANGLE
  elif ndims == 3:
    celltype = 10 # VTK_TETRA
  else:
    raise Exception('invalid point dimension: {}'.format(ndims))

  vtkdtype = {
    numpy.dtype('>i1'): 'char',  numpy.dtype('>u1'): 'unsigned_char',
    numpy.dtype('>i2'): 'short', numpy.dtype('>u2'): 'unsigned_short',
    numpy.dtype('>i4'): 'int',   numpy.dtype('>u4'): 'unsigned_int',
    numpy.dtype('>f4'): 'float', numpy.dtype('>f8'): 'double'}

  vtkndim = {1: 'SCALARS {} {} 1\nLOOKUP_TABLE default\n', 2: 'VECTORS {} {}\n', 3: 'TENSORS {} {}\n'}

  bigendian = lambda a: a.astype('>{0.kind}{0.itemsize}'.format(a.dtype))

  points = bigendian(points)
  if points.dtype not in vtkdtype:
    raise Exception('invalid data type for points: {}'.format(points.dtype))

  t_cells = numpy.empty((len(cells), ndims+2), dtype='>u4')
  t_cells[:,0] = ndims + 1
  t_cells[:,1:] = cells

  gathered = util.gather((len(array), (name, bigendian(array))) for name, array in kwargs.items())

  invalid_length = [name for n, arrays in gathered if n not in (len(points), len(cells)) for name, array in arrays]
  if invalid_length:
    raise Exception('data length matches neither points nor cells: {}'.format(', '.join(invalid_length)))

  invalid_dimension = [name for n, arrays in gathered for name, array in arrays if array.ndim not in vtkndim or any(n>3 for n in array.shape[1:])]
  if invalid_dimension:
    raise Exception('invalid array dimension: {}'.format(', '.join(invalid_dimension)))

  invalid_dtype = [name for n, arrays in gathered for name, array in arrays if array.dtype not in vtkdtype]
  if invalid_dtype:
    raise Exception('invalid array data type: {}'.format(', '.join(invalid_dtype)))

  name_vtk = name + '.vtk'
  with log.context(name_vtk), core.open_in_outdir(name_vtk, 'wb') as vtk:
    vtk.write(b'# vtk DataFile Version 3.0\nvtk output\nBINARY\nDATASET UNSTRUCTURED_GRID\n')
    vtk.write('POINTS {} {}\n'.format(len(points), vtkdtype[points.dtype]).encode('ascii'))
    points.tofile(vtk)
    vtk.write('CELLS {} {}\n'.format(len(t_cells), t_cells.size).encode('ascii'))
    t_cells.tofile(vtk)
    vtk.write('CELL_TYPES {}\n'.format(len(cells)).encode('ascii'))
    numpy.array(celltype, dtype='>u4').repeat(len(cells)).tofile(vtk)
    for n, arrays in gathered:
      vtk.write('{}_DATA {}\n'.format('POINT' if n == len(points) else 'CELL', n).encode('ascii'))
      for name, array in arrays:
        vtk.write(vtkndim[array.ndim].format(name, vtkdtype[array.dtype]).encode('ascii'))
        numpy.pad(array, [[0,0]] + [[0,3-n] for n in array.shape[1:]], mode='constant').tofile(vtk)

  log.user(name_vtk)

# vim:sw=2:sts=2:et
