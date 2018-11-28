from nutils.testing import *
import os, tempfile, pathlib, treelog
import nutils, numpy

class mplfigure(TestCase):

  def setUpContext(self, stack):
    super().setUpContext(stack)
    self.outdir = pathlib.Path(stack.enter_context(tempfile.TemporaryDirectory()))
    stack.enter_context(treelog.set(treelog.DataLog(str(self.outdir))))

  @nutils.testing.requires('matplotlib', 'PIL')
  def test_autodetect_imagetype(self):
    for (imagetype, test) in (('jpg', lambda data: self.assertEqual(data[:3], b'\xFF\xD8\xFF')),
                              ('png', lambda data: self.assertEqual(data[:8], b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A')),
                              ('pdf', lambda data: self.assertEqual(data[:4], b'\x25\x50\x44\x46')),
                              ('svg', lambda data: self.assertRegex(data, b'<svg[^<>]*>'))):
      with self.subTest(imagetype=imagetype):
        with nutils.export.mplfigure('test.{}'.format(imagetype)) as fig:
          ax = fig.add_subplot(111)
          ax.plot([1,2,3],[1,2,3])
        with (self.outdir/'test.{}'.format(imagetype)).open('rb') as f:
          test(f.read())

@parametrize
class vtk(TestCase):

  def setUp(self):
    if self.ndims == 2:
      self.x = numpy.array([[0,0],[0,1],[1,0],[1,1]], dtype=self.xtype)
      self.tri = numpy.array([[0,1,2],[1,2,3]])
    elif self.ndims == 3:
      self.x = numpy.array([[0,0,0],[0,1,0],[1,0,0],[0,0,1]], dtype=self.xtype)
      self.tri = numpy.array([[0,1,2,3]])
    else:
      raise Exception('invalid ndims {}'.format(self.ndims))
    if hasattr(self, 'ptype'):
      self.p = numpy.arange(len(self.x) * numpy.prod(self.pshape)).astype(self.ptype).reshape((len(self.x),)+self.pshape)
    else:
      self.p = None
    if hasattr(self, 'ctype'):
      self.c = numpy.arange(len(self.tri) * numpy.prod(self.cshape)).astype(self.ctype).reshape((len(self.tri),)+self.cshape)
    else:
      self.c = None

  @property
  def data(self):
    yield b'# vtk DataFile Version 3.0\nvtk output\nBINARY\nDATASET UNSTRUCTURED_GRID\n'
    if self.xtype == 'i4':
      yield b'POINTS 4 int\n'
    elif self.xtype == 'f4':
      yield b'POINTS 4 float\n'
    elif self.xtype == 'f8':
      yield b'POINTS 4 double\n'
    else:
      raise Exception('not supported: xtype={!r}'.format(self.xtype))
    if self.ndims == 2 and self.xtype == 'i4':
      yield bytes([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0])
    elif self.ndims == 2 and self.xtype == 'f4':
      yield bytes([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,63,128,0,0,63,128,0,0,0,0,0,0])
    elif self.ndims == 2 and self.xtype == 'f8':
      yield bytes([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,240,0,0,0,0,0,0,63,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif self.ndims == 3 and self.xtype == 'f4':
      yield bytes([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0,0,0,0,0,63,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,128,0,0])
    else:
      raise Exception('not supported: xtype={!r}, ndims={}'.format(self.xtype, self.ndims))
    if self.ndims == 2:
      yield b'CELLS 2 8\n'
      yield bytes([0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,1,0,0,0,2,0,0,0,3])
      yield b'CELL_TYPES 2\n'
      yield bytes([0,0,0,5,0,0,0,5])
    elif self.ndims == 3:
      yield b'CELLS 1 5\n'
      yield bytes([0,0,0,4,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,3])
      yield b'CELL_TYPES 1'
      yield bytes([10,0,0,0,10])
    else:
      raise Exception('invalid ndims {}'.format(self.ndims))
    if self.p is not None:
      yield b'POINT_DATA 4\n'
      if self.ptype == 'f4' and self.pshape == ():
        yield b'SCALARS p float 1\nLOOKUP_TABLE default\n'
        yield bytes([0,0,0,0,63,128,0,0,64,0,0,0,64,64,0,0])
      elif self.ptype == 'f8' and self.pshape == ():
        yield b'SCALARS p double 1\nLOOKUP_TABLE default\n'
        yield bytes([0,0,0,0,0,0,0,0,63,240,0,0,0,0,0,0,64,0,0,0,0,0,0,0,64,8,0,0,0,0,0,0])
      elif self.ptype == 'i1' and self.pshape == ():
        yield b'SCALARS p char 1\nLOOKUP_TABLE default\n'
        yield bytes([0,1,2,3])
      elif self.ptype == 'i2' and self.pshape == ():
        yield b'SCALARS p short 1\nLOOKUP_TABLE default\n'
        yield bytes([0,0,0,1,0,2,0,3])
      elif self.ptype == 'i4' and self.pshape == ():
        yield b'SCALARS p int 1\nLOOKUP_TABLE default\n'
        yield bytes([0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,3])
      elif self.ptype == 'i1' and self.pshape == (2,):
        yield b'VECTORS p char\n'
        yield bytes([0,1,0,2,3,0,4,5,0,6,7,0])
      elif self.ptype == 'f4' and self.pshape == (2,):
        yield b'VECTORS p float\n'
        yield bytes([0,0,0,0,63,128,0,0,0,0,0,0,64,0,0,0,64,64,0,0,0,0,0,0,64,128,0,0,64,160,0,0,0,0,0,0,64,192,0,0,64,224,0,0,0,0,0,0])
      elif self.ptype == 'i1' and self.pshape == (3,):
        yield b'VECTORS p char\n'
        yield bytes([0,1,2,3,4,5,6,7,8,9,10,11])
      elif self.ptype == 'i1' and self.pshape == (2,2):
        yield b'TENSORS p char\n'
        yield bytes([0,1,0,2,3,0,0,0,0,4,5,0,6,7,0,0,0,0,8,9,0,10,11,0,0,0,0,12,13,0,14,15,0,0,0,0])
      elif self.ptype == 'i1' and self.pshape == (3,3):
        yield b'TENSORS p char\n'
        yield bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
      else:
        raise Exception('not supported: ptype={}, udims={}'.format(self.ptype, self.udims))
    if self.c is not None:
      yield b'CELL_DATA 1\n'
      if self.ndims == 3 and self.ctype == 'i1' and self.cshape == ():
        yield b'SCALARS c char 1\nLOOKUP_TABLE default\n'
        yield bytes([0])
      else:
        raise Exception('not supported: ndims={}, ctype={}, cdims={}'.format(self.ndims, self.ctype, self.cdims))

  def test_data(self):
    with tempfile.TemporaryDirectory() as outdir, treelog.set(treelog.DataLog(outdir)):
      kwargs = {}
      if self.p is not None:
        kwargs['p'] = self.p
      if self.c is not None:
        kwargs['c'] = self.c
      nutils.export.vtk('test', self.tri, self.x, **kwargs)
      with open(os.path.join(outdir, 'test.vtk'), 'rb') as f:
        data = f.read()
    self.assertEqual(data, b''.join(self.data))

vtk(ndims=2, xtype='i4')
vtk(ndims=2, xtype='f4')
vtk(ndims=2, xtype='f8')
vtk(ndims=3, xtype='f4')
vtk(ndims=2, xtype='f4', ptype='f4', pshape=())
vtk(ndims=2, xtype='f4', ptype='f8', pshape=())
vtk(ndims=2, xtype='f4', ptype='i1', pshape=())
vtk(ndims=2, xtype='f4', ptype='i2', pshape=())
vtk(ndims=2, xtype='f4', ptype='i4', pshape=())
vtk(ndims=3, xtype='f4', ptype='i1', pshape=())
vtk(ndims=2, xtype='f4', ptype='i1', pshape=(2,))
vtk(ndims=2, xtype='f4', ptype='f4', pshape=(2,))
vtk(ndims=3, xtype='f4', ptype='i1', pshape=(3,))
vtk(ndims=2, xtype='f4', ptype='i1', pshape=(2,2))
vtk(ndims=3, xtype='f4', ptype='i1', pshape=(3,3))
vtk(ndims=3, xtype='f4', ctype='i1', cshape=())
