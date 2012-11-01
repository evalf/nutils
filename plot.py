from . import topology, util, numpy, function, element, _

import matplotlib
matplotlib.use( 'Agg' )

import os, tempfile

class Pylab( object ):
  'matplotlib figure'

  def __init__( self, title=None, filename=None ):
    'constructor'

    if isinstance( title, (list,tuple) ):
      self.title = numpy.array( title, dtype=object )
      self.shape = self.title.shape
      if self.title.ndim == 1:
        self.title = self.title[:,_]
      assert self.title.ndim == 2
    else:
      self.title = numpy.array( [[ title ]] )
      self.shape = ()
    self.filename = filename

  def __enter__( self ):
    'enter with block'

    from matplotlib import pyplot
    pyplot.figure()
    n, m = self.title.shape
    axes = [ PylabAxis( pyplot.subplot(n,m,iax+1), title ) for iax, title in enumerate( self.title.ravel() ) ]
    return numpy.array( axes, dtype=object ).reshape( self.shape ) if self.shape else axes[0]

  def __exit__( self, exc, msg, tb ):
    'exit with block'

    if exc:
      print 'ERROR: plot failed:', msg or exc
      return #True

    from matplotlib import pyplot
    n = len( os.listdir( util.DUMPDIR ) )
    if self.filename:
      name = self.filename
      if '.' not in name:
        name += '.png'
      fmt = name.split('.')[-1]
      path = os.path.join( util.DUMPDIR, name )
      fileobj = open( path, 'w' )
    else:
      fileobj = tempfile.NamedTemporaryFile( dir=util.DUMPDIR, prefix='f%04d' % n, suffix='.png', delete=False )
      path = fileobj.name
      name = os.path.basename( path )
      fmt = 'png'
    print 'saving to', name
    pyplot.savefig( fileobj, format=fmt )
    os.chmod( path, 0644 )
    pyplot.close()

class PylabAxis( object ):
  'matplotlib axis augmented with finity-specific functions'

  def __init__( self, ax, title ):
    'constructor'

    if title:
      ax.set_title( title )
    self._ax = ax

  def __getattr__( self, attr ):
    'forward getattr to axis'

    return getattr( self._ax, attr )

  def add_mesh( self, coords, topology, color=None, edgecolors='none', linewidth=1, xmargin=0, ymargin=0, aspect='equal', cbar='vertical', title=None, ischeme='gauss2', cscheme='contour3', clim=None, frame=True, usecolor=True ):
    'plot mesh'
  
    assert topology.ndims == 2
    from matplotlib import pyplot, collections
    poly = []
    values = []
    ndims, = coords.shape
    assert ndims in (2,3)
    if color:
      assert color.ndim == 0
      color = function.Tuple([ color, coords.iweights(ndims=2) ])
    for elem in util.progressbar( topology, title='plotting mesh' ):
      C = coords( elem, cscheme )
      if ndims == 3:
        C = project3d( C )
        cx, cy = numpy.hstack( [ C, C[:,:1] ] )
        if ( (cx[1:]-cx[:-1]) * (cy[1:]+cy[:-1]) ).sum() > 0:
          continue
      poly.append( C )
      if color:
        c, w = color( elem, ischeme )
        values.append( util.mean( c, weights=w, axis=0 ) )
  
    if values:
      elements = collections.PolyCollection( poly, edgecolors=edgecolors, linewidth=linewidth )
      elements.set_array( numpy.asarray(values) )
      if not usecolor:
        elements.set_cmap( pyplot.cm.gray )
      if cbar:
        pyplot.colorbar( elements, ax=self._ax, orientation=cbar )
    else:
      elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=linewidth )
    if clim:
      elements.set_clim( *clim )
    if ndims == 3:
      self.get_xaxis().set_visible( False )
      self.get_yaxis().set_visible( False )
      self.box( 'off' )
    self.add_collection( elements )
    vertices = numpy.concatenate( poly )
    xmin, ymin = vertices.min(0)
    xmax, ymax = vertices.max(0)
    if xmargin is not None:
      if not isinstance( xmargin, tuple ):
        xmargin = xmargin, xmargin
      self.set_xlim( xmin - xmargin[0], xmax + xmargin[1] )
    if ymargin is not None:
      if not isinstance( ymargin, tuple ):
        ymargin = ymargin, ymargin
      self.set_ylim( ymin - ymargin[0], ymax + ymargin[1] )
    if aspect:
      self.set_aspect( aspect )
      self.set_autoscale_on( False )
    if title:
      self.title( title )
    self.set_frame_on( frame )
  
  def add_quiver( self, coords, topology, quiver, sample='uniform3', scale=None ):
    'quiver builder'
  
    xyuv = function.Concatenate( [ coords, quiver ] )
    XYUV = [ xyuv(elem,sample) for elem in util.progressbar( topology, title='plotting quiver' ) ]
    self.quiver( *numpy.concatenate( XYUV, 0 ).T, scale=scale )

  def add_graph( self, xfun, yfun, topology, sample='contour10', **kwargs ):
    'plot graph of function on 1d topology'

    try:
      xfun = [ xf for xf in xfun ]
    except TypeError:
      xfun = [ xfun ]

    try:
      yfun = [ yf for yf in yfun ]
    except TypeError:
      yfun = [ yfun ]

    if len(xfun) == 1:
      xfun *= len(yfun)

    if len(yfun) == 1:
      yfun *= len(xfun)

    nfun = len(xfun)
    assert len(yfun) == nfun

    special_args = zip( *[ zip( [key]*nfun, val ) for (key,val) in kwargs.iteritems() if isinstance(val,list) and len(val) == nfun ] )
    XYD = [ ([],[],dict(d)) for d in special_args or [[]] * nfun ]
    xypairs = function.Zip( xfun, yfun, XYD )

    for elem in topology:
      for x, y, xyd in xypairs( elem, sample ):
        xyd[0].extend( x if x.ndim else [x] * y.size )
        xyd[0].append( numpy.nan )
        xyd[1].extend( y if y.ndim else [y] * x.size )
        xyd[1].append( numpy.nan )

    for x, y, d in XYD:
      kwargs.update(d)
      self.plot( x, y, **kwargs )

def project3d( C ):
  sqrt2 = numpy.sqrt( 2 )
  sqrt3 = numpy.sqrt( 3 )
  sqrt6 = numpy.sqrt( 6 )
  R = numpy.array( [[ sqrt3, 0, -sqrt3 ], [ 1, 2, 1 ], [ sqrt2, -sqrt2, sqrt2 ]] ) / sqrt6
  return util.transform( C, R[:,::2], axis=0 )

def writevtu( name, topology, coords, pointdata={}, celldata={} ):
  'write vtu from coords function'

  if not name.endswith( '.vtu' ):
    name += '.vtu'
  path = util.DUMPDIR + name
  import vtk
  vtkPoints = vtk.vtkPoints()
  vtkMesh = vtk.vtkUnstructuredGrid()
  pointdata_arrays = []
  for key, func in pointdata.iteritems():
    array = vtk.vtkFloatArray()
    array.SetName( key )
    if func.shape:
      assert len(func.shape) == 1
      array.SetNumberOfComponents( func.shape[0] )
    pointdata_arrays.append( function.Tuple([ array, func ]) )
    vtkMesh.GetPointData().AddArray( array )
  coords_pointdata = function.Tuple([ coords, function.Tuple( pointdata_arrays ) ])
  celldata_arrays = []
  for key, func in celldata.iteritems():
    assert func.ndim == 0
    array = vtk.vtkFloatArray()
    array.SetName( key )
    celldata_arrays.append( function.Tuple([ array, func, coords.iweights(topology.ndims) ]) )
    vtkMesh.GetCellData().AddArray( array )
  celldatafun = function.Tuple( celldata_arrays )
  for elem in util.progressbar( topology, title='saving %s' % name ):
    if isinstance( elem, element.TriangularElement ):
      vtkelem = vtk.vtkTriangle()
    elif isinstance( elem, element.QuadElement ) and elem.ndims == 2:
      vtkelem = vtk.vtkQuad()
    elif isinstance( elem, element.QuadElement ) and elem.ndims == 3:
      vtkelem = vtk.vtkVoxel() # TODO hexahedron for not rectilinear NOTE ordering changes!
    else:
      raise Exception, 'not sure what to do with element %r' % elem
    x, pdata = coords_pointdata( elem, 'contour0' )
    cellpoints = vtkelem.GetPointIds()
    for i, c in enumerate( x ):
      pointid = vtkPoints.InsertNextPoint( *c )
      cellpoints.SetId( i, pointid )
    vtkMesh.InsertNextCell( vtkelem.GetCellType(), cellpoints )
    for vtkArray, data in pdata:
      for v in data.flat:
        vtkArray.InsertNextValue( v )
    for vtkArray, data, iweights in celldatafun( elem, 'gauss1' ):
      vtkArray.InsertNextValue( util.mean( data, weights=iweights, axis=0 ) )
  vtkMesh.SetPoints( vtkPoints )
  vtkWriter = vtk.vtkXMLUnstructuredGridWriter()
  vtkWriter.SetInput( vtkMesh )
  vtkWriter.SetFileName( path )
  vtkWriter.SetDataModeToAscii()
  vtkWriter.Write()
  print 'saved', name

def preview( coords, topology, cscheme='contour8' ):
  'preview function'

  if topology.ndims == 3:
    topology = topology.boundary

  from matplotlib import pyplot, collections
  if coords.shape[0] == 2:
    mesh( coords, topology, cscheme=cscheme )
  elif coords.shape[0] == 3:
    polys = [ [] for i in range(4) ]
    for elem in topology:
      contour = coords( elem.eval(cscheme) )
      polys[0].append( project3d( contour ).T )
      polys[1].append( contour[:2].T )
      polys[2].append( contour[1:].T )
      polys[3].append( contour[::2].T )
    for iplt, poly in enumerate( polys ):
      elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=1 )
      ax = pyplot.subplot( 2, 2, iplt+1 )
      ax.add_collection( elements )
      xmin, ymin = numpy.min( [ numpy.min(p,axis=0) for p in poly ], axis=0 )
      xmax, ymax = numpy.max( [ numpy.max(p,axis=0) for p in poly ], axis=0 )
      d = .02 * (xmax-xmin+ymax-ymin)
      pyplot.axis([ xmin-d, xmax+d, ymin-d, ymax+d ])
      if iplt == 0:
        ax.get_xaxis().set_visible( False )
        ax.get_yaxis().set_visible( False )
        pyplot.box( 'off' )
      else:
        pyplot.title( '?ZXY'[iplt] )
  else:
    raise Exception, 'need 2D or 3D coordinates'
  pyplot.show()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
