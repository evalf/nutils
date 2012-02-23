from . import topology, util, numpy

def clf():
  'clear figure'

  from matplotlib import pyplot
  pyplot.clf()

def mesh( coords, topology, color=None, edgecolors='none', linewidth=1, xmargin=0, ymargin=0, ax=None, aspect='equal', cbar='horizontal' ):
  'plot mesh'

  from matplotlib import collections, pyplot
  poly = []
  values = []
  for elem in util.progressbar( topology, title='plotting mesh' ):
    poly.append( coords( elem('contour3') ).T )
    if color is not None:
      xi = elem('gauss2')
      values.append( util.mean( color(xi), weights=xi.weights ) )

  if values:
    elements = collections.PolyCollection( poly, edgecolors=edgecolors, linewidth=linewidth )
    elements.set_array( numpy.asarray(values) )
    if cbar:
      pyplot.colorbar( elements, orientation=cbar )
  else:
    elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=linewidth )
  if ax is None:
    ax = pyplot.gca()
  ax.add_collection( elements )
  vertices = numpy.concatenate( poly )
  xmin, ymin = vertices.min(0)
  xmax, ymax = vertices.max(0)
  if not isinstance( xmargin, tuple ):
    xmargin = xmargin, xmargin
  ax.set_xlim( xmin - xmargin[0], xmax + xmargin[1] )
  if not isinstance( ymargin, tuple ):
    ymargin = ymargin, ymargin
  ax.set_ylim( ymin - ymargin[0], ymax + ymargin[1] )
  if aspect:
    ax.set_aspect( aspect )
    ax.set_autoscale_on( False )

def quiver( coords, topology, quiver, sample='uniform3' ):
  'quiver builder'

  from matplotlib import collections, pyplot
  XYUV = []
  for elem in util.progressbar( topology, title='plotting quiver' ):
    xi = elem(sample)
    XYUV.append( numpy.concatenate( [ coords(xi), quiver(xi) ], axis=0 ) )
  pyplot.quiver( *numpy.concatenate( XYUV, 1 ) )

def show( block=True ):
  'show'

  from matplotlib import pyplot
  if block:
    pyplot.show()
  else:
    fig = pyplot.gcf()
    fig.show()
    fig.canvas.draw()
    fig.canvas.draw()

# OLD

class Quiver( object ):
  'quiver builder'

  def __init__( self ):
    'constructor'

    self.XYUV = []

  def add( self, xy, uv ):
    'add vectors'

    self.XYUV.append( numpy.concatenate([xy,uv]) )

  def plot( self, ax=None, color='k' ):
    'plot'

    if ax is None:
      ax = gca()
    quiver( *numpy.concatenate( self.XYUV, 1 ), color=color )

class Line( object ):
  'plot builder'

  def __init__( self, *lt ):
    'constructor'

    self.lt = lt
    self.XY = []

  def add( self, *xy ):
    'add vectors'

    xy = numpy.array(xy)
    self.XY.append( xy )

  def plot( self, xlim=None, ylim=None, ax=None, xlog=False, ylog=False, aspect=None ):
    'plot'

    XY = numpy.concatenate( self.XY, axis=1 )
    if ax is None:
      ax = gca()
    plotfun = loglog if xlog and ylog \
         else semilogx if xlog \
         else semilogy if ylog \
         else plot
    for i, lt in enumerate( self.lt ):
      plotfun( XY[0], XY[i+1], lt )
    if xlim is not None:
      ax.set_xlim( xlim )
    if ylim is not None:
      ax.set_ylim( ylim )
    if aspect:
      ax.set_aspect( aspect )
      ax.set_autoscale_on( False )

def build_image( coords, n=3, extent=(0,1,0,1), ax=None, ticks=True, clim=None, cbar=False, title='plotting' ):
  'image builder'

  if ax is None:
    ax = gca()
  assert isinstance( coords.topology, topology.StructuredTopology )
  assert coords.topology.ndims == 2
  image = numpy.zeros( coords.topology.structure.shape + (n,n) )
  scheme = 'uniform%d' % n
  items = zip( coords.topology, image.reshape(-1,n,n) )
  if title:
    items = util.progressbar( items, title=title )
  for elem, im in items:
    yield coords( elem(scheme) ), im.ravel()
  image = image.swapaxes(1,2).reshape( image.shape[0]*n, image.shape[1]*n )
  im = ax.imshow( image.T, extent=extent, origin='lower' )
  if not ticks:
    ax.xaxis.set_ticks( [] )
    ax.yaxis.set_ticks( [] )
  if clim:
    im.set_clim( *clim )
  if cbar:
    colorbar( im, orientation='vertical' )

def show_bg():
  'show in bg'

  fig = gcf()
  fig.show()
  fig.canvas.draw()
  fig.canvas.draw()

def plotmatr( A, **kwargs ):
  'Plot 10^log magnitudes of numpy matrix elements'

  A = numpy.log10( abs( A ) )
  if numpy.all( numpy.isinf( A ) ):
    A = numpy.zeros( A.shape )
  else:
    A[numpy.isinf( A )] = numpy.amin( A[~numpy.isinf( A )] ) - 1.
  pcolor( A, **kwargs )
  colorbar()
  ylim( ylim()[-1::-1] ) # invert y axis: equiv to MATLAB axis ij
  axis( 'tight' )

def savepdf( name, fig=None ):
  'save figure in plots dir'

  import os
  if fig is None:
    fig = gcf()
  path = os.path.join( 'plots', name + '.pdf' )
  dirname = os.path.dirname( path )
  if not os.path.isdir( dirname ):
    os.makedirs( dirname )
  fig.savefig( path, bbox_inches='tight', pad_inches=0 )

def writevtu( coords, path, topology=None, refine=1 ):
  'write vtu from coords function'

  topo = util.progressbar( ( topology or coords.topology ).refine( refine ), title='saving %s' % path )
  import vtk
  vtkPoints = vtk.vtkPoints()
  vtkMesh = vtk.vtkUnstructuredGrid()
  for elem in topo:
    xi = elem( 'contour2' )
    x = coords( xi )  
    cellpoints = vtk.vtkIdList()
    for c in x.fval.T:
      id = vtkPoints.InsertNextPoint( *c )
      cellpoints.InsertNextId( id )
    vtkMesh.InsertNextCell( vtk.VTK_QUAD, cellpoints )  
  vtkMesh.SetPoints( vtkPoints )
  vtkWriter = vtk.vtkXMLUnstructuredGridWriter()
  vtkWriter.SetInput( vtkMesh )
  vtkWriter.SetFileName( path )
  vtkWriter.SetDataModeToAscii()
  vtkWriter.Write()

def preview( coords, topology ):
  'preview function'

  if topology.ndims == 3:
    topology = topology.boundary
  assert topology.ndims == 2

  from matplotlib import collections
  figure()
  if coords.shape[0] == 2:
    mesh( coords, topology )
  elif coords.shape[0] == 3:
    polys = [ [] for i in range(4) ]
    sqrt2 = numpy.sqrt( 2 )
    sqrt3 = numpy.sqrt( 3 )
    sqrt6 = numpy.sqrt( 6 )
    R = numpy.array( [[ sqrt3, 0, -sqrt3 ], [ 1, 2, 1 ], [ sqrt2, -sqrt2, sqrt2 ]] ) / sqrt6
    for elem in topo:
      contour = coords( elem('contour8') )
      polys[0].append( util.transform( contour, R[:,::2], axis=0 ).T )
      polys[1].append( contour[:2].T )
      polys[2].append( contour[1:].T )
      polys[3].append( contour[::2].T )
    for iplt, poly in enumerate( polycolls ):
      elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=1 )
      ax = subplot( 2, 2, iplt+1 )
      ax.add_collection( elements )
      if iplt == 0:
        ax.get_xaxis().set_visible( False )
        ax.get_yaxis().set_visible( False )
        box( 'off' )
      else:
        title( '?ZXY'[iplt] )
  show()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
