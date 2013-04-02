from . import topology, util, numpy, function, element, log, prop, numeric, _
import os

class BasePlot( object ):
  'base class for plotting objects'

  def __init__ ( self, name, ndigits=0, index=None ):

    self.path = prop.dumpdir

    assert isinstance(ndigits,int) and ndigits >= 0, 'nonnegative integer required'
    if ndigits:
      if index is None:
        index = 1
        for filename in os.listdir( self.path ):
          if filename.startswith( name ):
            num = filename[len(name):].split('.')[0]
            if num.isdigit():
              index = max( index, int(num)+1 )
      name += str(index).rjust(ndigits,'0')

    self.name  = name
    self.names = None

  def __enter__( self ):
    'enter with block'

    self.oldlog = log.context( 'plotting', depth=1 )
    return self

  def __exit__( self, *exc_info ):
    'exit with block'

    exc_type = exc_info[0]
    if exc_type == KeyboardInterrupt:
      log.restore( self.oldlog, depth=1 )
      return False
    elif exc_type:
      log.exception( exc_info )
    else:
      if self.names:
        for name in self.names:
          self.save( name )
        log.path( *self.names )
    log.restore( self.oldlog, depth=1 )
    return True

  def save ( self, name ):
    return

class PyPlot( BasePlot ):
  'matplotlib figure'

  def __init__( self, name, imgtype=None, ndigits=3, index=None ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    import matplotlib

    matplotlib.use( 'Agg', warn=False )

    from matplotlib import pyplot

    imgtype = getattr( prop, 'imagetype', 'png' ) if imgtype is None else imgtype
    self.names = [ self.name + '.' + ext for ext in imgtype.split(',') ]

    self.__dict__.update( pyplot.__dict__ )
    fig = self.figure()
    fig.patch.set_alpha( 0 )

  def save( self, name ):
    'save images'

    self.savefig( os.path.join( self.path, name ) )
    #self.close()

  def polycol( self, verts, facecolors='none', **kwargs ):
    'add polycollection'
  
    from matplotlib import collections
    assert all( vert.ndim == 2 and vert.shape[1] == 2 for vert in verts )
    if facecolors != 'none':
      assert isinstance(facecolors,numpy.ndarray) and facecolors.shape == (len(verts),)
      array = facecolors
      facecolors = None
    polycol = collections.PolyCollection( verts, facecolors=facecolors, **kwargs )
    if facecolors is None:
      polycol.set_array( array )
    self.gca().add_collection( polycol )
    self.sci( polycol )
    return polycol

  def slope_triangle( self, x, y, fillcolor='0.9', edgecolor='k', xoffset=0, yoffset=0.1, slopefmt='{0:.1f}' ):
    '''Draw slope triangle for supplied y(x)
       - x, y: coordinates
       - xoffset, yoffset: distance graph & triangle (points)
       - fillcolor, edgecolor: triangle style
       - slopefmt: format string for slope number'''

    # TODO check for gca() loglog scale

    i, j = (-2,-1) if x[-1] < x[-2] else (-1,-2) # x[i] > x[j]

    from matplotlib import transforms
    shifttrans = self.gca().transData \
               + transforms.ScaledTranslation( xoffset, -yoffset, self.gcf().dpi_scale_trans )

    slope = numpy.log( y[-2]/y[-1] ) / numpy.log( x[-2]/x[-1] )

    self.fill( (x[i],x[j],x[i]), (y[j],y[j],y[i]),
      color=fillcolor,
      edgecolor=edgecolor,
      transform=shifttrans )

    self.text( x[i]**(2/3.) * x[j]**(1/3.), y[i]**(1/3.) * y[j]**(2/3.), slopefmt.format(slope),
      horizontalalignment='center',
      verticalalignment='center',
      transform=shifttrans )

    return slope

  def slope_trend( self, x, y, lt='k-', xoffset=.1, slopefmt='{0:.1f}' ):
    '''Draw slope triangle for supplied y(x)
       - x, y: coordinates
       - slopefmt: format string for slope number'''

    # TODO check for gca() loglog scale

    slope = numpy.log( y[-2]/y[-1] ) / numpy.log( x[-2]/x[-1] )
    C = y[-1] / x[-1]**slope

    self.loglog( x, C * x**slope, 'k-' )

    from matplotlib import transforms
    shifttrans = self.gca().transData \
               + transforms.ScaledTranslation( -xoffset if x[-1] < x[0] else xoffset, 0, self.gcf().dpi_scale_trans )

    self.text( x[-1], y[-1], slopefmt.format(slope),
      horizontalalignment='right' if x[-1] < x[0] else 'left',
      verticalalignment='center',
      transform=shifttrans )

    return slope

  def rectangle( self, x0, w, h, fc='none', ec='none', **kwargs ):
    'rectangle'

    from matplotlib import patches
    patch = patches.Rectangle( x0, w, h, fc=fc, ec=ec, **kwargs )
    self.gca().add_patch( patch )
    return patch

  def griddata( self, xlim, ylim, data ):
    'plot griddata'

    assert data.ndim == 2
    self.imshow( data.T, extent=(xlim[0],xlim[-1],ylim[0],ylim[-1]), origin='lower' )

  def cspy( self, A, **kwargs ): 
    'Like pyplot.spy, but coloring acc to 10^log of absolute values, where [0, inf, nan] show up in blue.'
    A = numpy.log10( numpy.abs( A ) )
    B = numpy.isinf( A ) | numpy.isnan( A ) # what needs replacement
    A[B] = ~B if numpy.all( B ) else numpy.amin( A[~B] ) - 1.
    self.pcolor( A, **kwargs )
    self.colorbar()
    self.ylim( self.ylim()[-1::-1] ) # invert y axis: equiv to MATLAB axis ij
    self.axis( 'tight' )

class VTKFile( BasePlot ):
  'vtk file'

  def __init__( self, name, index=None, ndigits=0, ascii=False ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    import vtk 

    if self.name.lower().endswith('.vtu'):
      self.names = [self.name]
    else:  
      self.names = [self.name+'.vtu']

    self.ascii   = ascii
    self.vtkMesh = vtk.vtkUnstructuredGrid()

  def save( self, name ):
    import vtk
    vtkWriter = vtk.vtkXMLUnstructuredGridWriter()
    vtkWriter.SetInput   ( self.vtkMesh )
    vtkWriter.SetFileName( os.path.join( self.path, name ) )
    if self.ascii:
      vtkWriter.SetDataModeToAscii()
    vtkWriter.Write()

  def vertices( self, points ):

    assert isinstance( points, (list,tuple,numpy.ndarray) ), 'Expected list of point arrays'

    import vtk

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints( sum(pts.shape[0] for pts in points) )

    cnt = 0
    for pts in points:
      if pts.shape[1] < 3:
        pts = numpy.concatenate([pts,numpy.zeros(shape=(pts.shape[0],3-pts.shape[1]))],axis=1)

      for point in pts:
        vtkPoints .SetPoint( cnt, point )
        cellpoints = vtk.vtkVertex().GetPointIds()
        cellpoints.SetId( 0, cnt )
        self.vtkMesh.InsertNextCell( vtk.vtkVertex().GetCellType(), cellpoints )
        cnt +=1

    self.vtkMesh.SetPoints( vtkPoints )

  def unstructuredgrid( self, points ):
    """add unstructured grid"""

    assert isinstance( points, (list,tuple,numpy.ndarray) ), 'Expected list of point arrays'

    import vtk

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints( sum(pts.shape[0] for pts in points) )

    cnt = 0
    for pts in points:

      np, ndims = pts.shape
      vtkelem   = None

      if np == 3:
        vtkelem = vtk.vtkTriangle()
      elif np == 4:  
        if ndims == 2:
          vtkelem = vtk.vtkQuad()
        elif ndims == 3:
          vtkelem = vtk.vtkTetra()
      elif np == 8:
        vtkelem = vtk.vtkVoxel() # TODO hexahedron for not rectilinear NOTE ordering changes!

      if not vtkelem:
        raise Exception('not sure what to do with cells with ndims=%d and npoints=%d' % (ndims,np))

      if ndims < 3:
        pts = numpy.concatenate([pts,numpy.zeros(shape=(pts.shape[0],3-ndims))],axis=1)

      cellpoints = vtkelem.GetPointIds()

      for i,point in enumerate(pts):
        vtkPoints .SetPoint( cnt, point )
        cellpoints.SetId( i, cnt )
        cnt +=1
    
      self.vtkMesh.InsertNextCell( vtkelem.GetCellType(), cellpoints )

    self.vtkMesh.SetPoints( vtkPoints )

  def celldataarray( self, name, data ):
    'add cell array'
    ncells = self.vtkMesh.GetNumberOfCells()
    assert ncells == data.shape[0], 'Cell data array should have %d entries' % ncells
    self.vtkMesh.GetCellData().AddArray( self.__vtkarray(name,data) )

  def pointdataarray( self, name, data ):
    'add cell array'
    npoints = self.vtkMesh.GetNumberOfPoints()
    data = numpy.concatenate( list(data), axis=0 )
    assert npoints == data.shape[0], 'Point data array should have %d entries' % npoints
    self.vtkMesh.GetPointData().AddArray( self.__vtkarray(name,data) )

  def __vtkarray( self, name, data ):
    import vtk
    if data.ndim == 1:
      data = data[:,_]
    array = vtk.vtkFloatArray()
    array.SetName( name )
    array.SetNumberOfComponents( data.shape[1] )
    array.SetNumberOfTuples( data.shape[0] )
    for i,d in enumerate(data):
      array.SetTuple( i, d )
    return array

def writevtu( name, topo, coords, pointdata={}, celldata={}, ascii=False, superelements=False, **kwargs ):
  'write vtu from coords function'

  with VTKFile( name, ascii=ascii ) as vtkfile:

    if not superelements:
      topo = topology.UnstructuredTopology( topo.get_simplices( **kwargs ), topo.ndims )
    else:
      topo = topology.UnstructuredTopology( filter(None,[elem if not isinstance(elem,element.TrimmedElement) else elem.elem for elem in topo]), topo.ndims )

    funcs = pointdata.values()
    funcs.append( coords )

    res = topo.elem_eval( funcs, ischeme='vtk' )

    vtkfile.unstructuredgrid( res.pop() )

    for name, data in zip( pointdata.keys(), res ):
      vtkfile.pointdataarray( name, data )

    res = topo.elem_mean( celldata.values(), coords, 'gauss1' )

    for name, data in zip( celldata.keys(), res ):
      vtkfile.celldataarray( name, data )

######## OLD PLOTTING INTERFACE ############

class Pylab( object ):
  'matplotlib figure'

  def __init__( self, title, name='graph{0:03x}' ):
    'constructor'

    import matplotlib
    matplotlib.use( 'Agg', warn=False )

    if '.' not in name.format(0):
      imgtype = getattr( prop, 'imagetype', 'png' )
      name += '.' + imgtype

    if isinstance( title, (list,tuple) ):
      self.title = numpy.array( title, dtype=object )
      self.shape = self.title.shape
      if self.title.ndim == 1:
        self.title = self.title[:,_]
      assert self.title.ndim == 2
    else:
      self.title = numpy.array( [[ title ]] )
      self.shape = ()
    self.name = name

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
      log.error( 'ERROR: plot failed:', msg or exc )
      return #True

    from matplotlib import pyplot
    dumpdir = prop.dumpdir
    n = len( os.listdir( dumpdir ) )
    imgpath = util.getpath( self.name )
    pyplot.savefig( imgpath, format=imgpath.split('.')[-1] )
    os.chmod( imgpath, 0644 )
    pyplot.close()
    log.path( os.path.basename(imgpath) )

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

  def add_mesh( self, coords, topology, deform=0, color=None, edgecolors='none', linewidth=1, xmargin=0, ymargin=0, aspect='equal', cbar='vertical', title=None, ischeme='gauss2', cscheme='contour3', clim=None, frame=True, colormap=None ):
    'plot mesh'
  
    log.context( 'plotting mesh' )

    assert topology.ndims == 2
    from matplotlib import pyplot, collections
    poly = []
    values = []
    ndims, = coords.shape
    assert ndims in (2,3)
    if color:
      assert color.ndim == 0
      color = function.Tuple([ color, coords.iweights(ndims=2) ])
    plotcoords = coords + deform
    for elem in topology:
      C = plotcoords( elem, cscheme )
      if ndims == 3:
        C = project3d( C )
        cx, cy = numpy.hstack( [ C, C[:,:1] ] )
        if ( (cx[1:]-cx[:-1]) * (cy[1:]+cy[:-1]) ).sum() > 0:
          continue
      if color:
        c, w = color( elem, ischeme )
        values.append( numeric.mean( c, weights=w, axis=0 ) if c.ndim > 0 else c )
      poly.append( C )
  
    if values:
      elements = collections.PolyCollection( poly, edgecolors=edgecolors, linewidth=linewidth, rasterized=True )
      elements.set_array( numpy.asarray(values) )
      if colormap is not None:
        elements.set_cmap( pyplot.cm.gray if colormap is False else colormap )
      if cbar:
        pyplot.colorbar( elements, ax=self._ax, orientation=cbar )
    else:
      elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=linewidth, rasterized=True )

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
    return elements
  
  def add_quiver( self, coords, topology, quiver, sample='uniform3', scale=None ):
    'quiver builder'
  
    xyuv = function.Concatenate( [ coords, quiver ] )
    XYUV = [ xyuv(elem,sample) for elem in log.ProgressBar( topology, title='plotting quiver' ) ]
    self.quiver( *numpy.concatenate( XYUV, 0 ).T, scale=scale )

  def add_graph( self, xfun, yfun, topology, sample='contour10', logx=False, logy=False, **kwargs ):
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
    xypairs = function.Tuple( [ function.Tuple(v) for v in zip( xfun, yfun, XYD ) ] )

    for elem in topology:
      for x, y, xyd in xypairs( elem, sample ):

        if y.ndim == 1 and y.shape[0] == 1:
          y = y[0]

        xyd[0].extend( x if x.ndim else [x] * y.size )
        xyd[0].append( numpy.nan )
        xyd[1].extend( y if y.ndim else [y] * x.size )
        xyd[1].append( numpy.nan )

    plotfun = self.loglog if logx and logy \
         else self.semilogx if logx \
         else self.semilogy if logy \
         else self.plot
    for x, y, d in XYD:
      kwargs.update(d)
      plotfun( x, y, **kwargs )

  def add_convplot( self, x, y, drop=0.8, shift=1.1, slope=True, **kwargs ): 
    """Convergence plot including slope triangle (below graph) for supplied y(x),
       drop  = distance graph & triangle,
       shift = distance triangle & text."""
    self.loglog( x, y, 'k.-', **kwargs )
    self.grid( True )
    if slope:
      if x[-1] < x[0]: # inverted order
        slx   = numpy.array( [x[-2], x[-2], x[-1], x[-2]] )
        sly   = numpy.array( [y[-2], y[-1], y[-1], y[-2]] )*drop
      if x[-1] > x[0]:
        slx   = numpy.array( [x[-1], x[-1], x[-2], x[-1]] )
        sly   = numpy.array( [y[-1], y[-2], y[-2], y[-1]] )/drop
      # slope = r'$%2.1f$' % (y[-2]*x[-1]/(x[-2]*y[-1]))
      slope = r'$%2.1f$' % (numpy.diff( numpy.log10( y[-2:] ) )/numpy.diff( numpy.log10( x[-2:] ) ))
      self.loglog( slx, sly, color='k', label='_nolegend_' )
      self.text( slx[-1]*shift, numpy.mean( sly[:2] )*drop, slope )

def project3d( C ):
  sqrt2 = numpy.sqrt( 2 )
  sqrt3 = numpy.sqrt( 3 )
  sqrt6 = numpy.sqrt( 6 )
  R = numpy.array( [[ sqrt3, 0, -sqrt3 ], [ 1, 2, 1 ], [ sqrt2, -sqrt2, sqrt2 ]] ) / sqrt6
  return numeric.transform( C, R[:,::2], axis=0 )

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
      contour = coords( elem, cscheme )
      polys[0].append( project3d( contour ).T )
      polys[1].append( contour[:2].T )
      polys[2].append( contour[1:].T )
      polys[3].append( contour[::2].T )
    for iplt, poly in enumerate( polys ):
      elements = collections.PolyCollection( poly, edgecolors='black', facecolors='none', linewidth=1, rasterized=True )
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
