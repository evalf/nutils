# -*- coding: utf8 -*-
#
# Module PLOT
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The plot module aims to provide a consistent interface to various plotting
backends. At this point `matplotlib <http://matplotlib.org/>`_ and `vtk
<http://vtk.org>`_ are supported.
"""

from __future__ import print_function, division
from . import topology, util, numpy, function, element, log, core, numeric, debug, _
import os, warnings
try:
  from scipy import spatial # for def mesh; import cannot be postponed apparently
except ImportError:
  pass

def _nansplit( data ):
  n, = numpy.where( numpy.isnan( data.reshape( data.shape[0], -1 ) ).any( axis=1 ) )
  N = numpy.concatenate( [ [-1], n, [data.shape[0]] ] )
  return [ data[a:b] for a, b in zip( N[:-1]+1, N[1:] ) ]

def _nanfilter( data ):
  return data[~numpy.isnan( data.reshape( data.shape[0], -1 ) ).all( axis=1 )]

class BasePlot( object ):
  'base class for plotting objects'

  def __init__ ( self, name, ndigits=0, index=None ):

    self.path = core.getprop( 'dumpdir' )

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

    return self

  def __exit__( self, *exc_info ):
    'exit with block'

    exc_type, exc_value, exc_tb = exc_info
    if exc_type == KeyboardInterrupt:
      pass
    elif exc_type:
      log.stack( exc_info )
    else:
      if self.names:
        for name in self.names:
          self.save( name )
        log.path( ', '.join( self.names ) )
      return True
    return False

  def save ( self, name ):
    return

class PyPlot( BasePlot ):
  'matplotlib figure'

  def __init__( self, name, imgtype=None, ndigits=3, index=None, **kwargs ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    import matplotlib

    matplotlib.use( 'Agg', warn=False )

    from matplotlib import pyplot

    imgtype = core.getprop( 'imagetype', 'png' ) if imgtype is None else imgtype
    self.names = [ self.name + '.' + ext for ext in imgtype.split(',') ]

    self.__dict__.update( pyplot.__dict__ )

    self._fig = self.figure( **kwargs )
    #self._fig.patch.set_alpha( 0 )

  def __exit__( self, *exc_info ):
    'exit with block'

    BasePlot.__exit__( self, *exc_info )
    try:
      self.close( self._fig )
    except:
      log.warning( 'failed to close figure' )

  def save( self, name ):
    'save images'

    self.savefig( os.path.join( self.path, name ) )
    #self.close()

  @staticmethod
  def _trimesh_class():
    'backport of TriMesh (function prevents unneccecary loading)'

    from matplotlib.collections import Collection
    from matplotlib.artist import allow_rasterization
    
    class TriMesh( Collection ):

      def __init__(self, xy, tri, **kwargs):
        Collection.__init__(self, **kwargs)
        self.xy = xy
        self.tri = tri
        self._facecolors = numpy.zeros([numpy.max(tri)+1,4]) # fully transparent

      def get_paths( self ):
        return []
    
      @allow_rasterization
      def draw(self, renderer):
        if not self.get_visible():
          return
        renderer.open_group(self.__class__.__name__)
        transform = self.get_transform()
        verts = self.xy.T[self.tri]
        self.update_scalarmappable()
        colors = self._facecolors[self.tri]
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])
        renderer.draw_gouraud_triangles(gc, verts, colors, transform.frozen())
        gc.restore()
        renderer.close_group(self.__class__.__name__)

    return TriMesh

  def mesh( self, points, colors=None, edgecolors='k', edgewidth=None, triangulate='delaunay', setxylim=True, **kwargs ):
    'plot elemtwise mesh'

    assert isinstance( points, numpy.ndarray ) and points.dtype == float
    if colors is not  None:
      assert isinstance( colors, numpy.ndarray ) and colors.dtype == float
      assert points.shape[:-1] == colors.shape

    if points.ndim == 3: # gridded data: nxpoints x nypoints x ndims

      assert points.shape[-1] == 2
      assert colors is not None
      data = colors.ravel()
      xy = points.reshape( -1, 2 ).T
      ind = numpy.arange( xy.shape[1] ).reshape( points.shape[:-1] )
      vert1 = numpy.array([ ind[:-1,:-1].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
      vert2 = numpy.array([ ind[1:,1:].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
      triangles = numpy.concatenate( [vert1,vert2], axis=0 )
      edges = None

    elif points.ndim == 2: # mesh: npoints x ndims

      ndims = points.shape[1]
      if ndims == 1:
        self.plot( points[:,0], colors )
        return
      else:
        assert ndims == 2, 'unsupported: ndims=%s' % ndims

      nans = numpy.isnan( points ).all( axis=1 )
      split, = numpy.where( nans )
      if colors is not None:
        assert numpy.isnan( colors[split] ).all()
  
      P = []
      N = []
      C = []
      E = []
      npoints = 0
  
      for a, b in zip( numpy.concatenate([[0],split+1]), numpy.concatenate([split,[nans.size]]) ):
        np = b - a
        if np == 0:
          continue
        epoints = points[a:b]
        if colors is not None:
          ecolors = colors[a:b]
        if triangulate == 'delaunay':
          tri = spatial.Delaunay( epoints )
          vertices = tri.vertices
          e0 = [ edge[0] for edge in tri.convex_hull ]
          e1 = [ edge[1] for edge in tri.convex_hull ]
          last = e1.pop()
          hull = [ e0.pop(), last ]
          while e0:
            try:
              index = e0.index( last )
              last = e1[index]
            except ValueError:
              index = e1.index( last )
              last = e0[index]
            e0.pop( index )
            e1.pop( index )
            hull.append( last )
          assert hull[0] == hull[-1]
        elif triangulate == 'bezier':
          nquad = int( numpy.sqrt(np) + .5 )
          ntri = int( numpy.sqrt((2*np)+.25) )
          if nquad**2 == np:
            ind = numpy.arange(np).reshape(nquad,nquad)
            vert1 = numpy.array([ ind[:-1,:-1].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
            vert2 = numpy.array([ ind[1:,1:].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
            vertices = numpy.concatenate( [vert1,vert2], axis=0 )
            hull = numpy.concatenate([ ind[:,0], ind[-1,1:], ind[-2::-1,-1], ind[0,-2::-1] ])
          elif ntri * (ntri+1) == 2 * np:
            vert1 = [ ((2*ntri-i+1)*i)//2+numpy.array([j,j+1,j+ntri-i]) for i in range(ntri-1) for j in range(ntri-i-1) ]
            vert2 = [ ((2*ntri-i+1)*i)//2+numpy.array([j+1,j+ntri-i+1,j+ntri-i]) for i in range(ntri-1) for j in range(ntri-i-2) ]
            vertices = numpy.concatenate( [vert1,vert2], axis=0 )
            hull = numpy.concatenate([ numpy.arange(ntri), numpy.arange(ntri-1,0,-1).cumsum()+ntri-1, numpy.arange(ntri+1,2,-1).cumsum()[::-1]-ntri-1 ])
          else:
            raise Exception( 'cannot match points to a bezier scheme' )
        else:
          raise Exception( 'unknown triangulation method %r' % triangulate )
        P.append( epoints.T )
        N.append( vertices + npoints )
        if colors is not None:
          C.append( ecolors )
        E.append( epoints[hull] )
        npoints += np
  
      xy = numpy.concatenate( P, axis=1 )
      triangles = numpy.concatenate( N, axis=0 )
      if colors is not None:
        data = numpy.concatenate( C )
      edges = E

    else:

      raise Exception( 'invalid points shape %r' % ( points.shape, ) )
  
    TriMesh = self._trimesh_class()
    polycol = TriMesh( xy, triangles, rasterized=True, **kwargs )
    if colors is not None:
      polycol.set_array( data )

    if edges and edgecolors != 'none':
      from matplotlib.collections import LineCollection
      linecol = LineCollection( edges, linewidths=(edgewidth,) if edgewidth is not None else None )
      linecol.set_color( edgecolors )
      self.gca().add_collection( linecol )

    self.gca().add_collection( polycol )
    self.sci( polycol )
    
    if setxylim:
      xmin, ymin = numpy.min( xy, axis=1 )
      xmax, ymax = numpy.max( xy, axis=1 )
      self.xlim( xmin, xmax )
      self.ylim( ymin, ymax )
    
    if edgecolors != 'none':
      return polycol, linecol

    return polycol

  def polycol( self, verts, facecolors='none', **kwargs ):
    'add polycollection'
  
    from matplotlib import collections
    assert verts.ndim == 2 and verts.shape[1] == 2
    verts = _nansplit( verts )
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

  def slope_marker( self, x, y, slope, width=.2, xoffset=0, yoffset=.2, color='0.3' ):

    ax = self.gca()

    if float(slope) > 0:
      width = -width

    xscale = ax.get_xscale()
    xmin, xmax = ax.get_xlim()
    if xscale == 'linear':
      W = ( xmax - xmin ) * width
      x0 = x - W
      xc = x - .5 * W
    elif xscale == 'log':
      W = numpy.log10( xmax / xmin ) * width
      x0 = x * 10**-W
      xc = x * 10**(-.5*W)
    else:
      raise Exception( 'unknown x-axis scale %r' % xscale )

    yscale = ax.get_yscale()
    H = W * float(slope)
    if yscale == 'linear':
      y0 = y - H
      yc = y - .5 * H
    elif yscale == 'log':
      y0 = y * 10**-H
      yc = y * 10**(-.5*H)
    else:
      raise Exception( 'unknown x-axis scale %r' % xscale )

    from matplotlib import transforms
    dpi = self.gcf().dpi_scale_trans
    shifttrans = ax.transData + transforms.ScaledTranslation( xoffset, numpy.sign(H) * yoffset, dpi )

    triangle = self.Polygon( [ (x0,y0), (x,y), (xc,y) ], closed=False, ec=color, fc='none', transform=shifttrans )
    ax.add_patch( triangle )

    self.text( xc, yc, str(slope), color=color,
      horizontalalignment = 'right' if W > 0 else 'left',
      verticalalignment = 'top' if H < 0 else 'bottom',
      transform = shifttrans + transforms.ScaledTranslation( numpy.sign(W) * -.05, numpy.sign(H) * .05, dpi ) )

  def slope_triangle( self, x, y, fillcolor='0.9', edgecolor='k', xoffset=0, yoffset=0.1, slopefmt='{0:.1f}' ):
    '''Draw slope triangle for supplied y(x)
       - x, y: coordinates
       - xoffset, yoffset: distance graph & triangle (points)
       - fillcolor, edgecolor: triangle style
       - slopefmt: format string for slope number'''

    i, j = (-2,-1) if x[-1] < x[-2] else (-1,-2) # x[i] > x[j]
    if not all(numpy.isfinite(x[-2:])) or not all(numpy.isfinite(y[-2:])):
      log.warning( 'Not plotting slope triangle for +/-inf or nan values' )
      return

    from matplotlib import transforms
    shifttrans = self.gca().transData \
               + transforms.ScaledTranslation( xoffset, -yoffset, self.gcf().dpi_scale_trans )
    xscale, yscale = self.gca().get_xscale(), self.gca().get_yscale()

    # delta() checks if either axis is log or lin scaled
    delta = lambda a, b, scale: numpy.log10(float(a)/b) if scale=='log' else float(a-b) if scale=='linear' else None
    slope = delta( y[-2], y[-1], yscale ) / delta( x[-2], x[-1], xscale )
    if slope in (numpy.nan, numpy.inf, -numpy.inf):
      warnings.warn( 'Cannot draw slope triangle with slope: %s, drawing nothing' % str( slope ) )
      return slope

    # handle positive and negative slopes correctly
    xtup, ytup = ((x[i],x[j],x[i]), (y[j],y[j],y[i])) if slope > 0 else ((x[j],x[j],x[i]), (y[i],y[j],y[i]))
    a, b = (2/3., 1/3.) if slope > 0 else (1/3., 2/3.)
    xval = a*x[i]+b*x[j] if xscale=='linear' else x[i]**a * x[j]**b
    yval = b*y[i]+a*y[j] if yscale=='linear' else y[i]**b * y[j]**a

    self.fill( xtup, ytup,
      color=fillcolor,
      edgecolor=edgecolor,
      transform=shifttrans )

    self.text( xval, yval,
      slopefmt.format(slope),
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
    if not isinstance( A, numpy.ndarray ):
      A = A.toarray()
    if A.size < 2: # trivial case of 1x1 matrix
      A = A.reshape( 1, 1 )
    else:
      A = numpy.log10( numpy.abs( A ) )
      B = numpy.isinf( A ) | numpy.isnan( A ) # what needs replacement
      A[B] = ~B if numpy.all( B ) else numpy.amin( A[~B] ) - 1.
    self.pcolormesh( A, **kwargs )
    self.colorbar()
    self.ylim( self.ylim()[-1::-1] ) # invert y axis: equiv to MATLAB axis ij
    self.xlabel( r'$j$' )
    self.ylabel( r'$i$' )
    self.title( r'$^{10}\log a_{ij}$' )
    self.axis( 'tight' )

  def image( self, image, location=[0,0], scale=1, alpha=1.0 ):
    image = image.resize( [int( scale*size ) for size in image.size ])
    dpi   = self._fig.get_dpi()
    self._fig.figimage( numpy.array( image ).astype(float)/255, location[0]*dpi, location[1]*dpi, zorder=10 ).set_alpha(alpha)
    
class DataFile( BasePlot ):
  """data file"""

  def __init__( self, name, index=None, ndigits=0, mode='w' ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    self.names = [name]
    self.fout  = open( os.path.join(self.path,name), mode )

  def save( self, name ):
    self.fout.close()

  def printline( self, line ):
    print(line,file=self.fout)

  def printlist( self, lst, delim=' ', start='', stop='' ):
    print(start + delim.join(str(s) for s in lst)  + stop,file=self.fout) 

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

  def unstructuredgrid( self, points, npars=None ):
    """add unstructured grid"""

    points = _nansplit( points )
    #assert isinstance( points, (list,tuple,numpy.ndarray) ), 'Expected list of point arrays'

    import vtk

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints( sum(pts.shape[0] for pts in points) )

    cnt = 0
    for pts in points:

      np, ndims = pts.shape
      if not npars:
        npars = ndims

      vtkelem   = None

      if np == 2:
        vtkelem = vtk.vtkLine()
      elif np == 3:
        vtkelem = vtk.vtkTriangle()
      elif np == 4:  
        if npars == 2:
          vtkelem = vtk.vtkQuad()
        elif npars == 3:
          vtkelem = vtk.vtkTetra()
      elif np == 8:
        vtkelem = vtk.vtkVoxel() # TODO hexahedron for not rectilinear NOTE ordering changes!

      if not vtkelem:
        raise Exception( 'not sure what to do with cells with ndims=%d and npoints=%d' % (ndims,np) )

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

    if npoints != data.shape[0]:
      data = _nanfilter( data )

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

def writevtu( name, topo, coords, pointdata={}, celldata={}, ascii=False, superelements=False, maxrefine=3, ndigits=0, ischeme='gauss1', **kwargs ):
  'write vtu from coords function'

  with VTKFile( name, ascii=ascii, ndigits=ndigits ) as vtkfile:

    if not superelements:
      topo = topo.simplex
    else:
      topo = topology.Topology( filter(None,[elem if not isinstance(elem,element.TrimmedElement) else elem.elem for elem in topo]) )

    points = topo.elem_eval( coords, ischeme='vtk', separate=True )
    vtkfile.unstructuredgrid( points, npars=topo.ndims )

    if pointdata:  
      keys, values = zip( *pointdata.items() )
      arrays = topo.elem_eval( values, ischeme='vtk', separate=False )
      for key, array in zip( keys, arrays ):
        vtkfile.pointdataarray( key, array )

    if celldata:  
      keys, values = zip( *celldata.items() )
      arrays = topo.elem_mean( values, coords=coords, ischeme=ischeme )
      for key, array in zip( keys, arrays ):
        vtkfile.celldataarray( key, array )

######## OLD PLOTTING INTERFACE ############

class Pylab( object ):
  'matplotlib figure'

  def __init__( self, title, name='graph{0:03x}' ):
    'constructor'

    import matplotlib
    matplotlib.use( 'Agg', warn=False )

    if '.' not in name.format(0):
      imgtype = core.getprop( 'imagetype', 'png' )
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
    dumpdir = core.getprop( 'dumpdir' )
    n = len( os.listdir( dumpdir ) )
    imgpath = util.getpath( self.name )
    pyplot.savefig( imgpath, format=imgpath.split('.')[-1] )
    os.chmod( imgpath, 0o644 )
    pyplot.close()
    log.path( os.path.basename(imgpath) )

class PylabAxis( object ):
  'matplotlib axis augmented with nutils-specific functions'

  def __init__( self, ax, title ):
    'constructor'

    if title:
      ax.set_title( title )
    self._ax = ax

  def __getattr__( self, attr ):
    'forward getattr to axis'

    return getattr( self._ax, attr )

  @log.title
  def add_mesh( self, coords, topology, deform=0, color=None, edgecolors='none', linewidth=1, xmargin=0, ymargin=0, aspect='equal', cbar='vertical', ischeme='gauss2', cscheme='contour3', clim=None, frame=True, colormap=None ):
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

    self.set_frame_on( frame )
    return elements
  
  def add_quiver( self, coords, topology, quiver, sample='uniform3', scale=None ):
    'quiver builder'
  
    xyuv = function.Concatenate( [ coords, quiver ] )
    XYUV = [ xyuv( elem, sample ) for elem in log.iter( 'elem', topology ) ]
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

    special_args = zip( *[ zip( [key]*nfun, val ) for (key,val) in kwargs.items() if isinstance(val,list) and len(val) == nfun ] )
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
    raise Exception( 'need 2D or 3D coordinates' )
  pyplot.show()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
