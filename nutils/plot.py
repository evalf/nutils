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
from . import numpy, log, core, _
import os, warnings, sys


class BasePlot( object ):
  'base class for plotting objects'

  def __init__ ( self, name=None, ndigits=0, index=None, outdir=None ):
    'constructor'

    self.path = outdir or core.getoutdir()
    self.name = name
    self.index = index
    self.ndigits = ndigits

  def getpath( self, name, index, ext ):
    if name is None:
      name = self.name
    if index is None:
      index = self.index
    if self.ndigits and index is None:
      index = _getnextindex( self.path, name, ext )
    if index is not None:
      name += str(index).rjust( self.ndigits, '0' )
    name += '.' + ext
    log.path( name )
    return os.path.join( self.path, name )

  def __enter__( self ):
    'enter with block'

    assert self.name, 'name must be set to use as with-context'
    return self

  def __exit__( self, exc_type, exc_value, exc_tb ):
    'exit with block'

    self.save( self.name, self.index )
    self.close()

  def __del__( self ):
    try:
      self.close()
    except Exception as e:
      log.error( 'failed to close:', e )

  def save( self, name=None, index=None ):
    pass

  def close( self ):
    pass

class PyPlot( BasePlot ):
  'matplotlib figure'

  def __init__( self, name=None, imgtype=None, ndigits=3, index=None, **kwargs ):
    'constructor'

    import matplotlib
    matplotlib.use( 'Agg', warn=False )
    from matplotlib import pyplot

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )
    self.imgtype = imgtype or core.getprop( 'imagetype', 'png' )
    self._fig = pyplot.figure( **kwargs )
    self._pyplot = pyplot

  def __getattr__( self, attr ):
    return getattr( self._pyplot, attr )

  def close( self ):
    'close figure'

    if not self._fig:
      return # already closed
    try:
      self._pyplot.close( self._fig )
    except:
      log.warning( 'failed to close figure' )
    self._fig = None

  def save( self, name=None, index=None ):
    'save images'

    assert self._fig, 'figure is closed'
    for ext in self.imgtype.split( ',' ):
      self.savefig( self.getpath(name,index,ext) )

  def mesh( self, points, colors=None, edgecolors='k', edgewidth=None, triangulate='delaunay', setxylim=True, aspect='equal', cmap='jet' ):
    'plot elemtwise mesh'

    assert isinstance( points, numpy.ndarray ) and points.dtype == float
    if colors is not None:
      assert isinstance( colors, numpy.ndarray ) and colors.dtype == float
      assert points.shape[:-1] == colors.shape

    import matplotlib.tri

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
  
      all_epoints = []
      all_vertices = []
      all_colors = []
      edges = []
      npoints = 0
  
      for a, b in zip( numpy.concatenate([[0],split+1]), numpy.concatenate([split,[nans.size]]) ):
        np = b - a
        if np == 0:
          continue
        epoints = points[a:b]
        if colors is not None:
          ecolors = colors[a:b]
        if triangulate == 'delaunay':
          import scipy.spatial
          tri = scipy.spatial.Delaunay( epoints )
          vertices = tri.vertices
          connectivity = {}
          for e0, e1 in tri.convex_hull: # build inverted data structure for fast lookups
            connectivity.setdefault( e0, [] ).append( e1 )
            connectivity.setdefault( e1, [] ).append( e0 )
          p = tri.convex_hull[0,0] # first point (arbitrary)
          q = connectivity.pop(p)[0] # second point (arbitrary orientation)
          hull = [ p ]
          while connectivity:
            hull.append( q )
            q, r = connectivity.pop( hull[-1] )
            if q == hull[-2]:
              q = r
          assert q == p
          hull.append( q )
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
        all_epoints.append( epoints.T )
        all_vertices.append( vertices + npoints )
        if colors is not None:
          all_colors.append( ecolors )
        edges.append( epoints[hull] )
        npoints += np
  
      xy = numpy.concatenate( all_epoints, axis=1 )
      triangles = numpy.concatenate( all_vertices, axis=0 )
      if colors is not None:
        data = numpy.concatenate( all_colors )

    else:

      raise Exception( 'invalid points shape %r' % ( points.shape, ) )

    if colors is not None:
      finite = numpy.isfinite(data)
      if not finite.all():
        data = data[finite]
        xy = xy[:,finite]
        triangles = (finite.cumsum()-1)[ triangles[finite[triangles].all(axis=1)] ]
      trimesh = self.tripcolor( xy[0], xy[1], triangles, data, shading='gouraud', rasterized=True, cmap=cmap )

    if edges and edgecolors != 'none':
      if edgewidth is None:
        edgewidth = .1 if colors is not None else .5
      from matplotlib.collections import LineCollection
      linecol = LineCollection( edges, linewidths=(edgewidth,) )
      linecol.set_color( edgecolors )
      self.gca().add_collection( linecol )

    if aspect:
      self.gca().set_aspect( aspect )

    if setxylim:
      self.autoscale( enable=True, axis='both', tight=True )
    
    return linecol if colors is None \
      else trimesh if edgecolors == 'none' \
      else (trimesh, linecol)

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

  def slope_marker( self, x, y, slope=None, width=.2, xoffset=0, yoffset=.2, color='0.5' ):

    ax = self.gca()

    if slope is None:
      x_, x = x[-2:]
      y_, y = y[-2:]
      slope = numpy.log(y/y_) / numpy.log(x/x_)
      slope = numpy.round( slope * 100 ) / 100.

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

  @staticmethod
  def _tickspacing( axis, base ):
    from matplotlib import ticker
    loc = ticker.MultipleLocator( base=base )
    axis.set_major_locator(loc)

  def xtickspacing( self, base ):
    self._tickspacing( self.gca().xaxis, base )

  def ytickspacing( self, base ):
    self._tickspacing( self.gca().yaxis, base )

class DataFile( BasePlot ):
  """data file"""

  def __init__( self, name=None, index=None, ext='txt', ndigits=0 ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )
    self.ext = ext
    self.lines = []

  def save( self, name=None, index=None ):
    with open( self.getpath(name,index,self.ext), 'w' ) as fout:
      fout.writelines( self.lines )

  def printline( self, line ):
    self.lines.append( line+'\n' )

  def printlist( self, lst, delim=' ', start='', stop='' ):
    self.lines.append( start + delim.join( str(s) for s in lst ) + stop + '\n' )

class VTKFile( BasePlot ):
  'vtk file'

  _vtkdtypes = (
    ( numpy.dtype('u1'), 'unsigned_char' ),
    ( numpy.dtype('i1'), 'char' ),
    ( numpy.dtype('u2'), 'unsigned_short' ),
    ( numpy.dtype('i2'), 'short' ),
    ( numpy.dtype('u4'), 'unsigned_int' ), # also 'unsigned_long_int'
    ( numpy.dtype('i4'), 'int' ), # also 'unsigned_int'
    ( numpy.float32, 'float' ),
    ( numpy.float64, 'double' ),
  )

  def __init__( self, name=None, index=None, ndigits=0, ascii=False ):
    'constructor'

    BasePlot.__init__( self, name, ndigits=ndigits, index=index )

    if ascii is True or ascii == 'ascii':
      self.ascii = True
    elif ascii is False or ascii == 'binary':
      self.ascii = False
    else:
      raise ValueError( 'unexpected value for argument `ascii`: {!r}' )

    self._mesh = None
    self._dataarrays = { 'points': [], 'cells': [] }

  def _getvtkdtype( self, data ):
    for dtype, vtkdtype in self._vtkdtypes:
      if dtype == data.dtype:
        return vtkdtype
    raise ValueError( 'No matching VTK dtype for {}.'.format( data.dtype ) )

  def _writearray( self, output, array ):
    if self.ascii:
      array.tofile( output, sep=' ' )
      output.write( b'\n' )
    else:
      if sys.byteorder != 'big':
        array = array.byteswap()
      array.tofile( output )

  def save( self, name=None, index=None ):
    assert self._mesh is not None, 'Grid not specified'
    with open( self.getpath(name,index,'vtk'), 'wb' ) as vtk:
      if sys.version_info.major == 2:
        write = vtk.write
      else:
        write = lambda s: vtk.write( s.encode( 'ascii' ) )

      # header
      write( '# vtk DataFile Version 3.0\n' )
      write( 'vtk output\n' )
      if self.ascii:
        write( 'ASCII\n' )
      else:
        write( 'BINARY\n' )

      # mesh
      if self._mesh[0] == 'unstructured':
        meshtype, ndims, npoints, ncells, points, cells, celltypes = self._mesh
        write( 'DATASET UNSTRUCTURED_GRID\n' )
        write( 'POINTS {} {}\n'.format( npoints, self._getvtkdtype( points ) ) )
        self._writearray( vtk, points )
        write( 'CELLS {} {}\n'.format( ncells, len( cells ) ) )
        self._writearray( vtk, cells )
        write( 'CELL_TYPES {}\n'.format( ncells ) )
        self._writearray( vtk, celltypes )
      elif self._mesh[0] == 'rectilinear':
        meshtype, ndims, npoints, ncells, coords = self._mesh
        write( 'DATASET RECTILINEAR_GRID\n' )
        write( 'DIMENSIONS {} {} {}\n'.format( *map( len, coords ) ) )
        for label, array in zip( 'XYZ', coords ):
          write( '{}_COORDINATES {} {}\n'.format( label, len(array), self._getvtkdtype( array ) ) )
          self._writearray( array )
      else:
        raise NotImplementedError

      # data
      for location in 'points', 'cells':
        if not self._dataarrays[location]:
          continue
        if location == 'points':
          write( 'POINT_DATA {}\n'.format( npoints ) )
        elif location == 'cells':
          write( 'CELL_DATA {}\n'.format( ncells ) )
        for name, vector, ncomponents, data in self._dataarrays[location]:
          vtkdtype = self._getvtkdtype( data )
          if vector:
            write( 'VECTORS {} {}\n'.format( name, vtkdtype ) )
          else:
            write( 'SCALARS {} {} {}\n'.format( name, vtkdtype, ncomponents ) )
            write( 'LOOKUP_TABLE default\n' )
          self._writearray( vtk, data )

  def rectilineargrid( self, coords ):
    """set rectilinear grid"""
    assert 1 <= len(coords) <= 3, 'Exptected a list of 1, 2 or 3 coordinate arrays, got {} instead'.format( len(coords) )
    ndims = len(coords)
    npoints = 1
    ncells = 1
    coords = list( coords )
    for i in range( ndims ):
      npoints *= len( coords[i] )
      ncells *= 1 - len( coords[i] )
      assert len( coords[i].shape ) == 1, 'Expected a one-dimensional array for coordinate {}, got an array with shape {!r}'.format( i, coords[i].shape )
    for i in range( ndims, 3 ):
      coords.append( numpy.array( [0], dtype=numpy.int32 ) )
    self._mesh = 'rectilinear', ndims, npoints, ncells, coords

  def unstructuredgrid( self, points, npars=None ):
    """set unstructured grid"""

    cellpoints = _nansplit( points )
    npoints = sum( map( len, cellpoints ) )
    ncells = len( cellpoints )

    points = numpy.zeros( [npoints, 3], dtype=points.dtype )
    cells = numpy.empty( [npoints+ncells], dtype=numpy.int32 )
    celltypes = numpy.empty( [ncells], dtype=numpy.int32 )

    j = 0
    for i, pts in enumerate( cellpoints ):

      np, ndims = pts.shape
      if not npars:
        npars = ndims

      if np == 2:
        celltype = 3
      elif np == 3:
        celltype = 5
      elif np == 4:
        if npars == 2:
          celltype = 9
        elif npars == 3:
          celltype = 10
      elif np == 8:
        celltype = 11 # TODO hexahedron for not rectilinear NOTE ordering changes!

      if not celltype:
        raise Exception( 'not sure what to do with cells with ndims={} and npoints={}'.format(ndims,np) )

      celltypes[i] = celltype
      cells[i+j] = np
      for k, p in enumerate( pts ):
        cells[i+j+k+1] = j+k
        points[j+k,:ndims] = p
      j += np

    self._mesh = 'unstructured', ndims, npoints, ncells, points.ravel(), cells, celltypes

  def celldataarray( self, name, data, vector=None ):
    'add cell array'
    self._adddataarray( name, data, 'cells', vector )

  def pointdataarray( self, name, data, vector=None ):
    'add cell array'
    self._adddataarray( name, data, 'points', vector )

  def _adddataarray( self, name, data, location, vector ):
    assert self._mesh is not None, 'Grid not specified'
    ndims, npoints, ncells = self._mesh[1:4]
    ncomponents = data.shape[1] if len( data.shape ) == 2 else 1
    if vector is None:
      vector = len( data.shape ) == 2

    if location == 'points':
      if npoints != data.shape[0]:
        data = _nanfilter( data )
      assert npoints == data.shape[0], 'Point data array should have {} entries'.format(npoints)
    elif location == 'cells':
      assert ncells == data.shape[0], 'Cell data array should have {} entries'.format(ncells)
    assert len( data.shape ) <= 2, 'Data array should have at most 2 axes: {} and components (optional)'.format(location)
    if vector:
      assert ncomponents == ndims, 'Data array should have {} components per entry'.format(ndims)
      if ndims != 3:
        data = numpy.concatenate( [data, numpy.zeros( [data.shape[0], 3-ndims], dtype=data.dtype ) ], axis=1 )
      ncomponents = 3

    self._dataarrays[location].append(( name, vector, ncomponents, data.ravel() ))


## AUXILIARY FUNCTIONS

def writevtu( name, topo, coords, pointdata={}, celldata={}, ascii=False, superelements=False, maxrefine=3, ndigits=0, ischeme='gauss1', **kwargs ):
  'write vtu from coords function'

  from . import element, topology

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

def _nansplit( data ):
  n, = numpy.where( numpy.isnan( data.reshape( data.shape[0], -1 ) ).any( axis=1 ) )
  N = numpy.concatenate( [ [-1], n, [data.shape[0]] ] )
  return [ data[a:b] for a, b in zip( N[:-1]+1, N[1:] ) ]

def _nanfilter( data ):
  return data[~numpy.isnan( data.reshape( data.shape[0], -1 ) ).all( axis=1 )]

def _getnextindex( path, name, ext ):
  index = 0
  for filename in os.listdir( path ):
    if filename.startswith(name) and filename.endswith('.'+ext):
      num = filename[len(name):-len(ext)-1]
      if num.isdigit():
        index = max( index, int(num)+1 )
  return index

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
