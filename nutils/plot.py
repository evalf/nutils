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
from . import numpy, log, core, cache, numeric, _
import os, warnings, sys, subprocess


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

    if not exc_type:
      self.save( self.name, self.index )
    try:
      self.close()
    except Exception as e:
      log.error( 'failed to close:', e )

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

  def __enter__( self ):
    'enter with block'

    # make this figure active
    self._pyplot.figure(self._fig.number)

    return super( PyPlot, self ).__enter__()

  def __getattr__( self, attr ):
    pyplot = self.__dict__['_pyplot'] # avoid recursion
    return getattr( pyplot, attr )

  def close( self ):
    'close figure'

    if not self._fig:
      return # already closed
    try:
      self._pyplot.close( self._fig )
    except Exception as e:
      log.warning( 'failed to close figure: {}'.format(e) )
    self._fig = None

  def save( self, name=None, index=None ):
    'save images'

    assert self._fig, 'figure is closed'
    for ext in self.imgtype.split( ',' ):
      self.savefig( self.getpath(name,index,ext) )

  def segments( self, points, color='black', **kwargs ):
    'plot line'

    segments = numpy.concatenate( [ numpy.array([xy[:-1],xy[1:]]).swapaxes(0,1) for xy in points ], axis=0 )
    from matplotlib.collections import LineCollection
    lc = LineCollection( segments, **kwargs )
    ax = self.gca()
    ax.add_collection( lc )
    if isinstance( color, str ):
      lc.set_color( color )
    else:
      array = numpy.concatenate( [ .5 * ( v[:-1] + v[1:] ) for v in color ], axis=0 )
      lc.set_array( array )
      self.sci( lc )
    return lc

  def mesh( self, points, values=None, edgecolors='k', edgewidth=.1, mergetol=0, setxylim=True, aspect='equal', tight=True, **kwargs ):
    'plot elemtwise mesh'

    kwargs.pop( 'triangulate', None ) # ignore deprecated argument

    if not isinstance( points, numpy.ndarray ) and points[0].shape[1] == 1: # line plot
      if values is not None:
        self.segments( [ numpy.concatenate( [x,y[:,_]], axis=1 ) for x, y in zip( points, values ) ], values )
      return

    if isinstance( points, numpy.ndarray ): # bulk data
      assert points.shape[-1] == 2
      import matplotlib.tri
      tri = matplotlib.tri.Triangulation( *points.reshape(-1,2).T )
      edgecolors = 'none'
      if values is not None:
        values = values.ravel()
    else: # mesh data
      tri, edges = triangulate( points, mergetol )
      if values is not None:
        values = numpy.concatenate( values, axis=0 )

    if values is not None:
      self.tripcolor( tri, values, shading='gouraud', **kwargs )
    if edgecolors != 'none':
      self.segments( edges, linewidth=edgewidth )
    if aspect:
      ( self.aspect if tight else self.axis )( aspect )
    if setxylim:
      self.autoscale( enable=True, axis='both', tight=True )

    return tri

  def aspect( self, *args, **kwargs ):
    self.gca().set_aspect( *args, **kwargs )

  def tripcolor( self, *args, **kwargs ):
    import matplotlib.tri
    assert len(args) >= 2
    if isinstance( args[0], numpy.ndarray ) and isinstance( args[1], numpy.ndarray ):
      # args = x, y[, triangles[, mask]], values
      tri = matplotlib.tri.Triangulation( *args[:-1] )
      values = args[-1]
    else:
      assert len(args) == 2
      tri, values = args
      if not isinstance( tri, matplotlib.tri.Triangulation ):
        tri, edges = triangulate( tri, mergetol )
      if not isinstance( values, numpy.ndarray ):
        values = numpy.concatenate( values, axis=0 )
    assert len(tri.x) == len(values)
    mask = ~numpy.isfinite( values )
    if mask.any():
      tri = matplotlib.tri.Triangulation( tri.x, tri.y, tri.triangles, mask[tri.triangles].any(axis=1) )
    return self._pyplot.tripcolor( tri, values, **kwargs )

  def tricontour( self, tri, values, every=None, levels=None, mergetol=0, **kwargs ):
    assert not every or levels is None, '"every" and "levels" arguments are mutually exclusive'
    import matplotlib.tri
    if not isinstance( tri, matplotlib.tri.Triangulation ):
      tri, edges = triangulate( tri, mergetol )
    if not isinstance( values, numpy.ndarray ):
      values = numpy.concatenate( values, axis=0 )
    assert len(tri.x) == len(values)
    if every:
      levels = numpy.arange( int(min(values)/every), int(max(values)/every)+1 ) * every
    return self._pyplot.tricontour( tri, values, levels=levels, **kwargs )

  def streamplot( self, tri, velo, spacing, bbox=None, mergetol=1e-5, linewidth=None, color=None, **kwargs ):
    if isinstance( spacing, numpy.ndarray ):
      # compatibility with original streamplot function definition
      x = tri
      y = velo
      u = spacing
      v = bbox
    else:
      import matplotlib.tri
      if not isinstance( tri, matplotlib.tri.Triangulation ):
        tri, edges = triangulate( tri, mergetol=mergetol )
      if not isinstance( velo, numpy.ndarray ):
        velo = numpy.concatenate( velo, axis=0 )
      assert len(tri.x) == len(velo)
      if bbox is None:
        xlim = min(tri.x), max(tri.x)
        ylim = min(tri.y), max(tri.y)
      else:
        xlim, ylim = bbox
      nx = int( ( xlim[-1] - xlim[0] ) / spacing )
      ny = int( ( ylim[-1] - ylim[0] ) / spacing )
      assert nx > 0 and ny > 0
      x = .5 * (xlim[0]+xlim[-1]) + ( numpy.arange(nx) - (nx-1)/2 ) * spacing
      y = .5 * (ylim[0]+ylim[-1]) + ( numpy.arange(ny) - (ny-1)/2 ) * spacing
      uv = interpolate( tri, numeric.meshgrid(x,y).T, velo, mergetol=mergetol )
      u = uv[...,0]
      v = uv[...,1]
    assert isinstance( x, numpy.ndarray ) and x.ndim == 1
    assert isinstance( y, numpy.ndarray ) and y.ndim == 1
    assert isinstance( u, numpy.ndarray ) and u.shape == (len(y),len(x))
    assert isinstance( v, numpy.ndarray ) and v.shape == (len(y),len(x))
    if linewidth is not None and linewidth < 0: # convention: negative linewidth is scaled with velocity magnitude
      linewidth = -linewidth * numpy.sqrt( u**2 + v**2 )
    if color is None: # default: color mapped to velocity magnitude
      color = numpy.sqrt( u**2 + v**2 )
    return self._pyplot.streamplot( x, y, u, v, linewidth=linewidth, color=color, **kwargs )

  def polycol( self, verts, facecolors='none', **kwargs ):
    'add polycollection'
  
    from matplotlib import collections
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
    'slope marker'

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

  def vectors( self, xy, uv, stems=True, **kwargs ):
    if not stems:
      uv = uv / numpy.linalg.norm( uv, axis=1 )[:,_]
      kwargs['width'] = 1e-3
      kwargs['headwidth'] = 3e3
      kwargs['headlength'] = 5e3
      kwargs['headaxislength'] = 2e3
    self.quiver( xy[:,0], xy[:,1], uv[:,0], uv[:,1], angles='xy', **kwargs )

class PyPlotVideo( PyPlot ):
  '''matplotlib based video generator

  Video generator based on matplotlib figures.  Follows the same syntax as
  `PyPlot`.

  Parameters
  ----------

  clearfigure: bool, default: True
    If True clears the matplotlib figure after writing each frame.

  framerate: int, float, default: 24
    Framerate in frames per second of the generated video.

  videotype: str, default: 'webm' unless overriden by property ``videotype``
    Video type of the generated video.  Note that not every video type supports
    playback before the video has been finalized, i.e. before ``close`` has
    been called.

  Nutils properties
  -----------------

  videotype: see parameter with the same name

  videoencoder: str, default: 'ffmpeg'
    Name or path of the video encoder.  The video encoder should take the same
    arguments as 'ffmpeg'.

  Examples
  --------

  Using a ``with``-statement:

    video = PyPlotVideo('video')
    for timestep in timesteps:
      ...
      with video:
        video.plot(...)
        video.title('frame {:04d}'.format(video.frame))
    video.close()

  Using ``saveframe``:

    video = PyPlotVideo('video')
    for timestep in timesteps:
      ...
      video.plot(...)
      video.title('frame {:04d}'.format(video.frame))
      video.saveframe()
    video.close()
  '''

  def __init__(self, name, videotype=None, clearfigure=True, framerate=24):
    'constructor'

    PyPlot.__init__( self, ndigits=0 )

    self.frame = 0
    self._clearfigure = clearfigure
    if videotype is None:
      videotype = core.getprop( 'videotype', 'webm' )
    self._encoder = subprocess.Popen([
        core.getprop( 'videoencoder', 'ffmpeg' ),
        '-loglevel', 'quiet',
        '-probesize', '1k',
        '-analyzeduration', '1',
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-r', str(framerate),
        '-i', '-',
        '-crf', '10', # constant quality (4-63, lower means better)
        '-b:v', '10M', # maximum allowed bitrate
        self.getpath( name, None, videotype ),
      ], stdin=subprocess.PIPE )

  def __enter__( self ):
    'enter with block'

    # make this figure active
    self._pyplot.figure(self._fig.number)

    return self

  def __exit__( self, exc_type, exc_value, exc_tb ):
    'exit with block'

    if not exc_type:
      self.saveframe()

  def saveframe( self ):
    'add a video frame'

    assert self._fig, 'video is closed'
    self.savefig( self._encoder.stdin, format='png' )
    if self._clearfigure:
      self._fig.clear()

  def close( self ):
    'finalize video'

    if not self._encoder:
      return # already closed
    self._encoder.stdin.close()
    self._encoder = None
    PyPlot.close( self )

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
        for name, data in self._dataarrays[location]:
          vtkdtype = self._getvtkdtype( data )
          if data.ndim==1:
            write( 'SCALARS {} {} {}\n'.format( name, vtkdtype, 1 ) )
            write( 'LOOKUP_TABLE default\n' )
          elif data.ndim==2:
            write( 'VECTORS {} {}\n'.format( name, vtkdtype ) )
          elif data.ndim==3:
            write( 'TENSORS {} {}\n'.format( name, vtkdtype ) )
          else:
            raise Exception('Unsupported data dimension')

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

  def unstructuredgrid( self, cellpoints, npars=None ):
    """set unstructured grid"""

    points = numpy.concatenate( cellpoints, axis=0 )
    npoints, ndims = points.shape

    if ndims == 2:
      points = numpy.concatenate( [ points, numpy.zeros_like(points[:,:1]) ], axis=1 )
    assert points.shape[1] == 3

    if npars is None:
      npars = ndims
    assert npars in (2,3)

    celltypemap = { 2: 3, 3: 5, 4: 9 if npars == 2 else 10, 5: 14, 8: 11 }

    ncells = len( cellpoints )
    cells = numpy.empty( npoints+ncells, dtype=numpy.int32 )
    celltypes = numpy.empty( ncells, dtype=numpy.int32 )

    j = 0
    for i, pts in enumerate( cellpoints ):
      np = len(pts)
      celltypes[i] = celltypemap[np]
      cells[i+j] = np
      cells[i+j+1:i+j+1+np] = j + numpy.arange(np)
      j += np

    self._mesh = 'unstructured', ndims, npoints, ncells, points.ravel(), cells, celltypes

  def celldataarray( self, name, data ):
    'add cell array'
    self._adddataarray( name, data, 'cells' )

  def pointdataarray( self, name, data ):
    'add cell array'
    self._adddataarray( name, data, 'points' )

  def _adddataarray( self, name, data, location ):
    assert self._mesh is not None, 'Grid not specified'
    ndims, npoints, ncells = self._mesh[1:4]

    assert len(data) == ncells, 'data mismatch: expected length {}, got {}'.format( len(data), ncells )

    if location == 'points':
      data = numpy.concatenate( data, axis=0 )
      assert npoints == data.shape[0], 'Point data array should have {} entries'.format(npoints)
    elif location != 'cells':
      raise Exception( 'invalid location: {}'.format( location ) )

    assert data.ndim <= 3, 'data array should have at most 3 axes: {} and components (optional)'.format(location)

    extshp  = (data.shape[0],)+(3,)*(data.ndim-1)
    if data.shape == extshp:
      extdata = data
    else:
      extdata = numpy.zeros( extshp, dtype=data.dtype )
      extdata[tuple(slice(sh) for sh in data.shape)] = data

    self._dataarrays[location].append(( name, extdata ))


## INTERNAL HELPER FUNCTIONS

def _getnextindex( path, name, ext ):
  index = 0
  for filename in os.listdir( path ):
    if filename.startswith(name) and filename.endswith('.'+ext):
      num = filename[len(name):-len(ext)-1]
      if num.isdigit():
        index = max( index, int(num)+1 )
  return index

def _triangulate_quad( n, m ):
  ind = numpy.arange( n*m ).reshape( n, m )
  vert1 = numpy.array([ ind[:-1,:-1].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
  vert2 = numpy.array([ ind[1:,1:].ravel(), ind[1:,:-1].ravel(), ind[:-1,1:].ravel() ]).T
  vertices = numpy.concatenate( [vert1,vert2], axis=0 )
  hull = numpy.concatenate([ ind[:,0], ind[-1,1:], ind[-2::-1,-1], ind[0,-2::-1] ])
  return vertices, numpy.array(numeric.overlapping(hull))

def _triangulate_tri( n ):
  vert1 = [ ((2*n-i+1)*i)//2+numpy.array([j,j+1,j+n-i]) for i in range(n-1) for j in range(n-i-1) ]
  vert2 = [ ((2*n-i+1)*i)//2+numpy.array([j+1,j+n-i+1,j+n-i]) for i in range(n-1) for j in range(n-i-2) ]
  vertices = numpy.array( vert1+vert2 )
  hull = numpy.concatenate([ numpy.arange(n), numpy.arange(n-1,0,-1).cumsum()+n-1, numpy.arange(n+1,2,-1).cumsum()[::-1]-n-1 ])
  return vertices, numpy.array(numeric.overlapping(hull))

def _triangulate_bezier( np ):
  nquad = int( numpy.sqrt(np) + .5 )
  if nquad**2 == np:
    return _triangulate_quad( nquad, nquad )
  ntri = int( numpy.sqrt((2*np)+.25) )
  if ntri * (ntri+1) == 2 * np:
    return _triangulate_tri( ntri )
  raise Exception( 'cannot match points to a bezier scheme' )


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
      arrays = topo.elem_eval( values, ischeme='vtk', separate=True )
      for key, array in zip( keys, arrays ):
        vtkfile.pointdataarray( key, array )

    if celldata:  
      keys, values = zip( *celldata.items() )
      arrays = topo.elem_mean( values, coords=coords, ischeme=ischeme )
      for key, array in zip( keys, arrays ):
        vtkfile.celldataarray( key, array )

def triangulate( points, mergetol=0 ):
  triangulate_bezier = cache.Wrapper(_triangulate_bezier)
  npoints = 0
  triangulation = []
  edges = []
  for epoints in points:
    np = len(epoints)
    assert epoints.shape == (np,2)
    if np == 0:
      continue
    etri, ehull = triangulate_bezier( np )
    triangulation.append( npoints + etri )
    edges.append( npoints + ehull )
    npoints += np
  triangulation = numpy.concatenate( triangulation, axis=0 )
  edges = numpy.concatenate( edges, axis=0 )
  points = numpy.concatenate( points, axis=0 )
  if mergetol:
    import scipy.spatial
    onedge = numpy.zeros( npoints, dtype=bool )
    onedge[edges] = True
    index, = onedge.nonzero()
    for i, j in sorted( scipy.spatial.cKDTree( points[onedge] ).query_pairs( mergetol ) ):
      assert i < j
      index[j] = index[i]
    renumber = numpy.arange( npoints )
    renumber[onedge] = index
    triangulation = renumber[triangulation]
    edges = numpy.sort( renumber[edges], axis=1 ) # order edge endpoints to recognize duplicates
    edges = edges[ numpy.lexsort( edges.T ) ] # sort edges lexicographically
    edges = edges[ numpy.concatenate( [ [True], numpy.diff( edges, axis=0 ).any(axis=1) ] ) ] # remove duplicates
  import matplotlib.tri
  return matplotlib.tri.Triangulation( points[:,0], points[:,1], triangulation ), points[edges]

def interpolate( tri, xy, values, mergetol=1e-5 ):
  assert xy.shape[-1] == 2
  import matplotlib.tri
  if not isinstance( tri, matplotlib.tri.Triangulation ):
    tri, edges = triangulate( tri, mergetol=mergetol )
  if not isinstance( values, numpy.ndarray ):
    values = numpy.concatenate( values, axis=0 )
  assert len(tri.x) == len(values)
  itri = tri.get_trifinder()( xy[...,0].ravel(), xy[...,1].ravel() )
  inside = itri != -1
  itri = itri[inside]
  interpvalues = numpy.empty( xy.shape[:-1] + values.shape[1:] )
  interpvalues[:] = numpy.nan
  xy1 = numpy.concatenate( [ xy.reshape(-1,2)[inside], numpy.ones([len(itri),1]) ], axis=1 )
  for iv, v in zip( interpvalues.reshape(len(inside),-1).T, values.reshape(len(values),-1).T ):
    plane_coefficients = tri.calculate_plane_coefficients(v)
    iv[inside] = numeric.contract( xy1, plane_coefficients[itri], axis=1 )
  return interpvalues


# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
