# -*- coding: utf8 -*-
#
# Module UTIL
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The util module provides a collection of general purpose methods. Most
importantly it provides the :func:`run` method which is the preferred entry
point of a nutils application, taking care of command line parsing, output dir
creation and initiation of a log file.
"""

from __future__ import print_function, division
from . import log, debug, core, version, numeric
import sys, os, time, numpy, hashlib, weakref, warnings, collections, inspect

def isiterable( obj ):
  'check for iterability'

  try:
    iter(obj)
  except TypeError:
    return False
  return True

class _SuppressedOutput( object ):
  'suppress all output by redirection to /dev/null'

  def __enter__( self ):
    sys.stdout.flush()
    sys.stderr.flush()
    self.stdout = os.dup( 1 )#sys.stdout.fileno() )
    self.stderr = os.dup( 2 )#sys.stderr.fileno() )
    devnull = os.open( os.devnull, os.O_WRONLY )
    os.dup2( devnull, 1 )#sys.stdout.fileno() )
    os.dup2( devnull, 2 )#sys.stderr.fileno() )
    os.close( devnull )

  def __exit__( self, exc_type, exc_value, traceback ):
    os.dup2( self.stdout, 1 )#sys.stdout.fileno() )
    os.dup2( self.stderr, 2 )#sys.stderr.fileno() )
    os.close( self.stdout )
    os.close( self.stderr )

suppressed_output = _SuppressedOutput()

class Product( object ):
  def __init__( self, iter1, iter2 ):
    self.iter1 = iter1
    self.iter2 = iter2
  def __len__( self ):
    return len( self.iter1 ) * len( self.iter2 )
  def __iter__( self ):
    return iter( item1 * item2 for item1 in self.iter1 for item2 in self.iter2 )

class _Unit( object ):
  def __mul__( self, other ): return other
  def __rmul__( self, other ): return other

unit = _Unit()

def delaunay( points, positive=False ):
  'delaunay triangulation'

  points = numpy.asarray( points )
  npoints, ndims = points.shape
  assert ndims >= 1, 'ndims should be at least 1'
  if npoints < 1 + ndims:
    return []
  if ndims == 1:
    indices = numpy.argsort( points[:,0] )
    return numeric.overlapping( indices )
  from scipy import spatial
  with suppressed_output:
    submesh = spatial.Delaunay( points )
  vertices = numpy.asarray( submesh.vertices )
  if positive:
    for tri in submesh.vertices:
      if numpy.linalg.det( points[tri[1:]] - points[tri[0]] ) < 0:
        tri[-2:] = tri[-1], tri[-2]
  return vertices

def withrepr( f ):
  'add string representation to generated function'

  class function_wrapper( object ):
    func = staticmethod( f )
    argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
    defaults = dict( zip( reversed(argnames), reversed(f.__defaults__) ) )
    def __init__( self, *args, **kwargs ):
      for name in self.__class__.argnames[len(args):]:
        try:
          arg = kwargs[name]
        except KeyError:
          arg = self.__class__.defaults[name]
        args += arg,
      self.__setstate__( args )
    def __setstate__( self, state ):
      assert len(state) == len(self.__class__.argnames)
      self.__args = state
      self.__func_instance = self.__class__.func( *state )
    def __getstate__( self ):
      return self.__args
    def __getattr__( self, attr ):
      try:
        index = self.__class__.argnames.index( attr )
      except ValueError:
        raise AttributeError( attr )
      return self.__args[index]
    def __call__( self, *args, **kwargs ):
      return self.__func_instance( *args, **kwargs )
    def __eq__( self, other ):
      return other.__class__ == self.__class__ \
         and other.__class__.func == self.__class__.func \
         and other.__args == self.__args
    def __str__( self ):
      argstr = ','.join( '%s=%s' % item for item in zip( self.__class__.argnames, self.__args ) )
      return '%s(%s)' % ( f.__name__, argstr )

  from functools import update_wrapper, WRAPPER_ASSIGNMENTS
  assignments = list( WRAPPER_ASSIGNMENTS )
  assignments.remove( '__doc__' ) # for python2
  return update_wrapper( function_wrapper, f, assignments, [] )

def profile( func ):
  import cProfile, pstats
  frame = sys._getframe(1)
  frame.f_locals['__profile_func__'] = func
  prof = cProfile.Profile()
  stats = prof.runctx( '__profile_retval__ = __profile_func__()', frame.f_globals, frame.f_locals )
  pstats.Stats( prof, stream=sys.stdout ).strip_dirs().sort_stats( 'time' ).print_stats()
  retval = frame.f_locals['__profile_retval__']
  del frame.f_locals['__profile_func__']
  del frame.f_locals['__profile_retval__']
  raw_input( 'press enter to continue' )
  return retval

def getpath( pattern ):
  'create file in outdir'

  outdir = core.getprop( 'outdir', '.' )
  if pattern == pattern.format( 0 ):
    return outdir + pattern
  prefix = pattern.split( '{' )[0]
  names = [ name for name in os.listdir( outdir ) if name.startswith(prefix) ]
  n = len(names)
  while True:
    n += 1
    newname = outdir + pattern.format( n )
    if not os.path.isfile( newname ):
      return newname

_sum = sum
def sum( seq ):
  'a better sum'

  seq = iter(seq)
  return _sum( seq, next(seq) )

def product( seq ):
  'multiply items in sequence'

  seq = iter(seq)
  prod = next(seq)
  for item in seq:
    prod = prod * item
  return prod

def allequal( seq1, seq2 ):
  seq1 = iter(seq1)
  seq2 = iter(seq2)
  for item1, item2 in zip( seq1, seq2 ):
    if item1 != item2:
      return False
  if list(seq1) or list(seq2):
    return False
  return True

def clone( obj ):
  'clone object'

  clone = object.__new__( obj.__class__ )
  clone.__dict__.update( obj.__dict__ )
  return clone

class NanVec( numpy.ndarray ):
  'nan-initialized vector'

  def __new__( cls, length ):
    'new'

    vec = numpy.empty( length ).view( cls )
    vec[:] = numpy.nan
    return vec

  @property
  def where( self ):
    'find non-nan items'

    return ~numpy.isnan( self.view(numpy.ndarray) )

  def __iand__( self, other ):
    'combine'

    where = self.where
    if numpy.isscalar( other ):
      self[ where ] = other
    else:
      assert isinstance( other, numpy.ndarray ) and other.shape == self.shape
      where &= ~numpy.isnan(other)
      self[ where ] = other[ where ]
    return self

  def __and__( self, other ):
    'combine'

    return self.copy().__iand__( other )

  def __ior__( self, other ):
    'combine'

    wherenot = ~self.where
    self[ wherenot ] = other if numpy.isscalar( other ) else other[ wherenot ]
    return self

  def __or__( self, other ):
    'combine'

    return self.copy().__ior__( other )

  def __invert__( self ):
    nanvec = NanVec( len(self) )
    nanvec[numpy.isnan(self)] = 0
    return nanvec

class Clock( object ):
  'simple interval timer'

  def __init__( self, dt=None, dtexp=None, dtmax=None ):
    'constructor'

    self.dt = core.getprop( 'progress_interval', 1. ) if dt is None else dt
    self.dtexp = core.getprop( 'progress_interval_scale', 2 ) if dtexp is None else dtexp
    self.dtmax = core.getprop( 'progress_interval_max', 0 ) if dtmax is None else dtmax
    self.reset()

  def reset( self ):
    self.tnext = time.time() + self.dt

  def check( self ):
    'check time'

    if time.time() < self.tnext:
      return False
    if self.dtexp != 1:
      self.dt *= self.dtexp
      if self.dt > self.dtmax > 0:
        self.dt = self.dtmax
    self.reset()
    return True

def tensorial( args ):
  'create n-dimensional array containing tensorial combinations of n args'

  shape = [ len(arg) for arg in args ]
  array = numpy.empty( shape, dtype=object )
  for index in numpy.lib.index_tricks.ndindex( *shape ):
    array[index] = tuple([ arg[i] for arg, i in zip(args,index) ])
  return array

def arraymap( f, dtype, *args ):
  'call f for sequence of arguments and cast to dtype'

  return numpy.array( [ f( arg ) for arg in args[0] ] if len( args ) == 1
                 else [ f( *arg ) for arg in numpy.broadcast( *args ) ], dtype=dtype )

def allunique( iterable ):
  iterable = tuple( iterable )
  return len(iterable) == len(set(iterable))

def minmax( iterable ):
  iterable = tuple( iterable )
  return min(iterable), max(iterable)

def objmap( func, *arrays ):
  'map numpy arrays'

  arrays = [ numpy.asarray( array, dtype=object ) for array in arrays ]
  return numpy.frompyfunc( func, len(arrays), 1 )( *arrays )

def fail( msg, *args ):
  'generate exception'

  raise Exception( msg % args )

class Locals( object ):
  'local namespace as object'

  def __init__( self ):
    'constructors'

    frame = sys._getframe( 1 )
    self.__dict__.update( frame.f_locals )

def _getkwargdefaults_new( func ):
  'helper for run'

  kwargs = collections.OrderedDict()
  signature = inspect.signature( func )
  for parameter in signature.parameters.values():
    if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
      continue
    if parameter.default is parameter.empty:
      raise ValueError( 'Function cannot be called without arguments.' )
    kwargs[parameter.name] = parameter.default
  return kwargs

def _getkwargdefaults_legacy( func ):
  'helper for run'

  args, varargs, keywords, defaults = inspect.getargspec( func )
  if defaults is None:
    defaults = []
  if len( defaults ) != len( args ):
    raise ValueError( 'Function cannot be called without arguments.' )
  return collections.OrderedDict(zip(args, defaults))

if sys.version_info >= (3,3):
  getkwargdefaults = _getkwargdefaults_new
else:
  getkwargdefaults = _getkwargdefaults_legacy

class Statm( object ):
  'memory statistics on systems that support it'

  __slots__ = 'size', 'resident', 'share', 'text', 'data'

  def __init__( self, rusage=None ):
    'constructor'

    if rusage is None:
      pid = os.getpid()
      self.size, self.resident, self.share, self.text, lib, self.data, dt = map( int, open( '/proc/%d/statm' % pid ).read().split() )
    else:
      self.size, self.resident, self.share, self.text, self.data = rusage

  def __sub__( self, other ):
    'subtract'

    diff = [ getattr(self,attr) - getattr(other,attr) for attr in self.__slots__ ]
    return Statm( diff )

  def __str__( self ):
    'string representation'

    return '\n'.join( [ 'STATM:     G  M  k  b' ]
      + [ attr + ' ' + (' %s'%getattr(self,attr)).rjust(20-len(attr),'-') for attr in self.__slots__ ] )

class Terminate( Exception ):
  pass

def githash( path, depth=0  ):
  abspath = os.path.abspath( path )
  for i in range( depth ):
    abspath = os.path.dirname( abspath )
  git = os.path.join( abspath, '.git' )
  with open( os.path.join( git, 'HEAD' ) ) as HEAD:
    head = HEAD.read()
  assert head.startswith( 'ref:' )
  ref = head[4:].strip()
  with open( os.path.join( git, ref ) ) as ref:
    githash, = ref.read().split()
  return githash

def run( *functions ):
  'call function specified on command line'

  assert functions

  if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    print( 'Usage: %s [FUNC] [ARGS]' % sys.argv[0] )
    print( '''
  --help                  Display this help
  --nprocs=%(nprocs)-14s Select number of processors
  --outrootdir=%(outrootdir)-10s Define the root directory for output
  --outdir=               Define custom directory for output
  --verbose=%(verbose)-13s Set verbosity level, 9=all
  --richoutput=%(richoutput)-10s Use rich output (colors, unicode)
  --tbexplore=%(tbexplore)-11s Start traceback explorer on error
  --imagetype=%(imagetype)-11s Set image type
  --symlink=%(symlink)-13s Create symlink to latest results
  --recache=%(recache)-13s Overwrite existing cache
  --dot=%(dot)-17s Set graphviz executable
  --profile=%(profile)-13s Show profile summary at exit''' % core.globalproperties )
    for i, func in enumerate( functions ):
      print()
      print( 'Arguments for %s%s' % ( func.__name__, '' if i else ' (default)' ) )
      print()
      for kwarg, default in getkwargdefaults( func ).items():
        print( '  --%s=%s' % ( kwarg, default ) )
    return

  func = functions[0]
  argv = sys.argv[1:]
  funcbyname = { func.__name__: func for func in functions }
  if argv and argv[0] in funcbyname:
    func = funcbyname[argv[0]]
    argv = argv[1:]

  kwargs = getkwargdefaults( func )
  properties = {}
  for arg in argv:
    arg = arg.lstrip('-')
    try:
      arg, val = arg.split( '=', 1 )
      val = eval( val, sys._getframe(1).f_globals )
    except ValueError: # split failed
      val = True
    except (SyntaxError,NameError): # eval failed
      pass
    if arg in kwargs:
      kwargs[ arg ] = val
    else:
      assert arg in core.globalproperties, 'invalid argument %r' % arg
      properties[arg] = val

  locals().update({ '__%s__' % name: value for name, value in properties.items() })

  scriptname = os.path.basename(sys.argv[0])
  outrootdir = os.path.expanduser( core.getprop( 'outrootdir' ) ).rstrip( os.sep ) + os.sep
  basedir = outrootdir + scriptname + os.sep
  localtime = time.localtime()
  timepath = time.strftime( '%Y/%m/%d/%H-%M-%S/', localtime )
  outdir = properties.get( 'outdir', None )

  if outdir is None:
    # `outdir` not specified on the commandline, use default directory layout

    outdir = basedir + timepath
    os.makedirs( outdir ) # asserts nonexistence

    if core.getprop( 'symlink' ):
      for i in range(2): # make two links
        target = outrootdir
        dest = ''
        if i: # global link
          target += scriptname + os.sep
        else: # script-local link
          dest += scriptname + os.sep
        target += core.getprop( 'symlink' )
        dest += timepath
        if os.path.islink( target ):
          os.remove( target )
        os.symlink( dest, target )

    logpath = os.path.join( os.path.dirname( log.__file__ ), '_log' ) + os.sep
    for filename in os.listdir( logpath ):
      if filename[0] != '.' and ( not os.path.isfile( outrootdir + filename ) or os.path.getmtime( outrootdir + filename ) < os.path.getmtime( logpath + filename ) ):
        print( 'updating', filename )
        open( outrootdir + filename, 'w' ).write( open( logpath + filename, 'r' ).read() )

    redirect = '<html>\n<head>\n<meta http-equiv="cache-control" content="max-age=0" />\n' \
             + '<meta http-equiv="cache-control" content="no-cache" />\n' \
             + '<meta http-equiv="expires" content="0" />\n' \
             + '<meta http-equiv="expires" content="Tue, 01 Jan 1980 1:00:00 GMT" />\n' \
             + '<meta http-equiv="pragma" content="no-cache" />\n' \
             + '<meta http-equiv="refresh" content="0;URL=%slog.html" />\n</head>\n</html>\n'

    print( redirect % ( scriptname + '/' + timepath ), file=open( outrootdir+'log.html', 'w' ) )
    print( redirect % ( timepath ), file=open( basedir+'log.html', 'w' ) )

  elif not os.path.isdir( outdir ):
    # use custom directory layout, skip creating symlinks, redirects
    os.makedirs( outdir )

  htmlfile = open( os.path.join( outdir, 'log.html' ), 'w' )
  htmlfile.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
  htmlfile.write( '<html><head>\n' )
  htmlfile.write( '<title>%s %s</title>\n' % ( scriptname, time.strftime( '%Y/%m/%d %H:%M:%S', localtime ) ) )
  htmlfile.write( '<script type="text/javascript" src="../../../../../viewer.js" ></script>\n' )
  htmlfile.write( '<link rel="stylesheet" type="text/css" href="../../../../../style.css">\n' )
  htmlfile.write( '<link rel="stylesheet" type="text/css" href="../../../../../custom.css">\n' )
  htmlfile.write( '</head><body><pre>\n' )
  htmlfile.write( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html">latest %s</a> | <a class="nav_latestall" href="../../../../../log.html">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>\n\n' % scriptname )
  htmlfile.flush()

  try:

    __log__ = log.Log( log.TeeStreamFactory( log.HtmlStreamFactory(htmlfile), log.StdoutStreamFactory() ) )
    __outdir__ = outdir
    __cachedir__ = basedir + 'cache'

    try:
      import signal
      signal.signal( signal.SIGTSTP, debug.signal_handler ) # start traceback explorer at ^Z
    except Exception as e:
      log.warning( 'failed to install signal handler:', e )

    try:
      gitversion = version + '.' + githash(__file__,2)[:8]
    except:
      gitversion = version
    log.info( 'nutils v%s\n' % gitversion )

    commandline = [ ' '.join([ scriptname, func.__name__ ]) ] + [ '  --%s=%s' % item for item in kwargs.items() ]
    log.info( ' \\\n'.join( commandline ) + '\n' )
    log.info( 'start %s\n' % time.ctime() )

    warnings.resetwarnings()

    t0 = time.time()

    if core.getprop( 'profile' ):
      import cProfile
      prof = cProfile.Profile()
      prof.enable()

    failed = 1
    frames = None
    try:
      func( **kwargs )
      failed = 0
    except KeyboardInterrupt:
      log.error( 'killed by user' )
    except Terminate as exc:
      log.error( 'terminated:', exc )
    except Exception:
      exc, frames = debug.exc_info()
      log.stack( repr(exc), frames )

    if core.getprop( 'profile' ):
      prof.disable()

    if hasattr( os, 'wait' ) and frames is None:
      # clear exception info, helps closing subprocesses that otherwise cause a
      # deadlock when calling `os.wait` below
      sys.exc_clear()
      exc = None
      try: # wait for child processes to die
        while True:
          pid, status = os.wait()
      except OSError: # no more children
        pass

    dt = time.time() - t0
    hours = dt // 3600
    minutes = dt // 60 - 60 * hours
    seconds = dt // 1 - 60 * minutes - 3600 * hours

    log.info()
    log.info( 'finish %s\n' % time.ctime() )
    log.info( 'elapsed %02.0f:%02.0f:%02.0f' % ( hours, minutes, seconds ) )

    if core.getprop( 'uncollected_summary', False ):
      debug.trace_uncollected()

    if core.getprop( 'profile' ):
      import pstats
      stream = log.getstream( 'warning' )
      stream.write( 'profile results:\n' )
      pstats.Stats( prof, stream=stream ).strip_dirs().sort_stats( 'time' ).print_stats()

    if frames:
      debug.write_html( htmlfile, repr(exc), frames )
      if core.getprop( 'tbexplore', False ):
        debug.explore( repr(exc), frames, '''
          Your program has died. The traceback explorer allows you to
          examine its post-mortem state to figure out why this happened.
          Type 'help' for an overview of commands to get going.''' )

    sys.exit( failed )

  finally:

    htmlfile.write( '</pre></body></html>\n' )
    htmlfile.close()

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
