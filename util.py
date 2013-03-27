from . import log, prop
import sys, os, time, numpy, cPickle, hashlib, weakref, traceback, core, warnings

class _SuppressedOutput( object ):
  'suppress all output by redirection to /dev/null'

  def __enter__( self ):
    sys.stdout.flush()
    sys.stderr.flush()
    self.stdout = os.dup( sys.stdout.fileno() )
    self.stderr = os.dup( sys.stderr.fileno() )
    devnull = os.open( os.devnull, os.O_WRONLY )
    os.dup2( devnull, sys.stdout.fileno() )
    os.dup2( devnull, sys.stderr.fileno() )
    os.close( devnull )

  def __exit__( self, exc_type, exc_value, traceback ):
    os.dup2( self.stdout, sys.stdout.fileno() )
    os.dup2( self.stderr, sys.stderr.fileno() )
    os.close( self.stdout )
    os.close( self.stderr )

suppressed_output = _SuppressedOutput()

class ImmutableArray( numpy.ndarray ):
  'immutable array'

  flags = None

  def __new__( self, arr ):
    'constructor'

    arr = numpy.asarray( arr )
    arr.flags.writeable = False
    arr = arr.view( ImmutableArray )
    return arr

  def __eq__( self, other ):
    'equals'

    return self is other

  def __hash__( self ):
    'hash'

    return hash( id(self) )

def delaunay( points ):
  'delaunay triangulation'

  from scipy import spatial
  with suppressed_output:
    return spatial.Delaunay( points )

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

class Cache( object ):
  'cache'

  def __init__( self, *args ):
    'constructor'

    name = sys._getframe(1).f_code.co_name
    import hashlib
    strhash = ','.join( str(arg) for arg in args )
    md5hash = hashlib.md5( strhash ).hexdigest() + '.' + name
    log.info( 'using cache:', md5hash )
    cachedir = getattr( prop, 'cachedir', 'cache' )
    if not os.path.exists( cachedir ):
      os.makedirs( cachedir )
    path = os.path.join( cachedir, md5hash )
    self.data = file( path, 'ab+' if not getattr( prop, 'recache', False ) else 'wb+' )

  def __call__( self, func, *args, **kwargs ):
    'call'

    name = func.__name__ + ''.join( ' %s' % arg for arg in args ) + ''.join( ' %s=%s' % item for item in kwargs.iteritems() )
    pos = self.data.tell()
    try:
      data = cPickle.load( self.data )
    except EOFError:
      data = func( *args, **kwargs)
      self.data.seek( pos )
      cPickle.dump( data, self.data, -1 )
      msg = 'written to'
    else:
      msg = 'loaded from'
    log.info( msg, 'cache:', name, '[%db]' % (self.data.tell()-pos) )
    return data

def getpath( pattern ):
  'create file in dumpdir'

  dumpdir = prop.dumpdir
  if pattern == pattern.format( 0 ):
    return dumpdir + pattern
  prefix = pattern.split( '{' )[0]
  names = [ name for name in os.listdir( dumpdir ) if name.startswith(prefix) ]
  n = len(names)
  while True:
    n += 1
    newname = dumpdir + pattern.format( n )
    if not os.path.isfile( newname ):
      return newname

_sum = sum
def sum( seq ):
  'a better sum'

  seq = iter(seq)
  return _sum( seq, seq.next() )

def product( seq ):
  'multiply items in sequence'

  seq = iter(seq)
  prod = seq.next()
  for item in seq:
    prod = prod * item
  return prod

def clone( obj ):
  'clone object'

  clone = object.__new__( obj.__class__ )
  clone.__dict__.update( obj.__dict__ )
  return clone

def iterate( context='iter', nmax=-1 ):
  'iterate forever'

  assert isinstance( nmax, int ), 'invalid value for nmax %r' % nmax
  i = 0
  while True:
    if i == nmax:
      break
    i += 1
    old = log.context( '%s %d' % (context,i), depth=1 )
    try:
      yield i
    finally:
      log.restore( old, depth=1 )

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

  @property
  def mask( self ):
    'find non-nan items'

    warnings.warn( '''Nanvec.mask will be removed in future
  Please use NanVec.~where instead.''', DeprecationWarning )
    return ~self.where

  def __iand__( self, other ):
    'combine'

    where = self.where
    if numpy.isscalar( other ):
      self[ where ] = other
    else:
      where &= other.where
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

class Clock( object ):
  'simpel interval timer'

  def __init__( self, interval ):
    'constructor'

    self.t = time.time()
    self.dt = interval

  def __nonzero__( self ):
    'check time'

    t = time.time()
    if t > self.t + self.dt:
      self.t = t
      return True
    return False

def arraymap( f, dtype, *args ):
  'call f for sequence of arguments and cast to dtype'

  return numpy.array( map( f, args[0] ) if len( args ) == 1
                 else [ f( *arg ) for arg in numpy.broadcast( *args ) ], dtype=dtype )

def objmap( func, *arrays ):
  'map numpy arrays'

  return numpy.frompyfunc( func, len(arrays), 1 )( *arrays )

def fail( msg, *args ):
  'generate exception'

  raise Exception, msg % args

def ipdb():
  'invoke debugger'

  from IPython import Debugger, Shell, ipapi
  
  Shell.IPShell( argv=[''] )
  ip = ipapi.get()
  def_colors = ip.options.colors
  frame = sys._getframe().f_back
  Debugger.BdbQuit_excepthook.excepthook_ori = sys.excepthook
  sys.excepthook = Debugger.BdbQuit_excepthook
  
  Debugger.Pdb( def_colors ).set_trace( frame )

class Locals( object ):
  'local namespace as object'

  def __init__( self ):
    'constructors'

    frame = sys._getframe( 1 )
    self.__dict__.update( frame.f_locals )

def getkwargdefaults( func ):
  'helper for run'

  defaults = func.func_defaults or []
  N = func.func_code.co_argcount - len( defaults )
  return zip( func.func_code.co_varnames[N:], defaults )

def run( *functions ):
  'call function specified on command line'

  assert functions

  properties = {
    'nprocs': 1,
    'outdir': '~/public_html',
    'verbose': 4,
    'imagetype': 'png',
    'symlink': False,
    'recache': False,
    'dot': False,
  }
  try:
    execfile( os.path.expanduser( '~/.finityrc' ), {}, properties )
  except IOError:
    pass # file does not exist
  except:
    print 'Error in .finityrc (skipping)'
    print traceback.format_exc()

  if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    print 'Usage: %s [FUNC] [ARGS]' % sys.argv[0]
    print '''
  --help                  Display this help
  --nprocs=%(nprocs)-14s Select number of processors
  --outdir=%(outdir)-14s Define directory for output
  --verbose=%(verbose)-13s Set verbosity level, 9=all
  --imagetype=%(imagetype)-11s Set image type
  --symlink=%(symlink)-13s Create symlink to latest results
  --recache=%(recache)-13s Overwrite existing cache
  --dot=%(dot)-17s Set graphviz executable''' % properties
    for i, func in enumerate( functions ):
      print
      print 'Arguments for %s%s' % ( func.func_name, '' if i else ' (default)' )
      print
      for kwarg, default in getkwargdefaults( func ):
        print '  --%s=%s' % ( kwarg, default )
    return

  if sys.argv[1:] and not sys.argv[1].startswith( '-' ):
    argv = sys.argv[2:]
    funcname = sys.argv[1]
    for func in functions:
      if func.func_name == funcname:
        break
    else:
      print 'error: invalid function name: %s' % funcname
      return
  else:
    func = functions[0]
    funcname = func.func_name
    argv = sys.argv[1:]
  kwargs = dict( getkwargdefaults( func ) )
  for arg in argv:
    assert arg.startswith('--'), 'invalid argument %r' % arg
    arg = arg[2:]
    try:
      arg, val = arg.split( '=', 1 )
      val = eval( val )
    except ValueError: # split failed
      val = True
    except (SyntaxError,NameError): # eval failed
      pass
    if arg in kwargs:
      kwargs[ arg ] = val
    else:
      assert arg in properties, 'invalid argument %r' % arg
      properties[arg] = val

  for name, value in properties.iteritems():
    setattr( prop, name, value )

  scriptname = os.path.basename(sys.argv[0])
  outdir = os.path.expanduser( prop.outdir ).rstrip( os.sep ) + os.sep
  basedir = outdir + scriptname + os.sep
  localtime = time.localtime()
  timepath = time.strftime( '%Y/%m/%d/%H-%M-%S/', localtime )

  dumpdir = basedir + timepath
  os.makedirs( dumpdir ) # asserts nonexistence

  if prop.symlink:
    for i in range(2): # make two links
      target = outdir
      dest = ''
      if i: # global link
        target += scriptname + os.sep
      else: # script-local link
        dest += scriptname + os.sep
      target += prop.symlink
      dest += timepath
      if os.path.islink( target ):
        os.remove( target )
      os.symlink( dest, target )

  logpath = os.path.join( os.path.dirname( log.__file__ ), '_log' ) + os.sep
  for filename in os.listdir( logpath ):
    if filename[0] != '.' and ( not os.path.isfile( outdir + filename ) or os.path.getmtime( outdir + filename ) < os.path.getmtime( logpath + filename ) ):
      print 'updating', filename
      open( outdir + filename, 'w' ).write( open( logpath + filename, 'r' ).read() )

  log.setup_html( maxlevel=prop.verbose, path=dumpdir+'log.html', title=scriptname + time.strftime( ' %Y/%m/%d %H:%M:%S', localtime ) )

  prop.dumpdir = dumpdir

  redirect = '<meta http-equiv="cache-control" content="no-cache">\n' \
           + '<meta http-equiv="refresh" content="0;URL=%slog.html">\n'

  print >> open( outdir+'log.html', 'w' ), redirect % ( scriptname + '/' + timepath )
  print >> open( basedir+'log.html', 'w' ), redirect % ( timepath )

  prop.cachedir = basedir + 'cache'

  commandline = [ ' '.join([ scriptname, funcname ]) ] + [ '  --%s=%s' % item for item in kwargs.items() ]

  log.info( ' \\\n'.join( commandline ) + '\n' )
  log.info( 'start %s\n' % time.ctime() )

  warnings.resetwarnings()

  t0 = time.time()
  try:
    func( **kwargs )
  except KeyboardInterrupt:
    log.error( 'killed by user' )
  except:
    log.exception()

  if hasattr( os, 'wait' ):
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
  log.info( 'elapsed %.0f:%.0f:%.0f' % ( hours, minutes, seconds ) )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
