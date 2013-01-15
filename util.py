from . import log, prop
import sys, os, time, numpy, cPickle, hashlib, weakref, traceback

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
    self.data = file( path, 'a+' if not getattr( prop, 'recache', False ) else 'w+' )

  def __call__( self, func, *args, **kwargs ):
    'call'

    name = func.__name__ + ''.join( ' %s' % arg for arg in args ) + ''.join( ' %s=%s' % item for item in kwargs.iteritems() )
    try:
      pos = self.data.tell()
      data = cPickle.load( self.data )
    except EOFError:
      log.info( 'not in cache:', name )
      data = func( *args, **kwargs)
      self.data.seek( pos )
      cPickle.dump( data, self.data, -1 )
    else:
      log.info( 'loaded from cache:', name )
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

def clone( obj ):
  'clone object'

  clone = object.__new__( obj.__class__ )
  clone.__dict__.update( obj.__dict__ )
  return clone

def iterate( nmax=-1, verbose=True ):
  'iterate forever'

  i = 0
  while True:
    i += 1
    if verbose:
      log.progress( 'iteration %d' % i )
    yield i
    if i == nmax:
      break

class NanVec( numpy.ndarray ):
  'nan-initialized vector'

  def __new__( cls, length ):
    'new'

    vec = numpy.empty( length ).view( cls )
    vec[:] = numpy.nan
    return vec

  def __ior__( self, other ):
    'combine'

    where = numpy.isnan( self )
    self[ where ] = other if numpy.isscalar( other ) else other[ where ]
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
    'linewidth': 60,
    'imagetype': 'png',
    'symlink': 'latest',
    'recache': False,
    'dot': 'dot',
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
  --linewidth=%(linewidth)-11s Set line width
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
  timepath = time.strftime( '%Y/%m/%d/%H-%M-%S/' )
  dumpdir = basedir + timepath
  os.makedirs( dumpdir ) # asserts nonexistence
  if prop.symlink:
    symlink = basedir + prop.symlink
    if os.path.islink( symlink ):
      os.remove( symlink )
    os.symlink( timepath, symlink )

  prop.dumpdir = dumpdir
  prop.html = log.HtmlWriter( dumpdir + 'log.html' )

  print >> open( outdir+'log.html', 'w' ), '<meta http-equiv="refresh" content="0;URL={}/log.html">'.format( scriptname )
  print >> open( basedir+'log.html', 'w' ), '<meta http-equiv="refresh" content="0;URL={}log.html">'.format( timepath )

  prop.cachedir = basedir + 'cache'

  commandline = [ ' '.join([ scriptname, funcname ]) ] + [ '  --%s=%s' % item for item in kwargs.items() ]

  log.info( ' \\\n'.join( commandline ), end='\n\n' )
  log.info( 'start %s\n' % time.ctime() )

  t0 = time.time()
  try:
    func( **kwargs )
  except:
    log.error( traceback.format_exc(), end='' )

  dt = time.time() - t0
  hours = dt // 3600
  minutes = dt // 60 - 60 * hours
  seconds = dt // 1 - 60 * minutes - 3600 * hours

  log.info( '\nfinish %s' % time.ctime() )
  log.info( 'elapsed %.0f:%.0f:%.0f' % ( hours, minutes, seconds ) )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
