from . import log, core
import sys, os, time, numpy, cPickle, hashlib, weakref, traceback

def getpath( pattern ):
  'create file in dumpdir'

  if pattern == pattern.format( 0 ):
    return dumpdir + pattern
  prefix = pattern.split( '{' )[0]
  dumpdir = core.getprop( 'dumpdir', False )
  assert dumpdir
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

  core.setprop( 'nprocs', 1 )
  core.setprop( 'outdir', '~/public_html' )
  core.setprop( 'verbose', 3 )
  core.setprop( 'linewidth', 60 )
  core.setprop( 'imagetype', 'png' )
  finityrc = os.path.expanduser( '~/.finityrc' )
  if os.path.isfile( finityrc ):
    d = {}
    try:
      execfile( finityrc, {}, d )
    except:
      log.error( 'Error in %s (skipping)' % finityrc )
      log.error( traceback.format_exc() )
    else:
      for key, val in d.iteritems():
        core.setprop( key, val )

  if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    print 'Usage: %s [FUNC] [ARGS]' % sys.argv[0]
    print
    print '  -h    --help         Display this help'
    print '  -p P  --nprocs=P     Select number of processors [%d]' % core.getprop( 'nprocs' )
    print '  -o O  --outdir=O     Define directory for output [%s]' % core.getprop( 'outdir' )
    print '  -v V  --verbose=V    Set verbosity level [%d]' % core.getprop( 'verbose' )
    print '  -l L  --linewidth=L  Set line width [%d]' % core.getprop( 'linewidth' )
    print '  -i I  --imagetype=I  Set image type [%s]' % core.getprop( 'imagetype' )
    print '  -g G  --graphviz=G   Set graphviz executable location'
    for i, func in enumerate( functions ):
      print
      print 'Arguments for %s%s' % ( func.func_name, '' if i else ' (default)' )
      print
      for kwarg, default in getkwargdefaults( func ):
        tmp = '--%s=%s' % ( kwarg.lower(), kwarg[0].upper() )
        print '  %-20s Default: %s' % ( tmp, default )
    return

  if sys.argv[1:] and not sys.argv[1].startswith( '-' ):
    argiter = iter( sys.argv[2:] )
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
    argiter = iter( sys.argv[1:] )

  args = {}
  for arg in argiter:
    if arg.startswith('--'):
      arg = arg[2:]
      if '=' in arg:
        arg, val = arg.split('=')
      else:
        val = True
    else:
      try:
        arg = { '-p':'parallel', '-o':'outdir', '-v':'verbose', '-l':'linewidth', '-i':'imagetype' }[arg]
      except KeyError:
        print 'invalid argument %r' % arg
        return
      val = argiter.next()
    assert arg not in args, 'argument encountered twice: %r' % arg
    args[arg] = val

  if 'parallel' in args:
    core.setprop( 'nprocs', int( args.pop('parallel') ) )
  if 'outdir' in args:
    core.setprop( 'outdir', args.pop('outdir') )
  if 'verbose' in args:
    core.setprop( 'verbose', int( args.pop('verbose') ) )
  if 'linewidth' in args:
    core.setprop( 'linewidth', int( args.pop('linewidth') ) )
  if 'imagetype' in args:
    core.setprop( 'imagetype', args.pop('imagetype') )
  if 'graphviz' in args:
    core.setprop( 'graphviz', args.pop('graphviz') )

  kwargs = dict( getkwargdefaults( func ) )
  for arg, val in args.iteritems():
    for kwarg, default in kwargs.iteritems():
      if kwarg.lower() == arg.lower():
        break
    else:
      print 'error: invalid argument for %s: %s' % ( funcname, arg )
      return
    try:
      val = eval( val )
    except:
      pass
    kwargs[ kwarg ] = val

  scriptname = os.path.basename(sys.argv[0])
  outdir = os.path.expanduser( core.getprop( 'outdir', None ) ).rstrip( os.sep ) + os.sep
  basedir = outdir + scriptname + os.sep
  dumpdir = basedir + time.strftime( '%Y/%m/%d/%H-%M-%S/' )
  os.makedirs( dumpdir )

  core.setprop( 'dumpdir', dumpdir )
  core.setprop( 'html', log.HtmlWriter( dumpdir + 'index.html' ) )

  for directory in outdir, basedir:
    link = directory + 'latest'
    if os.path.islink( link ):
      os.remove( link )
    os.symlink( dumpdir, link )

  commandline = [ ' '.join([ scriptname, funcname ]) ] \
              + [ '  --%s=%s' % ( arg.lower(), val ) for arg, val in kwargs.items() ]

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
