import sys, os, time, numpy, cPickle, hashlib
from numpyextra import *

LINEWIDTH = 50
BASEPATH = os.path.expanduser( '~/public_html/' )
DUMPDIR = BASEPATH + 'latest'

class Cache( object ):
  'data cacher'

  fmt = 'cache/%s.npz'

  def __init__( self, *args ):
    'constructor'

    m = hashlib.md5()
    for arg in args:
      m.update( '%s\0' % arg )
    self.myhash = m.hexdigest()
    self.path = self.fmt % self.myhash

  def load( self ):
    'load or create snapshot'

    print 'loading data:',
    try:
      npzobj = numpy.load( self.path, mmap_mode='c' )
      arrays = [ val for (key,val) in sorted( npzobj.items() ) ]
      print self.format_arrays( arrays )
    except IOError, e:
      print 'failed: not in cache.'
      raise
    except Exception, e:
      print 'failed:', e
      raise

    return arrays if len( arrays ) > 1 else arrays[0]

  @staticmethod
  def format_arrays( arrays ):
    'format as string'

    return ', '.join( '%s(%s)' % ( arr.dtype, 'x'.join( str(n) for n in arr.shape ) ) for arr in arrays )

  def chain( self, *args ):

    return Cache( self.myhash, *args )

  def save( self, *arrays ):

    print 'saving data:', self.format_arrays( arrays )
    dirname = os.path.dirname( self.path )
    if not os.path.isdir( dirname ):
      os.makedirs( dirname )
    numpy.savez( self.path, *arrays )

class UsableArray( numpy.ndarray ):
  'array wrapper that can be compared'

  def __new__( cls, array ):
    'new'

    return numpy.asarray( array ).view( cls )

  def __eq__( self, other ):
    'compare'

    return self is other \
        or isinstance( other, numpy.ndarray ) \
       and other.shape == self.shape \
       and numpy.ndarray.__eq__( self, other ).all()

def iterate():
  'iterate forever'

  i = 0
  while True:
    i += 1
    print 'iteration %d' % i
    yield i

def obj2str( obj ):
  'convert object to string'

  if isinstance( obj, numpy.ndarray ) and obj.ndim > 0:
    return 'array[%s]' % 'x'.join( map( str, obj.shape ) )
  if isinstance( obj, (list,dict,tuple) ):
    if len(obj) > 4:
      return '%s[(%s items)]' % ( obj.__class__.__name__, len(obj) )
    if isinstance( obj, list ):
      return '[%s]' % ','.join( obj2str(o) for o in obj )
    if isinstance( obj, tuple ):
      return '(%s)' % ','.join( obj2str(o) for o in obj )
    if isinstance( obj, dict ):
      return '{%s}' % ','.join( '%s:%s' % ( obj2str(k), obj2str(v) ) for k, v in obj.iteritems() )
  if isinstance( obj, slice ):
    I = ''
    if obj.start is not None:
      I += str(obj.start)
    I += ':'
    if obj.stop is not None:
      I += str(obj.stop)
    if obj.step is not None:
      I += ':' + str(obj.step)
    return I
  if obj is None:
    return '_'
  return str(obj)

def cacheprop( func ):
  'cached property'

  key = func.func_name
  def wrapped( self ):
    value = self.__dict__.get( key )
    if value is None:
      value = func( self )
      self.__dict__[ key ] = value
    return value

  return property( wrapped )

def cachefunc( func ):
  'cached property'

  def wrapped( self, *args, **kwargs ):
    funcache = self.__dict__.setdefault( '_funcache', {} )

    unspecified = object()
    argcount = func.func_code.co_argcount - 1 # minus self
    args = list(args) + [unspecified] * ( argcount - len(args) ) if func.func_defaults is None \
      else list(args) + list(func.func_defaults[ len(args) + len(func.func_defaults) - argcount: ]) if len(args) + len(func.func_defaults) > argcount \
      else list(args) + [unspecified] * ( argcount - len(func.func_defaults) - len(args) ) + list(func.func_defaults)
    try:
      for kwarg, val in kwargs.items():
        args[ func.func_code.co_varnames.index(kwarg)-1 ] = val
    except ValueError:
      raise TypeError, '%s() got an unexpected keyword argument %r' % ( func.func_name, kwarg )
    args = tuple( args )
    if unspecified in args:
      raise TypeError, '%s() not all arguments were specified' % func.func_name
    key = (func.func_name,) + args
    value = funcache.get( key )
    if value is None:
      value = func( self, *args )
      funcache[ key ] = value
    return value

  return wrapped

def classcache( fun ):
  'wrapper to cache return values'

  cache = {}
  def wrapped_fun( cls, *args ):
    data = cache.get( args )
    if data is None:
      data = fun( cls, *args )
      cache[ args ] = data
    return data
  return wrapped_fun if fun.func_name == '__new__' \
    else classmethod( wrapped_fun )

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

class Iter_CallBack:
  def __init__( self, tol, title ):
    self.t = time.time()
    self.progressbar = progressbar( n=numpy.log(tol), title=title )
  def __call__( self, arg ):
    if time.time() > self.t:
      self.t = time.time() + .1
      if isinstance( arg, numpy.ndarray ):
        arg = numpy.linalg.norm( b - A * arg ) # for cg
      self.progressbar.update( numpy.log(arg) )

def solve_system( matrix, rhs, title='solving system', symmetric=False, tol=0, maxiter=99999 ):
  'solve linear system iteratively'

  assert matrix.shape[:-1] == rhs.shape
  if not tol:
    print 'solving system, condition number:', numpy.linalg.cond( matrix )
    return numpy.linalg.solve( matrix, rhs )

  from scipy.sparse import linalg
  solver = linalg.cg if symmetric else linalg.gmres
  lhs, status = solver( matrix, rhs,
                        callback=title and Iter_CallBack(tol,'%s [%s:%d]' % (title,symmetric and 'CG' or 'GMRES', matrix.shape[0]) ),
                        tol=tol,
                        maxiter=maxiter )
  assert status == 0, 'solution failed to converge'
  return lhs

def solve( A, b=None, constrain=None, lconstrain=None, rconstrain=None, **kwargs ):
  'solve'

  assert A.ndim == 2
  if constrain is None and lconstrain is None and rconstrain is None:
    assert b is not None
    return solve_system( A, b, **kwargs )

  rcons = numpy.ones( A.shape[0], dtype=bool )
  lcons = NanVec( A.shape[1] )
  if constrain is not None:
    rcons[:constrain.size] = numpy.isnan( constrain )
    lcons[:constrain.size] = constrain
  if lconstrain is not None:
    lcons[:lconstrain.size] = lconstrain
  if rconstrain is not None:
    assert isinstance( rconstrain, numpy.ndarray ) and rconstrain.dtype == bool
    rcons[:rconstrain.size] = rconstrain

  where = numpy.isnan( lcons )
  matrix = A[ numpy.ix_(rcons,where) ]
  rhs = -numpy.dot( A[ numpy.ix_(rcons,~where) ], lcons[~where] )
  if b is not None:
    tmp = b[ rcons[:b.size] ]
    rhs[:tmp.size] += tmp
  lhs = lcons.view( numpy.ndarray )
  lhs[ where ] = solve_system( matrix, rhs, **kwargs )
  return lhs

def transform( arr, trans, axis ):
  'transform one axis by matrix multiplication'

  if trans is 1:
    return arr

  if isinstance( trans, (float,int) ):
    return arr * trans

# assert trans.ndim == 2
# return numpy.dot( arr.swapaxes(axis,-1), trans ).swapaxes(axis,-1)

  if axis < 0:
    axis += arr.ndim

  if arr.shape[axis] == 1:
    trans = trans.sum(0)[numpy.newaxis]

  assert arr.shape[axis] == trans.shape[0]

  transformed = numpy.tensordot( arr, trans, [axis,0] )
  if trans.ndim > 1 and axis != arr.ndim-1:
    order = range(axis) + range(arr.ndim-trans.ndim+1,arr.ndim) + range(axis,arr.ndim-trans.ndim+1)
    transformed = transformed.transpose( order )

  return transformed

def inv( arr, axes ):
  'linearized inverse'

  L = map( numpy.arange, arr.shape )

  ax1, ax2 = sorted( ax + arr.ndim if ax < 0 else ax for ax in axes ) # ax2 > ax1
  L.pop( ax2 )
  L.pop( ax1 )

  indices = list( numpy.ix_( *L ) )
  indices.insert( ax1, slice(None) )
  indices.insert( ax2, slice(None) )

  invarr = numpy.empty_like( arr )
  for index in numpy.broadcast( *indices ):
    invarr[index] = numpy.linalg.inv( arr[index] )

  return invarr

def arraymap( f, dtype, *args ):
  'call f for sequence of arguments and cast to dtype'

  return numpy.array( map( f, args[0] ) if len( args ) == 1
                 else [ f( *arg ) for arg in numpy.broadcast( *args ) ], dtype=dtype )

def det( A, ax1, ax2 ):
  'determinant'

  assert isinstance( A, numpy.ndarray )
  ax1, ax2 = sorted( ax + A.ndim if ax < 0 else ax for ax in (ax1,ax2) ) # ax2 > ax1
  assert A.shape[ax1] == A.shape[ax2]
  T = range(A.ndim)
  T.pop(ax2)
  T.pop(ax1)
  T.extend([ax1,ax2])
  A = A.transpose( T )
  if A.shape[-1] == 2:
    det = A[...,0,0] * A[...,1,1] - A[...,0,1] * A[...,1,0]
  else:
    det = numpy.empty( A.shape[:-2] )
    for I in numpy.broadcast( *numpy.ix_( *[ range(n) for n in A.shape[:-2] ] ) ) if A.ndim > 3 else range( A.shape[0] ):
      det[I] = numpy.linalg.det( A[I] )
  return det

def reshape( A, *shape ):
  'more useful reshape'

  newshape = []
  i = 0
  for s in shape:
    if isinstance( s, (tuple,list) ):
      assert numpy.product( s ) == A.shape[i]
      newshape.extend( s )
      i += 1
    elif s == 0:
      newshape.extend( A.shape[i:] )
      i = A.ndim
    elif s == 1:
      newshape.append( A.shape[i] )
      i += 1
    else:
      iprev, i = i, i+s if s > 0 else A.ndim
      newshape.append( numpy.product( A.shape[iprev:i] ) )
  assert i == A.ndim
  return A.reshape( newshape )

def mean( A, weights=None, axis=-1 ):
  'generalized mean'

  return A.mean( axis ) if weights is None else transform( A, weights / weights.sum(), axis )

def norm2( A, axis=-1 ):
  'L2 norm over specified axis'

  return numpy.sqrt( contract( A, A, axis ) )

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

class progressbar( object ):
  'progress bar class'

  def __init__( self, iterable=None, n=0, title='iterating' ):
    'constructor'

    self.iterable = iterable
    self.n = n or len( iterable )
    self.x = 0
    self.t0 = time.time()
    sys.stdout.write( title + ' ' )
    sys.stdout.flush()
    self.length = LINEWIDTH - len(title)

  def __iter__( self ):
    'iterate'

    for i, item in enumerate( self.iterable ):
      self.update( i )
      yield item

  def update( self, i ):
    'update'

    x = self.length if self.n == 1 else int( (i+1) * self.length ) // (self.n+1)
    if self.x < x <= self.length:
      sys.stdout.write( '-' * (x-self.x) )
      sys.stdout.flush()
      self.x = x

  def __del__( self ):
    'destructor'

    sys.stdout.write( '-' * (self.length-self.x) )
    dt = '%.2f' % ( time.time() - self.t0 )
    dts = dt[1:] if dt[0] == '0' else \
          dt[:3] if len(dt) <= 6 else \
          '%se%d' % ( dt[0], len(dt)-3 )
    sys.stdout.write( ' %s\n' % dts )
    sys.stdout.flush()

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

class StdOut( object ):
  'stdout wrapper'

  def __init__( self, stdout, html ):
    'constructor'

    self.html = html
    self.html.write( '<html><body><pre>' )
    self.html.flush()
    self.stdout = stdout
    self.linebuf = ''

  def write( self, s ):
    'write string'

    self.stdout.write( s )

    for line in s.splitlines( True ):
      self.html.write( line )
      if not line.endswith( '\n' ):
        self.linebuf = line
        continue
      if self.linebuf:
        line = self.linebuf + line
        self.linebuf = ''
      if '.png' in line:
        for word in line.split():
          if word.endswith( '.png' ):
            self.html.write( '\n<a href="%s"><img src="%s" width="435pt" border="1px"/></a>\n\n' % ( word, word ) )
    self.html.flush()

  def flush( self ):
    'flush'

    self.stdout.flush()
    self.html.flush()

def run( *functions ):
  'call function specified on command line'

  assert functions
  args = sys.argv[1:]
  if '-h' in args or '--help' in args:
    print 'Usage: %s [OPTIONS]' % sys.argv[0]
    print
    print '  -h    --help         Display this help.'
    print '  -p P  --parallel=P   Select number of processors.'
    print '  -f F  --function=F   Select function.'
    for i, func in enumerate( functions ):
      print
      print 'Arguments for -f %s%s' % ( func.func_name, '' if i else ' (default)' )
      print
      for kwarg, default in getkwargdefaults( func ):
        tmp = '--%s=%s' % ( kwarg.lower(), kwarg[0].upper() )
        print >> sys.stderr, '  %-20s Default: %s' % ( tmp, default )
    return

  for index, arg in enumerate( args ):
    if arg == '-p' or arg.startswith( '--parallel=' ):
      args.pop( index )
      import parallel
      parallel.nprocs = int( args.pop( index ) if arg == '-p' else arg[11:] )
      break

  for index, arg in enumerate( args ):
    if arg == '-f' or arg.startswith( '--function=' ):
      args.pop( index )
      funcname = args.pop( index ) if arg == '-f' else arg[11:]
      break
  else:
    funcname = functions[0].func_name

  for func in functions:
    if func.func_name == funcname:
      break
  else:
    print 'error: invalid function name: %s' % funcname
    return

  kwargs = dict( getkwargdefaults( func ) )
  for arg in args:
    if arg[:2] != '--' or '=' not in arg:
      print 'error: function arguments must be of type --key=value'
      return
    key, value = arg[2:].split( '=', 1 )
    for kwarg, default in kwargs.iteritems():
      if kwarg.lower() == key.lower():
        break
    else:
      print 'error: invalid argument for %s: %s' % ( funcname, key )
      return
    try:
      value = eval( value )
    except:
      pass
    kwargs[ kwarg ] = value

  title = '%s.%s' % ( sys.argv[0].split('/')[-1].lower(), funcname.lower() )

  path = time.strftime( '%Y-%m-%d/%H-%M-%S-%%s/' ) % title
  os.makedirs( BASEPATH + path )
  if os.path.islink( DUMPDIR ):
    os.remove( DUMPDIR )
  os.symlink( path, DUMPDIR )
  output = open( BASEPATH + path + 'index.html', 'w' )

  sys.stdout = StdOut( sys.stdout, output )

  print title, ( ' ' + time.ctime() ).rjust( LINEWIDTH-len(title), '=' ), '|>|'
  for arg, val in kwargs.items():
    print '.'.rjust( len(title) ), '%s = %s' % ( arg.lower(), val )

  try:
    func( **kwargs )
  finally:
    print ( ' ' + time.ctime() ).rjust( LINEWIDTH+1, '=' ), '|<|'

  output.write( '</body></html>' )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
