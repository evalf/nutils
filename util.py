import sys, os, time, numpy, cPickle, hashlib
from numpyextra import *

LINEWIDTH = 50
BASEPATH = os.path.expanduser( '~/public_html/' )
DUMPDIR = BASEPATH + time.strftime( '%Y-%m-%d/%H-%M-%S/' )

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

def iterate( nmax=-1 ):
  'iterate forever'

  i = 0
  while True:
    i += 1
    print 'iteration %d' % i
    yield i
    if i == nmax:
      break

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

def solve_system( matrix, rhs, title=True, symmetric=False, tol=0, maxiter=999999 ):
  'solve linear system iteratively'

  assert matrix.shape[:-1] == rhs.shape

  if title is True:
    title = 'solve system'

  progress = None
  if not tol:
    if title:
      progress = progressbar( title=title, n=1 )
    #print 'solving system:'#, condition number:', numpy.linalg.cond( matrix )
    lhs = numpy.linalg.solve( matrix, rhs )
  else:
    from scipy.sparse import linalg
    callback = None
    if title:
      progress = progressbar( n=numpy.log(tol), title='%s [%s:%d]' % ( title, symmetric and 'CG' or 'GMRES', matrix.shape[0] ) )
      clock = Clock( .1 )
      callback = lambda vec: clock and progress.update(
        numpy.log( vec if not symmetric else numpy.linalg.norm( rhs - matrix * vec ) ) )
    solver = linalg.cg if symmetric else linalg.gmres
    lhs, status = solver( matrix, rhs, callback=callback, tol=tol, maxiter=maxiter )
    assert status == 0, 'solution failed to converge (status=%d)' % status
  if progress:
    progress.finish()
  return lhs

def solve( A, b=0, constrain=None, lconstrain=None, rconstrain=None, tol=0, **kwargs ):
  'solve'

  b = numpy.asarray( b )
  if ( b == 0 ).all():
    b = None

  assert A.ndim == 2
  if constrain is None and lconstrain is None and rconstrain is None:
    assert b is not None
    return solve_system( A, b, **kwargs )

  rcons = numpy.ones( A.shape[0], dtype=bool )
  lcons = numpy.empty( A.shape[1] )
  lcons[:] = numpy.nan
  if constrain is not None:
    rcons[:constrain.size] = numpy.isnan( constrain )
    lcons[:constrain.size] = constrain
  if lconstrain is not None:
    lcons[:lconstrain.size] = lconstrain
  if rconstrain is not None:
    assert isinstance( rconstrain, numpy.ndarray ) and rconstrain.dtype == bool
    rcons[:rconstrain.size] = rconstrain

  if tol == 0 and not isinstance( A, numpy.ndarray ):
    A = A.toarray()

  where = numpy.isnan( lcons )
  if isinstance( A, numpy.ndarray ):
    matrix = A[ numpy.ix_(rcons,where) ]
    rhs = -numpy.dot( A[ numpy.ix_(rcons,~where) ], lcons[~where] )
  else:
    from scipy.sparse import linalg
    tmpvec = numpy.zeros( A.shape[1] )
    def matvec( v ):
      tmpvec[where] = v
      return ( A * tmpvec )[rcons]
    shape = numpy.sum(rcons), numpy.sum(where)
    matrix = linalg.LinearOperator( matvec=matvec, shape=shape, dtype=float )
    lcons[where] = 0
    rhs = -( A * lcons )[rcons]

  assert matrix.shape[0] == matrix.shape[1]
  if b is not None:
    tmp = b[ rcons[:b.size] ]
    rhs[:tmp.size] += tmp

  lcons[ where ] = solve_system( matrix, rhs, tol=tol, **kwargs )
  return lcons

def transform( arr, trans, axis ):
  'transform one axis by matrix multiplication'

  if trans is 1:
    return arr

  if isinstance( trans, (float,int) ) or trans.ndim == 0:
    return arr * trans

  if axis < 0:
    axis += arr.ndim

  if arr.shape[axis] == 1:
    trans = trans.sum(0)[numpy.newaxis]

  assert arr.shape[axis] == trans.shape[0]

  s1 = [ slice(None) ] * arr.ndim
  s1[axis+1:axis+1] = [ numpy.newaxis ] * (trans.ndim-1)
  s1 = tuple(s1)

  s2 = [ numpy.newaxis ] * (arr.ndim-1)
  s2[axis:axis] = [ slice(None) ] * trans.ndim
  s2 = tuple(s2)

  return contract( arr[s1], trans[s2], axis )

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
    elif s == 1:
      newshape.append( A.shape[i] )
      i += 1
    else:
      assert s > 1
      newshape.append( numpy.product( A.shape[i:i+s] ) )
      i += s
  assert i <= A.ndim
  newshape.extend( A.shape[i:] )
  return A.reshape( newshape )

def mean( A, weights=None, axis=-1 ):
  'generalized mean'

  return A.mean( axis ) if weights is None else transform( A, weights / weights.sum(), axis )

def fail( msg, *args ):
  'generate exception'

  raise Exception, msg % args

def norm2( A, axis=-1 ):
  'L2 norm over specified axis'

  return numpy.sqrt( contract( A, A, axis ) )

def align_arrays( *funcs ):
  'align shapes'

  shape = []
  for f in funcs:
    d = len(shape) - len(f.shape)
    if d < 0:
      shape = list(f.shape[:-d]) + shape
      d = 0
    for i, sh in enumerate( f.shape ):
      if shape[d+i] == 1:
        shape[d+i] = sh
      else:
        assert sh == shape[d+i] or sh == 1
  ndim = len(shape)
  return tuple(shape), tuple( f if ndim == f.ndim
                         else f[ (numpy.newaxis,)*(ndim-f.ndim) + (slice(None),) * f.ndim ] for f in funcs )

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
    self.finish()

  def update( self, i ):
    'update'

    x = self.length if self.n == 1 else int( (i+1) * self.length ) // (self.n+1)
    if self.x < x <= self.length:
      sys.stdout.write( '-' * (x-self.x) )
      sys.stdout.flush()
      self.x = x

  def finish( self ):
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

# var timeout;
# $(window).focus(function (){
#   timeout = timedRefresh(1000);
# })
# .blur(function() {
#   window.clearTimeout(timeout);
# })
# .load(function (){
#   timeout = timedRefresh(1000);
# });
# function timedRefresh(timeoutPeriod) {
#   return window.setTimeout("location.reload(true);",timeoutPeriod);
# }
# 
# $(window).scroll(function(){
#   if ($(window).scrollTop() == $(window).height()){
#     location.reload(true);
#   }
# });

  HEAD = '''\
<html>
<head>
<script type="text/javascript" src="http://code.jquery.com/jquery-1.8.0.min.js"></script>
<script type="text/javascript">

this.imagePreview = function(){ 
  $("a.preview").hover(function(e){
    $("body").append("<p id='preview'><img src='"+ this.href +"' alt='Image preview' width='600' /></p>");                
    $("#preview").fadeIn("fast");            
  },
  function(){
    this.title = this.t;  
    $("#preview").remove();
  }); 
};

// starting the script on page load
$(document).ready(function(){
  imagePreview();
});

</script>
<style>

a{
  text-decoration: none;
}

#preview {
  position: fixed;
  top: 10px;
  right: 10px;
  border: 1px solid gray;
  padding: 0px;
  display: none;
}

</style>
</head>
<body>
<pre>
'''

  TAIL = '''\
</pre>
</body>
</html>
'''

  def __init__( self, stdout, html ):
    'constructor'

    self.html = html
    self.html.write( self.HEAD )
    #self.html.write( '<a href="../%s">permanent</a>\n' % os.readlink(DUMPDIR) )
    self.html.flush()
    self.stdout = stdout

    import re
    self.pattern = re.compile( r'\b(\w+[.]\w+)\b' )

  @staticmethod
  def filerep( match ):
    'replace file occurrences'

    name = match.group(0)
    path = DUMPDIR + name
    if not os.path.isfile( path ):
      return name
    if name.endswith('.png'):
      return r'<a href="%s" class="preview">%s</a>' % (name,name)
    return r'<a href="%s">%s</a>' % (name,name)

  def write( self, s ):
    'write string'

    self.stdout.write( s )
    self.html.write( self.pattern.sub( self.filerep, s ) )
    self.html.flush()

  def flush( self ):
    'flush'

    self.stdout.flush()
    self.html.flush()

  def __del__( self ):
    'destructor'

    self.html.write( self.TAIL )

def run( *functions ):
  'call function specified on command line'

  assert functions
  args = sys.argv[1:]
  if '-h' in args or '--help' in args:
    print 'Usage: %s [FUNC] [ARGS]' % sys.argv[0]
    print
    print '  -h    --help         Display this help.'
    print '  -p P  --parallel=P   Select number of processors.'
    for i, func in enumerate( functions ):
      print
      print 'Arguments for %s%s' % ( func.func_name, '' if i else ' (default)' )
      print
      for kwarg, default in getkwargdefaults( func ):
        tmp = '--%s=%s' % ( kwarg.lower(), kwarg[0].upper() )
        print >> sys.stderr, '  %-20s Default: %s' % ( tmp, default )
    return

  if args and not args[0].startswith( '-' ):
    funcname = args.pop(0)
  else:
    funcname = functions[0].func_name

  for index, arg in enumerate( args ):
    if arg == '-p' or arg.startswith( '--parallel=' ):
      args.pop( index )
      import parallel
      parallel.nprocs = int( args.pop( index ) if arg == '-p' else arg[11:] )
      break

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

  LINK = BASEPATH + 'latest'
  os.makedirs( DUMPDIR )
  if os.path.islink( LINK ):
    os.remove( LINK )
  os.symlink( DUMPDIR, LINK )
  output = open( DUMPDIR + 'index.html', 'w' )

  sys.stdout = StdOut( sys.stdout, output )

  print title, ( ' ' + time.ctime() ).rjust( LINEWIDTH-len(title), '=' ), '|>|'
  for arg, val in kwargs.items():
    print '.'.rjust( len(title) ), '%s = %s' % ( arg.lower(), val )

  try:
    func( **kwargs )
  finally:
    print ( ' ' + time.ctime() ).rjust( LINEWIDTH+1, '=' ), '|<|'

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
