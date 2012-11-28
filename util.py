import sys, os, time, numpy, cPickle, hashlib, weakref
from numpyextra import *

LINEWIDTH = 50
BASEPATH = os.path.expanduser( '~/public_html/' )
DUMPDIR = BASEPATH + time.strftime( '%Y/%m/%d/%H-%M-%S/' )

def getpath( pattern ):
  'create file in DUMPDIR'

  if pattern == pattern.format( 0 ):
    return DUMPDIR + pattern
  prefix = pattern.split( '{' )[0]
  names = [ name for name in os.listdir( DUMPDIR ) if name.startswith(prefix) ]
  n = len(names)
  while True:
    n += 1
    newname = DUMPDIR + pattern.format( n )
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
      print 'iteration %d' % i
    yield i
    if i == nmax:
      break

def weakcacheprop( func ):
  'weakly cached property'

  key = func.func_name
  def wrapped( self ):
    value = self.__dict__.get( key )
    value = value and value()
    if value is None:
      value = func( self )
      self.__dict__[ key ] = weakref.ref(value)
    return value

  return property( wrapped )

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

def transform( arr, trans, axis ):
  'transform one axis by matrix multiplication'

  if trans is 1:
    return arr

  trans = numpy.asarray( trans )
  if trans.ndim == 0:
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
  return numpy.asarray( det ).view( A.__class__ )

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

  return numpy.asarray( numpy.sqrt( contract( A, A, axis ) ) )

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
        assert sh == shape[d+i] or sh == 1, 'incompatible shapes: %s' % ' & '.join( str(f.shape) for f in funcs )
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

class ProgressBar( object ):
  'progress bar class'

  def __init__( self ):
    'constructor'

    self.x = 0
    self.t0 = time.time()
    self.length = LINEWIDTH
    self.endtext = ''

  def add( self, text ):
    'add text'

    self.length -= len(text) + 1
    sys.stdout.write( text + ' ' )
    sys.stdout.flush()

  def add_to_end( self, text ):
    'add to after progress bar'

    self.length -= len(text) + 1
    self.endtext += ' ' + text

  def bar( self, iterable, n=None ):
    'iterate'

    if n is None:
      n = len( iterable )
    for i, item in enumerate( iterable ):
      self.update( i, n )
      yield item

  def update( self, i, n ):
    'update'

    x = int( (i+1) * self.length ) // (n+1)
    if self.x < x <= self.length:
      sys.stdout.write( '-' * (x-self.x) )
      sys.stdout.flush()
      self.x = x

  def __del__( self ):
    'destructor'

    sys.stdout.write( '-' * (self.length-self.x) )
    sys.stdout.write( self.endtext )
    dt = '%.2f' % ( time.time() - self.t0 )
    dts = dt[1:] if dt[0] == '0' else \
          dt[:3] if len(dt) <= 6 else \
          '%se%d' % ( dt[0], len(dt)-3 )
    sys.stdout.write( ' %s\n' % dts )
    sys.stdout.flush()

def progressbar( iterable, title='iterating' ):
  'show progressbar while iterating'

  progress = ProgressBar()
  progress.add( title )
  return progress.bar( iterable )

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

  HEAD = '''\
<html>
<head>
<script type='application/javascript'>

var i_focus = 0; // currently focused anchor element
var anchors; // list of all anchors (ordered by height)
var focus; // = anchors[i_focus] after first mouse move
var preview; // preview div element
var Y = 0; // current mouse height relative to window

findclosest = function () {
  y = Y + document.body.scrollTop - anchors[0].offsetHeight / 2;
  var dy = y - anchors[i_focus].offsetTop;
  if ( dy > 0 ) {
    for ( var i = i_focus; i < anchors.length-1; i++ ) {
      var yd = anchors[i+1].offsetTop - y;
      if ( yd > 0 ) return i + ( yd < dy );
      dy = -yd;
    }
    return anchors.length - 1;
  }
  else {
    for ( var i = i_focus; i > 0; i-- ) {
      var yd = anchors[i-1].offsetTop - y;
      if ( yd < 0 ) return i - ( yd > dy );
      dy = -yd;
    }
    return 0;
  }
}

refocus = function () {
  // update preview image if necessary
  var newfocus = anchors[ findclosest() ];
  if ( focus ) {
    if ( focus == newfocus ) return;
    focus.classList.remove( 'highlight' );
    focus.classList.remove( 'loading' );
  }
  focus = newfocus;
  focus.classList.add( 'loading' );
  newobj = document.createElement( 'img' );
  newobj.setAttribute( 'width', '600px' );
  newobj.onclick = function () { document.location.href=focus.getAttribute('href'); };
  newobj.onload = function () {
    preview.innerHTML='';
    preview.appendChild(this);
    focus.classList.add( 'highlight' )
    focus.classList.remove( 'loading' );
  };
  newobj.setAttribute( 'src', focus.getAttribute('href') );
}

window.onload = function() {
  // set up anchor list, preview pane, document events
  nodelist = document.getElementsByTagName('a');
  anchors = []
  for ( i = 0; i < nodelist.length; i++ ) {
    var url = nodelist[i].getAttribute('href');
    var ext = url.split('.').pop();
    var idx = ['png','svg','jpg','jpeg'].indexOf(ext);
    if ( idx != -1 ) anchors.push( nodelist[i] );
  }
  if ( anchors.length == 0 ) return;
  preview = document.createElement( 'div' );
  preview.setAttribute( 'id', 'preview' );
  document.body.appendChild( preview );
  document.onmousemove = function (event) { Y=event.clientY; refocus(); };
  document.onscroll = refocus;
};

</script>
<style>

body {
  padding: 10px;
  margin: 0px;
}

a {
  text-decoration: none;
  color: blue;
}

a.loading {
  color: green;
}

a.highlight {
  color: red;
}

p.pre {
  white-space: pre;
  font-family: monospace;
  padding: 2px;
  margin: 0px;
}

#preview {
  position: fixed;
  top: 10px;
  right: 10px;
  border: 1px solid gray;
  padding: 0px;
}

</style>
</head>
<body>
<p class="pre">'''

  TAIL = '''\
</p>
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
      return r'<a href="%s">%s</a>' % (name,name)
    return r'<a href="%s">%s</a>' % (name,name)

  def write( self, s ):
    'write string'

    self.stdout.write( s )
    self.html.write( self.pattern.sub( self.filerep, s ).replace('\n','</p>\n<p class="pre">' ) )
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
