from . import prop
import sys, time, os

error    = lambda *args, **kwargs: log( 0, *args, **kwargs )
warning  = lambda *args, **kwargs: log( 1, *args, **kwargs )
info     = lambda *args, **kwargs: log( 2, *args, **kwargs )
progress = lambda *args, **kwargs: log( 3, *args, **kwargs )
debug    = lambda *args, **kwargs: log( 9, *args, **kwargs )

def log( level, *args, **kwargs ):
  'log text (modeled after python3 print)'

  if level > getattr( prop, 'verbose', None ):
    return False

  sep = kwargs.pop( 'sep', ' ' )
  end = kwargs.pop( 'end', '\n' )
  out = kwargs.pop( 'file', getattr( prop, 'html', sys.stdout ) )
  assert not kwargs, 'invalid log argument: %s=%s' % kwargs.popitem()
  out.write( sep.join( map( str, args ) ) + end )
  out.flush()

  return True

class ProgressBar( object ):
  'progress bar class'

  def __init__( self, iterable, title ):
    'constructor'

    try:
      self.iterable = iter(iterable)
    except TypeError:
      self.iterable = None
      self.n = iterable
    else:
      self.n = len(iterable)

    self.index = 0
    self.x = 0
    self.t0 = time.time()
    self.length = getattr( prop, 'linewidth', 50 )
    self.out = getattr( prop, 'verbose', None ) >= 3 and getattr( prop, 'html', sys.stdout )
    self.add( title )

  def add( self, text ):
    'add text'

    if not self.out:
      return

    self.length -= len(text) + 1
    self.out.write( text + ' ' )
    self.out.flush()

  def __iter__( self ):
    'iterate'

    return self.iterator() if self.out else self.iterable

  def iterator( self ):
    'iterate'

    for item in self.iterable:
      self.update()
      yield item
    self.close()

  def update( self, index=None ):
    'update'

    if not self.out:
      return

    if index is None:
      self.index += 1
      index = self.index
    else:
      self.index = index

    x = int( (index+1) * self.length ) // (self.n+1)
    if not self.x < x <= self.length:
      return

    self.out.write( '-' * (x-self.x) )
    self.out.flush()
    self.x = x

  def close( self ):
    'destructor'

    if not self.out:
      return

    dt = '%.2f' % ( time.time() - self.t0 )
    dts = dt[1:] if dt[0] == '0' else \
          dt[:3] if len(dt) <= 6 else \
          '%se%d' % ( dt[0], len(dt)-3 )
    self.out.write( '-' * (self.length-self.x) + ' ' + dts + '\n' )

class HtmlWriter( object ):
  'html writer'

  html = None

  def __init__( self, htmlfile, stdout=sys.stdout ):
    'constructor'

    self.basedir = os.path.dirname( htmlfile )
    self.html = open( htmlfile, 'w' )
    self.html.write( HTMLHEAD )
    if 'public_html/' in htmlfile:
      import pwd
      username = pwd.getpwuid( os.getuid() ).pw_name
      permanent = '/~%s/%s' % ( username, htmlfile.split('public_html/',1)[1] )
    else:
      permanent = 'file://%s' % htmlfile
    self.html.write( '<a href="%s">[permalink]</a>\n\n' % permanent )
    self.html.flush()
    self.stdout = stdout

    import re
    self.pattern = re.compile( r'\b(\w+[.]\w+)\b' )

  def filerep( self, match ):
    'replace file occurrences'

    name = match.group(0)
    path = os.path.join( self.basedir, name )
    if not os.path.isfile( path ):
      return name
    return r'<a href="%s">%s</a>' % (name,name)

  def write( self, s ):
    'write string'

    self.stdout.write( s )
    self.html.write( self.pattern.sub( self.filerep, s ) )

  def flush( self ):
    'flush'

    self.stdout.flush()
    self.html.flush()

  def __del__( self ):
    'destructor'

    if self.html is not None:
      self.html.write( HTMLFOOT )

HTMLHEAD = '''\
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
  newobj.setAttribute( 'width', '520px' );
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

a { text-decoration: none; color: blue; }
a.loading { color: green; }
a.highlight { color: red; }

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
<pre>'''

HTMLFOOT = '''\
</pre>
</body>
</html>
'''

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
