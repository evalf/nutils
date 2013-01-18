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
      self.setmax( iterable )
    else:
      self.setmax( len(iterable) )

    self.index = 0
    self.x = 0
    self.t0 = time.time()
    self.length = getattr( prop, 'linewidth', 50 )
    self.out = getattr( prop, 'verbose', None ) >= 3 and getattr( prop, 'html', sys.stdout )
    self.add( title )

  def setmax( self, n ):
    'set maximum pbar value'

    self.n = n

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

  def write( self, s ):
    'write string'

    s = str(s)
    if not s:
      return

    self.out.write( s )
    self.out.flush()
    self.x += len(s)

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
    self.write( '-' * (x-self.x) )

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

  def __init__( self, title, htmlfile, stdout=sys.stdout ):
    'constructor'

    self.basedir = os.path.dirname( htmlfile )
    self.html = open( htmlfile, 'w' )
    self.html.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
    self.html.write( '<html><head>\n' )
    self.html.write( '<title>{}</title>\n'.format(title) )
    self.html.write( '<script type="text/javascript" src="../../../../../viewer.js" ></script>\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../style.css">\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../custum.css">\n' )
    self.html.write( '</head><body><pre>\n' )
    self.html.write( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html">latest %s</a> | <a class="nav_latestall" href="../../../../../log.html">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>\n\n' % title.split()[0] )
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
    return r'<a href="%s" class="plot">%s</a>' % (name,name)

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
      self.html.write( '</pre></body></html>\n' )
      self.html.flush()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
