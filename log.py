from . import core, prop
import sys, time, os

error     = lambda *args: TextLog( 0, None, *args )
warning   = lambda *args: TextLog( 1, None, *args )
info      = lambda *args: TextLog( 2, None, *args )
debug     = lambda *args: TextLog( 9, None, *args )
iter      = lambda *args, **kwargs: IterLog( 3, None, *args, **kwargs )
enumerate = lambda *args, **kwargs: IterLog( 3, None, *args, enum=True, **kwargs )

class Log( object ):
  'log object'

  error     = lambda self, *args: TextLog( 0, self, *args )
  warning   = lambda self, *args: TextLog( 1, self, *args )
  info      = lambda self, *args: TextLog( 2, self, *args )
  debug     = lambda self, *args: TextLog( 9, self, *args )
  iter      = lambda self, *args, **kwargs: IterLog( 3, self, *args, **kwargs )
  enumerate = lambda self, *args, **kwargs: IterLog( 3, self, *args, enum=True, **kwargs )

  def __init__( self, level, parent=None ):
    'constructor'

    self.parent = parent
    self.out = None if level >= getattr( prop, 'verbose', None ) \
          else getattr( prop, 'html', sys.stdout )

  def getparenttext( self ):
    'get parent text'

    text = '%d' % os.getpid() if self.parent is None \
      else self.parent.gettext()
    return text + ' > '

  def display( self ):
    'display text'

    if self.out:
      self.out.write( self.gettext() + '\n' )
      self.out.flush()

class TextLog( Log ):
  'simple text logger'

  def __init__( self, level, parent, *args ):
    'constructor'

    self.text = ' '.join( map( str, args ) )
    Log.__init__( self, level, parent )
    self.display()

  def gettext( self ):
    'get text'

    return self.getparenttext() + self.text

class IterLog( Log ):
  'progress bar'

  def __init__( self, level, parent, text, iterable=None, target=None, nchar=20, tint=1, texp=2, enum=False ):
    'constructor'

    self.text = text
    self.iterable = iterable
    self.tint = tint
    self.texp = texp
    self.t0 = time.time()
    self.tnext = self.t0 + self.tint
    self.target = target if target is not None else len(iterable)
    self.current = 0
    self.nchar = nchar
    self.enum = enum
    Log.__init__( self, level, parent )

  def __iter__( self ):
    'iterate'
  
    current = 0
    for item in self.iterable:
      yield item if not self.enum else (current,item)
      current += 1
      self.update( current )

  def update( self, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      self.display()

  def gettext( self ):
    'show'

    self.tint *= self.texp
    self.tnext = time.time() + self.tint
    text = self.getparenttext()
    text += self.text
    text += ' %.0f/%.0f' % ( self.current, self.target )
    pct = self.current / float(self.target)
    nblk = int( pct * self.nchar )
    bar = ( '%.0f%%' % (100*pct) ).center( self.nchar, '.' )
    text += ' [%s]' % ( bar[:nblk].replace('.','#') + bar[nblk:] )
    return text

# DEPRECATED

progress = info

class ProgressBar( IterLog ):
  'temporary construct for backwards compatibility'

  @core.deprecated( old='ProgressBar(n,title)', new='progressbar(title,n)' )
  def __init__( self, n, title ):
    'constructor'

    if isinstance( n, int ):
      iterable = None
      target = n
    else:
      iterable = n
      target = len(n)
    IterLog.__init__( self, 3, None, title, iterable=iterable, target=target )

  def add( self, text ):
    'add to text'

    self.text += ' %s' % text

  def close( self ):
    'does nothing'

    pass


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
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../custom.css">\n' )
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
