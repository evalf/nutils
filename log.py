from . import core, prop
import sys, time, os

_KEY = '__logger__'
_ERROR, _WARNING, _PATH, _INFO, _PROGRESS, _DEBUG = range(6)

_makestr = lambda args: ' '.join( str(arg) for arg in args )

def _findlogger():
  'find logger in call stack'

  frame = sys._getframe(1)
  while frame:
    if _KEY in frame.f_locals:
      return frame.f_locals[_KEY]
    frame = frame.f_back
  return SimpleLog()

info    = lambda *args: _findlogger().write( _INFO,    _makestr(args) )
path    = lambda *args: _findlogger().write( _PATH,    _makestr(args) )
error   = lambda *args: _findlogger().write( _ERROR,   _makestr(args) )
warning = lambda *args: _findlogger().write( _WARNING, _makestr(args) )

def context( *args, **kwargs ):
  'context'

  level = kwargs.pop( 'level', 0 )
  assert not kwargs
  sys._getframe(level+1).f_locals[_KEY] = ContextLog( _makestr(args) )

def popcontext( level=0 ):
  'pop context'

  frame = sys._getframe(level+1)
  frame.f_locals[_KEY] = frame.f_locals[_KEY].parent

def iterate( text, iterable, target=None, **kwargs ):
  'iterate'
  
  logger = ProgressLog( text, target if target is not None else len(iterable), **kwargs )
  try:
    frame = sys._getframe(1).f_locals[_KEY] = logger
    for i, item in enumerate( iterable ):
      yield item
      logger.update( level=_PROGRESS, current=i )
  finally:
    frame = sys._getframe(1).f_locals[_KEY] = logger.parent

class SimpleLog( object ):
  'simple log'

  def __init__( self ):
    'constructor'

    self.out = getattr( prop, 'html', sys.stdout )

  def write( self, level, *text ):
    'write'

    print ' > '.join( text )

def setup_html( maxlevel, path, title, depth=0 ):
  'setup html logging'

  sys._getframe(depth+1).f_locals[_KEY] = HtmlLog( maxlevel, path, title )

class HtmlLog( object ):
  'html log'

  def __init__( self, maxlevel, path, title ):
    'constructor'

    self.html = open( path, 'w' )
    self.html.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
    self.html.write( '<html><head>\n' )
    self.html.write( '<title>{}</title>\n'.format(title) )
    self.html.write( '<script type="text/javascript" src="../../../../../viewer.js" ></script>\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../style.css">\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../custom.css">\n' )
    self.html.write( '</head><body><pre>\n' )
    self.html.write( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html">latest %s</a> | <a class="nav_latestall" href="../../../../../log.html">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>\n\n' % title.split()[0] )
    self.html.flush()
    self.maxlevel = maxlevel

  def __del__( self ):
    'destructor'

    self.html.write( '</pre></body></html>\n' )
    self.html.flush()

  def write( self, level, *text ):
    'write'

    if level <= self.maxlevel:
      print ' > '.join( text )
    if level == _PATH:
      text = text[:-1] + ( '<a href="%s" class="plot">%s</a>' % (text[-1],text[-1]), )
    self.html.write( ' > '.join(text) + '\n' )
    self.html.flush()

class ContextLog( object ):
  'simple text logger'

  def __init__( self, text ):
    'constructor'

    self.text = text
    self.parent = _findlogger()

  def write( self, level, *text ):
    'write'

    self.parent.write( level, self.text, *text )

class ProgressLog( object ):
  'progress bar'

  def __init__( self, text, target, nchar=20, tint=1, texp=2 ):
    'constructor'

    self.text = text
    self.tint = tint
    self.texp = texp
    self.t0 = time.time()
    self.tnext = self.t0 + self.tint
    self.target = target
    self.current = 0
    self.nchar = nchar
    self.parent = _findlogger()

  def update( self, level, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      self.write( level )

  def write( self, level, *text ):
    'write'

    self.tint *= self.texp
    self.tnext = time.time() + self.tint
    pbar = self.text + ' %.0f/%.0f' % ( self.current, self.target )
    pct = self.current / float(self.target)
    nblk = int( pct * self.nchar )
    bar = ( '%.0f%%' % (100*pct) ).center( self.nchar, '.' )
    pbar += ' [%s]' % ( bar[:nblk].replace('.','#') + bar[nblk:] )
    self.parent.write( level, pbar, *text )

# DEPRECATED

progress = info

class ProgressBar( ProgressLog ):
  'temporary construct for backwards compatibility'

  @core.deprecated( old='ProgressBar(n,text)', new='iterate(text,n)' )
  def __init__( self, n, title ):
    'constructor'

    if isinstance( n, int ):
      iterable = None
      target = n
    else:
      iterable = n
      target = len(n)
    ProgressLog.__init__( self, 3, None, title, iterable=iterable, target=target )

  def add( self, text ):
    'add to text'

    self.text += ' %s' % text

  def close( self ):
    'does nothing'

    pass

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
