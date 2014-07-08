# -*- coding: utf8 -*-
#
# Module LOG
#
# Part of Nutils: open source numerical utilities for Python. Jointly developed
# by HvZ Computational Engineering, TU/e Multiscale Engineering Fluid Dynamics,
# and others. More info at http://nutils.org <info@nutils.org>. (c) 2014

"""
The log module provides print methods ``debug``, ``info``, ``user``,
``warning``, and ``error``, in increasing order of priority. Output is sent to
stdout as well as to an html formatted log file if so configured.
"""

from __future__ import print_function
from . import prop, debug
import sys, time, os, warnings, numpy, re

_KEY = '__logger__'


def _findlogger( frame=None ):
  'find logger in call stack'

  if frame is None:
    frame = sys._getframe(1)
  while frame:
    logger = frame.f_locals.get(_KEY)
    if logger:
      return logger
    frame = frame.f_back
  return SimpleLog


warnings.showwarning = lambda message, category, filename, lineno, *args: warning( '%s: %s\n  In %s:%d' % ( category.__name__, message, filename, lineno ) )


def stack( msg, frames=None ):
  'print stack trace'

  if frames is None:
    frames = debug.callstack( depth=2 )
  stream = getstream( attr='error', frame=frames[-1].frame )
  print( msg, *reversed(frames), sep='\n', file=stream )


def SimpleLog( chunks=('',), attr=None ):
  'just write to stdout'
  
  sys.stdout.write( ' > '.join( chunks ) )
  return sys.stdout


def _path2href( match ):
  whitelist = ['.jpg','.png','.svg','.txt'] + getattr( prop, 'plot_extensions', [] )
  filename = match.group(0)
  ext = match.group(1)
  return '<a href="%s">%s</a>' % (filename,filename) if ext not in whitelist \
    else '<a href="%s" name="%s" class="plot">%s</a>' % (filename,filename,filename)


class HtmlStream( object ):
  'html line stream'

  def __init__( self, chunks, attr, html ):
    'constructor'

    self.out = SimpleLog( chunks, attr=attr )
    self.attr = attr
    self.head = ' &middot; '.join( chunks )
    self.body = ''
    self.html = html

  def write( self, text ):
    'write to out and buffer for html'

    self.out.write( text )
    self.body += text.replace( '<', '&lt;' ).replace( '>', '&gt;' )

  def __del__( self ):
    'postprocess buffer and write to html'

    body = self.body
    if self.attr == 'path':
      body = re.sub( r'\b\w+([.]\w+)\b', _path2href, body )
    if self.attr:
      body = '<span class="%s">%s</span>' % ( self.attr, body )
    line = '<span class="line">%s</span>' % ( self.head + body )

    self.html.write( line )
    self.html.flush()


class HtmlLog( object ):
  'html log'

  def __init__( self, fileobj, title, depth=1 ):
    'constructor'

    self.html = fileobj
    self.html.write( '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "DTD/xhtml1-strict.dtd">\n' )
    self.html.write( '<html><head>\n' )
    self.html.write( '<title>%s</title>\n' % title )
    self.html.write( '<script type="text/javascript" src="../../../../../viewer.js" ></script>\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../style.css">\n' )
    self.html.write( '<link rel="stylesheet" type="text/css" href="../../../../../custom.css">\n' )
    self.html.write( '</head><body><pre>\n' )
    self.html.write( '<span id="navbar">goto: <a class="nav_latest" href="../../../../log.html">latest %s</a> | <a class="nav_latestall" href="../../../../../log.html">latest overall</a> | <a class="nav_index" href="../../../../../">index</a></span>\n\n' % title.split()[0] )
    self.html.flush()

    sys._getframe(depth).f_locals[_KEY] = self

  def __call__( self, chunks=('',), attr=None ):
    return HtmlStream( chunks, attr, self.html )

  def __del__( self ):
    'destructor'

    self.html.write( '</pre></body></html>\n' )
    self.html.close()
    

class ContextLog( object ):
  'base class'

  def __init__( self, depth=1 ):
    'constructor'

    frame = sys._getframe(depth)

    parent = _findlogger( frame )
    while isinstance( parent, ContextLog ) and not parent.__enabled:
      parent = parent.parent

    self.parent = parent
    self.__enabled = True

    frame.f_locals[_KEY] = self

  def __call__( self, chunks=('',), attr=None ):
    if self.__enabled:
      chunks = (self.text,) + chunks
    return self.parent( chunks, attr=attr )

  def disable( self ):
    'disable this logger'

    self.__enabled = False

  def __repr__( self ):
    return '%s(%s)' % ( self.__class__.__name__, self )

  def __str__( self ):
    return '%s > %s' % ( self.parent, self.text ) if self.__enabled else str(self.parent)


class StaticContextLog( ContextLog ):
  'simple text logger'

  def __init__( self, text, depth=1 ):
    'constructor'

    self.text = text
    ContextLog.__init__( self, depth=depth+1 )


class ProgressContextLog( ContextLog ):
  'progress bar'

  def __init__( self, text, iterable=None, target=None, showpct=True, depth=1 ):
    'constructor'

    self.msg = text
    self.showpct = showpct
    self.tint = getattr(prop,'progress_interval',1.)
    self.tmax = getattr(prop,'progress_interval_max',numpy.inf)
    self.texp = getattr(prop,'progress_interval_scale',2.)
    self.t0 = time.time()
    self.tnext = self.t0 + min( self.tint, self.tmax )
    self.iterable = iterable
    self.target = len(iterable) if target is None else target
    self.current = 0
    ContextLog.__init__( self, depth=depth+1 )

  def __iter__( self ):
    try:
      for i, item in enumerate( self.iterable ):
        self.update( i )
        yield item
    finally:
      self.disable()

  def update( self, current ):
    'update progress'

    self.current = current
    if time.time() > self.tnext:
      print( file=self(()) )

  @property
  def text( self ):
    'get text'

    self.tint = min( self.tint*self.texp, self.tmax )
    self.tnext = time.time() + self.tint
    pbar = self.msg + ' %.0f/%.0f' % ( self.current, self.target )
    if self.showpct:
      pct = 100 * self.current / float(self.target)
      pbar += ' (%.0f%%)' % pct
    return pbar


# historically grown
context = StaticContextLog
progress = iterate = ProgressContextLog
setup_html = HtmlLog


def getstream( attr=None, frame=None ):
  logger = _findlogger(frame)
  return logger( attr=attr )

def _mklog( attr, frame=None ):
  return lambda *args: print( *args, file=getstream(attr,frame) )

path    = _mklog( 'path'    )
error   = _mklog( 'error'   )
warning = _mklog( 'warning' )
user    = _mklog( 'user'    )
info    = _mklog( 'info'    )
debug   = _mklog( 'debug'   )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
