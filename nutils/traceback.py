from . import core
import sys, cmd, re, os, linecache

class Frame( object ):
  'frame info'

  def __init__( self, frame, lineno=None ):
    'constructor'

    self.frame = frame
    self.lineno = lineno or self.frame.f_lineno

  @staticmethod
  def _name( frame ):
    # If frame is a class method try to add the class name
    # http://stackoverflow.com/questions/2203424/python-how-to-retrieve-class-information-from-a-frame-object/15704609#15704609
    name = frame.f_code.co_name
    for classname, obj in frame.f_globals.iteritems():
      try:
        assert obj.__dict__[name].func_code is frame.f_code
      except:
        pass
      else:
        return '%s.%s' % ( classname, name )
    return name

  @property
  @core.cache
  def context( self ):
    path = self.frame.f_code.co_filename
    name = self._name( self.frame )
    context = '  File "%s", line %d, in %s' % ( os.path.relpath(path), self.lineno, name )
    lineno = self.lineno
    line = linecache.getline( path, lineno )
    indent = len(line) - len(line.lstrip())
    context += '\n    ' + line[indent:].rstrip()
    while context.endswith( '\\' ):
      lineno += 1
      line = linecache.getline( path, lineno )
      context += '\n    ' + line[indent:].rstrip()
    return context

  @property
  @core.cache
  def source( self ):
    path = self.frame.f_code.co_filename
    lineno = self.frame.f_code.co_firstlineno
    line = linecache.getline( path, lineno )
    if not line:
      return '<not avaliable>'
    indent = len(line) - len(line.lstrip())
    source = line[indent:]
    pointing = False
    while True:
      lineno += 1
      line = linecache.getline( path, lineno )
      if not line or ( line.strip() and line[:indent+1].strip() ):
        break
      if pointing or lineno == self.lineno:
        source += '>' + line[indent+1:]
        pointing = line.rstrip().endswith( '\\' )
      else:
        source += line[indent:]
    return source.rstrip() # strip trailing empty lines

  def __str__( self ):
    'string representation'

    return self.context

class TracebackExplorer( cmd.Cmd ):
  'traceback explorer'

  def __init__( self, exc, frames, intro ):
    'constructor'

    cmd.Cmd.__init__( self, completekey='tab' )

    lines = [ 'WELCOME TO TRACEBACK EXPLORER.' ]
    maxlen = 44
    nextline = ''
    for word in intro.split():
      if not nextline or len(nextline) + 1 + len(word) > maxlen:
        lines.append( nextline )
        nextline = word
      else:
        nextline += ' ' + word
    lines.append( nextline )
    rule = '+-' + '-' * maxlen + '-+'
    self.intro = '\n'.join( [ rule ] + [ '| %s |' % line.ljust(maxlen) for line in lines ] + [ rule ] )

    self.exc = exc
    self.frames = frames
    self.index = len(frames) - 1
    self.prompt = '\n>>> '

  def show_context( self ):
    'show traceback up to index'

    for i, f in enumerate(self.frames):
      print ' *'[i == self.index] + f.context[1:]
    print ' ', self.exc

  def do_s( self, arg ):
    '''Show source code of the currently focussed frame.'''

    print self.frames[self.index].source

  def do_l( self, arg ):
    '''List the stack and exception type'''

    self.show_context()

  def do_q( self, arg ):
    '''Quit traceback exploror.'''

    print 'quit.'
    return True

  def do_u( self, arg ):
    '''Shift focus to the frame above the current one.'''

    if self.index > 0:
      self.index -= 1
      self.show_context()

  def do_d( self, arg ):
    '''Shift focus to the frame below the current one.'''

    if self.index < len(self.frames)-1:
      self.index += 1
      self.show_context()

  def do_w( self, arg ):
    '''Show overview of local variables.'''

    frame = self.frames[self.index].frame
    maxlen = max( len(name) for name in frame.f_locals )
    fmt = '  %' + str(maxlen) + 's : %s'
    for item in frame.f_locals.iteritems():
      print fmt % item

  def do_p( self, arg ):
    '''Print local of global variable, or function evaluation.'''

    frame = self.frames[self.index].frame
    print eval(arg,frame.f_globals,frame.f_locals)

  def onecmd( self, text ):
    'wrap command handling to avoid a second death'

    try:
      return cmd.Cmd.onecmd( self, text )
    except Exception, e:
      print '%s in %r:' % ( e.__class__.__name__, text ), e

  def do_pp( self, arg ):
    '''Pretty-print local of global variable, or function evaluation.'''

    import pprint
    frame = self.frames[self.index].frame
    pprint.pprint( eval(arg,frame.f_globals,frame.f_locals) )

  def completedefault( self, text, line, begidx, endidx ):
    'complete object names'

    frame = self.frames[self.index].frame

    objs = {}
    objs.update( frame.f_globals )
    objs.update( frame.f_locals )

    base = ''
    while '.' in text:
      objname, attr = text.split( '.', 1 )
      try:
        obj = objs[ objname ]
      except KeyError:
        return []
      objs = {}
      for attrname in dir(obj):
        try:
          objs[attrname] = getattr(obj,attrname)
        except:
          pass
      base += objname + '.'
      text = attr

    return [ base+name for name in objs if name.startswith(text) ]

def traceback():
  'constructor'

  tb = sys.exc_traceback
  frames = []
  while tb:
    frames.append( Frame( tb.tb_frame, tb.tb_lineno ) )
    tb = tb.tb_next
  return frames

def format_exc():
  return '\n'.join( [ repr(sys.exc_value) ] + [ str(f) for f in traceback() ] )

def callstack( depth=1 ):
  frame = sys._getframe( depth )
  frames = []
  while frame:
    frames.append( Frame( frame ) )
    frame = frame.f_back
  return frames[::-1]

def write_html( out, exc, frames ):
  'write exception info to html file'

  out.write( '<span class="info">' )
  out.write( '\n<hr/>' )
  out.write( '<b>EXHAUSTIVE POST-MORTEM DUMP FOLLOWS</b>\n' )
  out.write( '\n'.join( [ repr(exc) ] + [ str(f) for f in frames ] ) )
  out.write( '<hr/>\n' )
  for f in reversed(frames):
    out.write( f.context.splitlines()[0] + '\n' )
    for line in f.source.splitlines():
      if line.startswith( '>' ):
        fmt = '<span class="error"> %s</span>\n'
        line = line[1:]
      else:
        fmt = '%s\n'
      line = re.sub( r'\b(def|if|elif|else|for|while|with|in|return)\b', r'<b>\1</b>', line.replace('<','&lt;').replace('>','&gt;') )
      out.write( fmt % line )
    out.write( '\n\n' )
    out.write( '<table border="1" style="border:none; margin:0px; padding:0px;">\n' )
    for key, val in f.frame.f_locals.iteritems():
      try:
        val = str(val).replace('<','&lt;').replace('>','&gt;')
      except:
        val = 'ERROR'
      out.write( '<tr><td>%s</td><td>%s</td></tr>\n' % ( key, val ) )
    out.write( '</table>\n' )
    out.write( '<hr/>\n' )
  out.write( '</span>' )
  out.flush()

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
