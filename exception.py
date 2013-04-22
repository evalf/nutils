import sys, cmd, re, os

class FrameInfo( object ):
  'frame info'

  def __init__( self, context, source, frame ):
    'constructor'

    self.context = context
    self.source = source
    self.frame = frame

  def __str__( self ):
    'string representation'

    return self.context

class ExcInfo( object ):
  'gather exception info'

  def __init__( self, excinfo=None ):
    'constructor'

    if isinstance( excinfo, ExcInfo ):
      self.tb = excinfo.tb
      self.exctype = excinfo.exctype
      self.excvalue = excinfo.excvalue
      return

    if excinfo is None:
      excinfo = sys.exc_info()

    self.exctype, self.excvalue, tb = excinfo
  
    filecache = {}
    self.tb = []

    while tb:
      frame = tb.tb_frame
      code = frame.f_code
      name = code.co_name
  
      # http://stackoverflow.com/questions/2203424/python-how-to-retrieve-class-information-from-a-frame-object/15704609#15704609
      for classname, obj in frame.f_globals.iteritems():
        try:
          assert obj.__dict__[name].func_code is code
        except:
          pass
        else:
          name = '%s.%s' % ( classname, name )
          break
  
      path = os.path.relpath( code.co_filename )
      if path in filecache:
        lines = filecache[path]
      else:
        try:
          fileobj = open(path)
        except IOError:
          lines = None
        else:
          lines = fileobj.readlines()
          filecache[path] = lines
  
      lineno = tb.tb_lineno - 1
      context = 'File "%s", line %d, in %s' % ( path, lineno+1, name )
      if lines:
        indent = len(lines[lineno]) - len(lines[lineno].lstrip())
        counterr = 0
        while True:
          context += '\n    ' + lines[lineno+counterr][indent:].rstrip()
          counterr += 1
          if not context.endswith( '\\' ):
            break
        iline = code.co_firstlineno-1
        indent = len(lines[iline]) - len(lines[iline].lstrip())
        source = lines[iline][indent:].rstrip()
        while iline+1 < len(lines) and ( iline < lineno or not lines[iline+1][:indent+1].strip() ):
          iline += 1
          if lineno<=iline<lineno+counterr:
            source += '\n>' + lines[iline][indent+1:].rstrip()
          else:
            source += '\n' + lines[iline][indent:].rstrip()
        source = source.rstrip()
      else:
        source = '<not avaliable>'
        
      self.tb.append( FrameInfo(context,source,frame) )
      tb = tb.tb_next

  def __len__( self ):
    'number of items in traceback'

    return len( self.tb )

  def __getitem__( self, item ):
    'index / slice traceback'

    tb = self.tb[item]
    if isinstance( tb, FrameInfo ):
      return tb

    excinfo = object.__new__( ExcInfo )
    excinfo.tb = tb
    excinfo.exctype = self.exctype
    excinfo.excvalue = self.excvalue
    return excinfo

  def __str__( self ):
    'string representation'

    return '%s: %s' % ( self.exctype.__name__, str(self.excvalue) or '<empty>' )

  __repr__ = __str__
  
  def write_html( self, out ):
    'write exception info to html file'

    out.write( '<span class="info">' )
    out.write( '\n<hr/>' )
    out.write( '<b>EXHAUSTIVE POST-MORTEM DUMP FOLLOWS</b>\n' )
    out.write( str(self) )
    out.write( '<hr/>\n' )
    for f in reversed(self):
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

  def summary( self ):
    'simple traceback akin to python default'

    return [ '  %s\n' % f for f in self ] + [ '%s\n' % self ]

  def explore( self ):
    'start traceback explorer'

    TracebackExplorer( self ).cmdloop()

class TracebackExplorer( cmd.Cmd ):
  'traceback explorer'

  intro = '''
  +----------------------------------------------+
  | WELCOME TO TRACEBACK EXPLORER.               |
  |                                              |
  | Your program has died. The traceback exporer |
  | allows you to examine its post-mortem state  |
  | to figure out why this happened. Type 'help' |
  | for an overview of commands to get going.    |
  +----------------------------------------------+'''

  def __init__( self, excinfo ):
    'constructor'

    cmd.Cmd.__init__( self, completekey='tab' )

    self.excinfo = excinfo
    self.index = len(excinfo) - 1
    self.prompt = '\n>>> '

  def show_context( self ):
    'show traceback up to index'

    for i, f in enumerate(self.excinfo):
      print ' *'[i == self.index], f.context
    print ' ', self.excinfo

  def do_s( self, arg ):
    '''Show source code of the currently focussed frame.'''

    print self.excinfo[self.index].source

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

    if self.index < len(self.excinfo)-1:
      self.index += 1
      self.show_context()

  def do_w( self, arg ):
    '''Show overview of local variables.'''

    frame = self.excinfo[self.index].frame
    maxlen = max( len(name) for name in frame.f_locals )
    fmt = '  %' + str(maxlen) + 's : %s'
    for item in frame.f_locals.iteritems():
      print fmt % item

  def do_p( self, arg ):
    '''Print local of global variable, or function evaluation.'''

    frame = self.excinfo[self.index].frame
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
    frame = self.excinfo[self.index].frame
    pprint.pprint( eval(arg,frame.f_globals,frame.f_locals) )

  def completedefault( self, text, line, begidx, endidx ):
    'complete object names'

    frame = self.excinfo[self.index].frame

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

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
