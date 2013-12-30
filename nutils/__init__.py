import numpy, sys

class _Prop( object ):
  prefix = '__property__'
  def __getattr__( self, attr ):
    frame = sys._getframe(1)
    key = self.prefix + attr
    while frame:
      if key in frame.f_locals:
        return frame.f_locals[key]
      frame = frame.f_back
    raise AttributeError, attr
  def __setattr__( self, attr, value ):
    frame = sys._getframe(1)
    key = self.prefix + attr
    frame.f_locals[key] = value
  def __str__( self ):
    props = {}
    frame = sys._getframe(1)
    n = len(self.prefix)
    while frame:
      props.update( (key[n:],value) for key, value in frame.f_locals.iteritems() if key[:n] == self.prefix and key[n:] not in props )
      frame = frame.f_back
    return '\n . '.join( [ 'properties:' ] + [ '%s: %s' % item for item in props.iteritems() ] )

prop = _Prop()
_ = numpy.newaxis

__all__ = [
  '_',
  'prop',
  'numpy',
  'core',
  'numeric',
  'element',
  'function',
  'mesh',
  'plot',
  'library',
  'topology',
  'util',
  'matrix',
  'parallel',
  'log',
  'debug',
]
