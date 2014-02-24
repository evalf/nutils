import sys

_nodefault = object()
def prop( name, default=_nodefault ):
  frame = sys._getframe(1)
  key = '__%s__' % name
  while frame:
    if key in frame.f_locals:
      return frame.f_locals[key]
    frame = frame.f_back
  assert default is not _nodefault, 'property %r is not defined' % name
  return default
