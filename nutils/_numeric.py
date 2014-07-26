import warnings

warnings.warn( '''Failed to load the _numeric C module.
Falling back on equivalent Python implementation.''', stacklevel=2 )

def _contract( A, B, axes ):
  assert A.shape == B.shape and axes > 0
  return ((A*B).reshape(A.shape[:-axes]+(-1,))).sum(-1)

