from ._base import BackendNotAvailable

def setassemble(sets):
  try:
    from ._mkl import setassemble
  except BackendNotAvailable:
    try:
      from ._scipy import setassemble
    except BackendNotAvailable:
      from ._numpy import setassemble
  return setassemble(sets)
