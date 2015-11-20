from nutils import *
from . import register, unittest


@register
def indexedarray():

  domain, geom = mesh.rectilinear( [3,3,3] )
  basis = domain.basis( 'spline', degree=3 )

  a = geom
  b = geom**2
  ab_outer = a[:,_]*b[_,:]
  c = basis.grad(geom)

  @unittest
  def outer_product():
    assert (a['i']*b['j']).unwrap() == a[:,_]*b[_,:]

  @unittest
  def inner_product():
    assert (a['i']*b['i']).unwrap() == function.dot(a, b, axes=[0])

  @unittest
  def matvec():
    assert (c['ij']*b['j']).unwrap() == function.dot(c, b[_,:], axes=[1])

  @unittest
  def transpose():
    assert (ab_outer['ij']+ab_outer['ji']).unwrap() == ab_outer+function.transpose(ab_outer)

  @unittest
  def neg():
    assert (-a['i']).unwrap() == -a

  @unittest
  def add():
    assert (a['i'] + b['i']).unwrap() == a + b

  @unittest
  def sub():
    assert (a['i'] - b['i']).unwrap() == a - b

  @unittest
  def mul_scalar():
    assert (2 * a['i']).unwrap() == 2 * a

  @unittest( raises=ValueError )
  def div():
    a['i'] / b['i']

  @unittest
  def div_scalar():
    assert (a['i'] / 2).unwrap() == a / 2

  @unittest( raises=ValueError )
  def not_enough_indices():
    c['i']

  @unittest( raises=ValueError )
  def too_many_indices():
    c['ijk']

  @unittest( raises=ValueError )
  def triple_index():
    c.grad(geom)['iii']
