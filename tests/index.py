from nutils import *
from . import register, unittest


@register
def indexedarray():

  domain, geom = mesh.rectilinear( [3,3,3] )
  basis = domain.basis( 'spline', degree=3 )
  surface = function.concatenate([geom, [1]])

  a = geom
  b = geom**2
  ab_outer = a[:,_]*b[_,:]
  c = basis.grad(geom)
  d = basis
  e = basis.vector(domain.ndims)

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

  @unittest
  def repeated_indices():
    assert ab_outer['ii'].unwrap() == function.trace( ab_outer )

  @unittest( raises=ValueError )
  def not_enough_indices():
    c['i']

  @unittest( raises=ValueError )
  def too_many_indices():
    c['ijk']

  @unittest( raises=ValueError )
  def triple_index():
    c.grad(geom)['iii']

  @unittest
  def grad1():
    assert d['i,j'].unwrap(geom) == d.grad(geom)

  @unittest
  def grad2():
    assert d['i'][',j'].unwrap(geom) == d.grad(geom)

  @unittest
  def grad3():
    assert d['i,jk'].unwrap(geom) == d.grad(geom).grad(geom)

  @unittest
  def grad4():
    assert e['ij,kj'].unwrap(geom) == function.trace( e.grad(geom).grad(geom), 1, 3 )

  @unittest( raises=ValueError )
  def grad_triple_index1():
    d['i,ii'].unwrap(geom)

  @unittest( raises=ValueError )
  def grad_triple_index2():
    d['i'][',ii'].unwrap(geom)

  @unittest
  def surfgrad():
    assert d['i;j'].unwrap(geom) == d.grad(geom, -1)

  @unittest
  def domain_integrate():
    x = domain.integrate( d['i,k']*d['j,k'], geometry=geom, ischeme='gauss3' )
    y = domain.integrate( function.outer( d.grad(geom) ).sum(2), geometry=geom, ischeme='gauss3' )
    numpy.testing.assert_almost_equal( x.toscipy().todense(), y.toscipy().todense() )

  @unittest
  def surface_integrate():
    x = domain.integrate( d['i;k']*d['j;k'], geometry=surface, ischeme='gauss3' )
    y = domain.integrate( function.outer( d.grad(surface, -1) ).sum(2), geometry=surface, ischeme='gauss3' )
    numpy.testing.assert_almost_equal( x.toscipy().todense(), y.toscipy().todense() )

  @unittest
  def number1():
    assert b['0'].unwrap() == b[0]

  @unittest
  def number2():
    assert ab_outer['01'].unwrap() == ab_outer[0,1]

  @unittest
  def grad_number1():
    assert a['i,0'].unwrap(geom) == a.grad(geom)[:,0]

  @unittest
  def grad_number2():
    assert a['i,01'].unwrap(geom) == a.grad(geom).grad(geom)[:,0,1]

  @unittest
  def reindex():
    assert a['j,i']['kj'].unwrap(geom) == a.grad(geom)

  @unittest
  def reindex_number():
    assert a['i,j']['j0'].unwrap(geom) == a.grad(geom)[:,0]

  @unittest
  def reindex_repeated_indices():
    assert a['i,j']['ii'].unwrap(geom) == function.trace( a.grad(geom) )

  @unittest
  def reindex_with_grad():
    assert a['i,j']['ij,k'].unwrap(geom) == a.grad(geom).grad(geom)

  @unittest( raises=ValueError )
  def reindex_not_enough_indices():
    a['i,j']['i']

  @unittest( raises=ValueError )
  def reindex_too_many_indices():
    a['i,j']['ijk']

  @unittest( raises=ValueError )
  def reindex_triple_index():
    a['i,j']['iii']

  @unittest
  def shapes1():
    (function.eye['ij']*a['i,j']).unwrap(geom) == function.trace( a.grad(geom) )

  @unittest
  def shapes2():
    (a['i']*function.eye['ij']*function.eye['jk']*function.eye['kl']*function.normal['l']).unwrap(geom) == function.dot( a, geom.normal(), 0 )

  @unittest
  def shapes3():
    (function.eye['ij'] + a['i']*a['j']).unwrap() == function.eye( len(geom) ) + function.outer( a )

  @unittest( raises=ValueError )
  def shape_mismatch():
    # `d` and `normal` have different lengths
    (d['i']*function.eye['ij']*function.normal['j']).unwrap(geom)

  @unittest( raises=ValueError )
  def undetermined_shape1():
    # shape of `eye` cannot be determined
    function.eye['ij'].unwrap(geom)

  @unittest( raises=ValueError )
  def undetermined_shape2():
    # shape of `eye` cannot be determined
    function.eye['ii'].unwrap(geom)

  @unittest( raises=ValueError )
  def undetermined_shape3():
    # shape of product of `eye`s cannot be determined
    (function.eye['ij'] * function.eye['jk']).unwrap(geom)

  @unittest
  def opposite():
    function.opposite(d['i']).unwrap() == function.opposite(d)
