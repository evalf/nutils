from nutils import *
from . import register, unittest


def _unittest_raises( exception ):
  def wrapper1( func ):
    def wrapper2():
      try:
        func()
      except exception:
        pass
      except:
        assert False, 'incorrect exception raised'
      else:
        assert False, 'no exception raised'
    wrapper2.__name__ = func.__name__
    return unittest( wrapper2 )
  return wrapper1


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

  @_unittest_raises( ValueError )
  def div():
    a['i'] / b['i']

  @unittest
  def div_scalar():
    assert (a['i'] / 2).unwrap() == a / 2

  @_unittest_raises( ValueError )
  def not_enough_indices():
    c['i']

  @_unittest_raises( ValueError )
  def too_many_indices():
    c['ijk']

  @_unittest_raises( ValueError )
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

  @_unittest_raises( ValueError )
  def grad_triple_index1():
    d['i,ii'].unwrap(geom)

  @_unittest_raises( ValueError )
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
