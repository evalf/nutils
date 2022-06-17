# Integrals

A central operation in any Finite Element application is to integrate a
function over a physical domain. In Nutils, integration starts with the
topology, in particular the `integral()` method.

The integral method takes a `nutils.function.Array` function as first argument
and the degree as keyword argument. The function should contain the Jacobian of
the geometry against which the function should be integrated, using either
`nutils.function.J` or `dV` in a namespace expression (assuming the jacobian
has been added to the namespace using `ns.define_for(..., jacobians=('dV',
'dS'))`). For example, the following integrates `1` against geometry `x`:

```python
I = topo.integral('1 dV' @ ns, degree=0)
I
# Array<>
```

The resulting `nutils.function.Array` object is a representation of the
integral, as yet unevaluated. To compute the actual numbers, call the
`Array.eval()` method:

```python
I.eval()
# 1.0±1e-15
```

Be careful with including the Jacobian in your integrands.  The following two
integrals are different:

```python
topo.integral('(1 + 1) dV' @ ns, degree=0).eval()
# 2.0±1e-15
topo.integral('1 + 1 dV' @ ns, degree=0).eval()
# 5.0±1e-15
```

Like any other `nutils.function.Array`, the integrals can be added or
subtracted:

```python
J = topo.integral('x_0 dV' @ ns, degree=1)
(I+J).eval()
# 1.5±1e-15
```

Recall that a topology boundary is also a `nutils.topology.Topology` object,
and hence it supports integration.  For example, to integrate the geometry `x`
over the entire boundary, write

```python
topo.boundary.integral('x_0 dS' @ ns, degree=1).eval()
# 1.0±1e-15
```

To limit the integral to the right boundary, write

```python
topo.boundary['right'].integral('x_0 dS' @ ns, degree=1).eval()
# 1.0±1e-15
```

Note that this boundary is simply a point and the integral a point evaluation.

Integrating and evaluating a 1D `nutils.function.Array` results in a 1D
`numpy.ndarray`:

```python
>>> topo.integral('basis_i dV' @ ns, degree=1).eval()
array([0.125, 0.25 , 0.25 , 0.25 , 0.125])±1e-15
```

Since the integrals of 2D `nutils.function.Array` functions are usually sparse,
the `Array.eval() <nutils.function.Array.eval>` method does not return a dense
`numpy.ndarray`, but a Nutils sparse matrix object: a subclass of
`nutils.matrix.Matrix`.  Nutils interfaces several linear solvers (more on this
in Section [solvers](solvers.md) below) but if you want to use a custom solver
you can export the matrix to a dense, compressed sparse row or coordinate
representation via the `Matrix.export()` method.  An example:

```python
M = topo.integral('∇_i(basis_m) ∇_i(basis_n) dV' @ ns, degree=1).eval()
M.export('dense')
# array([[ 4., -4.,  0.,  0.,  0.],
#        [-4.,  8., -4.,  0.,  0.],
#        [ 0., -4.,  8., -4.,  0.],
#        [ 0.,  0., -4.,  8., -4.],
#        [ 0.,  0.,  0., -4.,  4.]])±1e-15
M.export('csr') # (data, column indices, row pointers) # doctest: +NORMALIZE_WHITESPACE
# (array([ 4., -4., -4.,  8., -4., -4.,  8., -4., -4.,  8., -4., -4.,  4.])±1e-15,
#  array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4])±1e-15,
#  array([ 0,  2,  5,  8, 11, 13])±1e-15)
M.export('coo') # (data, (row indices, column indices)) # doctest: +NORMALIZE_WHITESPACE
# (array([ 4., -4., -4.,  8., -4., -4.,  8., -4., -4.,  8., -4., -4.,  4.])±1e-15,
#  (array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])±1e-15,
#   array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4])±1e-15))
```
