# Solvers

Using topologies, bases and integrals, we now have the tools in place to start
performing some actual functional-analytical operations. We start with what is
perhaps the simplest of its kind, the least squares projection, demonstrating
the different implementations now available to us and working our way up from
there.

Taking the geometry component \\( x_0 \\) as an example, to project it onto the
basis \\( \{φ_n\} \\) means finding the coefficients \\( \hat{u}_n \\) such
that

\\[ \left(\int_Ω φ_n φ_m \ dV\right) \hat u_m = \int_Ω φ_n x_0 \ dV \\]

for all \\( φ_n \\), or \\( A_{nm} \hat{u}_m = f_n \\). This is implemented as
follows:

```python
A = topo.integral('basis_m basis_n dV' @ ns, degree=2).eval()
f = topo.integral('basis_n x_0 dV' @ ns, degree=2).eval()
A.solve(f)
# solve > solving 5 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 3e-17±1e-15
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

Alternatively, we can write this in the slightly more general form

\\[ R_n := \int_Ω φ_n (u - x_0) \ dV = 0. \\]

```python
res = topo.integral('basis_n (u - x_0) dV' @ ns, degree=2)
```

Taking the derivative of \\( R_n \\) to \\( \hat{u}_m \\) gives the above
matrix \\( A_{nm} \\), and substituting for \\( \hat{u} \\) the zero vector
yields \\( -f_n \\).  Nutils can compute those derivatives for you, using the
method `Array.derivative()` to compute the derivative with respect to an
`nutils.function.Argument`, returning a new `nutils.function.Array`.

```python
A = res.derivative('lhs').eval()
f = -res.eval(lhs=numpy.zeros(5))
A.solve(f)
# solve > solving 5 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 3e-17±1e-15
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

The above three lines are so common that they are combined in the function
`nutils.solver.solve_linear`:

```python
solver.solve_linear('lhs', res)
# solve > solving 5 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 3e-17±1e-15
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

We can take this formulation one step further.  Minimizing

\\[ S := \int_Ω (u - x_0)^2 \ dV \\]

for \\( \hat{u} \\) is equivalent to the above two variants.  The derivative of
\\( S \\) to \\( \hat{u}_n \\) gives \\( 2 R_n \\):

```python
sqr = topo.integral('(u - x_0)^2 dV' @ ns, degree=2)
solver.solve_linear('lhs', sqr.derivative('lhs'))
# solve > solving 5 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 6e-17±1e-15
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

The optimization problem can also be solved by the
`nutils.solver.optimize` function, which has the added benefit that
\\( S \\) may be nonlinear in \\( \hat{u} \\) --- a property not used here.

```python
solver.optimize('lhs', sqr)
# optimize > solve > solving 5 dof system to machine precision using arnoldi solver
# optimize > solve > solver returned with residual 0e+00±1e-15
# optimize > optimum value 0.00e+00±1e-15
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

Nutils also supports solving a partial optimization problem.  In the Laplace
problem stated above, the Dirichlet boundary condition at \\( Γ_\text{left} \\)
minimizes the following functional:

```python
sqr = topo.boundary['left'].integral('(u - 0)^2 dS' @ ns, degree=2)
```

By passing the `droptol` argument, `nutils.solver.optimize` returns an
array with `nan` ('not a number') for every entry for which the optimization
problem is invariant, or to be precise, where the variation is below
`droptol`:

```python
cons = solver.optimize('lhs', sqr, droptol=1e-15)
# optimize > constrained 1/5 dofs
# optimize > optimum value 0.00e+00
cons
# array([ 0., nan, nan, nan, nan])±1e-15
```

Consider again the Laplace problem stated above. The
[residual](theory.md#weak-form) is implemented as

```python
res = topo.integral('∇_i(basis_n) ∇_i(u) dV' @ ns, degree=0)
res -= topo.boundary['right'].integral('basis_n dS' @ ns, degree=0)
```

Since this problem is linear in argument `lhs`, we can use the
`nutils.solver.solve_linear` method to solve this problem.  The constraints
`cons` are passed via the keyword argument `constrain`:

```python
lhs = solver.solve_linear('lhs', res, constrain=cons)
# solve > solving 4 dof system to machine precision using arnoldi solver
# solve > solver returned with residual 9e-16±1e-15
lhs
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])±1e-15
```

For nonlinear residuals you can use `nutils.solver.newton`.
