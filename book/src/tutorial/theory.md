# A Little Bit of Theory

We will introduce fundamental Nutils concepts based on the 1D homogeneous
Laplace problem,

\\[ u''(x) = 0 \\]

with boundary conditions \\( u(0) = 0 \\) and \\( u'(1) = 1 \\). Even though
the solution is trivially found to be \\( u(x) = x \\), the example serves to
introduce many key concepts in the Nutils paradigm, concepts that can then be
applied to solve a wide class of physics problems.

## Weak Form

A key step to solving a problem using the Finite Element Method is to cast it
into weak form.

Let \\( Ω \\) be the unit line \\( [0,1] \\) with boundaries \\( Γ_\text{left}
\\) and \\( Γ_\text{right} \\), and let \\( H_0(Ω) \\) be a suitable function
space such that any \\( u ∈ H_0(Ω) \\) satisfies \\( u = 0 \\) in \\(
Γ_\text{left} \\). The Laplace problem is solved uniquely by the element \\( u
∈ H_0(Ω) \\) for which \\( R(v, u) = 0 \\) for all test functions \\( v ∈
H_0(Ω) \\), with \\( R \\) the bilinear functional

\\[ R(v, u) := ∫_Ω \frac{∂v}{∂x_i} \frac{∂u}{∂x_i} \ dV - ∫_{Γ_\text{right}} v \ dS. \\]

## Discrete Solution

The final step before turning to code is to make the problem discrete.

To restrict ourselves to a finite dimensional subspace we adopt a set of Finite
Element basis functions \\( φ_n ∈ H_0(Ω) \\). In this space, the Finite Element
solution is established by solving the linear system of equations \\(
R_n(\hat{u}) = 0 \\), with residual vector \\( R_n(\hat{u}) := R(φ_n, \hat{u})
\\), and discrete solution

\\[ \hat{u}(x) = φ_n(x) \hat{u}_n. \\]

Note that discretization inevitably implies approximation, i.e. \\( u ≠ \hat{u}
\\) in general. In this case, however, we choose \\( \{φ_n\} \\) to be the
space of piecewise linears, which contains the exact solution. We therefore
expect our Finite Element solution to be exact.
