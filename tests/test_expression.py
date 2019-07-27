import nutils.expression
from nutils.testing import *

_ = lambda arg: (None, arg)

class Array:
  def __init__(self, text, shape):
    self.text = text
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
  def __len__(self):
    return self.shape[0]
  def __str__(self):
    return self.text
  def __repr__(self):
    return self.text
  def __eq__(self, other):
    return type(self) == type(other) and self.text == other.text
  def __hash__(self):
    return hash(self.text)

class Variables:
  def __init__(self, x, altgeom, funcoverride):
    self.x = x
    self.altgeom = altgeom
    self.funcoverride = funcoverride
    self._lengths = {str(i): i for i in range(10)}
  def __getitem__(self, name):
    if not name.startswith('_'):
      try:
        return getattr(self, name)
      except AttributeError:
        pass
    raise KeyError(name)
  def __contains__(self, name):
    try:
      self[name]
      return True
    except KeyError:
      return False
  def __getattr__(self, name):
    if name.startswith('_'):
      return _(getattr(self, name[1:]))
    elif name.startswith('a'):
      return Array(name, tuple(self._lengths.get(i, nutils.expression._Length(ord(i))) for i in name[1:]))
    else:
      raise AttributeError(name)
  def get(self, name, default):
    try:
      return self[name]
    except KeyError:
      return default

v = Variables(x=Array('x', [2]), altgeom=Array('altgeom', [3]), funcoverride=Array('funcoverride', []))
functions = dict(func1=1, func2=2, func3=3, funcoverride=1)

class parse(TestCase):

  def assert_ast(self, expression, indices, ast, variables=None, **parse_kwargs):
    if variables is None:
      variables = v
    self.assertEqual(nutils.expression.parse(expression, variables, functions, indices, **parse_kwargs)[0], ast)

  def assert_syntax_error(self, msg, expression, indices, highlight, arg_shapes={}, fixed_lengths=None, exccls=nutils.expression.ExpressionSyntaxError):
    with self.assertRaises(exccls) as cm:
      nutils.expression.parse(expression, v, functions, indices, arg_shapes, fixed_lengths=fixed_lengths)
    self.assertEqual(str(cm.exception), msg + '\n' + expression + '\n' + highlight)

  # OTHER

  def test_no_indices_0(self): self.assert_ast('a', None, v._a)
  def test_no_indices_1(self): self.assert_ast('a2_i', None, v._a2)

  def test_ambiguous_alignment(self):
    self.assert_syntax_error(
      "Cannot unambiguously align the array because the array has more than one dimension.",
      "a23_ij", None,
      "^^^^^^",
      exccls=nutils.expression.AmbiguousAlignmentError)

  def test_mul_2(self):
    self.assert_ast('a2_i a3_j', 'ij',
      ('mul',
        ('append_axis', v._a2, _(3)),
        ('transpose',
          ('append_axis', v._a3, _(2)),
          _((1,0)))))

  def test_mul_add_sub(self):
    self.assert_ast('1_j a2_i + 1_i a3_j - a23_ij', 'ij',
      ('transpose',
        ('sub',
          ('add',
            ('mul',
              ('append_axis', ('append_axis', _(1), _(3)), _(2)),
              ('transpose', ('append_axis', v._a2, _(3)), _((1,0)))),
            ('transpose',
              ('mul',
                ('append_axis', ('append_axis', _(1), _(2)), _(3)),
                ('transpose', ('append_axis', v._a3, _(2)), _((1,0)))),
              _((1,0)))),
          ('transpose', v._a23, _((1,0)))),
        _((1,0))))

  def test_mul_reduce_2(self):
    self.assert_ast('a2_i a23_ij a3_j', '',
      ('sum',
        ('mul',
          ('sum',
            ('mul',
              ('append_axis', v._a2, _(3)),
              v._a23),
            _(0)),
          v._a3),
        _(0)))

  def test_dupl_indices_1(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "a3_j a2_i,ii a2_k", "i",
      "     ^^^^^^^")

  def test_missing_indices_1(self):
    self.assert_syntax_error(
      "Expected 1 index, got 2.",
      "a2_i a3_ij a2_j", "",
      "     ^^^^^")

  def test_missing_indices_2(self):
    self.assert_syntax_error(
      "Expected 2 indices, got 0.",
      "a2_i a23 a2_j", "ij",
      "     ^^^")

  def test_wrap_array_trace(self): self.assert_ast('a222_ijj', 'i', ('trace', v._a222, _(1), _(2)))
  def test_div_const_scalar(self): self.assert_ast('a2_i / 1', 'i', ('truediv', v._a2, _(1)))
  def test_div_scalar_variable(self): self.assert_ast('a2_i / a', 'i', ('truediv', v._a2, v._a))
  def test_div_scalar_sum(self): self.assert_ast('a2_i / 2 a3_j a3_j', 'i', ('truediv', v._a2, ('sum', ('mul', ('mul', ('append_axis', _(2), _(3)), v._a3), v._a3), _(0))))

  def test_array_denominator(self):
    self.assert_syntax_error(
      "A denominator must have dimension 0.",
      "a2_i / a3_j", "ij",
      "       ^^^^")

  def test_duplicate_indices_numerator_denominator(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "1 + a2_i a2_i / a3_i a3_i", "",
      "    ^^^^^^^^^^^^^^^^^^^^^")

  def test_duplicate_indices_2(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "a2_i (a3_i a23_ji)", "ij",
      "^^^^^^^^^^^^^^^^^^")

  def test_duplicate_indices_3(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "a2_i (a3_i a23_ji)", "ij",
      "^^^^^^^^^^^^^^^^^^")

  def test_duplicate_indices_4(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "a222_iii", "",
      "^^^^^^^^")

  def test_leading_zeros_int(self):
    self.assert_syntax_error(
      "Leading zeros are forbidden.",
      "1 + 01", "",
      "    ^^")

  def test_leading_zeros_float(self):
    self.assert_syntax_error(
      "Leading zeros are forbidden.",
      "1 + 01.0", "",
      "    ^^^^")

  def test_missing_indices_3(self):
    self.assert_syntax_error(
      "Missing indices.",
      "a22_ij + a2_", "ij",
      "            ^")

  def test_missing_gradient_indices_1(self):
    self.assert_syntax_error(
      "Missing indices.",
      "a22_ij + a2_,", "ij",
      "            ^")

  def test_missing_whitespace_add_right(self):
    self.assert_syntax_error(
      "Missing whitespace.",
      "a2_i +a2_i", "i",
      "      ^")

  def test_missing_whitespace_add_left(self):
    self.assert_syntax_error(
      "Missing whitespace.",
      "a2_i+ a2_i", "i",
      "    ^")

  def test_missing_whitespace_sub_right(self):
    self.assert_syntax_error(
      "Missing whitespace.",
      "a2_i -a2_i", "i",
      "      ^")

  def test_missing_whitespace_sub_left(self):
    self.assert_syntax_error(
      "Missing whitespace.",
      "a2_i- a2_i", "i",
      "    ^")

  def test_int_float_syntax(self):
    self.assert_ast('1 + 1.1 + 1. + 0.12', '',
      ('add', ('add', ('add', _(1), _(1.1)), _(1.)), _(0.12)))

  def test_jump_mean(self):
    self.assert_ast('[a2_i,i] + {a2_j,j}', '',
      ('add',
        ('jump', ('trace', ('grad', v._a2, v._x), _(0), _(1))),
        ('mean', ('trace', ('grad', v._a2, v._x), _(0), _(1)))))

  def test_jump_normal(self): self.assert_ast('[a]_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(2)), ('normal', v._x)))
  def test_jump_normal_altgeom(self): self.assert_ast('[a]_altgeom_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(3)), ('normal', v._altgeom)))

  def test_laplace_of_group(self):
    self.assert_ast('(2 a2_i)_,jj', 'i',
      ('trace',
        ('grad',
          ('grad',
            ('group',
              ('mul',
                ('append_axis', _(2), _(2)),
                v._a2)),
            v._x),
          v._x),
        _(1), _(2)))

  def test_indices_on_group(self):
    self.assert_syntax_error(
      "Indices can only be specified for variables, e.g. 'a_ij', not for groups, e.g. '(a+b)_ij'.",
      "1 + (a2_i)_j + 1", "ij",
      "           ^")

  def test_unknown_symbol(self):
    self.assert_syntax_error(
      "Unknown symbol: '#'.",
      "1 + # + 1", "",
      "    ^")

  def test_invalid_group_end_partial_expression(self):
    self.assert_syntax_error(
      "Expected a variable, group or function call.",
      "1 + (2 + )", "",
      "         ^")

  def test_invalid_group_end_wrong_bracket_no_whitespace(self):
    self.assert_syntax_error(
      "Expected ')'.",
      "1 + (2 + 3] + 4", "",
      "          ^")

  def test_invalid_group_end_wrong_bracket_whitespace(self):
    self.assert_syntax_error(
      "Expected ')'.",
      "1 + (2 + 3 ] + 4", "",
      "           ^")

  def test_invalid_group_end_eof(self):
    self.assert_syntax_error(
      "Expected ')'.",
      "1 + (2 + 3", "",
      "          ^")

  def test_expected_EOF(self):
    self.assert_syntax_error(
      "Unexpected symbol at end of expression.",
      "1 ) 1", "",
      "  ^")

  def test_shape_mismatch(self):
    self.assert_syntax_error(
      "Shapes at index 'i' differ: 2, 4.",
      "1_j + a234_iji + 1_j", "j",
      "      ^^^^^^^^")

  def test_unknown_variable(self):
    self.assert_syntax_error(
      "Unknown variable: 'b'.",
      "1 + b + 1", "",
      "    ^")

  def test_const_numeric_indices(self):
    self.assert_syntax_error(
      "Numeric indices are not allowed on constant values.",
      "1 + 1_i0 + 1", "",
      "    ^^^^")

  def test_const_repeated_indices(self):
    self.assert_syntax_error(
      "Indices of a constant value may not be repeated.",
      "1 + 1_ii + 1", "",
      "    ^^^^")

  def test_const_index_pos(self):
    self.assert_syntax_error(
      "Length of axis cannot be determined from the expression.",
      "1_i", "i",
      "  ^")

  # NEG

  def test_neg_no_whitspace(self): self.assert_ast('-a2_i', 'i', ('neg', v._a2))
  def test_neg_whitespace(self): self.assert_ast('- a2_i', 'i', ('neg', v._a2))
  def test_neg_in_group(self): self.assert_ast('(- a2_i)', 'i', ('group', ('neg', v._a2)))

  # ADD SUB

  def test_add_sub_unmatched_indices(self):
    self.assert_syntax_error(
      "Cannot add arrays with unmatched indices: 'i', 'j'.",
      "a22_ij + (a2_i + a2_j + a2_ij)", "ij",
      "          ^^^^^^^^^^^")

  # POW

  def test_array_pow_pos(self): self.assert_ast('a2_i^2', 'i', ('pow', v._a2, _(2)))
  def test_array_pow_neg(self): self.assert_ast('a2_i^-2', 'i', ('pow', v._a2, ('neg', _(2))))
  def test_array_pow_scalar_expr(self): self.assert_ast('a2_i^(1 / 3)', 'i', ('pow', v._a2, ('truediv', _(1), _(3))))

  def test_array_pow_nonconst(self):
    self.assert_syntax_error(
      "Expected a number.",
      "a2_i + a2_i^a + a2_i", "i",
      "            ^")

  def test_array_pow_vector_expr(self):
    self.assert_syntax_error(
      "An exponent must have dimension 0.",
      "1_i + a2_i^(a2_j) + 1_i", "ij",
      "      ^^^^^^^^^^^")

  def test_array_pow_repeated_indices(self):
    self.assert_syntax_error(
      "Index 'i' occurs more than twice.",
      "1_i + a2_i^(a22_ii) + 1_i", "i",
      "      ^^^^^^^^^^^^^")

  # NUMERIC INDEX

  def test_numeric_index(self): self.assert_ast('a23_i0', 'i', ('getitem', v._a23, _(1), _(0)))
  def test_numeric_index_grad(self): self.assert_ast('a2_i,1', 'i', ('getitem', ('grad', v._a2, v._x), _(1), _(1)))

  def test_numeric_index_out_of_range(self):
    self.assert_syntax_error(
      "Index of dimension 1 with length 4 out of range.",
      "1 + a343_i4i + 1", "",
      "    ^^^^^^^^")

  def test_numeric_index_out_of_range_grad(self):
    self.assert_syntax_error(
      "Index of dimension 0 with length 2 out of range.",
      "1 + a2_1,2 + 1", "",
      "    ^^^^^^")

  # EYE

  def test_single_eye(self):
    for eye in 'δ$':
      with self.subTest(eye=eye):
        self.assert_ast('a2_i {}_ij'.format(eye), 'j', ('sum', ('mul', ('append_axis', v._a2, _(2)), ('eye', _(2))), _(0)))

  def test_multiple_eye(self): self.assert_ast('δ_ij δ_jk a2_i a2_k', '',
    ('sum',
      ('mul',
        ('sum',
          ('mul',
            ('sum',
              ('mul',
                ('append_axis', ('eye', _(2)), _(2)),
                ('transpose', ('append_axis', ('eye', _(2)), _(2)), _((2,0,1)))),
              _(1)),
            ('append_axis', v._a2, _(2))),
          _(0)),
        v._a2),
      _(0)))

  def test_eye_missing_indices(self):
    self.assert_syntax_error(
      "Expected 2 indices, got 0.",
      "1 + δ + 1", "",
      "    ^")

  def test_eye_invalid_number_of_indices(self):
    self.assert_syntax_error(
      "Expected 2 indices, got 3.",
      "1 + δ_ijk + 1", "",
      "    ^^^^^")

  def test_eye_same_index(self):
    self.assert_syntax_error(
      "Length of axis cannot be determined from the expression.",
      "1 + δ_ii + 1", "",
      "      ^")

  def test_eye_shape_mismatch(self):
    self.assert_syntax_error(
      "Shapes at index 'k' differ: 2, 3.",
      "1 + δ_ij δ_jk a2_i a3_k + 1", "",
      "    ^^^^^^^^^^^^^^^^^^^")

  def test_variable_startswith_eye(self):
    b = Array('b', [2,2])
    δx = Array('δx', [2,2])
    self.assert_ast('b_ij + δx_ij', 'ij', ('add', _(b), _(δx)), variables=dict(b=b, δx=δx))

  def test_variable_startswith_dollar_eye(self):
    self.assert_syntax_error(
      "Expected 2 indices, got 0.",
      "a22_ij + $x_ij", "ij",
      "         ^")

  # GRAD

  def test_gradient_default(self): self.assert_ast('a2_i,j', 'ij', ('grad', v._a2, v._x))
  def test_gradient_other_default(self): self.assert_ast('a2_i,j', 'ij', ('grad', v._a2, v._altgeom), default_geometry_name='altgeom')
  def test_gradient_default_trace(self): self.assert_ast('a2_i,i', '', ('trace', ('grad', v._a2, v._x), _(0), _(1)))
  def test_gradient_default_double_trace(self): self.assert_ast('a422_ijk,jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a422, v._x), _(1), _(3)), v._x), _(1), _(2)))
  def test_gradient_altgeom(self): self.assert_ast('a3_i,altgeom_j', 'ij', ('grad', v._a3, v._altgeom))
  def test_gradient_altgeom_trace(self): self.assert_ast('a3_i,altgeom_i', '', ('trace', ('grad', v._a3, v._altgeom), _(0), _(1)))
  def test_gradient_altgeom_double_trace(self): self.assert_ast('a433_ijk,altgeom_jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a433, v._altgeom), _(1), _(3)), v._altgeom), _(1), _(2)))
  def test_surfgrad_default(self): self.assert_ast('a2_i;j', 'ij', ('surfgrad', v._a2, v._x))
  def test_surfgrad_default_trace(self): self.assert_ast('a2_i;i', '', ('trace', ('surfgrad', v._a2, v._x), _(0), _(1)))

  def test_gradient_invalid_geom_0dim(self):
    self.assert_syntax_error(
      "Invalid geometry: expected 1 dimension, but 'a' has 0.",
      "1 + a2_i,a_i + 1", "",
      "    ^^^^^^^^")

  def test_gradient_invalid_geom_2dim(self):
    self.assert_syntax_error(
      "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
      "1 + a2_i,a22_i + 1", "",
      "    ^^^^^^^^^^")

  def test_gradient_const_scalar(self):
    self.assert_syntax_error(
      "Taking a derivative of a constant is not allowed.",
      "1_i + 1_,i + 1_i", "i",
      "      ^^^^")

  def test_gradient_const_array(self):
    self.assert_syntax_error(
      "Taking a derivative of a constant is not allowed.",
      "1 + 1_i,i + 1", "",
      "    ^^^^^")

  # NEW GRAD

  def test_newgradient(self): self.assert_ast('dx_j:a2_i', 'ij', ('grad', v._a2, v._x))
  def test_newgradient_trace(self): self.assert_ast('dx_i:a2_i', '', ('trace', ('grad', v._a2, v._x), _(0), _(1)))
  def test_newgradient_double_trace(self): self.assert_ast('dx_k:(dx_j:a422_ijk)', 'i', ('trace', ('grad', ('group', ('trace', ('grad', v._a422, v._x), _(1), _(3))), v._x), _(1), _(2)))

  # DERIVATIVE

  def test_derivative0(self): self.assert_ast('(2 ?arg + 1)_,?arg', '', ('derivative', ('group', ('add', ('mul', _(2), ('arg', _('arg'))), _(1))), ('arg', _('arg'))))
  def test_derivative1(self): self.assert_ast('(a2_i + ?arg_i)_,?arg_j', 'ij', ('derivative', ('group', ('add', v._a2, ('arg', _('arg'), _(2)))), ('arg', _('arg'), _(2))))
  def test_derivative2(self): self.assert_ast('(a23_ij + ?arg_ij)_,?arg_kj', 'ik', ('trace', ('derivative', ('group', ('add', v._a23, ('arg', _('arg'), _(2), _(3)))), ('arg', _('arg'), _(2), _(3))), _(1), _(3)))

  # NEW DERIVATIVE

  def test_newderivative0(self): self.assert_ast('d?arg:(2 ?arg + 1)', '', ('derivative', ('group', ('add', ('mul', _(2), ('arg', _('arg'))), _(1))), ('arg', _('arg'))))
  def test_newderivative1(self): self.assert_ast('d?arg_j:(a2_i + ?arg_i)', 'ij', ('derivative', ('group', ('add', v._a2, ('arg', _('arg'), _(2)))), ('arg', _('arg'), _(2))))
  def test_newderivative2(self): self.assert_ast('d?arg_kj:(a23_ij + ?arg_ij)', 'ik', ('trace', ('derivative', ('group', ('add', v._a23, ('arg', _('arg'), _(2), _(3)))), ('arg', _('arg'), _(2), _(3))), _(1), _(3)))

  # NORMAL

  def test_normal(self): self.assert_ast('n:x_i', 'i', ('normal', v._x))

  def test_normal_default(self): self.assert_ast('n_i', 'i', ('normal', v._x))
  def test_normal_altgeom(self): self.assert_ast('n_altgeom_i', 'i', ('normal', v._altgeom))
  def test_normal_default_grad_default(self): self.assert_ast('n_i,j', 'ij', ('grad', ('normal', v._x), v._x))
  def test_normal_altgeom_grad_default(self): self.assert_ast('n_altgeom_i,x_j', 'ij', ('grad', ('normal', v._altgeom), v._x))
  def test_normal_altgeom_grad_altgeom(self): self.assert_ast('n_altgeom_i,altgeom_j', 'ij', ('grad', ('normal', v._altgeom), v._altgeom))

  def test_normal_altgeom_grad_nogeom(self):
    self.assert_syntax_error(
      "Missing geometry, e.g. ',altgeom_i' or ',x_i'.",
      "1 + n_altgeom_i,i + 1", "",
      "               ^")

  def test_normal_missing_indices(self):
    self.assert_syntax_error(
      "Expected 1 index, got 0.",
      "1 + n + 1", "",
      "    ^")

  def test_normal_too_many_indices(self):
    self.assert_syntax_error(
      "Expected 1 index, got 2.",
      "1 + n_ij + 1", "",
      "    ^^^^")

  def test_normal_invalid_geom_0dim(self):
    self.assert_syntax_error(
      "Invalid geometry: expected 1 dimension, but 'a' has 0.",
      "1 + n_a_i + 1", "",
      "    ^^^^")

  def test_normal_invalid_geom_2dim(self):
    self.assert_syntax_error(
      "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
      "1 + n_a22_i + 1", "",
      "    ^^^^^^")

  def test_variable_startswith_normal(self):
    nx = Array('nx', [2])
    self.assert_ast('nx_i', 'i', _(nx), variables=dict(nx=nx))

  # JACOBIAN

  def test_jacobian(self): self.assert_ast('J:x', '', ('jacobian', v._x, _(2)))
  def test_jacobian_boundary(self): self.assert_ast('J^:x', '', ('jacobian', v._x, _(1)))
  def test_jacobian_double_boundary(self): self.assert_ast('J^^:x', '', ('jacobian', v._x, _(0)))
  def test_old_jacobian(self): self.assert_ast('d:x', '', ('jacobian', v._x, _(None)))

  # VARIABLE LENGTH TESTS

  def test_variable_lengths_shape_mismatch1(self):
    self.assert_syntax_error(
      "Axes have different lengths: 2, 3.",
      "aXY_ii + aX2_ii + aY3_ii", "",
      "^^^^^^^^^^^^^^^^^^^^^^^^")

  def test_variable_lengths_shape_mismatch2(self):
    self.assert_syntax_error(
      "Shapes at index 'j' differ: 2, 3.",
      "aX2X3_iijj", "",
      "^^^^^^^^^^")

  # FIXED LENGTHS

  def test_fixed_lengths(self): self.assert_ast('δ_ij', 'ij', ('eye', _(3)), fixed_lengths=dict(i=3))

  def test_fixed_lengths_invalid(self):
    self.assert_syntax_error(
      'Length of index i is fixed at 3 but the expression has length 2.',
      'a2_i', 'i',
      '   ^',
      fixed_lengths=dict(i=3))

  def test_fixed_lengths_invalid_linked(self):
    self.assert_syntax_error(
      'Axes have different lengths: 2, 3.',
      'a2_i δ_ij', 'j',
      '^^^^^^^^^',
      fixed_lengths=dict(j=3))

  # FALLBACK LENGHT

  def test_fallback_length(self): self.assert_ast('1_i', 'i', ('append_axis', _(1), _(2)), fallback_length=2)

  # ARG

  def test_arg0(self): self.assert_ast('a ?coeffs', '', ('mul', v._a, ('arg', _('coeffs'))))
  def test_arg1(self): self.assert_ast('a2_i ?coeffs_i', '', ('sum', ('mul', v._a2, ('arg', _('coeffs'), _(2))), _(0)))
  def test_arg2(self): self.assert_ast('a23_ij ?coeffs_ij', '', ('sum', ('sum', ('mul', v._a23, ('arg', _('coeffs'), _(2), _(3))), _(1)), _(0)))

  def test_arg_reshape(self):
    self.assert_syntax_error(
      "Argument 'arg' previously defined with 1 axis instead of 2.",
      "a2_i (a2_j + ?arg_j) + ?arg_ij", "ij",
      "                       ^^^^^^^")

  def test_arg_shape_mismatch(self):
    self.assert_syntax_error(
      "Axes have different lengths: 2, 3.",
      "1 + a2_i ?arg_i + a3_j ?arg_j + 1", "",
      "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

  def test_arg_reshape_external(self):
    self.assert_syntax_error(
      "Argument 'arg' previously defined with 2 axes instead of 1.",
      "1 + a3_j ?arg_j + 1", "",
      "         ^^^^^^",
      {'arg': (2,3)})

  def test_arg_shape_mismatch_external(self):
    self.assert_syntax_error(
      "Shapes at index 'j' differ: 3, 2.",
      "1 + a3_j ?arg_j + 1", "",
      "    ^^^^^^^^^^^",
      {'arg': (2,)})

  def test_arg_index_pos1(self):
    self.assert_syntax_error(
      "Length of axis cannot be determined from the expression.",
      "?arg_n", "n",
      "     ^")

  def test_arg_index_pos2(self):
    self.assert_syntax_error(
      "Length of axis cannot be determined from the expression.",
      "?foo_,?bar_n", "n",
      "           ^")

  # SUBSTITUTE

  def test_arg_subs_0d_const(self): self.assert_ast('?arg_,?arg(arg=1)', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), _(1)))
  def test_arg_subs_0d_var(self): self.assert_ast('?arg_,?arg(arg=a )', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), v._a))
  def test_arg_subs_1d_var(self): self.assert_ast('?arg_i,?arg_j(arg_i = a2_i)', 'ij', ('substitute', ('derivative', ('arg', _('arg'), _(2)), ('arg', _('arg'), _(2))), ('arg', _('arg'), _(2)), v._a2))
  def test_arg_subs_2d_var(self): self.assert_ast('?arg_ij,?arg_kl( arg_ij =a23_ji)', 'ijkl', ('substitute', ('derivative', ('arg', _('arg'), _(3), _(2)), ('arg', _('arg'), _(3), _(2))), ('arg', _('arg'), _(3), _(2)), ('transpose', v._a23, _((1,0)))))
  def test_arg_multisubs(self): self.assert_ast('(1 + ?x + ?y)(x=1 + a, y=2)', '', ('substitute', ('group', ('add', ('add', _(1), ('arg', _('x'))), ('arg', _('y')))), ('arg', _('x')), ('add', _(1), v._a), ('arg', _('y')), _(2)))

  def test_arg_subs_missing_equals(self):
    self.assert_syntax_error(
      "Expected '='.",
      "(1 + ?x)(x + 2)", "",
      "           ^")

  def test_arg_subs_unmatched_indices(self):
    self.assert_syntax_error(
      "Left and right hand side should have the same indices, got 'kl' and 'jk'.",
      "a23_ij + ?x_ij(x_kl=a23_jk) + a23_ij", "ij",
      "               ^^^^^^^^^^^")

  def test_arg_subs_lhs_repeated_index(self):
    self.assert_syntax_error(
      "Repeated indices are not allowed on the left hand side.",
      "a23_ij + ?x_ij(x_kk=a23_jk) + 2", "ij",
      "               ^^^^")

  def test_arg_subs_lhs_numeric_index(self):
    self.assert_syntax_error(
      "Numeric indices are not allowed on the left hand side.",
      "a23_ij + ?x_ij(x_k0=a23_0k) + 2", "ij",
      "               ^^^^")

  def test_arg_subs_lhs_with_questionmark(self):
    self.assert_syntax_error(
      "The argument name at the left hand side of a substitution must not be prefixed by a '?'.",
      "?x_ij(?x_ij=1_ij)", "ij",
      "      ^^^")

  def test_arg_subs_lhs_not_an_argument(self):
    self.assert_syntax_error(
      "Expected an argument, e.g. 'argname'.",
      "?x(1=2)", "",
      "   ^")

  def test_arg_subs_double_occurence(self):
    self.assert_syntax_error(
      "Argument 'x' occurs more than once.",
      "?x(x=1, x=2)", "",
      "        ^")

  def test_arg_subs_zero(self):
    self.assert_syntax_error(
      "Zero substitutions are not allowed.",
      "?x_ij()", "ij",
      "^^^^^^^")

  # TRANSPOSE

  def test_transpose_duplicate_indices(self):
    self.assert_syntax_error(
      "Cannot transpose from 'ij' to 'jii': duplicate indices.",
      "a23_ij", "jii",
      "^^^^^^")

  def test_transpose_indices_differ(self):
    self.assert_syntax_error(
      "Cannot transpose from 'ij' to 'jk': indices differ.",
      "a23_ij", "jk",
      "^^^^^^")

  # STACK

  def test_stack_1_0d(self): self.assert_ast('<a>_i', 'i', ('append_axis', v._a, _(1)))
  def test_stack_1_1di_1(self): self.assert_ast('<a2_i>_i', 'i', v._a2)
  def test_stack_2_0d_0d(self): self.assert_ast('<a, a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
  def test_stack_2_1di_0d(self): self.assert_ast('<a2_i, a>_i', 'i', ('concatenate', v._a2, ('append_axis', v._a, _(1))))
  def test_stack_3_2di_1d_1d(self): self.assert_ast('<a23_ij, a3_j, 1_j>_i', 'ij', ('concatenate', v._a23, ('transpose',  ('append_axis', v._a3, _(1)), _((1,0))), ('transpose', ('append_axis', ('append_axis', _(1), _(3)), _(1)), _((1,0)))))

  def test_stack_no_indices(self):
    self.assert_syntax_error(
      "Expected 1 index.",
      "<1, a2_i> + a3_i", "i",
      "         ^")

  def test_stack_too_many_indices(self):
    self.assert_syntax_error(
      "Expected 1 index, got 2.",
      "<1, a2_i>_ij + a3_i", "i",
      "          ^^")

  def test_stack_numeric_index(self):
    self.assert_syntax_error(
      "Expected a non-numeric index, got '1'.",
      "<1, a2_i>_1 + a3_i", "i",
      "          ^")

  def test_stack_0(self):
    self.assert_syntax_error(
      "Cannot stack 0 arrays.",
      "1_i + <>_i + 1_i", "i",
      "      ^^^^")

  def test_stack_unmatched_indices(self):
    self.assert_syntax_error(
      "Cannot stack arrays with unmatched indices (excluding the stack index 'i'): j, ijk.",
      "1_ij + <a2_j, a222_ijk>_i + 1_ij", "ij",
      "       ^^^^^^^^^^^^^^^^^^")

  def test_stack_undetermined_length(self):
    self.assert_syntax_error(
      "Cannot determine the length of the stack axis, because the length at 12 is unknown.",
      "1_i + <a, 1_i>_i + 1_i", "i",
      "            ^")

  def test_stack_whitespace_left(self): self.assert_ast('< a, a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
  def test_stack_whitespace_right(self): self.assert_ast('<a, a >_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
  def test_stack_whitespace_before_comma(self): self.assert_ast('<a , a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))

  # FIXME: the following should work
  # 'a2_j a2_i + <0j, δ_ij>_i'

  # FUNCTION

  def test_function_0d(self): self.assert_ast('func1(a)', '', ('call', _('func1'), v._a))
  def test_function_1d(self): self.assert_ast('func1(a2_i)', 'i', ('call', _('func1'), v._a2))
  def test_function_2d(self): self.assert_ast('func1(a23_ij)', 'ij', ('call', _('func1'), v._a23))
  def test_function_0d_0d(self): self.assert_ast('func2(a, a)', '', ('call', _('func2'), v._a, v._a))
  def test_function_1d_1d(self): self.assert_ast('func2(a2_i, a2_i)', 'i', ('call', _('func2'), v._a2, v._a2))
  def test_function_2d_2d(self): self.assert_ast('func2(a23_ij, a32_ji)', 'ij', ('call', _('func2'), v._a23, ('transpose', v._a32, _((1,0)))))
  def test_function_2d_2d_2d(self): self.assert_ast('func3(a23_ij, a22_ik a23_kj, a23_ij)', 'ij', ('call', _('func3'), v._a23, ('sum', ('mul', ('append_axis', v._a22, _(3)), ('transpose', ('append_axis', v._a23, _(2)), _((2,0,1)))), _(1)), v._a23))

  def test_function_invalid_nargs(self):
    self.assert_syntax_error(
      "Function 'func1' takes 1 argument, got 2.",
      "1 + func1(a, a) + 1", "",
      "    ^^^^^^^^^^^")

  def test_function_unmatched_indices(self):
    self.assert_syntax_error(
      "Cannot align arrays with unmatched indices: ij, ij, jk.",
      "1_ij + func3(a23_ij, a23_ij, a23_jk) + 1_ij", "ij",
      "       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

  def test_function_unmatched_shape(self):
    self.assert_syntax_error(
      "Shapes at index 'i' differ: 2, 3.",
      "1_ij + func2(a23_ij, a33_ij) + 1_ij", "ij",
      "       ^^^^^^^^^^^^^^^^^^^^^")

  def test_function_unknown(self):
    self.assert_syntax_error(
      "Unknown variable: 'funcX'.",
      "1_ij + funcX(a23_ij) + 1_ij", "ij",
      "       ^^^^^")

  def test_function_override(self):
    self.assert_syntax_error(
      "Expected '='.",
      "1_ij + funcoverride(a23_ij) + 1_ij", "ij",
      "                          ^")

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
