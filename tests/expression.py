import nutils.expression
from . import register, unittest

_ = lambda arg: (None, arg)

class Array:
  def __init__(self, text, shape):
    self.text = text
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
  def __str__(self):
    return self.text
  def __repr__(self):
    return self.text
  def __eq__(self, other):
    return type(self) == type(other) and self.text == other.text
  def __hash__(self):
    return hash(self.text)

class Variables:
  def __init__(self, x, altgeom):
    self.x = x
    self.altgeom = altgeom
    self._lengths = {str(i): i for i in range(10)}
  def __getitem__(self, name):
    if name == 'x':
      return self.x
    elif name == 'altgeom':
      return self.altgeom
    elif name.startswith('a'):
      return Array(name, tuple(self._lengths.setdefault(i, nutils.expression._Length(0)) for i in name[1:]))
    else:
      raise KeyError(name)
  def __getattr__(self, name):
    if name.startswith('_'):
      return _(getattr(self, name[1:]))
    elif name.startswith('a'):
      return Array(name, tuple(self._lengths.setdefault(i, nutils.expression._Length(0)) for i in name[1:]))
    else:
      raise AttributeError(name)
  def get(self, name, default):
    try:
      return self[name]
    except KeyError:
      return default

@register
def parse():

  v = Variables(x=Array('x', [2]), altgeom=Array('altgeom', [3]))
  functions = dict(func1=1, func2=2, func3=3)

  def assert_ast(name, expression, indices, ast, **parse_kwargs):
    @unittest(name=name)
    def test():
      assert nutils.expression.parse(expression, v, functions, indices, **parse_kwargs)[0] == ast
    return test

  def assert_syntax_error(name, msg, expression, indices, highlight, arg_shapes={}, exccls=nutils.expression.ExpressionSyntaxError):
    @unittest(name=name)
    def test():
      try:
        nutils.expression.parse(expression, v, functions, indices, arg_shapes)
      except exccls as e:
        e_msg = str(e)
        assert e_msg == msg + '\n' + expression + '\n' + highlight
      else:
        raise ValueError('Expected an {!r}.'.format(exccls))
    return test

  # OTHER

  assert_ast('no_indices_0', 'a', None, v._a)
  assert_ast('no_indices_1', 'a2_i', None, v._a2)

  assert_syntax_error('ambiguous_alignment',
    "Cannot unambiguously align the array because the array has more than one dimension.",
    "a23_ij", None,
    "^^^^^^",
    exccls=nutils.expression.AmbiguousAlignmentError)

  assert_ast('mul_2', 'a2_i a3_j', 'ij',
    ('mul',
      ('append_axis', v._a2, _(3)),
      ('transpose',
        ('append_axis', v._a3, _(2)),
        _((1,0)))))

  assert_ast('mul_add_sub', '1_j a2_i + 1_i a3_j - a23_ij', 'ij',
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

  assert_ast('mul_reduce_2', 'a2_i a23_ij a3_j', '',
    ('sum',
      ('mul',
        ('sum',
          ('mul',
            ('append_axis', v._a2, _(3)),
            v._a23),
          _(0)),
        v._a3),
      _(0)))

  assert_syntax_error('dupl_indices_1',
    "Index 'i' occurs more than twice.",
    "a3_j a2_i,ii a2_k", "i",
    "     ^^^^^^^")

  assert_syntax_error('missing_indices_1',
    "Expected 1 index, got 2.",
    "a2_i a3_ij a2_j", "",
    "     ^^^^^")

  assert_syntax_error('missing_indices_2',
    "Expected 2 indices, got 0.",
    "a2_i a23 a2_j", "ij",
    "     ^^^")

  assert_ast('wrap_array_trace', 'a222_ijj', 'i', ('trace', v._a222, _(1), _(2)))
  assert_ast('div_const_scalar', 'a2_i / 1', 'i', ('truediv', v._a2, _(1)))
  assert_ast('div_scalar_variable', 'a2_i / a', 'i', ('truediv', v._a2, v._a))
  assert_ast('div_scalar_sum', 'a2_i / 2 a3_j a3_j', 'i', ('truediv', v._a2, ('sum', ('mul', ('mul', ('append_axis', _(2), _(3)), v._a3), v._a3), _(0))))

  assert_syntax_error('array_denominator',
    "A denominator must have dimension 0.",
    "a2_i / a3_j", "ij",
    "       ^^^^")

  assert_syntax_error('duplicate_indices_numerator_denominator',
    "Index 'i' occurs more than twice.",
    "1 + a2_i a2_i / a3_i a3_i", "",
    "    ^^^^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('duplicate_indices_2',
    "Index 'i' occurs more than twice.",
    "a2_i (a3_i a23_ji)", "ij",
    "^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('duplicate_indices_3',
    "Index 'i' occurs more than twice.",
    "a2_i (a3_i a23_ji)", "ij",
    "^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('duplicate_indices_4',
    "Index 'i' occurs more than twice.",
    "a222_iii", "",
    "^^^^^^^^")

  assert_syntax_error('leading_zeros_int',
    "Leading zeros are forbidden.",
    "1 + 01", "",
    "    ^^")

  assert_syntax_error('leading_zeros_float',
    "Leading zeros are forbidden.",
    "1 + 01.0", "",
    "    ^^^^")

  assert_syntax_error('missing_indices_3',
    "Missing indices.",
    "a22_ij + a2_", "ij",
    "            ^")

  assert_syntax_error('missing_gradient_indices_1',
    "Missing indices.",
    "a22_ij + a2_,", "ij",
    "            ^")

  assert_syntax_error('missing_whitespace_add_right',
    "Missing whitespace.",
    "a2_i +a2_i", "i",
    "      ^")

  assert_syntax_error('missing_whitespace_add_left',
    "Missing whitespace.",
    "a2_i+ a2_i", "i",
    "    ^")

  assert_syntax_error('missing_whitespace_sub_right',
    "Missing whitespace.",
    "a2_i -a2_i", "i",
    "      ^")

  assert_syntax_error('missing_whitespace_sub_left',
    "Missing whitespace.",
    "a2_i- a2_i", "i",
    "    ^")

  assert_ast('int_float_syntax', '1 + 1.1 + 1. + 0.12', '',
    ('add', ('add', ('add', _(1), _(1.1)), _(1.)), _(0.12)))

  assert_ast('jump_mean', '[a2_i,i] + {a2_j,j}', '',
    ('add',
      ('jump', ('trace', ('grad', v._a2, v._x), _(0), _(1))),
      ('mean', ('trace', ('grad', v._a2, v._x), _(0), _(1)))))

  assert_ast('jump_normal', '[a]_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(2)), ('normal', v._x)))
  assert_ast('jump_normal_altgeom', '[a]_altgeom_i', 'i', ('mul', ('append_axis', ('jump', v._a), _(3)), ('normal', v._altgeom)))

  assert_ast('laplace_of_group', '(2 a2_i)_,jj', 'i',
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

  assert_syntax_error('indices_on_group',
    "Indices can only be specified for variables, e.g. 'a_ij', not for groups, e.g. '(a+b)_ij'.",
    "1 + (a2_i)_j + 1", "ij",
    "           ^")

  assert_syntax_error('unknown symbol',
    "Unknown symbol: '#'.",
    "1 + # + 1", "",
    "    ^")

  assert_syntax_error('invalid_group_end_partial_expression',
    "Expected a variable, group or function call.",
    "1 + (2 + )", "",
    "         ^")

  assert_syntax_error('invalid_group_end_wrong_bracket_no_whitespace',
    "Expected ')'.",
    "1 + (2 + 3] + 4", "",
    "     ^^^^^")

  assert_syntax_error('invalid_group_end_wrong_bracket_whitespace',
    "Expected ')'.",
    "1 + (2 + 3 ] + 4", "",
    "     ^^^^^")

  assert_syntax_error('invalid_group_end_eof',
    "Expected ')'.",
    "1 + (2 + 3", "",
    "     ^^^^^")

  assert_syntax_error('shape_mismatch',
    "Shapes at index 'i' differ: 2, 4.",
    "1_j + a234_iji + 1_j", "j",
    "      ^^^^^^^^")

  assert_syntax_error('unknown_variable',
    "Unknown variable: 'b'.",
    "1 + b + 1", "",
    "    ^")

  assert_syntax_error('const_numeric_indices',
    "Numeric indices are not allowed on constant values.",
    "1 + 1_i0 + 1", "",
    "    ^^^^")

  assert_syntax_error('const_repeated_indices',
    "Indices of a constant value may not be repeated.",
    "1 + 1_ii + 1", "",
    "    ^^^^")

  # NEG

  assert_ast('neg_no_whitspace', '-a2_i', 'i', ('neg', v._a2))
  assert_ast('neg_whitespace', '- a2_i', 'i', ('neg', v._a2))
  assert_ast('neg_in_group', '(- a2_i)', 'i', ('group', ('neg', v._a2)))

  # ADD SUB

  assert_syntax_error('add_sub_unmatched_indices',
    "Cannot add arrays with unmatched indices: 'i', 'j'.",
    "a22_ij + (a2_i + a2_j + a2_ij)", "ij",
    "          ^^^^^^^^^^^")

  # POW

  assert_ast('array_pow_pos', 'a2_i^2', 'i', ('pow', v._a2, _(2)))
  assert_ast('array_pow_neg', 'a2_i^-2', 'i', ('pow', v._a2, ('neg', _(2))))
  assert_ast('array_pow_scalar_expr', 'a2_i^(1 / 3)', 'i', ('pow', v._a2, ('truediv', _(1), _(3))))

  assert_syntax_error('array_pow_nonconst',
    "Expected a number.",
    "a2_i + a2_i^a + a2_i", "i",
    "            ^")

  assert_syntax_error('array_pow_vector_expr',
    "An exponent must have dimension 0.",
    "1_i + a2_i^(a2_j) + 1_i", "ij",
    "      ^^^^^^^^^^^")

  assert_syntax_error('array_pow_repeated_indices',
    "Index 'i' occurs more than twice.",
    "1_i + a2_i^(a22_ii) + 1_i", "i",
    "      ^^^^^^^^^^^^^")

  # NUMERIC INDEX

  assert_ast('numeric_index', 'a23_i0', 'i', ('getitem', v._a23, _(1), _(0)))
  assert_ast('numeric_index_grad', 'a2_i,1', 'i', ('getitem', ('grad', v._a2, v._x), _(1), _(1)))

  assert_syntax_error('numeric_index_out_of_range',
    "Index of dimension 1 with length 4 out of range.",
    "1 + a343_i4i + 1", "",
    "    ^^^^^^^^")

  assert_syntax_error('numeric_index_out_of_range_grad',
    "Index of dimension 0 with length 2 out of range.",
    "1 + a2_1,2 + 1", "",
    "    ^^^^^^")

  # EYE

  assert_ast('single_eye', 'a2_i δ_ij', 'j', ('sum', ('mul', ('append_axis', v._a2, _(2)), ('eye', _(2))), _(0)))
  assert_ast('multiple_eye', 'δ_ij δ_jk a2_i a2_k', '',
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

  assert_syntax_error('eye_missing_indices',
    "Expected 2 indices, got 0.",
    "1 + δ + 1", "",
    "    ^")

  assert_syntax_error('eye_invalid_number_of_indices',
    "Expected 2 indices, got 3.",
    "1 + δ_ijk + 1", "",
    "    ^^^^^")

  assert_syntax_error('eye_same_index',
    "Length of axis cannot be determined from the expression.",
    "1 + δ_ii + 1", "",
    "      ^")

  assert_syntax_error('eye_shape_mismatch',
    "Shapes at index 'k' differ: 2, 3.",
    "1 + δ_ij δ_jk a2_i a3_k + 1", "",
    "    ^^^^^^^^^^^^^^^^^^^")

  # GRAD

  assert_ast('gradient_default', 'a2_i,j', 'ij', ('grad', v._a2, v._x))
  assert_ast('gradient_other_default', 'a2_i,j', 'ij', ('grad', v._a2, v._altgeom), default_geometry_name='altgeom')
  assert_ast('gradient_default_trace', 'a2_i,i', '', ('trace', ('grad', v._a2, v._x), _(0), _(1)))
  assert_ast('gradient_default_double_trace', 'a422_ijk,jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a422, v._x), _(1), _(3)), v._x), _(1), _(2)))
  assert_ast('gradient_altgeom', 'a3_i,altgeom_j', 'ij', ('grad', v._a3, v._altgeom))
  assert_ast('gradient_altgeom_trace', 'a3_i,altgeom_i', '', ('trace', ('grad', v._a3, v._altgeom), _(0), _(1)))
  assert_ast('gradient_altgeom_double_trace', 'a433_ijk,altgeom_jk', 'i', ('trace', ('grad', ('trace', ('grad', v._a433, v._altgeom), _(1), _(3)), v._altgeom), _(1), _(2)))
  assert_ast('surfgrad_default', 'a2_i;j', 'ij', ('surfgrad', v._a2, v._x))
  assert_ast('surfgrad_default_trace', 'a2_i;i', '', ('trace', ('surfgrad', v._a2, v._x), _(0), _(1)))

  assert_syntax_error('gradient_invalid_geom_0dim',
    "Invalid geometry: expected 1 dimension, but 'a' has 0.",
    "1 + a2_i,a_i + 1", "",
    "    ^^^^^^^^")

  assert_syntax_error('gradient_invalid_geom_2dim',
    "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
    "1 + a2_i,a22_i + 1", "",
    "    ^^^^^^^^^^")

  assert_syntax_error('gradient_const_scalar',
    "Taking a derivative of a constant is not allowed.",
    "1_i + 1_,i + 1_i", "i",
    "      ^^^^")

  assert_syntax_error('gradient_const_array',
    "Taking a derivative of a constant is not allowed.",
    "1 + 1_i,i + 1", "",
    "    ^^^^^")

  # DERIVATIVE

  assert_ast('derivative0', '(2 ?arg + 1)_,?arg', '', ('derivative', ('group', ('add', ('mul', _(2), ('arg', _('arg'))), _(1))), ('arg', _('arg'))))
  assert_ast('derivative1', '(a2_i + ?arg_i)_,?arg_j', 'ij', ('derivative', ('group', ('add', v._a2, ('arg', _('arg'), _(2)))), ('arg', _('arg'), _(2))))
  assert_ast('derivative2', '(a23_ij + ?arg_ij)_,?arg_kj', 'ik', ('trace', ('derivative', ('group', ('add', v._a23, ('arg', _('arg'), _(2), _(3)))), ('arg', _('arg'), _(2), _(3))), _(1), _(3)))

  # NORMAL

  assert_ast('normal_default', 'n_i', 'i', ('normal', v._x))
  assert_ast('normal_altgeom', 'n_altgeom_i', 'i', ('normal', v._altgeom))
  assert_ast('normal_default_grad_default', 'n_i,j', 'ij', ('grad', ('normal', v._x), v._x))
  assert_ast('normal_altgeom_grad_default', 'n_altgeom_i,x_j', 'ij', ('grad', ('normal', v._altgeom), v._x))
  assert_ast('normal_altgeom_grad_altgeom', 'n_altgeom_i,altgeom_j', 'ij', ('grad', ('normal', v._altgeom), v._altgeom))

  assert_syntax_error('normal_altgeom_grad_nogeom',
    "Missing geometry, e.g. ',altgeom_i' or ',x_i'.",
    "1 + n_altgeom_i,i + 1", "",
    "               ^")

  assert_syntax_error('normal_missing_indices',
    "Expected 1 index, got 0.",
    "1 + n + 1", "",
    "    ^")

  assert_syntax_error('normal_too_many_indices',
    "Expected 1 index, got 2.",
    "1 + n_ij + 1", "",
    "    ^^^^")

  assert_syntax_error('normal_invalid_geom_0dim',
    "Invalid geometry: expected 1 dimension, but 'a' has 0.",
    "1 + n_a_i + 1", "",
    "    ^^^^")

  assert_syntax_error('normal_invalid_geom_2dim',
    "Invalid geometry: expected 1 dimension, but 'a22' has 2.",
    "1 + n_a22_i + 1", "",
    "    ^^^^^^")

  # VARIABLE LENGTH TESTS

  assert_syntax_error('variable_lengths_shape_mismatch1',
    "Axes have different lengths: 2, 3.",
    "aXY_ii + aX2_ii + aY3_ii", "",
    "^^^^^^^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('variable_lengths_shape_mismatch2',
    "Shapes at index 'j' differ: 2, 3.",
    "aX2X3_iijj", "",
    "^^^^^^^^^^")

  # ARG

  assert_ast('arg0', 'a ?coeffs', '', ('mul', v._a, ('arg', _('coeffs'))))
  assert_ast('arg1', 'a2_i ?coeffs_i', '', ('sum', ('mul', v._a2, ('arg', _('coeffs'), _(2))), _(0)))
  assert_ast('arg2', 'a23_ij ?coeffs_ij', '', ('sum', ('sum', ('mul', v._a23, ('arg', _('coeffs'), _(2), _(3))), _(1)), _(0)))

  assert_syntax_error('arg_reshape',
    "Argument 'arg' previously defined with 1 axis instead of 2.",
    "a2_i (a2_j + ?arg_j) + ?arg_ij", "ij",
    "                       ^^^^^^^")

  assert_syntax_error('arg_shape_mismatch',
    "Axes have different lengths: 2, 3.",
    "1 + a2_i ?arg_i + a3_j ?arg_j + 1", "",
    "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('arg_reshape_external',
    "Argument 'arg' previously defined with 2 axes instead of 1.",
    "1 + a3_j ?arg_j + 1", "",
    "         ^^^^^^",
    {'arg': (2,3)})

  assert_syntax_error('arg_shape_mismatch_external',
    "Shapes at index 'j' differ: 3, 2.",
    "1 + a3_j ?arg_j + 1", "",
    "    ^^^^^^^^^^^",
    {'arg': (2,)})

  # SUBSTITUTE

  assert_ast('arg_subs_0d_const', '?arg_,?arg | ?arg = 1', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), _(1)))
  assert_ast('arg_subs_0d_var', '?arg_,?arg | ?arg = a', '', ('substitute', ('derivative', ('arg', _('arg')), ('arg', _('arg'))), ('arg', _('arg')), v._a))
  assert_ast('arg_subs_1d_var', '?arg_i,?arg_j | ?arg_i = a2_i', 'ij', ('substitute', ('derivative', ('arg', _('arg'), _(2)), ('arg', _('arg'), _(2))), ('arg', _('arg'), _(2)), v._a2))
  assert_ast('arg_subs_2d_var', '?arg_ij,?arg_kl | ?arg_ij = a23_ji', 'ijkl', ('substitute', ('derivative', ('arg', _('arg'), _(3), _(2)), ('arg', _('arg'), _(3), _(2))), ('arg', _('arg'), _(3), _(2)), ('transpose', v._a23, _((1,0)))))
  assert_ast('arg_multisubs', '1 + ?x + ?y | ?x = 1 + a | ?y = 2', '', ('substitute', ('substitute', ('add', ('add', _(1), ('arg', _('x'))), ('arg', _('y'))), ('arg', _('x')), ('add', _(1), v._a)), ('arg', _('y')), _(2)))
  assert_ast('arg_multisubs', '1 + ?x + ?y | ?x = 1 + a | ?y = 2', '', ('substitute', ('substitute', ('add', ('add', _(1), ('arg', _('x'))), ('arg', _('y'))), ('arg', _('x')), ('add', _(1), v._a)), ('arg', _('y')), _(2)))

  assert_syntax_error('arg_subs_missing_equals',
    "Expected a '='.",
    "1 + ?x | ?x + 2", "",
    "           ^")

  assert_syntax_error('arg_subs_unmatched_indices',
    "Left and right hand side should have the same indices, got 'kl' and 'jk'.",
    "a23_ij + (?x_ij | ?x_kl = a23_jk) + a23_ij", "ij",
    "                  ^^^^^^^^^^^^^^")

  assert_syntax_error('arg_subs_lhs_repeated_index',
    "Repeated indices are not allowed on the left hand side.",
    "a23_ij + (?x_ij | ?x_kk = a23_jk) + 2", "ij",
    "                  ^^^^^")

  assert_syntax_error('arg_subs_lhs_numeric_index',
    "Numeric indices are not allowed on the left hand side.",
    "a23_ij + (?x_ij | ?x_k0 = a23_0k) + 2", "ij",
    "                  ^^^^^")

  assert_syntax_error('arg_subs_not_an_argument',
    "Expected an argument, e.g. '?argname'.",
    "?x_ij | x_ij = 1_ij", "ij",
    "        ^^")

  # TRANSPOSE

  assert_syntax_error('transpose_duplicate_indices',
    "Cannot transpose from 'ij' to 'jii': duplicate indices.",
    "a23_ij", "jii",
    "^^^^^^")

  assert_syntax_error('transpose_indices_differ',
    "Cannot transpose from 'ij' to 'jk': indices differ.",
    "a23_ij", "jk",
    "^^^^^^")

  # STACK

  assert_ast('stack_1_0d', '<a>_i', 'i', ('append_axis', v._a, _(1)))
  assert_ast('stack_1_1di_1', '<a2_i>_i', 'i', v._a2)
  assert_ast('stack_2_0d_0d', '<a, a>_i', 'i', ('concatenate', ('append_axis', v._a, _(1)), ('append_axis', v._a, _(1))))
  assert_ast('stack_2_1di_0d', '<a2_i, a>_i', 'i', ('concatenate', v._a2, ('append_axis', v._a, _(1))))
  assert_ast('stack_3_2di_1d_1d', '<a23_ij, a3_j, 1_j>_i', 'ij', ('concatenate', v._a23, ('transpose',  ('append_axis', v._a3, _(1)), _((1,0))), ('transpose', ('append_axis', ('append_axis', _(1), _(3)), _(1)), _((1,0)))))

  assert_syntax_error('stack_no_indices',
    "Expected 1 index.",
    "<1, a2_i> + a3_i", "i",
    "         ^")

  assert_syntax_error('stack_too_many_indices',
    "Expected 1 index, got 2.",
    "<1, a2_i>_ij + a3_i", "i",
    "          ^^")

  assert_syntax_error('stack_numeric_index',
    "Expected a non-numeric index, got '1'.",
    "<1, a2_i>_1 + a3_i", "i",
    "          ^")

  assert_syntax_error('stack_0',
    "Cannot stack 0 arrays.",
    "1_i + <>_i + 1_i", "i",
    "      ^^^^")

  assert_syntax_error('stack_unmatched_indices',
    "Cannot stack arrays with unmatched indices (excluding the stack index 'i'): j, ijk.",
    "1_ij + <a2_j, a222_ijk>_i + 1_ij", "ij",
    "       ^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('stack_undetermined_length',
    "Cannot determine the length of the stack axis, because the length at 12 is unknown.",
    "1_i + <a, 1_i>_i + 1_i", "i",
    "            ^")

  # FIXME: the following should work
  # 'a2_j a2_i + <0j, δ_ij>_i'

  # FUNCTION

  assert_ast('function_0d', 'func1(a)', '', ('call', _('func1'), v._a))
  assert_ast('function_1d', 'func1(a2_i)', 'i', ('call', _('func1'), v._a2))
  assert_ast('function_2d', 'func1(a23_ij)', 'ij', ('call', _('func1'), v._a23))
  assert_ast('function_0d_0d', 'func2(a, a)', '', ('call', _('func2'), v._a, v._a))
  assert_ast('function_1d_1d', 'func2(a2_i, a2_i)', 'i', ('call', _('func2'), v._a2, v._a2))
  assert_ast('function_2d_2d', 'func2(a23_ij, a32_ji)', 'ij', ('call', _('func2'), v._a23, ('transpose', v._a32, _((1,0)))))
  assert_ast('function_2d_2d_2d', 'func3(a23_ij, a22_ik a23_kj, a23_ij)', 'ij', ('call', _('func3'), v._a23, ('sum', ('mul', ('append_axis', v._a22, _(3)), ('transpose', ('append_axis', v._a23, _(2)), _((2,0,1)))), _(1)), v._a23))

  assert_syntax_error('function_invalid_nargs',
    "Function 'func1' takes 1 argument, got 2.",
    "1 + func1(a, a) + 1", "",
    "    ^^^^^^^^^^^")

  assert_syntax_error('function_unmatched_indices',
    "Cannot align arrays with unmatched indices: ij, ij, jk.",
    "1_ij + func3(a23_ij, a23_ij, a23_jk) + 1_ij", "ij",
    "       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('function_unmatched_shape',
    "Shapes at index 'i' differ: 2, 3.",
    "1_ij + func2(a23_ij, a33_ij) + 1_ij", "ij",
    "       ^^^^^^^^^^^^^^^^^^^^^")

  assert_syntax_error('function_unmatched_shape',
    "Unknown function 'funcX'.",
    "1_ij + funcX(a23_ij) + 1_ij", "ij",
    "       ^^^^^")

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
