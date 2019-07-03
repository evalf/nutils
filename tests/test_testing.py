from nutils.testing import *
from nutils import numeric
import numpy


class almostequal64(TestCase):

  maxDiff = 5000
  actual = numpy.arange(16).reshape(2, 4, 2)
  desired = 'eNpjYAgyX2tRbsluVW51yUreOsl6rvVFa0YbXZtQmyqbOTYAmAYJvQ=='

  def test_equal(self):
    self.assertAlmostEqual64(self.actual, self.desired)

  def test_notequal(self):
    with self.assertRaises(AssertionError) as cm:
      self.assertAlmostEqual64(self.actual, 'eNpjYFhrwW51ySrJ+qK1rk2VzR6b7zY6tjG27bbrba/YfrcFALIODB0=')
    self.assertEqual(str(cm.exception), '''15/16 values do not match up to atol=2.00e-15, rtol=2.00e-03:
[0, 0, 1] desired: +2.0014e+00, actual: +1.0000e+00, spacing: 4.0e-03
[0, 1, 0] desired: +3.9981e+00, actual: +2.0000e+00, spacing: 8.0e-03
[0, 1, 1] desired: +6.0004e+00, actual: +3.0000e+00, spacing: 1.2e-02
[0, 2, 0] desired: +8.0031e+00, actual: +4.0000e+00, spacing: 1.6e-02
[0, 2, 1] desired: +9.9925e+00, actual: +5.0000e+00, spacing: 2.0e-02
...
[1, 1, 1] desired: +2.2017e+01, actual: +1.1000e+01, spacing: 4.4e-02
[1, 2, 0] desired: +2.3995e+01, actual: +1.2000e+01, spacing: 4.8e-02
[1, 2, 1] desired: +2.5993e+01, actual: +1.3000e+01, spacing: 5.2e-02
[1, 3, 0] desired: +2.7990e+01, actual: +1.4000e+01, spacing: 5.6e-02
[1, 3, 1] desired: +3.0019e+01, actual: +1.5000e+01, spacing: 6.0e-02
If this is expected, update the base64 string to:
''' + self.desired)

  def test_fail(self):
    with self.assertRaises(AssertionError) as cm:
      self.assertAlmostEqual64(self.actual, 'invalid')
    self.assertEqual(str(cm.exception), '''failed to decode data: Incorrect padding
If this is expected, update the base64 string to:
''' + self.desired)
