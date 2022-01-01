import os
import warnings

_env = dict.fromkeys(filter(None, os.getenv('NUTILS_DEBUG', '').lower().split(':')), True)
_all = _env.pop('all', False)

sparse = _env.pop('sparse', _all or __debug__)  # check sparse chunks in evaluable
lower = _env.pop('lower', _all or __debug__)  # check lowered shape, dtype in function
evalf = _env.pop('evalf', _all)  # check evaluated arrays in evaluable

if _env:
    warnings.warn('unused debug flags: {}'.format(', '.join(_env)))
