from .. import log
import sys, site, os

libname = 'libmkl_rt.so'
libsubdir = 'lib'
prefixes = list(site.PREFIXES)
if hasattr(site, 'getuserbase'):
  prefixes.append(site.getuserbase())
for prefix in prefixes:
  libdir = os.path.join(prefix, libsubdir)
  if not os.path.exists(os.path.join(libdir, libname)):
    continue
  log.set_output('libdir', libdir)
  break
else:
  log.error('cannot find {} in any of the following dirs: {}'.format(libname, ' '.join(prefixes)))
