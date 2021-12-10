from .. import log
import sys, site, os

libsubdir = 'lib'
prefixes = list(site.PREFIXES)
if hasattr(site, 'getuserbase'):
  prefixes.append(site.getuserbase())

candidates = [os.path.join(prefix, libsubdir, 'libmkl_rt.so' + ext) for prefix in prefixes for ext in ('.1', '.2')]
for path in candidates:
  if os.path.exists(path):
    break
else:
  log.error('cannot find any of {}'.format(' '.join(candidates)))
  raise SystemExit(1)

lib = os.path.splitext(path)[0]
if not os.path.exists(lib):
  os.symlink(path, lib)

ld_library_path = os.pathsep.join(filter(None, (os.environ.get('LD_LIBRARY_PATH', ''), os.path.dirname(path))))
with open(os.environ['GITHUB_ENV'], 'a') as f:
  print('LD_LIBRARY_PATH={}'.format(ld_library_path), file=f)
