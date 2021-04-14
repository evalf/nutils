from .. import log
import sys, site, os

libsubdir = 'lib'
prefixes = list(site.PREFIXES)
if hasattr(site, 'getuserbase'):
  prefixes.append(site.getuserbase())
for prefix in prefixes:
  libdir = os.path.join(prefix, libsubdir)
  if not os.path.exists(os.path.join(libdir, 'libmkl_rt.so.1')):
    continue
  log.set_output('libdir', libdir)
  break
else:
  log.error('cannot find {} in any of the following dirs: {}'.format('libmkl_rt.so.1', ' '.join(prefixes)))

lib = os.path.join(libdir, 'libmkl_rt.so')
if not os.path.exists(lib):
  os.symlink('libmkl_rt.so.1', lib)

github_env = os.environ.get('GITHUB_ENV')
assert github_env
ld_library_path = ':'.join(filter(None, (os.environ.get('LD_LIBRARY_PATH', ''), libdir)))
with open(github_env, 'a') as f:
  print('LD_LIBRARY_PATH={}'.format(ld_library_path), file=f)
