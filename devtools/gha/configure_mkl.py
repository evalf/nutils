from .. import log
import sys
import site
import os
from pathlib import Path

libsubdir = 'lib'
prefixes = list(map(Path, site.PREFIXES))
if hasattr(site, 'getuserbase'):
    prefixes.append(Path(site.getuserbase()))

candidates = [prefix / libsubdir / f'libmkl_rt.so{ext}' for prefix in prefixes for ext in ('.1', '.2')]
for path in candidates:
    if path.exists():
        break
else:
    log.error('cannot find any of {}'.format(' '.join(map(str, candidates))))
    raise SystemExit(1)

ld_library_path = os.pathsep.join(filter(None, (os.environ.get('LD_LIBRARY_PATH', ''), str(path.parent))))
with open(os.environ['GITHUB_ENV'], 'a') as f:
    print('LD_LIBRARY_PATH={}'.format(ld_library_path), file=f)
