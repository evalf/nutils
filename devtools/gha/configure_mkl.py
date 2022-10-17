from .. import log
import re
import sys
import site
import os
from pathlib import Path

if sys.platform == 'linux':
    libsubdir = 'lib',
    re_libmkl = re.compile('libmkl_rt[.]so[.][0-9]+')
elif sys.platform == 'darwin':
    libsubdir = 'lib',
    re_libmkl = re.compile('libmkl_rt[.][0-9]+[.]dylib')
elif sys.platform == 'win32':
    libsubdir = 'Library', 'bin'
    re_libmkl = re.compile('mkl_rt[.][0-9]+[.]dll')
else:
    log.error(f'unsupported platform: {sys.platform}')
    raise SystemExit(1)

prefixes = list(map(Path, site.PREFIXES))
if hasattr(site, 'getuserbase'):
    prefixes.append(Path(site.getuserbase()))

libdirs = {libdir := prefix.joinpath(*libsubdir).resolve() for prefix in prefixes}
libs = {file for libdir in libdirs if libdir.is_dir() for file in libdir.iterdir() if re_libmkl.match(file.name)}
if len(libs) == 0:
    log.error('cannot find MKL in any of {}'.format(', '.join(map(str, libdirs))))
    raise SystemExit(1)
elif len(libs) != 1:
    log.error('found MKL at more than one location: {}'.format(', '.join(map(str, libs))))
    raise SystemExit(1)
else:
    lib, = libs
    log.info(f'using MKL at {lib}')

with open(os.environ['GITHUB_ENV'], 'a') as f:
    print(f'NUTILS_MATRIX_MKL_LIB={lib}', file=f)
