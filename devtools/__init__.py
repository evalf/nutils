#! /usr/bin/env python3

import sys
import os
import subprocess
import shlex
import typing
from typing import Mapping, Dict, Optional

GITHUB_ACTIONS = bool(os.environ.get('GITHUB_ACTIONS'))

if GITHUB_ACTIONS:

    from . import _log_gha as log

    orig_excepthook = sys.excepthook

    def new_excepthook(exctype, value, tb):
        if exctype is SystemExit and value.code not in (0, None):
            log.error(*value.args)
        else:
            orig_excepthook(exctype, value, tb)

    sys.excepthook = new_excepthook

else:

    from . import _log_default as log


def run(*args: str,
        check: bool = True,
        env: Mapping[str, str] = {},
        stdin: int = subprocess.DEVNULL,
        stdout: Optional[int] = None,
        capture_output: bool = False,
        print_cmdline: bool = True,
        cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    if print_cmdline:
        log.debug('running {}'.format(' '.join(map(shlex.quote, args))))
    if env:
        fullenv = typing.cast(Dict[str, str], dict(os.environ))
        fullenv.update(env)
    else:
        fullenv = None
    proc = subprocess.run(args, env=fullenv, stdin=stdin, stdout=stdout, capture_output=capture_output, cwd=cwd)
    if check and proc.returncode:
        if capture_output:
            log.error(proc.stderr.decode().rstrip())
        raise SystemExit('process exited with code {}'.format(proc.returncode))
    return proc
