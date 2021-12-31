from typing import Generator
from contextlib import contextmanager
from subprocess import CompletedProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from . import run


class Git:

    def __init__(self, root: str = '.') -> None:
        self._root = root
        self.path = Path(root)

    def _run(self, *args: str, **kwargs) -> CompletedProcess:
        return run(*args, cwd=self._root, **kwargs)

    @property
    def head_ref(self):
        return self._run('git', 'symbolic-ref', 'HEAD', capture_output=True).stdout.decode().strip()

    def get_commit_from_rev(self, rev: str) -> str:
        return self._run('git', 'rev-parse', '--verify', rev, capture_output=True).stdout.decode().strip()

    def get_commit_timestamp(self, rev: str) -> int:
        return int(self._run('git', 'show', '-s', '--format=%ct', rev, capture_output=True).stdout.decode().strip())

    @contextmanager
    def worktree(self, rev: str, *, detach: bool = False) -> Generator['Git', None, None]:
        wt = ''
        try:
            with TemporaryDirectory() as wt:
                self._run('git', 'worktree', 'add', *(['--detach'] if detach else []), wt, rev)
                yield Git(wt)
        finally:
            if wt:
                self._run('git', 'worktree', 'remove', wt)
