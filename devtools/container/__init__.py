import uuid
import json
import shlex
from typing import List, Sequence, Optional, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import NamedTemporaryFile
from os import PathLike
from .. import run, log

OFFICIAL_CONTAINER_REPO = 'ghcr.io/evalf/nutils'


@dataclass
class Mount:
    src: Path
    dst: str
    rw: bool = False


class Container:

    built_images: List[str] = []

    @classmethod
    @contextmanager
    def new_from(cls, from_image: str, *, mounts: Sequence[Mount] = (), network: Optional[str] = None) -> Generator['Container', None, None]:
        id = f'work-{uuid.uuid4()}'
        args = ['buildah', 'from', '--name', id]
        if network:
            args += ['--network', network]
        for mnt in mounts:
            args += ['--volume', f'{mnt.src.resolve()}:{mnt.dst}:{"rw" if mnt.rw else "ro"}']
        args += [from_image]
        log.debug('FROM', from_image)
        run(*args, print_cmdline=False)
        try:
            yield cls(id)
        finally:
            log.debug('destroy container')
            run('buildah', 'rm', id, print_cmdline=False)
            del id

    def __init__(self, id: str) -> None:
        self._id = id

    def run(self, *args: str, env: Mapping[str, str] = {}, capture_output=False) -> CompletedProcess:
        cmdline = []
        if env:
            cmdline.append('env')
            for key, value in env.items():
                assert '-' not in key
                cmdline.append(f'{key}={value}')
        cmdline.extend(args)
        log.debug('RUN', *(f'{key}={value}' for key, value in env.items()), *args)
        return run('buildah', 'run', '--', self._id, *cmdline, print_cmdline=False, capture_output=capture_output)

    def copy(self, *src: PathLike, dst: str) -> None:
        log.debug('COPY', *src, dst)
        run('buildah', 'copy', self._id, *map(str, src), dst, print_cmdline=False)

    def add_env(self, key: str, value: str) -> None:
        log.debug('ENV', f'{key}={value}')
        run('buildah', 'config', '--env', f'{key}={value}', self._id, print_cmdline=False)

    def add_label(self, key: str, value: str) -> None:
        log.debug('LABEL', f'{key}={value}')
        run('buildah', 'config', '--label', f'{key}={value}', self._id, print_cmdline=False)

    def add_volume(self, path: str) -> None:
        log.debug('VOLUME', path)
        run('buildah', 'config', '--volume', path, self._id, print_cmdline=False)

    def set_workingdir(self, path: str) -> None:
        log.debug('WORKDIR', path)
        run('buildah', 'config', '--workingdir', path, self._id, print_cmdline=False)

    def set_entrypoint(self, *cmd: str) -> None:
        log.debug('ENTRYPOINT', json.dumps(cmd))
        run('buildah', 'config', '--entrypoint', json.dumps(cmd), self._id, print_cmdline=False)

    def set_cmd(self, *cmd: str) -> None:
        log.debug('CMD', json.dumps(cmd))
        run('buildah', 'config', '--cmd', ' '.join(map(shlex.quote, cmd)), self._id, print_cmdline=False)

    def commit(self, name: Optional[str] = None) -> str:
        with NamedTemporaryFile('r') as f:
            args = ['buildah', 'commit', '--iidfile', f.name, '--format', 'oci', self._id]
            if name:
                args.append(name)
            run(*args)
            image_id = f.read()
            assert image_id
            self.built_images.append(image_id)
            log.debug(f'created container image with id {image_id}')
            return image_id


def get_container_tag_from_ref(ref: str) -> str:
    if not ref.startswith('refs/'):
        raise SystemExit(f'expected an absolute ref, e.g. `refs/heads/master`, but got `{ref}`')
    elif ref == 'refs/heads/master':
        return 'latest'
    elif ref.startswith('refs/heads/release/'):
        return ref[19:]
    else:
        raise SystemExit(f'cannot determine container tag from ref `{ref}`')
