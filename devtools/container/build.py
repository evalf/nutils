import argparse, textwrap, typing, sys, json
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from .. import log, run
from .._git import Git
from . import OFFICIAL_CONTAINER_REPO, Container, Mount, get_container_tag_from_ref

parser = argparse.ArgumentParser(description='build an OCI compatible container image')
parser.add_argument('--base', required=True, help='the base image to build upon; example: `docker.io/evalf/nutils` or `docker.io/evalf/nutils:_base_latest`')
parser.add_argument('--name', metavar='NAME', help='the name to attach to the image; defaults to `{OFFICIAL_CONTAINER_REPO}:TAG` where TAG is based on REV')
parser.add_argument('--revision', '--rev', metavar='REV', help='set image label `org.opencontainers.image.revision` to the commit hash refered to by REV')
parser.add_argument('--build-from-worktree', action='store_const', const=True, default=False, help='build from the worktree')
args = parser.parse_args()

if not args.build_from_worktree and not args.revision:
  raise SystemExit('either `--revision` or `--build-from-worktree` must be specified')

rev = args.revision or 'HEAD'
git = Git()
commit = git.get_commit_from_rev(rev)

if args.build_from_worktree and args.revision and (head_commit := git.get_commit_from_rev('HEAD')) != commit:
  raise SystemExit(f'`HEAD` points to `{head_commit}` but `--revision={args.revision}` points to `{commit}`')

if args.name and ':' in args.name:
  image_name = args.name
  tag = image_name.rsplit(':', 1)[-1]
else:
  tag = get_container_tag_from_ref(rev)
  image_name = f'{args.name or OFFICIAL_CONTAINER_REPO}:{tag}'

base = args.base
if ':' not in base.split('/')[-1]:
  base = f'{base}:_base_{tag}'

with ExitStack() as stack:

  if args.build_from_worktree:
    dist = Path()/'dist'
    examples = Path()/'examples'
    log.info(f'installing Nutils from {dist}')
    log.info(f'using examples from {examples}')
  else:
    # Check out Nutils in a clean working tree, build a wheel and use the working tree as `src`.
    log.info(f'using examples from commit {commit}')
    src = stack.enter_context(git.worktree(typing.cast(str, commit)))
    log.info(f'building wheel from commit {commit}')
    run(sys.executable, 'setup.py', 'bdist_wheel', cwd=str(src.path), env=dict(SOURCE_DATE_EPOCH=str(src.get_commit_timestamp('HEAD'))))
    dist = src.path/'dist'
    examples = src.path/'examples'

  dist = Path(dist)
  if not dist.exists():
    raise SystemExit(f'cannot find dist dir: {dist}')
  examples = Path(examples)
  if not examples.exists():
    raise SystemExit(f'cannot find example dir: {examples}')

  container = stack.enter_context(Container.new_from(base, network='none', mounts=[Mount(src=dist, dst='/mnt')]))

  container.run('pip', 'install', '--no-cache-dir', '--no-index', '--find-links', 'file:///mnt/', 'nutils', env=dict(PYTHONHASHSEED='0'))
  container.add_label('org.opencontainers.image.url', 'https://github.com/evalf/nutils')
  container.add_label('org.opencontainers.image.source', 'https://github.com/evalf/nutils')
  container.add_label('org.opencontainers.image.authors', 'Evalf')
  if commit:
    container.add_label('org.opencontainers.image.revision', commit)
  container.add_volume('/app')
  container.add_volume('/log')
  container.set_workingdir('/app')
  container.set_entrypoint('/usr/bin/python3', '-u')
  container.set_cmd('help')
  container.add_env('NUTILS_MATRIX', 'mkl')
  container.add_env('NUTILS_OUTDIR', '/log')
  container.add_env('OMP_NUM_THREADS', '1')
  # Copy examples and generate a help message.
  msg = textwrap.dedent('''\
    Usage
    =====

    This container includes the following examples:

    ''')
  for example in sorted(examples.glob('*.py')):
    if example.name == '__init__.py':
      continue
    container.copy(example, dst=f'/app/{example.stem}')
    msg += f'*   {example.stem}\n'
  msg += textwrap.dedent(f'''\

    To run an example, add the name of the example and any additional arguments to the command line.
    For example, you can run example `laplace` with

        docker run --rm -it {image_name} laplace

    HTML log files are generated in the `/log` directory of the container. If
    you want to store the log files in `/path/to/log` on the
    host, add `-v /path/to/log:/log` to the command line before the
    name of the image. Extending the previous example:

        docker run --rm -it -v /path/to/log:/log {image_name} laplace

    To run a Python script in this container, bind mount the directory
    containing the script, including all files necessary to run the script,
    to `/app` in the container and add the relative path to the script and
    any arguments to the command line. For example, you can run
    `/path/to/script/example.py` with Docker using

        docker run --rm -it -v /path/to/script:/app:ro {image_name} example.py

    Installed software
    ==================

    ''')

  pip_list = {item['name']: item['version'] for item in json.loads(container.run('python3', '-m', 'pip', 'list', '--format', 'json', capture_output=True).stdout.decode())}
  v = dict(
    nutils=pip_list['nutils'] + (f'  (git: {commit})' if commit else ''),
    python=container.run('python3', '--version', capture_output=True).stdout.decode().replace('Python', '').strip(),
    **{name: pip_list[name] for name in ('numpy', 'scipy', 'matplotlib')})
  msg += ''.join(f'{name:18}{version}\n' for name, version in v.items())
  with NamedTemporaryFile('w') as f:
    f.write(f'print({msg!r})')
    f.flush()
    container.copy(f.name, dst='/app/help')

  image_id = container.commit(image_name)

log.set_output('id', image_id)
log.set_output('name', image_name)
log.set_output('tag', tag)
log.set_output('base', base)
