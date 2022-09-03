import argparse
from .. import log
from .._git import Git
from . import OFFICIAL_CONTAINER_REPO, Container, get_container_tag_from_ref

parser = argparse.ArgumentParser(description='build an OCI compatible container base image')
parser.add_argument('--name', metavar='NAME', help=f'the name to attach to the image; defaults to a {OFFICIAL_CONTAINER_REPO}:TAG where TAG is based on the current HEAD')
args = parser.parse_args()

if args.name and ':' in args.name:
    image_name = args.name
else:
    image_name = f'{args.name or OFFICIAL_CONTAINER_REPO}:_base_{get_container_tag_from_ref(Git().head_ref)}'

log.info(f'building container base image with name `{image_name}`')

with Container.new_from('debian:bullseye', network='host') as container:
    container.run('sed', '-i', 's/ main$/ main contrib non-free/', '/etc/apt/sources.list')
    container.run('apt', 'update')
    # Package `libtbb2` is required when using Intel MKL with environment
    # variable `MKL_THREADING_LAYER` set to `TBB`, which is nowadays the default.
    container.run('apt', 'install', '-y', '--no-install-recommends', 'python3', 'python3-pip', 'python3-wheel', 'python3-ipython', 'python3-numpy', 'python3-scipy', 'python3-matplotlib', 'python3-pil', 'libmkl-rt', 'libomp-dev', 'libtbb2', 'python3-gmsh', env=dict(DEBIAN_FRONTEND='noninteractive'))
    container.add_label('org.opencontainers.image.url', 'https://github.com/evalf/nutils')
    container.add_label('org.opencontainers.image.source', 'https://github.com/evalf/nutils')
    container.add_label('org.opencontainers.image.authors', 'Evalf')

    image_id = container.commit(image_name)

log.set_output('id', image_id)
log.set_output('name', image_name)
