import os
import argparse
from .. import log
from ..container import get_container_tag_from_ref

argparse.ArgumentParser().parse_args()

if os.environ.get('GITHUB_EVENT_NAME') == 'pull_request':
    ref = os.environ.get('GITHUB_BASE_REF')
    if not ref:
        raise SystemExit('`GITHUB_BASE_REF` environment variable is empty')
    base = '_base_' + get_container_tag_from_ref('refs/heads/' + ref)
    if sha := os.environ.get("GITHUB_SHA", ''):
        image = '_git_' + sha
    else:
        image = '_pr'
else:
    ref = os.environ.get('GITHUB_REF')
    if not ref:
        raise SystemExit('`GITHUB_REF` environment variable is empty')
    image = get_container_tag_from_ref(ref)
    base = '_base_' + image

log.set_output('base', base)
log.set_output('image', image)
