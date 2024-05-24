from . import api
from .. import log
import os

run_id = os.environ.get('GITHUB_RUN_ID')
if not run_id:
    raise RuntimeError('ERROR: environment variable GITHUB_RUN_ID not set')

for artifact in api.list_workflow_run_artifacts(run_id):
    if artifact['name'].startswith('_coverage_'):
        log.debug(f'deleting {artifact["name"]}')
        api.delete_artifact(artifact['id'])
