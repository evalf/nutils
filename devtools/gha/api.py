from http.client import HTTPSConnection
import json
import os
import sys

_token = os.environ.get('GITHUB_TOKEN')
if not _token:
    import getpass
    _token = getpass.getpass('GitHub token: ')

repo = os.environ.get('GITHUB_REPOSITORY', 'evalf/nutils')

host = 'api.github.com'
_conn = HTTPSConnection(host)

def _request(method, url, *, desired_status=200):
    _conn.request(
        method,
        url,
        headers={
            'Host': host,
            'User-Agent': 'Nutils devtools',
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {_token}',
            'X-GitHub-Api-Version': '2022-11-28',
        },
    )
    response = _conn.getresponse()
    if response.status != desired_status:
        raise RuntimeError(f'ERROR: {method} https://{host}{url} failed: {response.status} {response.reason}')
    return response.read()

def list_workflow_run_artifacts(run_id: str):
    # TODO: implement pagination: https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api?apiVersion=2022-11-28
    return json.loads(_request('GET', f'/repos/{repo}/actions/runs/{run_id}/artifacts'))['artifacts']

def delete_artifact(artifact_id: str):
    _request('DELETE', f'/repos/{repo}/actions/artifacts/{artifact_id}', desired_status=204)
