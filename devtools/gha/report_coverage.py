from .. import log
import array
import itertools
import json
import os
from pathlib import Path
import subprocess

cov_dir = Path() / 'target' / 'coverage'

# Load and merge coverage data.
coverage = {}
for part in cov_dir.glob('*.json'):
    with part.open('r') as f:
        part = json.load(f)
        for file_name, part_file_coverage in part.items():
            coverage.setdefault(file_name, []).append(part_file_coverage)
coverage = {file_name: array.array('B', list(map(max, *file_coverage)) if len(file_coverage) > 1 else file_coverage[0]) for file_name, file_coverage in coverage.items()}

# Generate lcov.
with (cov_dir / 'coverage.info').open('w') as f:
    print('TN:unittest', file=f)
    for file_name, file_coverage in sorted(coverage.items()):
        print(f'SF:{file_name}', file=f)
        print('FNF:0', file=f)
        print('FNH:0', file=f)
        print('BRF:0', file=f)
        print('BRH:0', file=f)
        for i, status in enumerate(file_coverage[1:], 1):
            if status:
                print(f'DA:{i},{status - 1}', file=f)
        hit = sum(status == 2 for status in file_coverage)
        found = sum(status != 0 for status in file_coverage)
        print(f'LH:{hit}', file=f)
        print(f'LF:{found}', file=f)
        print('end_of_record', file=f)

# If this is a PR, build patch coverage data.
patch_coverage = {}
if os.environ.get('GITHUB_EVENT_NAME', None) == 'pull_request':
    base = os.environ.get('GITHUB_BASE_REF')
    subprocess.run(['git', 'fetch', '--depth=1', 'origin', base], check=True, stdin=subprocess.DEVNULL)
    patch = iter(subprocess.run(['git', 'diff', '-U0', f'origin/{base}', '--'], check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True).stdout.splitlines())
    for line in patch:
        # Skip to a file with coverage.
        if not line.startswith('+++ b/'):
            continue
        file_name = line[6:].rstrip()
        if (file_coverage := coverage.get(file_name)) is None:
            continue
        # Copy the full coverage and mask out unchanged lines.
        patch_coverage[file_name] = patch_file_coverage = array.array('B', file_coverage)
        prev_offset = 0
        for line in patch:
            if line.startswith('--- '):
                break
            if line.startswith('@@ '):
                chunk = line.split(' ')[2]
                assert chunk.startswith('+')
                if ',' in chunk:
                    offset, count = map(int, chunk[1:].split(','))
                else:
                    offset = int(chunk[1:])
                    count = 1
                for i in range(prev_offset, offset):
                    patch_file_coverage[i] = 0
                prev_offset = offset + count
        for i in range(prev_offset, len(patch_file_coverage)):
            patch_file_coverage[i] = 0

    # Annotate lines without coverage.
    for file_name, file_coverage in sorted(patch_coverage.items()):
        i = 0
        while i < len(file_coverage):
            j = i
            if file_coverage[i] == 1:
                while j + 1 < len(file_coverage) and file_coverage[j + 1] == 1:
                    j += 1
                if i == j:
                    log.warning(f'Line {i} of `{file_name}` is not covered by tests.', file=file_name, line=i, title='Line not covered')
                else:
                    log.warning(f'Lines {i}–{j} of `{file_name}` are not covered by tests.', file=file_name, line=range(i, j+1), title='Lines not covered')
            i = j + 1

# Generate summary.
header = ['Name', 'Stmts', 'Miss', 'Cover']
align = ['<', '>', '>', '>']
if patch_coverage:
    header += ['Patch stmts', 'Patch miss', 'Patch cover']
    align += ['>'] * 3
table = []
def row_stats(*data):
    hit = 0
    miss = 0
    for data in data:
        hit += data.count(2)
        miss += data.count(1)
    total = hit + miss
    percentage = 100 * hit / (hit + miss) if hit + miss else 100.
    return [str(total), str(miss), f'{percentage:.1f}%']
for file_name, file_coverage in sorted(coverage.items()):
    row = [f'`{file_name}`'] + row_stats(file_coverage)
    if (patch_file_coverage := patch_coverage.get(file_name)):
        row += row_stats(patch_file_coverage)
    elif patch_coverage:
        row += [''] * 3
    table.append(row)
row = ['TOTAL'] + row_stats(*coverage.values())
if patch_coverage:
    row += row_stats(*patch_coverage.values())
table.append(row)
with open(os.environ.get('GITHUB_STEP_SUMMARY', None) or cov_dir / 'summary.md', 'w') as f:
    width = tuple(max(map(len, columns)) for columns in zip(header, *table))
    print('| ' + ' | '.join(f'{{:<{w}}}'.format(h) for w, h in zip(width, header)) + ' |', file=f)
    print('| ' + ' | '.join(':' + '-' * (w - 1) if a == '<' else '-' * (w - 1) + ':' for a, w in zip(align, width)) + ' |', file=f)
    fmt = '| ' + ' | '.join(f'{{:{a}{w}}}' for a, w in zip(align, width)) + ' |'
    for row in table:
        print(fmt.format(*row), file=f)
