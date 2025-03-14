import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import unittest

source = importlib.util.find_spec('nutils').origin
assert source.endswith(os.sep + '__init__.py')
source = source[:-11]
# Dictionary of file names to line coverage data. The line coverage data is a
# list of `int`s, one for each line, with the following meaning:
#
#     0: line contains no statements
#     1: line is not hit
#     2: line is hit
#
# The first entry of the line coverage data is unused; hence line 1 is entry 1
# in the line coverage data.
coverage = {}

if hasattr(sys, 'monitoring'):

    def start(code, _):
        if isinstance(code.co_filename, str) and code.co_filename.startswith(source) and not sys.monitoring.get_local_events(sys.monitoring.COVERAGE_ID, code):
            if (file_coverage := coverage.get(code.co_filename)) is None:
                with open(code.co_filename, 'rb') as f:
                    nlines = sum(1 for _ in f)
                coverage[code.co_filename] = file_coverage = [0] * (nlines + 1)
            for _, _, l in code.co_lines():
                if l:
                    file_coverage[l] = 1
            sys.monitoring.set_local_events(sys.monitoring.COVERAGE_ID, code, sys.monitoring.events.LINE)
            for obj in code.co_consts:
                if inspect.iscode(obj):
                    start(obj, None)
        return sys.monitoring.DISABLE

    def line(code, line_number):
        coverage[code.co_filename][line_number] = 2
        return sys.monitoring.DISABLE

    sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START, start)
    sys.monitoring.register_callback(sys.monitoring.COVERAGE_ID, sys.monitoring.events.LINE, line)
    sys.monitoring.use_tool_id(sys.monitoring.COVERAGE_ID, 'test')
    sys.monitoring.set_events(sys.monitoring.COVERAGE_ID, sys.monitoring.events.PY_START)

loader = unittest.TestLoader()
suite = loader.discover('tests', top_level_dir='.')
runner = unittest.TextTestRunner(buffer=True)
result = runner.run(suite)

if hasattr(sys, 'monitoring'):
    sys.monitoring.free_tool_id(sys.monitoring.COVERAGE_ID)

coverage = {file_name[len(source) - 7:].replace('\\', '/'): file_coverage for file_name, file_coverage in coverage.items()}
cov_dir = (Path() / 'target' / 'coverage')
cov_dir.mkdir(parents=True, exist_ok=True)
cov_file = cov_dir / (os.environ.get('COVERAGE_ID', 'coverage') + '.json')
with cov_file.open('w') as f:
    json.dump(coverage, f)

sys.exit(0 if result.wasSuccessful() else 1)
