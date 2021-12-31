import sys
import re
import os.path
from typing import Sequence
from xml.etree import ElementTree
from pathlib import Path
from coverage import Coverage

paths = []
for path in sys.path:
    try:
        paths.append(str(Path(path).resolve()).lower()+os.path.sep)
    except FileNotFoundError:
        pass
paths = list(sorted(paths, key=len, reverse=True))
unix_paths = tuple(p.replace('\\', '/') for p in paths)
packages = tuple(p.replace('/', '.') for p in unix_paths)

dst = Path('coverage.xml')

# Generate `coverage.xml` with absolute file and package names.
cov = Coverage()
cov.load()
cov.xml_report(outfile=str(dst))

# Load the report, remove the largest prefix in `packages` from attribute
# `name` of element `package`, if any, and similarly the largest prefix in
# `paths` from attribute `filename` of element `class` and from the content of
# element `source`. Matching prefixes is case insensitive for case insensitive
# file systems.


def remove_prefix(value: str, prefixes: Sequence[str]) -> str:
    lvalue = value.lower()
    for prefix in prefixes:
        if lvalue.startswith(prefix):
            return value[len(prefix):]
    return value


root = ElementTree.parse(str(dst))
for elem in root.iter('package'):
    for package in packages:
        name = elem.get('name')
        if name:
            elem.set('name', remove_prefix(name, packages))
    for elem in root.iter('class'):
        filename = elem.get('filename')
        if filename:
            elem.set('filename', remove_prefix(filename, unix_paths))
    for elem in root.iter('source'):
        text = elem.text
        if text:
            elem.text = remove_prefix(text, paths)
root.write('coverage.xml')
