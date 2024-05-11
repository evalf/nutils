import functools
import os
from pathlib import Path
from typing import Any, Optional, Union


def _log_msg(*msg, type: str, title: Optional[str] = None, file: Optional[str] = None, line: Union[int, range, None] = None, column: Union[int, range, None] = None):
    params = []

    if file:
        params.append(f'file={file}')

    if isinstance(line, range) and line.start == line.stop + 1 and line.step == 1:
        line = line.stop
    if isinstance(line, int):
        params.append(f'line={line}')
    elif isinstance(line, range) and line.start < line.stop and line.step == 1:
        params.append(f'line={line.start}')
        params.append(f'endLine={line.stop-1}')

    if isinstance(column, range) and column.start == column.stop + 1 and column.step == 1:
        column = column.stop
    if isinstance(column, int):
        params.append(f'col={column}')
    elif isinstance(column, range) and column.start < column.stop and column.step == 1:
        params.append(f'col={column.start}')
        params.append(f'endColumn={column.stop-1}')

    if title:
        params.append(f'title={title}')

    prefix = f'::{type} {",".join(params)}::'

    for line in ' '.join(map(str, msg)).split('\n'):
        print(prefix + line)


debug = functools.partial(_log_msg, type='debug')
notice = functools.partial(_log_msg, type='notice')
warning = functools.partial(_log_msg, type='warning')
error = functools.partial(_log_msg, type='error')


def info(*args: Any) -> None:
    print('\033[1;37m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print(line)
    print('\033[0m', end='', flush=True)


def set_output(key: str, value: str) -> None:
    Path(os.environ['GITHUB_OUTPUT']).open('a').write(f'{key}={value}\n')
    print('\033[1;35mOUTPUT: {}={}\033[0m'.format(key, value))
