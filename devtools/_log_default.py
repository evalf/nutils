import functools
from typing import Any, Optional, Union


def _log_msg(*msg, color: Optional[str] = None, title: Optional[str] = None, file: Optional[str] = None, line: Union[int, range, None] = None, column: Union[int, range, None] = None):
    params = []

    if file:
        params.append(f'file={file}')

    if isinstance(line, range) and line.start == line.stop + 1 and line.step == 1:
        line = line.stop
    if isinstance(line, int):
        params.append(f'line={line}')
    elif isinstance(line, range) and line.start < line.stop and line.step == 1:
        params.append(f'lines={line.start}-{line.stop-1}')

    if isinstance(column, range) and column.start == column.stop + 1 and column.step == 1:
        column = column.stop
    if isinstance(column, int):
        params.append(f'column={column}')
    elif isinstance(column, range) and column.start < column.stop and column.step == 1:
        params.append(f'columns={column.start}-{column.stop-1}')

    if title:
        params.append(f'title={title}')

    if color:
        print(f'\033[{color}m', end='')

    if params:
        print('--', ','.join(params))

    print(*msg)

    if color:
        print(f'\033[0m', end='')


debug = functools.partial(_log_msg)
notice = functools.partial(_log_msg, color='1;36')
warning = functools.partial(_log_msg, color='1;33')
error = functools.partial(_log_msg, color='1;31')


def info(*args: Any) -> None:
    print('\033[1;37m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print(line)
    print('\033[0m', end='', flush=True)


def set_output(key: str, value: str) -> None:
    print('\033[1;35mOUTPUT: {}={}\033[0m'.format(key, value))
