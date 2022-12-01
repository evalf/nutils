import os
from pathlib import Path
from typing import Any

debug = print


def info(*args: Any) -> None:
    print('\033[1;37m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print(line)
    print('\033[0m', end='', flush=True)


def warning(*args: Any) -> None:
    for line in ' '.join(map(str, args)).split('\n'):
        print('::warning ::{}'.format(line))


def error(*args: Any) -> None:
    for line in ' '.join(map(str, args)).split('\n'):
        print('::error ::{}'.format(line))


def set_output(key: str, value: str) -> None:
    Path(os.environ['GITHUB_OUTPUT']).open('a').write(f'{key}={value}\n')
    print('\033[1;35mOUTPUT: {}={}\033[0m'.format(key, value))
