from typing import Any

debug = print


def info(*args: Any) -> None:
    print('\033[1;37m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print(line)
    print('\033[0m', end='', flush=True)


def warning(*args: Any) -> None:
    print('\033[1;33m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print('WARNING: {}'.format(line))
    print('\033[0m', end='', flush=True)


def error(*args: Any) -> None:
    print('\033[1;31m', end='')
    for line in ' '.join(map(str, args)).split('\n'):
        print('ERROR: {}'.format(line))
    print('\033[0m', end='', flush=True)


def set_output(key: str, value: str) -> None:
    print('\033[1;35mOUTPUT: {}={}\033[0m'.format(key, value))
