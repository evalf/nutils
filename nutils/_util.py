"""
The util module provides a collection of general purpose methods.
"""

from . import numeric, warnings
from .types import arraydata
import stringly
import sys
import os
import numpy
import collections.abc
import inspect
import functools
import operator
import numbers
import pathlib
import ctypes
import io
import contextlib
import treelog
import datetime
import site
import re
from typing import Iterable, Sequence, Tuple

supports_outdirfd = os.open in os.supports_dir_fd and os.listdir in os.supports_fd

sum = functools.partial(functools.reduce, operator.add)
product = functools.partial(functools.reduce, operator.mul)


def cumsum(seq):
    offset = 0
    for i in seq:
        yield offset
        offset += i


def deep_reduce(f, a):
    '''Recursively apply function to lists or tuples, depth first.'''

    return f([deep_reduce(f, v) for v in a]) if isinstance(a, (list, tuple)) else a


def gather(items):
    gathered = collections.defaultdict(list)
    # NOTE: defaultdict is subclass of dict, so it maintains the insertion order
    for key, value in items:
        gathered[key].append(value)
    return gathered.items()


def pairwise(items, *, periodic=False):
    items = iter(items)
    try:
        first = a = next(items)
    except StopIteration:
        return
    for b in items:
        yield a, b
        a = b
    if periodic:
        yield a, first


def allequal(seq1, seq2):
    seq1 = iter(seq1)
    seq2 = iter(seq2)
    for item1, item2 in zip(seq1, seq2):
        if item1 != item2:
            return False
    if list(seq1) or list(seq2):
        return False
    return True


class NanVec(numpy.ndarray):
    'nan-initialized vector'

    def __new__(cls, length):
        vec = numpy.empty(length, dtype=float).view(cls)
        vec[:] = numpy.nan
        return vec

    @property
    def where(self):
        return ~numpy.isnan(self.view(numpy.ndarray))

    def __iand__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__iand__(other)
        where = self.where
        if numpy.isscalar(other):
            self[where] = other
        else:
            assert numeric.isarray(other) and other.shape == self.shape
            self[where] = other[where]
        return self

    def __and__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__and__(other)
        return self.copy().__iand__(other)

    def __ior__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__ior__(other)
        wherenot = ~self.where
        self[wherenot] = other if numpy.isscalar(other) else other[wherenot]
        return self

    def __or__(self, other):
        if self.dtype != float:
            return self.view(numpy.ndarray).__or__(other)
        return self.copy().__ior__(other)

    def __invert__(self):
        if self.dtype != float:
            return self.view(numpy.ndarray).__invert__()
        nanvec = NanVec(len(self))
        nanvec[numpy.isnan(self)] = 0
        return nanvec


def obj2str(obj):
    '''compact, lossy string representation of arbitrary object'''
    return '['+','.join(obj2str(item) for item in obj)+']' if isinstance(obj, collections.abc.Iterable) \
        else str(obj).strip('0').rstrip('.') or '0' if isinstance(obj, numbers.Real) \
        else str(obj)


class single_or_multiple:
    """
    Method wrapper, converts first positional argument to tuple: tuples/lists
    are passed on as tuples, other objects are turned into tuple singleton.
    Return values should match the length of the argument list, and are unpacked
    if the original argument was not a tuple/list.

    >>> class Test:
    ...   @single_or_multiple
    ...   def square(self, args):
    ...     return [v**2 for v in args]
    ...
    >>> T = Test()
    >>> T.square(2)
    4
    >>> T.square([2,3])
    (4, 9)

    Args
    ----
    f: :any:`callable`
        Method that expects a tuple as first positional argument, and that
        returns a list/tuple of the same length.

    Returns
    -------
    :
        Wrapped method.
    """

    def __init__(self, f):
        functools.update_wrapper(self, f)

    def __get__(self, instance, owner):
        return single_or_multiple(self.__wrapped__.__get__(instance, owner))

    def __call__(self, arg, *args, **kwargs):
        if isinstance(arg, map) or inspect.isgenerator(arg):
            arg = tuple(arg)
        # 1. flatten arg = [a, (b, [c, d]), e] to flatarg = [a, b, c, d, e].
        flatarg = []
        slices = []
        stack = [arg]
        while stack:
            obj = stack.pop()
            if isinstance(obj, (tuple, list)):
                stack.extend(reversed(obj))
                slices.append((len(flatarg), len(flatarg) + len(obj)))
            else:
                flatarg.append(obj)
        # 2. call wrapped function with flattened first argument
        retvals = tuple(self.__wrapped__(tuple(flatarg), *args, **kwargs))
        # 3. reconstruct nested sequences as tuples
        for i, j in reversed(slices):
            retvals = *retvals[:i], retvals[i:j], *retvals[j:]
        assert len(retvals) == 1
        return retvals[0]


def loadlib(name):
    '''Find and load a dynamic library.

    This routine will try to load the requested library from any of the
    platform default locations. Only the unversioned name is queried, assuming
    that this is an alias to the most recent version. If this fails then
    site-package directories are searched for both versioned and unversioned
    files. The library is returned upon first succes or ``None`` otherwise.

    Example
    -------

    To load the Intel MKL runtime library, write simply::

        libmkl = loadlib('mkl_rt')
    '''

    if sys.platform == 'linux':
        libsubdir = 'lib'
        libname = f'lib{name}.so'
        versioned = f'lib{name}.so.(\d+)'
    elif sys.platform == 'darwin':
        libsubdir = 'lib'
        libname = f'lib{name}.dylib'
        versioned = f'lib{name}.(\d+).dylib'
    elif sys.platform == 'win32':
        libsubdir = r'Library\bin'
        libname = f'{name}.dll'
        versioned = f'{name}.(\d+).dll'
    else:
        return

    try:
        return ctypes.CDLL(libname)
    except:
        pass

    for prefix in dict.fromkeys(site.PREFIXES): # stable deduplication
        if os.path.isdir(libdir := os.path.join(prefix, libsubdir)):
            if os.path.isfile(path := os.path.join(libdir, libname)):
                return ctypes.CDLL(path)
            if match := max(map(re.compile(versioned).fullmatch, os.listdir(libdir)), key=lambda m: int(m.group(1)) if m else -1, default=None):
                return ctypes.CDLL(os.path.join(libdir, match.group(0)))


def readtext(path):
    '''Read file and return contents

    Args
    ----
    path: :class:`os.PathLike`, :class:`str` or :class:`io.TextIOBase`
        Path-like or file-like object pointing to the data to be read.

    Returns
    -------
    :
        File data as :class:`str`.
    '''

    if isinstance(path, pathlib.Path):
        with path.open() as f:
            return f.read()

    if isinstance(path, str):
        with open(path) as f:
            return f.read()

    if isinstance(path, io.TextIOBase):
        return path.read()

    raise TypeError('readtext requires a path-like or file-like argument')


def binaryfile(path):
    '''Open file for binary reading

    Args
    ----
    path: :class:`os.PathLike`, :class:`str` or :class:`io.BufferedIOBase`
        Path-like or file-like object pointing to the data to be read.

    Returns
    -------
    :
        Context that returns a :class:`io.BufferedReader` upon entry.
    '''

    if isinstance(path, pathlib.Path):
        return path.open('rb')

    if isinstance(path, str):
        return open(path, 'rb')

    if isinstance(path, io.BufferedIOBase):
        return contextlib.nullcontext(path)

    raise TypeError('binaryfile requires a path-like or file-like argument')


def set_current(f):
    '''Decorator for setting global state.

    The decorator turns a function into a context that holds the return value
    in its ``.current`` attribute. All function arguments are required to have
    a default value, and the corresponding return value is the initial value of
    the ``.current`` attribute.

    Example:

    >>> @set_current
    ... def state(x=1, y=2):
    ...     return f'x={x}, y={y}'
    >>> state.current
    'x=1, y=2'
    >>> with state(10):
    ...     state.current
    'x=10, y=2'
    >>> state.current
    'x=1, y=2'
    '''

    @functools.wraps(f)
    @contextlib.contextmanager
    def set_current(*args, **kwargs):
        previous = set_current.current
        set_current.current = f(*args, **kwargs)
        try:
            yield
        finally:
            set_current.current = previous

    set_current.current = f()
    return set_current


def index(sequence, item):
    '''Index of first occurrence.

    Generalization of `tuple.index`.
    '''

    if isinstance(sequence, (list, tuple)):
        return sequence.index(item)
    for i, v in enumerate(sequence):
        if v == item:
            return i
    raise ValueError('index(sequence, item): item not in sequence')


def unique(items, key=None):
    '''Deduplicate items in sequence.

    Return a tuple `(unique, indices)` such that `items[i] == unique[indices[i]]`
    and `unique` does not contain duplicate items. An optional `key` is applied
    to all items before testing for equality.
    '''

    seen = {}
    unique = []
    indices = []
    for item in items:
        k = item if key is None else key(item)
        try:
            index = seen[k]
        except KeyError:
            index = seen[k] = len(unique)
            unique.append(item)
        indices.append(index)
    return unique, indices


def defaults_from_env(f):
    '''Decorator for changing function defaults based on environment.

    This decorator searches the environment for variables matching the pattern
    ``NUTILS_MYPARAM``, where ``myparam`` is a parameter of the decorated
    function. Only parameters with type annotation and a default value are
    considered, and the string value is deserialized using `Stringly
    <https://pypi.org/project/stringly/>`_. In case deserialization fails, a
    warning is emitted and the original default is maintained.'''

    sig = inspect.signature(f)
    params = []
    changed = False
    for param in sig.parameters.values():
        envname = f'NUTILS_{param.name.upper()}'
        if envname in os.environ and param.annotation != param.empty and param.default != param.empty:
            try:
                v = stringly.loads(param.annotation, os.environ[envname])
            except Exception as e:
                warnings.warn(f'ignoring environment variable {envname}: {e}')
            else:
                param = param.replace(default=v)
                changed = True
        params.append(param)
    if not changed:
        return f
    sig = sig.replace(parameters=params)
    @functools.wraps(f)
    def defaults_from_env(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return f(*bound.args, **bound.kwargs)
    defaults_from_env.__signature__ = sig
    return defaults_from_env


def format_timedelta(d):
    m, s = divmod(int(d.total_seconds()), 60)
    if m >= 60:
        m = '{}:{:02d}'.format(*divmod(m, 60))
    return f'{m}:{s:02d}'


@contextlib.contextmanager
def timeit():
    '''Context that logs begin time, end time, and duration.'''

    t0 = datetime.datetime.now()
    treelog.info(f'start {t0:%Y-%m-%d %H:%M:%S}')
    yield
    te = datetime.datetime.now()
    treelog.info(f'finish {te:%Y-%m-%d %H:%M:%S}, elapsed {format_timedelta(te-t0)}')


class elapsed:
    '''Timer that returns elapsed seconds as string representation.'''

    def __init__(self):
        self.t0 = datetime.datetime.now()

    def __str__(self):
        return format_timedelta(datetime.datetime.now() - self.t0)


class memory: # pragma: no cover
    '''Memory usage of the current process as string representation.'''

    def __init__(self):
        import psutil
        self.total = psutil.virtual_memory().total
        self.process = psutil.Process()

    def __str__(self):
        rss = self.process.memory_info().rss
        return f'{rss>>20:,}M ({100*rss/self.total:.0f}%)'


def in_context(context):
    '''Decorator to run a function in a context.

    Context arguments are added as position-only arguments to the signature of
    the decorated function. Any overlap between parameters of the function and
    context results in a ``ValueError``.'''

    params = []
    for param in inspect.signature(context).parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            param = param.replace(kind=param.KEYWORD_ONLY)
            # kind serves to make sure that we can always append parameters to the existing signature
        elif param.kind != param.KEYWORD_ONLY:
            raise Exception(f'context parameter {param.name!r} cannot be specified as keyword argument')
        params.append(param)

    def in_context_wrapper(f):

        @functools.wraps(f)
        def in_context(*args, **kwargs):
            with context(**{param.name: kwargs.pop(param.name) for param in params if param.name in kwargs}):
                return f(*args, **kwargs)

        sig = inspect.signature(f)
        in_context.__signature__ = sig.replace(parameters=(*sig.parameters.values(), *params))
        return in_context

    return in_context_wrapper


def log_version(f):

    from . import version, version_name

    @functools.wraps(f)
    def log_version(*args, **kwargs):
        treelog.info(f'NUTILS {version} "{version_name.title()}"')
        return f(*args, **kwargs)

    return log_version


def log_arguments(f):
    '''Decorator to log a function's arguments.

    The arguments are logged in the 'arguments' context. ``Stringly.loads``
    will be used whenever an argument supports it, transparently falling back
    on ``str`` otherwise.'''

    sig = inspect.signature(f)

    @functools.wraps(f)
    def log_arguments(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        with treelog.context('arguments'):
            for k, v in bound.arguments.items():
                try:
                    s = stringly.dumps(_infer_type(sig.parameters[k]), v)
                except:
                    s = str(v)
                treelog.info(f'{k}={s}')
        return f(*args, **kwargs)

    return log_arguments


@contextlib.contextmanager
@defaults_from_env
def post_mortem(pdb: bool = False): # pragma: no cover
    '''Context to activate post mortem debugging upon error.'''

    try:
        yield
    except Exception as e:
        if pdb:
            print(f'{type(e).__name__}: {e}')
            from pdb import post_mortem
            post_mortem()
        raise


@contextlib.contextmanager
@defaults_from_env
def log_traceback(gracefulexit: bool = True):
    '''Context to log traceback information to the active logger.

    Afterwards ``SystemExit`` is raised to avoid reprinting of the traceback by
    Python's default error handler.'''

    import traceback

    if not gracefulexit:
        yield
        return

    try:
        yield
    except SystemExit:
        raise
    except:
        exc = traceback.TracebackException(*sys.exc_info())
        prefix = ''
        while True:
            treelog.error(prefix + ''.join(exc.format_exception_only()).rstrip())
            treelog.debug('Traceback (most recent call first):\n' + ''.join(reversed(exc.stack.format())).rstrip())
            if exc.__cause__ is not None:
                exc = exc.__cause__
                prefix = '.. caused by '
            elif exc.__context__ is not None and not exc.__suppress_context__:
                exc = exc.__context__
                prefix = '.. while handling '
            else:
                break
        raise SystemExit(1)


@contextlib.contextmanager
def signal_handler(sig, handler):
    '''Context to temporarily replace a signal handler.

    The original handler is restored upon exit. A handler value of None
    disables the signal handler.'''

    import signal

    sig = signal.Signals[sig]
    oldhandler = signal.signal(sig, handler or signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(sig, oldhandler)


def trap_sigint(): # pragma: no cover
    '''Context to handle the SIGINT signal with a quit/continue/debug menu.'''

    @signal_handler('SIGINT', None)
    def handler(sig, frame):
        while True:
            answer = input('interrupted. quit, continue or start debugger? [q/c/d]')
            if answer == 'q':
                raise KeyboardInterrupt
            if answer == 'c' or answer == 'd':
                break
        if answer == 'd':  # after break, to minimize code after set_trace
            from pdb import Pdb
            pdb = Pdb()
            pdb.message('tracing activated; type "c" to continue normal execution.')
            pdb.set_trace(frame)

    return signal_handler('SIGINT', handler)


@defaults_from_env
def set_stdoutlog(richoutput: bool = sys.stdout.isatty(), verbose: int = 4): # pragma: no cover
    '''Context to replace the active logger with a StdoutLog or RichOutputLog.'''

    try:
        Level = treelog.proto.Level
    except AttributeError:  # treelog version < 1.0b6
        levels = 4, 3, 2, 1
    else:
        levels = Level.error, Level.warning, Level.user, Level.info

    stdoutlog = treelog.RichOutputLog() if richoutput else treelog.StdoutLog()
    if 0 <= verbose-1 < len(levels):
        stdoutlog = treelog.FilterLog(stdoutlog, minlevel=levels[verbose-1])

    return treelog.set(stdoutlog)


def name_of_main():
    '''Best-effort routine to establish the name of the running program.

    The name is constructed from the __main__ module. Since this module is
    loaded late, name_of_main cannot be used during module initialization.'''

    import __main__
    name = getattr(__main__, '__package__', None)
    if not name:
        path = getattr(__main__, '__file__', None)
        if not path:
            name = 'interactive'
        else:
            name = os.path.basename(__main__.__file__)
            if name.endswith('.py'):
                name = name[:-3]
    return name


@contextlib.contextmanager
@defaults_from_env
def add_htmllog(outrootdir: str = '~/public_html', outrooturi: str = '', scriptname: str = '', outdir: str = '', outuri: str = ''):
    '''Context to add a HtmlLog to the active logger.'''

    import html, base64

    if not scriptname and (not outdir or outrooturi and not outuri):
        scriptname = name_of_main()

    # the outdir argument exists for backwards compatibility; outrootdir
    # and scriptname are ignored if outdir is defined
    if outdir:
        outdir = pathlib.Path(outdir).expanduser()
    else:
        outdir = pathlib.Path(outrootdir).expanduser() / scriptname

    # the outuri argument exists for backwards compatibility; outrooturi is
    # ignored if outuri is defined
    if not outuri:
        outuri = outrooturi.rstrip('/') + '/' + scriptname if outrooturi else outdir.as_uri()

    nutils_logo = (
      '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" style="vertical-align: middle;" width="24" height="24" viewBox="-12 -12 24 24">'
        '<path d="M -9 3 v -6 a 6 6 0 0 1 12 0 v 6 M 9 -3 v 6 a 6 6 0 0 1 -12 0 v -6" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
      '</svg>')
    favicon = 'data:image/svg+xml;base64,' + base64.b64encode(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>' + nutils_logo.encode()).decode()
    htmltitle = '<a href="http://www.nutils.org">{}</a> {}'.format(nutils_logo, html.escape(scriptname))

    with treelog.HtmlLog(outdir, title=scriptname, htmltitle=htmltitle, favicon=favicon) as htmllog:
        loguri = outuri + '/' + htmllog.filename
        try:
            import bottombar
        except ImportError:
            treelog.info(f'writing log to: {loguri}')
            status = contextlib.nullcontext()
        else:
            status = bottombar.add(loguri, label='writing log to')
        try:
            with treelog.add(htmllog), status:
                yield
        except Exception as e:
            with treelog.set(htmllog):
                treelog.error(f'{e.__class__.__name__}: {e}')
            raise
        finally:
            treelog.info(f'log written to: {loguri}')


def _infer_type(param):
    '''Infer argument type from annotation or default value.'''

    if param.annotation is not param.empty:
        return param.annotation
    if param.default is not param.empty:
        return type(param.default)
    raise Exception(f'cannot determine type for argument {param.name!r}')


def cli(f, *, argv=None):
    '''Call a function using command line arguments.'''

    import textwrap

    progname, *args = argv or sys.argv
    doc = stringly.util.DocString(f)
    serializers = {}
    booleans = set()
    mandatory = set()

    for param in inspect.signature(f).parameters.values():
        T = _infer_type(param)
        try:
            s = stringly.serializer.get(T)
        except Exception as e:
            raise Exception(f'stringly cannot deserialize argument {param.name!r} of type {T}') from e
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise Exception(f'argument {param.name!r} is positional-only')
        if param.default is param.empty and param.name not in doc.defaults:
            mandatory.add(param.name)
        if T == bool:
            booleans.add(param.name)
        serializers[param.name] = s

    usage = [f'USAGE: {progname}']
    if doc.presets:
        usage.append(f'[{"|".join(doc.presets)}]')
    usage.extend(('{}' if arg in mandatory else '[{}]').format(f'{arg}={arg[0].upper()}') for arg in serializers)
    usage = '\n'.join(textwrap.wrap(' '.join(usage), subsequent_indent='  '))

    if '-h' in args or '--help' in args:
        help = [usage]
        if doc.text:
            help.append('')
            help.append(inspect.cleandoc(doc.text))
        if doc.argdocs:
            help.append('')
            for k, d in doc.argdocs.items():
                if k in serializers:
                    help.append(f'{k} (default: {doc.defaults[k]})' if k in doc.defaults else k)
                    help.extend(textwrap.wrap(doc.argdocs[k], initial_indent='    ', subsequent_indent='    '))
        sys.exit('\n'.join(help))

    kwargs = doc.defaults
    if args and args[0] in doc.presets:
        kwargs.update(doc.presets[args.pop(0)])
    for arg in args:
        name, sep, value = arg.partition('=')
        kwargs[name] = value if sep else 'yes' if name in booleans else None

    for name, s in kwargs.items():
        if name not in serializers:
            sys.exit(f'{usage}\n\nError: invalid argument {name!r}')
        if s is None:
            sys.exit(f'{usage}\n\nError: argument {name!r} requires a value')
        try:
            value = serializers[name].loads(s)
        except Exception as e:
            sys.exit(f'{usage}\n\nError: invalid value {s!r} for {name}: {e}')
        kwargs[name] = value

    for name in mandatory.difference(kwargs):
        sys.exit(f'{usage}\n\nError: missing argument {name}')

    return f(**kwargs)


def merge_index_map(nin: int, merge_sets: Iterable[Sequence[int]], condense: bool = True) -> Tuple[numpy.ndarray, int]:
    '''Returns an index map relating ``nin`` unmerged elements to ``nout`` merged elements.

    The index map is an array of length ``nin`` satisfying the condition that
    for every pair of indices ``i`` and ``j`` in a merge set, ``index_map[i] ==
    index_map[j]``. In code, the following is true:

            all(index_map[i] == index_map[j] for i, *js in merge_sets for j in js)

    If ``condense`` is true (the default) then the indices are remapped onto
    the smallest range [0,nout), and ``nout`` is returned along the the index
    map. In this case, selecting the first occurences of indices in
    ``index_map`` gives the sequence ``range(nout)``.

    Args
    ----
    nin : :class:`int`
        The number of elements before merging.
    merge_sets : iterable of sequences of at least one :class:`int`
        An iterable of merge sets, where each merge set lists the indices of
        input elements that should be merged. Every merge set should have at
        least one index.
    condense : :class:`bool`
        If true (default), then the returned indices form a permutation of the
        smallest possible range. Otherwise, precicely one index in every merged
        set maps onto itself.

    Returns
    -------
    index_map : :class:`numpy.ndarray`
        Index map with satisfying the above conditions.
    nout : :class:`int`
        The number of output indices.
    '''

    index_map = numpy.arange(nin)
    for merge_set in merge_sets:
        resolved = []
        for index in merge_set:
            while (parent := index_map[index]) != index:
                index = parent
            resolved.append(parent)
        index_map[resolved] = min(resolved)
    count = 0
    for iin, ptr in enumerate(index_map):
        if iin == ptr:
            if condense:
                index_map[iin] = count
            count += 1
        else:
            index_map[iin] = index_map[ptr]
    return index_map, count


def nutils_dispatch(f):
    '''Decorator for nutils-dispatching based on argument types.'''

    sig = inspect.signature(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        seen = set()
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for arg in bound.args:
            T = type(arg)
            if hasattr(T, '__nutils_dispatch__') and T not in seen:
                retval = T.__nutils_dispatch__(wrapper, bound.args, bound.kwargs)
                if retval is not NotImplemented:
                    return retval
                seen.add(T)
        return f(*args, **kwargs)

    return wrapper


class IDSetView:

    def __init__(self, init=()):
        self._dict = init._dict if isinstance(init, IDSetView) else {id(obj): obj for obj in init}

    def __len__(self):
        return len(self._dict)

    def __bool__(self):
        return bool(self._dict)

    def __iter__(self):
        return iter(self._dict.values())

    def __and__(self, other):
        return self.copy().__iand__(other)

    def __or__(self, other):
        return self.copy().__ior__(other)

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def isdisjoint(self, other):
        return self._dict.keys().isdisjoint(IDSetView(other)._dict)

    def intersection(self, other):
        return self.__and__(IDSetView(other))

    def difference(self, other):
        return self.__sub__(IDSetView(other))

    def union(self, other):
        return self.__or__(IDSetView(other))

    def __repr__(self):
        return '{' + ', '.join(map(repr, self)) + '}'

    def copy(self):
        return IDSet(self)


class IDSet(IDSetView):

    def __init__(self, init=()):
        self._dict = init._dict.copy() if isinstance(init, IDSetView) else {id(obj): obj for obj in init}

    def __iand__(self, other):
        if not isinstance(other, IDSetView):
            return NotImplemented
        if not other._dict:
            self._dict.clear()
        elif self._dict:
            for k in set(self._dict) - set(other._dict):
                del self._dict[k]
        return self

    def __ior__(self, other):
        if not isinstance(other, IDSetView):
            return NotImplemented
        self._dict.update(other._dict)
        return self

    def __isub__(self, other):
        if not isinstance(other, IDSetView):
            return NotImplemented
        for k in other._dict:
            self._dict.pop(k, None)
        return self

    def add(self, obj):
        self._dict[id(obj)] = obj

    def pop(self):
        return self._dict.popitem()[1]

    def intersection_update(self, other):
        self.__iand__(IDSetView(other))

    def difference_update(self, other):
        self.__isub__(IDSetView(other))

    def update(self, other):
        self.__ior__(IDSetView(other))

    def view(self):
        return IDSetView(self)


class IDDict:
    '''Mapping from instance (is, not ==) to value. Keys need not be hashable.'''

    def __init__(self):
        self.__dict = {}

    def __setitem__(self, key, value):
        self.__dict[id(key)] = key, value

    def __getitem__(self, key):
        key_, value = self.__dict[id(key)]
        assert key_ is key
        return value

    def get(self, key, default=None):
        kv = self.__dict.get(id(key))
        if kv is None:
            return default
        key_, value = kv
        assert key_ is key
        return value

    def __delitem__(self, key):
        del self.__dict[id(key)]

    def __len__(self):
        return len(self.__dict)

    def keys(self):
        return (key for key, value in self.__dict.values())

    def values(self):
        return (value for key, value in self.__dict.values())

    def items(self):
        return self.__dict.values()

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        return self.__dict.__contains__(id(key))

    def __repr__(self):
        return '{' + ', '.join(f'{k!r}: {v!r}' for k, v in self.items()) + '}'


def _tuple(*args):
    return args

_container_types = frozenset({tuple, list, dict, set, frozenset})
_terminal_types = frozenset({type(None), bool, int, float, complex, str, bytes, arraydata})

def _reduce(obj):
    'helper function for deep_replace_property and shallow_replace'

    T = type(obj)
    if T in _container_types:
        if not obj: # empty containers need not be entered
            return
        elif T is tuple:
            return _tuple, obj
        elif T is dict:
            return T, (tuple(obj.items()),)
        else:
            return T, (tuple(obj),)
    if T in _terminal_types:
        return
    try:
        f, args = obj.__reduce__()
    except:
        return
    else:
        return f, args


class deep_replace_property:
    '''decorator for deep object replacement

    Generates a cached property for deep replacement of reduceable objects,
    based on a callable that is applied depth first and recursively on
    individual constructor arguments. Intermediate values are stored in the
    attribute by the same name of any object that is a descendent of the class
    that owns the property.

    Args
    ----
    func
        Callable which maps an object onto a new object, or onto itself if no
        replacement is made. It must have precisely one positional argument for
        the object.
    '''

    identity = object()
    recreate = collections.namedtuple('recreate', ['f', 'nargs'])

    def __init__(self, func):
        self.func = func

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def __set__(self, obj, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, obj):
        raise AttributeError("can't delete attribute")

    def __get__(self, obj, objtype=None):
        fstack = [obj] # stack of unprocessed objects and command tokens
        rstack = [] # stack of processed objects
        ostack = IDSet() # stack of original objects to cache new value into

        while fstack:
            obj = fstack.pop()

            if isinstance(obj, self.recreate): # recreate object from rstack
                f, nargs = obj
                r = f(*[rstack.pop() for _ in range(nargs)])
                if isinstance(r, self.owner) and (newr := self.func(r)) is not r:
                    fstack.append(newr) # recursion
                else:
                    rstack.append(r)

            elif obj is ostack: # store new representation
                orig = ostack.pop()
                r = rstack[-1]
                if r is orig: # this may happen if obj is memoizing
                    r = self.identity # prevent cyclic reference
                orig.__dict__[self.name] = r

            elif isinstance(obj, self.owner):
                if (r := obj.__dict__.get(self.name)) is not None: # in cache
                    rstack.append(r if r is not self.identity else obj)
                elif obj in ostack:
                    raise Exception(f'{type(obj).__name__}.{self.name} is caught in a loop')
                else:
                    ostack.add(obj)
                    fstack.append(ostack)
                    f, args = obj.__reduce__()
                    fstack.append(self.recreate(f, len(args)))
                    fstack.extend(args)

            elif reduced := _reduce(obj):
                f, args = reduced
                fstack.append(self.recreate(f, len(args)))
                fstack.extend(args)

            else:
                rstack.append(obj)

        assert not ostack
        assert len(rstack) == 1
        return rstack[0]


def shallow_replace(func, *funcargs, **funckwargs):
    '''decorator for deep object replacement

    Generates a deep replacement method for reduceable objects based on a
    callable that is applied on individual constructor arguments. The
    replacement takes a shallow first approach and stops as soon as the
    callable returns a value that is not ``None``. Intermediate values are
    flushed upon return.

    Args
    ----
    func
        Callable which maps an object onto a new object, or ``None`` if no
        replacement is made. It must have one positional argument for the object,
        and may have any number of additional positional and/or keyword
        arguments.

    Returns
    -------
    :any:`callable`
        The method that searches the object to perform the replacements.
    '''

    if not funcargs and not funckwargs: # decorator
        # it would be nice to use partial here but then the decorator doesn't work with methods
        return functools.wraps(func)(lambda *args, **kwargs: shallow_replace(func, *args, **kwargs))

    target, *funcargs = funcargs
    recreate = collections.namedtuple('recreate', ['f', 'nargs', 'orig'])

    fstack = [target] # stack of unprocessed objects and command tokens
    rstack = [] # stack of processed objects
    cache = funckwargs.pop('__persistent_cache__', IDDict()) # cache of seen objects

    while fstack:
        obj = fstack.pop()

        if isinstance(obj, recreate):
            f, nargs, orig = obj
            r = f(*[rstack.pop() for _ in range(nargs)])
            cache[orig] = r
            rstack.append(r)

        elif (r := cache.get(obj)) is not None:
            rstack.append(r)

        elif (r := func(obj, *funcargs, **funckwargs)) is not None:
            cache[obj] = r
            rstack.append(r)

        elif reduced := _reduce(obj):
            f, args = reduced
            fstack.append(recreate(f, len(args), obj))
            fstack.extend(args)

        else: # obj cannot be reduced
            rstack.append(obj)

    assert len(rstack) == 1
    return rstack[0]


def tree_walk(visit_node, /, *roots):
    '''calls ``visit_node`` for every node in ``roots``

    The callable ``visit_node`` takes one node as argument and should return an
    iterator over the child nodes to visit. Every identical node is visited
    only once.

    Examples
    --------

    >>> def visit(node):
    ...     if isinstance(node, list):
    ...         return node
    ...     else:
    ...         print(node)
    ...         return ()
    >>> tree_walk(visit, [1, [2, 3]])
    3
    2
    1

    Identical nodes, ``a`` in the following example, are visited once:

    >>> a = [1, 2]
    >>> tree_walk(visit, [a, a])
    2
    1
    '''

    stack = []
    seen = set()
    for root in roots:
        if id(root) not in seen:
            stack.append(root)
            seen.add(id(root))
    while stack:
        node = stack.pop()
        for dep in visit_node(node):
            if id(dep) not in seen:
                stack.append(dep)
                seen.add(id(dep))


def untake(indices, items=None):
    '''shuffle items (default: range(len(indices))) into tuple untake, such
    that untake[indices[i]] == items[i] for all 0 <= i < len(items).'''

    if items is None:
        items_indices = enumerate(indices)
        nil = None
    else:
        assert len(items) == len(indices), 'items and indices do not match'
        items_indices = zip(items, indices)
        nil = object()
    untake = [nil] * len(indices)
    for item, index in items_indices:
        assert untake[index] is nil, f'index {index} occurs twice in indices'
        untake[index] = item
    return tuple(untake)


class abstract_property:
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, obj, objtype=None):
        raise NotImplementedError(f'class {obj.__class__.__name__} fails to implement {self.name}')


# vim:sw=4:sts=4:et
