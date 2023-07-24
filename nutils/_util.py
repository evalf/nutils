"""
The util module provides a collection of general purpose methods.
"""

from . import numeric, warnings
import stringly
import sys
import os
import numpy
import collections.abc
import inspect
import itertools
import functools
import operator
import numbers
import pathlib
import ctypes
import io
import contextlib
import treelog
import datetime
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

    def __call__(self, *args, **kwargs):
        if not args:
            raise TypeError('{} requires at least 1 positional argument'.format(self.__wrapped__.__name__))
        ismultiple = isinstance(args[0], (list, tuple, map))
        retvals = tuple(self.__wrapped__(tuple(args[0]) if ismultiple else args[:1], *args[1:], **kwargs))
        if not ismultiple:
            retvals, = retvals
        return retvals


class positional_only:
    '''Change all positional-or-keyword arguments to positional-only.

    Python introduces syntax to define positional-only parameters in version 3.8,
    but the same effect can be achieved in older versions by using a wrapper with
    a var-positional argument. The :func:`positional_only` decorator uses this
    technique to treat all positional-or-keyword arguments as positional-only. In
    order to avoid name clashes between the positional-only arguments and
    variable keyword arguments, the wrapper additionally introduces the
    convention that the last argument receives the variable keyword argument
    dictionary in case is has a default value of ... (ellipsis).

    Example:

    >>> @positional_only
    ... def f(x, *, y):
    ...   pass
    >>> inspect.signature(f)
    <Signature (x, /, *, y)>

    >>> @positional_only
    ... def f(x, *args, y, kwargs=...):
    ...   pass
    >>> inspect.signature(f)
    <Signature (x, /, *args, y, **kwargs)>

    Args
    ----
    f : :any:`callable`
        Function to be wrapped.
    '''

    def __init__(self, f):
        signature = inspect.signature(f)
        parameters = list(signature.parameters.values())
        keywords = []
        varkw = None
        for i, param in enumerate(parameters):
            if param.kind is param.VAR_KEYWORD:
                raise Exception('positional_only decorated function must use ellipses to mark a variable keyword argument')
            if i == len(parameters)-1 and param.default is ...:
                parameters[i] = param.replace(kind=inspect.Parameter.VAR_KEYWORD, default=inspect.Parameter.empty)
                varkw = param.name
            elif param.kind is param.POSITIONAL_OR_KEYWORD:
                parameters[i] = param.replace(kind=param.POSITIONAL_ONLY)
            elif param.kind is param.KEYWORD_ONLY:
                keywords.append(param.name)
        self.__keywords = tuple(keywords)
        self.__varkw = varkw
        self.__signature__ = signature.replace(parameters=parameters)
        functools.update_wrapper(self, f)

    def __get__(self, instance, owner):
        return positional_only(self.__wrapped__.__get__(instance, owner))

    def __call__(self, *args, **kwargs):
        wrappedkwargs = {name: kwargs.pop(name) for name in self.__keywords if name in kwargs}
        if self.__varkw:
            wrappedkwargs[self.__varkw] = kwargs
        elif kwargs:
            raise TypeError('{}() got an unexpected keyword argument {!r}'.format(self.__wrapped__.__name__, *kwargs))
        return self.__wrapped__(*args, **wrappedkwargs)


def loadlib(**libname):
    '''
    Find and load a dynamic library using :any:`ctypes.CDLL`.  For each
    (supported) platform the name of the library should be specified as a keyword
    argument, including the extension, where the keywords should match the
    possible values of :any:`sys.platform`.

    Example
    -------

    To load the Intel MKL runtime library, write::

        loadlib(linux='libmkl_rt.so', darwin='libmkl_rt.dylib', win32='mkl_rt.dll')
    '''

    try:
        return ctypes.CDLL(libname[sys.platform])
    except (OSError, KeyError):
        pass


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


class timer:
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
                    s = stringly.dumps(sig.parameters[k].annotation, v)
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

    import html, base64, bottombar

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
            with treelog.add(htmllog), bottombar.add(loguri, label='writing log to'):
                yield
        except Exception as e:
            with treelog.set(htmllog):
                treelog.error(f'{e.__class__.__name__}: {e}')
            raise
        finally:
            treelog.info(f'log written to: {loguri}')


def cli(f, *, argv=None):
    '''Call a function using command line arguments.'''

    import textwrap

    progname, *args = argv or sys.argv
    doc = stringly.util.DocString(f)
    serializers = {}
    booleans = set()
    mandatory = set()

    for param in inspect.signature(f).parameters.values():
        T = param.annotation
        if T == param.empty and param.default != param.empty:
            T = type(param.default)
        if T == param.empty:
            raise Exception(f'cannot determine type for argument {param.name!r}')
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


def merge_index_map(nin: int, merge_sets: Iterable[Sequence[int]]) -> Tuple[numpy.ndarray, int]:
    '''Returns an index map relating ``nin`` unmerged elements to ``nout`` merged elements.

    The index map, an array of length ``nin``, satisfies the following conditions:

    *   For every merge set in ``merge_sets``: for every pair of indices ``i``
        and ``j`` in a merge set, ``index_map[i] == index_map[j]`` is true.

        In code, the following is true:

            all(index_map[i] == index_map[j] for i, *js in merge_sets for j in js)

    *   Selecting the first occurences of indices in ``index_map`` gives the
        sequence ``range(nout)``.

    Args
    ----
    nin : :class:`int`
        The number of elements before merging.
    merge_sets : iterable of sequences of at least one :class:`int`
        An iterable of merge sets, where each merge set lists the indices of
        input elements that should be merged. Every merge set should have at
        least one index.

    Returns
    -------
    index_map : :class:`numpy.ndarray`
        Index map with satisfying the above conditions.
    nout : :class:`int`
        The number of output indices.
    '''

    index_map = numpy.arange(nin)
    def resolve(index):
        parent = index_map[index]
        while index != parent:
            index = parent
            parent = index_map[index]
        return index
    for merge_set in merge_sets:
        resolved = list(map(resolve, merge_set))
        index_map[resolved] = min(resolved)
    new_indices = itertools.count()
    for iin, ptr in enumerate(index_map):
        index_map[iin] = next(new_indices) if iin == ptr else index_map[ptr]
    return index_map, next(new_indices)


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


# vim:sw=4:sts=4:et
