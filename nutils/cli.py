"""
The cli (command line interface) module provides the `cli.run` function that
can be used set up properties, initiate an output environment, and execute a
python function based arguments specified on the command line.
"""


def run(f, *, argv=None):
    '''Command line interface for a single function.'''

    import treelog, bottombar
    from . import _util as util, matrix, parallel, cache, warnings

    decorators = (
        util.trap_sigint(),
        bottombar.add(util.timer(), label='runtime', right=True, refresh=1), \
        bottombar.add(util.memory(), label='memory', right=True, refresh=1), \
        util.in_context(cache.caching),
        util.in_context(parallel.maxprocs),
        util.in_context(matrix.backend),
        util.in_context(util.set_stdoutlog),
        util.in_context(util.add_htmllog),
        util.in_context(util.log_traceback),
        util.in_context(util.post_mortem),
        warnings.via(treelog.warning),
        util.log_arguments,
        util.timeit(),
    )

    for decorator in reversed(decorators):
        f = decorator(f)

    return util.cli(f, argv=argv)


def choose(*functions, argv=None):
    '''Command line interface for multiple functions.'''

    import sys

    progname, *args = argv or sys.argv
    fmap = {f.__name__: f for f in functions}
    try:
        choice = args.pop(0)
        f = fmap[choice]
    except:
        sys.exit(f'USAGE: {progname} {"|".join(fmap)} [...]')
    else:
        return run(f, argv=(f'{progname} {choice}', *args))


# vim:sw=4:sts=4:et
