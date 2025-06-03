import contextlib
from dataclasses import dataclass, field
import itertools
import time
import treelog, treelog._io

@dataclass
class Context:
    title: str
    start: float = field(default_factory=time.time)

    def __str__(self):
        return self.title

    @property
    def duration(self):
        return time.time() - self.start

class ContextDuration:

    def __init__(self):
        self._contexts = []

    def pushcontext(self, title):
        self._contexts.append(Context(title))

    def popcontext(self):
        context = self._contexts.pop()
        contexts = ' > '.join(map(str, itertools.chain(self._contexts, [context])))
        print(f'{contexts} [duration: {context.duration:.1f}s]')

    def recontext(self, title: str):
        self._contexts[-1].title = title

    def write(self, text, level):
        print(' > '.join(itertools.chain(map(str, self._contexts), [text])))

    @contextlib.contextmanager
    def open(self, filename, mode, level):
        with treelog._io.devnull(mode) as f:
            yield f
        self.write(filename, level=level)

if __name__ == '__main__':
    import runpy
    import sys
    import nutils._util
    glob = runpy.run_path(sys.argv[1])
    with treelog.set(ContextDuration()):
        nutils._util.cli(glob[sys.argv[2]], argv=sys.argv[2:])
