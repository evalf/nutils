# Copyright (c) 2014 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from . import config, log, core, util
import contextlib

@contextlib.contextmanager
@util.positional_only('name')
def mplfigure(*args, **kwargs):
  '''Matplotlib figure context, convenience function.

  Returns a :class:`matplotlib.figure.Figure` object suitable for
  `object-oriented plotting`_. Upon exit the result is saved using the
  agg-backend in all formats configured via :attr:`nutils.config.imagetype`,
  and the resulting filenames written to log.

  .. _`object-oriented plotting`: https://matplotlib.org/gallery/api/agg_oo_sgskip.html

  Args
  ----
  name : :class:`str`
      The filename (without extension) of the resulting figure(s)
  **kwargs :
      Keyword arguments are passed on unchanged to the constructor of the
      :class:`matplotlib.figure.Figure` object.
  '''

  name, = args
  formats = config.imagetype.split(',')
  with log.context(name):
    import matplotlib.backends.backend_agg
    fig = matplotlib.figure.Figure(**kwargs)
    matplotlib.backends.backend_agg.FigureCanvas(fig) # sets reference via fig.set_canvas
    yield fig
    for fmt in formats:
      with log.context(fmt), core.open_in_outdir(name+'.'+fmt, 'wb') as f:
        fig.savefig(f, format=fmt)
    fig.set_canvas(None) # break circular reference
  log.user(', '.join(name+'.'+fmt for fmt in formats))

# vim:sw=2:sts=2:et
