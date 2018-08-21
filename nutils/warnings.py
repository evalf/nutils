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

import warnings

class NutilsWarning(Warning):
  pass

class NutilsDeprecationWarning(NutilsWarning):
  pass

def warn(message, category=NutilsWarning, stacklevel=1):
  warnings.warn(message, category, stacklevel=stacklevel)

def deprecation(message):
  warnings.warn(message, NutilsDeprecationWarning, stacklevel=2)

class via:
  '''context manager to set/reset warnings.showwarning'''

  def __init__(self, print):
    self.print = print

  def __enter__(self):
    self.oldshowwarning = warnings.showwarning
    warnings.showwarning = self.showwarning

  def __exit__(self, *args):
    warnings.showwarning = self.oldshowwarning

  def showwarning(self, message, category, filename, lineno, *args):
    self.print('{}: {}\n  In {}:{}'.format(category.__name__, message, filename, lineno))

# vim:sw=2:sts=2:et
