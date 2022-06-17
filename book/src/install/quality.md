# Quality of Life

Here we list some modules that are not direct requirements, but that can be
used in conjunction with Nutils to make life a little bit better.

## BottomBar

[BottomBar](https://github.com/evalf/bottombar) is a context manager for Python
that prints a status line at the bottom of a terminal window. When it is
installed, `cli.run` automatically activates it to display the location of the
html log (rather than only logging it at the beginning and end of the
simulation) as well as runtime and memory usage information.

```sh
python -m pip install bottombar
```
