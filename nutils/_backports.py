"""
The backports module provides minimal fallback implementations for functions
that are introduced in versions of Python that are newer than the minimum
required version. Function behavour is equal to its Python counterpart only to
the extent that the Nutils code base requires it to be. As such,
implementations found in this module should not be relied upon as general
drop-on replacements.
"""


if False: # awaiting introduction

    from doctest import DocTestFinder

else:

    # This is a modified version of doctest.DocTestFinder to fix issue
    # https://github.com/python/cpython/issues/107715, which prevents doctest
    # operation for the SI module. The modification assumes that `find` relies
    # on the internal `_find_lineno` method.

    import doctest, inspect, re

    class DocTestFinder(doctest.DocTestFinder):

        def _find_lineno(self, obj, source_lines):
            """
            Return a line number of the given object's docstring.  Note:
            this method assumes that the object has a docstring.
            """
            lineno = None

            # Find the line number for modules.
            if inspect.ismodule(obj):
                lineno = 0

            # Find the line number for classes.
            # Note: this could be fooled if a class is defined multiple
            # times in a single file.
            if inspect.isclass(obj):
                if source_lines is None:
                    return None
                pat = re.compile(r'^\s*class\s*%s\b' %
                                 re.escape(getattr(obj, '__name__', '-')))
                for i, line in enumerate(source_lines):
                    if pat.match(line):
                        lineno = i
                        break

            # Find the line number for functions & methods.
            if inspect.ismethod(obj): obj = obj.__func__
            if inspect.isfunction(obj): obj = obj.__code__
            if inspect.istraceback(obj): obj = obj.tb_frame
            if inspect.isframe(obj): obj = obj.f_code
            if inspect.iscode(obj):
                lineno = getattr(obj, 'co_firstlineno', None)-1

            # Find the line number where the docstring starts.  Assume
            # that it's the first line that begins with a quote mark.
            # Note: this could be fooled by a multiline function
            # signature, where a continuation line begins with a quote
            # mark.
            if lineno is not None:
                if source_lines is None:
                    return lineno+1
                pat = re.compile(r'(^|.*:)\s*\w*("|\')')
                for lineno in range(lineno, len(source_lines)):
                    if pat.match(source_lines[lineno]):
                        return lineno

            # We couldn't find the line number.
            return None
