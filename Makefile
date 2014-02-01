_numeric.so: _numeric.c
	python setup.py build
	cp build/lib*/$@ .
	rm -r -f build

_numeric_d.so: _numeric.c
	python-dbg setup.py build
	cp build/lib*/$@ .
	rm -r -f build

test: test_py test_c

test_c: _numeric.so
	python test.py

test_py: clean
	python test.py

clean:
	rm -r -f build _numeric.so _numeric_d.so

.PHONY: clean test test_c test_py

# vim:noexpandtab
