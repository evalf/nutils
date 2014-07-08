build:
	python setup.py build clean

dist:
	python setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

dev: build
	cp build/lib*/nutils/_numeric.so nutils

test: dev
	nosetests tests

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc nutils/_numeric.so
	${MAKE} -C docs clean

.PHONY: build dist docs dev test clean

# vim:noexpandtab
