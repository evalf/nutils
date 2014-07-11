build:
	python setup.py build clean

dist:
	python setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

dev: build
	cp build/lib*/nutils/_numeric.so nutils

test: test_nose test_examples

test_nose: dev
	nosetests tests

test_examples: dev
	for script in examples/*; do \
		python $$script unittest --tbexplore=False; \
	done

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc nutils/_numeric.so
	${MAKE} -C docs clean

.PHONY: build dist docs dev test test_nose test_examples clean

# vim:noexpandtab
