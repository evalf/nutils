dev: build
	cp -vu build/lib*/nutils/_numeric.so nutils

build:
	python setup.py build clean

dist:
	python setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

test: test_nose test_examples

test_nose: dev
	nosetests -v tests

test_examples: dev
	@for script in examples/*; do \
		echo $$script; \
		python $$script unittest --tbexplore=False --verbose=3; \
	done

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc nutils/_numeric.so
	${MAKE} -C docs clean

.PHONY: build dist docs dev test test_nose test_examples clean

# vim:noexpandtab
