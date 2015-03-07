build:
	python setup.py build clean

dist:
	python setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

test: test_unit test_examples

test_unit:
	python -m tests

test_examples:
	@for script in examples/*; do \
		echo $$script; \
		python $$script unittest --tbexplore=False --verbose=3; \
	done

test3: test3_unit test3_examples

test3_unit:
	python3 -m tests

test3_examples:
	@for script in examples/*; do \
		echo $$script; \
		python3 $$script unittest --tbexplore=False --verbose=3; \
	done

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc
	${MAKE} -C docs clean

.PHONY: build dist docs test test_unit test_examples clean

# vim:noexpandtab
