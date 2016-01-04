PYTHON?=python
COVERAGE?=python-coverage

build:
	${PYTHON} setup.py build clean

dist:
	${PYTHON} setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

test: test_unit test_examples

test_unit:
	${PYTHON} -m tests

test_examples:
	@for script in examples/*; do \
		echo $$script; \
		${PYTHON} $$script unittest --tbexplore=False --verbose=3; \
	done

test3: test3_unit test3_examples

test3_unit:
	python3 -m tests

test3_examples:
	@for script in examples/*; do \
		echo $$script; \
		python3 $$script unittest --tbexplore=False --verbose=3; \
	done

coverage:
	${COVERAGE} erase
	$(MAKE) test "PYTHON=${COVERAGE} run -a"

htmlcov: coverage
	rm -rf htmlcov
	${COVERAGE} html

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc
	${MAKE} -C docs clean

.PHONY: build dist docs test test_unit test_examples coverage htmlcov clean

# vim:noexpandtab
