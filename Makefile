PYTHON?=python3
COVERAGE?=python3-coverage

EXAMPLES = $(wildcard examples/*)

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

test_examples: $(EXAMPLES)

$(EXAMPLES):
	${PYTHON} $@ unittest --tbexplore=False --verbose=3

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

.PHONY: build dist docs test test_unit test_examples $(EXAMPLES) coverage htmlcov clean

# vim:noexpandtab
