PYTHON?=python3

build:
	${PYTHON} setup.py build clean

dist:
	${PYTHON} setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

test:
	${PYTHON} -m unittest -b

coverage:
	${PYTHON} -m coverage erase
	$(MAKE) test "PYTHON=${PYTHON} -m coverage run -a"

htmlcov: coverage
	rm -rf htmlcov
	${PYTHON} -m coverage html

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc
	${MAKE} -C docs clean

.PHONY: build dist docs test coverage htmlcov clean

# vim:noexpandtab
