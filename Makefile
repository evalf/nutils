build:
	python setup.py build clean

dist:
	python setup.py sdist
	rm -rf nutils.egg-info

docs:
	${MAKE} -C docs html

test: test_nose test_examples

test_nose:
	nosetests -v tests

test_examples:
	@for script in examples/*; do \
		echo $$script; \
		python $$script unittest --tbexplore=False --verbose=3; \
	done

test3: test3_nose test3_examples

test3_nose:
	nosetests3 -v tests

test3_examples:
	@for script in examples/*; do \
		echo $$script; \
		python3 $$script unittest --tbexplore=False --verbose=3; \
	done

clean:
	rm -fr build dist
	rm -f MANIFEST nutils/*.pyc
	${MAKE} -C docs clean

.PHONY: build dist docs test test_nose test_examples clean

# vim:noexpandtab
