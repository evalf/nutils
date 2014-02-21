all: numeric

numeric:
	git submodule init nutils/numeric
	git submodule update nutils/numeric
	$(MAKE) -C nutils/numeric test_c

test:
	python tests/test.py

clean:
	$(MAKE) -C nutils/numeric clean
	rm -f nutils/*.pyc tests/*.pyc

.PHONY: all numeric test clean

# vim:noexpandtab
