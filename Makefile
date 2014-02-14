all: numeric

numeric:
	$(MAKE) -C nutils/numeric test_c

test:
	python tests/test.py

clean:
	$(MAKE) -C nutils/numeric clean

.PHONY: all numeric test clean

# vim:noexpandtab
