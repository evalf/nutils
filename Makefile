nutils:
	${MAKE} -C nutils

docs:
	${MAKE} -C docs html

all: nutils docs

clean:
	${MAKE} -C docs clean
	${MAKE} -C nutils clean

.PHONY: all docs nutils clean

# vim:noexpandtab
