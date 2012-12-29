MODULE=_numeric

all: $(MODULE).so $(MODULE)_d.so clean

$(MODULE).so: $(MODULE).c
	python setup.py build
	cp build/lib*/$(MODULE).so .

$(MODULE)_d.so: $(MODULE).c
	python-dbg setup.py build
	cp build/lib*/$(MODULE)_d.so .

clean:
	rm -r -f build

.PHONY: all clean

# vim:noexpandtab
